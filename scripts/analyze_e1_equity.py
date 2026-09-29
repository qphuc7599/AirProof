from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP_COLUMNS = [f"group_{index}_rmse" for index in range(4)]
E1_KEYS = ["availability", "outage_median_hours", "outage_p95_hours", "participation_skew"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario_name(row: pd.Series) -> str:
    if float(row["availability"]) == 0.75:
        return "anchor"
    if float(row["participation_skew"]) == 1.0:
        return "outage"
    return "severe"


def _e3_scenario_name(row: pd.Series) -> str:
    return "nominal_clean" if row["attack_kind"] == "clean" else "severe_drift20"


def _bootstrap_mean(values: np.ndarray, *, seed: int, replicates: int = 10_000) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 1_000):
        stop = min(start + 1_000, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "bootstrap_95_ci": [float(low), float(high)],
        "bootstrap_replicates": replicates,
        "n_paired_worlds": len(values),
    }


def _paired_contrast(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    *,
    seed: int,
) -> dict[str, Any]:
    pivot = frame.pivot(index="seed", columns="method", values=metric).dropna()
    delta = pivot[reference].to_numpy() - pivot[comparator].to_numpy()
    relative = delta / np.maximum(np.abs(pivot[comparator].to_numpy()), 1e-12)
    return {
        "reference": reference,
        "comparator": comparator,
        "metric": metric,
        "absolute_difference": _bootstrap_mean(delta, seed=seed),
        "relative_difference": _bootstrap_mean(relative, seed=seed + 1),
    }


def build_report(e1_path: Path, e3_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    e1 = pd.read_parquet(e1_path)
    e3 = pd.read_parquet(e3_path)
    e1["scenario"] = e1.apply(_scenario_name, axis=1)
    e3["scenario"] = e3.apply(_e3_scenario_name, axis=1)

    means = (
        e1.groupby(["scenario", "method"], sort=True)[
            ["rmse", "worst_group_rmse", "coverage_gap", *GROUP_COLUMNS]
        ]
        .mean()
        .reset_index()
    )
    contrasts: list[dict[str, Any]] = []
    contrast_seed = 20260902
    for scenario, subset in e1.groupby("scenario", sort=True):
        for comparator in ("air_quality_dt", "opportunistic_ttl"):
            for metric in ("rmse", "worst_group_rmse", "coverage_gap", *GROUP_COLUMNS):
                item = _paired_contrast(
                    subset,
                    "airproof",
                    comparator,
                    metric,
                    seed=contrast_seed,
                )
                contrasts.append({"scenario": scenario, **item})
                contrast_seed += 2

    pof_rows: list[dict[str, Any]] = []
    for scenario, scenario_frame in e3.groupby("scenario", sort=True):
        for relay in (0, 1):
            for huber in (0, 1):
                unfair = f"factorial_0{relay}{huber}"
                fair = f"factorial_1{relay}{huber}"
                pair = (
                    scenario_frame[scenario_frame["method"].isin([unfair, fair])]
                    .pivot(
                        index="seed",
                        columns="method",
                        values=["rmse", "worst_group_rmse", "coverage_gap"],
                    )
                    .dropna()
                )
                rmse_pof = (
                    pair[("rmse", fair)].to_numpy() - pair[("rmse", unfair)].to_numpy()
                ) / pair[("rmse", unfair)].to_numpy()
                worst_pof = (
                    pair[("worst_group_rmse", fair)].to_numpy()
                    - pair[("worst_group_rmse", unfair)].to_numpy()
                ) / pair[("worst_group_rmse", unfair)].to_numpy()
                coverage_gain = (
                    pair[("coverage_gap", unfair)].to_numpy()
                    - pair[("coverage_gap", fair)].to_numpy()
                ) / np.maximum(pair[("coverage_gap", unfair)].to_numpy(), 1e-12)
                pof_rows.append(
                    {
                        "scenario": scenario,
                        "relay": relay,
                        "huber": huber,
                        "fair_method": fair,
                        "unconstrained_method": unfair,
                        "price_of_fairness_rmse": _bootstrap_mean(
                            rmse_pof,
                            seed=contrast_seed,
                        ),
                        "price_of_fairness_worst_group": _bootstrap_mean(
                            worst_pof,
                            seed=contrast_seed + 1,
                        ),
                        "relative_coverage_gap_reduction": _bootstrap_mean(
                            coverage_gain,
                            seed=contrast_seed + 2,
                        ),
                    }
                )
                contrast_seed += 3

    severe = means[means["scenario"] == "severe"].set_index("method")
    severe_gate = all(
        severe.loc["airproof", "worst_group_rmse"] < severe.loc[method, "worst_group_rmse"]
        for method in ("air_quality_dt", "opportunistic_ttl")
    )
    full_stack = [row for row in pof_rows if row["relay"] == 1 and row["huber"] == 1]
    report = {
        "status": "measured-superseded-v2-diagnostic-not-v3-primary",
        "interpretation_boundary": (
            "The frozen E1 rows evaluate the predecessor fixed-Huber architecture. "
            "They diagnose group behavior and size the v3 equity protocol; they must not be "
            "presented as primary evidence for predictive-residual AirProof v3."
        ),
        "inputs": {
            str(e1_path).replace("\\", "/"): _sha256(e1_path),
            str(e3_path).replace("\\", "/"): _sha256(e3_path),
        },
        "e1_rows": len(e1),
        "e3_rows": len(e3),
        "e1_worlds_per_cell": int(e1.groupby(["scenario", "method"])["seed"].nunique().min()),
        "group_means": means.to_dict(orient="records"),
        "paired_group_contrasts": contrasts,
        "factorial_price_of_fairness": pof_rows,
        "full_stack_price_of_fairness": full_stack,
        "diagnostic_gate": {
            "v2_severe_worst_group_superiority_over_both_e1_baselines": severe_gate,
            "v3_primary_equity_matrix_required": True,
            "continuous_fairness_frontier_required": True,
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report, means


def write_outputs(report: dict[str, Any], means: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    means.to_csv(output.with_suffix(".csv"), index=False)
    severe = means[means["scenario"] == "severe"].set_index("method")
    order = ["airproof", "air_quality_dt", "opportunistic_ttl"]
    labels = ["AirProof v2", "Air-quality DT", "Opportunistic TTL"]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), constrained_layout=True)
    width = 0.25
    for offset, method, label in zip((-width, 0.0, width), order, labels):
        axes[0].bar(
            x + offset,
            severe.loc[method, GROUP_COLUMNS].to_numpy(dtype=float),
            width,
            label=label,
        )
    axes[0].set_xticks(x, ["G0", "G1", "G2", "G3"])
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Frozen E1 severe regime: per-group error")
    axes[0].legend(frameon=False, fontsize=8)
    pof = pd.DataFrame(
        [
            {
                "scenario": row["scenario"],
                "relay": row["relay"],
                "huber": row["huber"],
                "pof": row["price_of_fairness_rmse"]["estimate"],
                "coverage": row["relative_coverage_gap_reduction"]["estimate"],
            }
            for row in report["factorial_price_of_fairness"]
        ]
    )
    markers = {"nominal_clean": "o", "severe_drift20": "s"}
    for scenario, subset in pof.groupby("scenario"):
        axes[1].scatter(
            100 * subset["pof"],
            100 * subset["coverage"],
            marker=markers[scenario],
            s=48,
            label=scenario.replace("_", " "),
        )
    axes[1].axvline(0, color="0.6", linewidth=0.8)
    axes[1].axhline(0, color="0.6", linewidth=0.8)
    axes[1].set_xlabel("Price of Fairness in RMSE (%)")
    axes[1].set_ylabel("Coverage-gap reduction (%)")
    axes[1].set_title("Matched E3 fairness contrasts")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    lines = [
        "# Frozen E1 equity diagnostic",
        "",
        f"Status: **{report['status']}**",
        "",
        report["interpretation_boundary"],
        "",
        "## Per-group means",
        "",
        "| Scenario | Method | RMSE | Worst-group RMSE | Coverage gap | G0 | G1 | G2 | G3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["group_means"]:
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['rmse']:.4f} | "
            f"{row['worst_group_rmse']:.4f} | {row['coverage_gap']:.4f} | "
            f"{row['group_0_rmse']:.4f} | {row['group_1_rmse']:.4f} | "
            f"{row['group_2_rmse']:.4f} | {row['group_3_rmse']:.4f} |"
        )
    lines += [
        "",
        "## Gate decision",
        "",
        (
            "The predecessor architecture does **not** satisfy severe-regime worst-group "
            "superiority over both E1 comparators. This result is retained as diagnosis; the "
            "powered v3 primary matrix must replace it before a headline prediction-equity claim."
        ),
        "",
        (
            "The E3 matched contrasts provide a valid two-point Price-of-Fairness diagnostic, "
            "but not a continuous Pareto frontier. A frozen fairness-strength sweep is therefore "
            "required."
        ),
        "",
        f"Artifact SHA-256: `{report['artifact_sha256']}`",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e1",
        type=Path,
        default=Path("results/fullpaper30h_20260902/results/e1.parquet"),
    )
    parser.add_argument(
        "--e3",
        type=Path,
        default=Path("results/fullpaper30h_20260902/results/e3.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/analysis/e1_equity_diagnostic"),
    )
    args = parser.parse_args()
    report, means = build_report(args.e1, args.e3)
    write_outputs(report, means, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "e1_rows": report["e1_rows"],
                "e3_rows": report["e3_rows"],
                "severe_gate": report["diagnostic_gate"][
                    "v2_severe_worst_group_superiority_over_both_e1_baselines"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
