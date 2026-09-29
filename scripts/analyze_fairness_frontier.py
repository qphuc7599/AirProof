from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airproof.config import config_hash, load_config, with_overrides


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 1_000):
        stop = min(start + 1_000, replicates)
        sample = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "bootstrap_95_ci": [float(low), float(high)],
        "bootstrap_replicates": replicates,
        "n_paired_worlds": len(values),
    }


def _matrix_config_map(matrix_path: Path) -> dict[str, dict[str, float | int]]:
    matrix = load_config(matrix_path)
    base = load_config(matrix_path.parent / matrix["base_config"])
    result: dict[str, dict[str, float | int]] = {}
    for strength in matrix["factors"]["scheduler.fairness_strength"]:
        for target in matrix["factors"]["scheduler.target_contributors"]:
            config = with_overrides(
                base,
                {
                    "scheduler.fairness_strength": strength,
                    "scheduler.target_contributors": target,
                },
            )
            result[config_hash(config)] = {
                "fairness_strength": float(strength),
                "target_contributors": int(target),
            }
    return result


def _flatten(path: Path, config_map: dict[str, dict[str, float | int]]) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["config_hash"] not in config_map:
                raise ValueError(f"unregistered configuration hash {record['config_hash']}")
            factors = config_map[record["config_hash"]]
            row = {
                "seed": int(record["seed"]),
                "fairness_strength": float(factors["fairness_strength"]),
                "target_contributors": int(factors["target_contributors"]),
                "source_tree_sha256": record["source_tree_sha256"],
                "registry_sha256": record["registry_sha256"],
            }
            row.update(record["metrics"])
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_effect(
    frame: pd.DataFrame,
    *,
    target: int,
    strength: float,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    subset = frame[frame["target_contributors"] == target]
    pivot = subset.pivot(index="seed", columns="fairness_strength", values=metric).dropna()
    baseline = pivot[0.0].to_numpy(dtype=float)
    treated = pivot[strength].to_numpy(dtype=float)
    relative = (treated - baseline) / np.maximum(np.abs(baseline), 1e-12)
    return _bootstrap_mean(relative, seed=seed)


def build_report(path: Path, matrix_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = _flatten(path, _matrix_config_map(matrix_path))
    expected_strengths = {0.0, 0.25, 0.5, 0.75, 1.0}
    expected_targets = {10, 20, 30}
    if set(frame["fairness_strength"].unique()) != expected_strengths:
        raise ValueError("frontier does not contain the frozen fairness-strength grid")
    if set(frame["target_contributors"].unique()) != expected_targets:
        raise ValueError("frontier does not contain the frozen contributor-target grid")
    if frame.groupby(["fairness_strength", "target_contributors"])["seed"].nunique().min() != 8:
        raise ValueError("frontier cell is incomplete")
    if frame["source_tree_sha256"].nunique() != 1 or frame["registry_sha256"].nunique() != 1:
        raise ValueError("provenance digest changed within the frontier")

    metrics = [
        "rmse",
        "worst_group_rmse",
        "coverage_gap",
        "coverage_p10",
        "mean_max_service_deficit",
        "globally_feasible_epochs",
        "constraint_satisfied_epochs",
        "constraint_violation_when_feasible_epochs",
        "scoring_steps",
        "runtime_seconds",
    ]
    means = (
        frame.groupby(["target_contributors", "fairness_strength"], sort=True)[metrics]
        .mean()
        .reset_index()
    )

    contrasts: list[dict[str, Any]] = []
    bootstrap_seed = 20260902
    for target in sorted(expected_targets):
        for strength in sorted(expected_strengths - {0.0}):
            item: dict[str, Any] = {
                "target_contributors": target,
                "fairness_strength": strength,
            }
            for metric in ("rmse", "worst_group_rmse", "coverage_gap", "coverage_p10"):
                item[f"relative_{metric}_change"] = _paired_effect(
                    frame,
                    target=target,
                    strength=strength,
                    metric=metric,
                    seed=bootstrap_seed,
                )
                bootstrap_seed += 1
            contrasts.append(item)

    candidates = []
    for contrast in contrasts:
        mean_row = means[
            (means["target_contributors"] == contrast["target_contributors"])
            & (means["fairness_strength"] == contrast["fairness_strength"])
        ].iloc[0]
        pof = contrast["relative_rmse_change"]["estimate"]
        gap_reduction = -contrast["relative_coverage_gap_change"]["estimate"]
        feasibility_rate = (
            mean_row["globally_feasible_epochs"] / max(mean_row["scoring_steps"], 1.0)
        )
        eligible = bool(
            pof <= 0.05
            and mean_row["constraint_violation_when_feasible_epochs"] == 0
            and feasibility_rate >= 0.90
        )
        candidates.append(
            {
                "target_contributors": contrast["target_contributors"],
                "fairness_strength": contrast["fairness_strength"],
                "effective_floor": int(
                    np.ceil(
                        contrast["target_contributors"] * contrast["fairness_strength"]
                    )
                ),
                "global_feasibility_rate": float(feasibility_rate),
                "price_of_fairness_rmse": pof,
                "relative_coverage_gap_reduction": gap_reduction,
                "eligible_under_frozen_rule": eligible,
            }
        )
    eligible = [item for item in candidates if item["eligible_under_frozen_rule"]]
    selected = (
        max(
            eligible,
            key=lambda item: (
                item["relative_coverage_gap_reduction"],
                -item["price_of_fairness_rmse"],
                item["fairness_strength"],
            ),
        )
        if eligible
        else None
    )
    report = {
        "status": "completed-validation-frontier-not-primary-inference",
        "selection_rule": (
            "Among nonzero fairness settings with paired mean RMSE price <=5%, zero "
            "violations whenever the executable certificate declares global feasibility, "
            "and global feasibility in >=90% of scored epochs, maximize paired coverage-gap "
            "reduction; break ties by lower RMSE price and then full-strength semantics. "
            "Freeze the selected setting before primary runs."
        ),
        "interpretation_boundary": (
            "This 8-seed severe-network clean-data matrix tunes the operational fairness "
            "setting. Its uncertainty intervals and Pareto surface are descriptive validation "
            "evidence and are not reused as confirmatory primary inference."
        ),
        "inputs": {
            str(path).replace("\\", "/"): _sha256(path),
            str(matrix_path).replace("\\", "/"): _sha256(matrix_path),
        },
        "rows": len(frame),
        "worlds_per_cell": 8,
        "source_tree_sha256": frame["source_tree_sha256"].iat[0],
        "registry_sha256": frame["registry_sha256"].iat[0],
        "cell_means": means.to_dict(orient="records"),
        "paired_contrasts_against_lambda_zero": contrasts,
        "selection_candidates": candidates,
        "selected_for_primary_freeze": selected,
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

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    palette = {10: "#2f6690", 20: "#3a7d44", 30: "#bc4749"}
    for target, subset in means.groupby("target_contributors", sort=True):
        subset = subset.sort_values("fairness_strength")
        axes[0].plot(
            subset["coverage_gap"],
            subset["rmse"],
            "o-",
            color=palette[int(target)],
            label=f"target {int(target)}",
        )
        axes[1].plot(
            subset["fairness_strength"],
            subset["worst_group_rmse"],
            "o-",
            color=palette[int(target)],
            label=f"target {int(target)}",
        )
    axes[0].set_xlabel("Coverage gap (lower is better)")
    axes[0].set_ylabel("Overall RMSE (lower is better)")
    axes[0].set_title("Operational fairness frontier")
    axes[1].set_xlabel(r"Fairness strength $\lambda_f$")
    axes[1].set_ylabel("Worst-group RMSE")
    axes[1].set_title("Worst-group response")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    lines = [
        "# AirProof v3 operational fairness frontier",
        "",
        f"Status: **{report['status']}**",
        "",
        report["interpretation_boundary"],
        "",
        "## Frozen selection rule",
        "",
        report["selection_rule"],
        "",
        "## Cell means",
        "",
        "| Target | Strength | RMSE | Worst-group RMSE | Coverage gap | Coverage p10 | Feasible epochs | Violations when feasible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cell_means"]:
        lines.append(
            f"| {int(row['target_contributors'])} | {row['fairness_strength']:.2f} | "
            f"{row['rmse']:.4f} | {row['worst_group_rmse']:.4f} | "
            f"{row['coverage_gap']:.4f} | {row['coverage_p10']:.4f} | "
            f"{row['globally_feasible_epochs']:.2f} | "
            f"{row['constraint_violation_when_feasible_epochs']:.2f} |"
        )
    lines += ["", "## Primary freeze decision", ""]
    selected = report["selected_for_primary_freeze"]
    if selected is None:
        lines.append("No nonzero setting passed the frozen validation rule.")
    else:
        lines.append(
            f"Freeze target={selected['target_contributors']} and "
            f"fairness_strength={selected['fairness_strength']:.2f} "
            f"(effective floor {selected['effective_floor']}); global feasibility is "
            f"{100 * selected['global_feasibility_rate']:.2f}%, paired mean RMSE price "
            f"is {100 * selected['price_of_fairness_rmse']:.2f}% and paired coverage-gap "
            f"reduction is {100 * selected['relative_coverage_gap_reduction']:.2f}%."
        )
    lines += ["", f"Artifact semantic SHA-256: `{report['artifact_sha256']}`", ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/matrices/v3_fairness_frontier.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, means = build_report(args.input, args.matrix)
    write_outputs(report, means, args.output)


if __name__ == "__main__":
    main()
