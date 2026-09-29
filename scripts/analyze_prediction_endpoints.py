from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(values: np.ndarray, seed: int, replicates: int = 10_000) -> dict[str, Any]:
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
        "n_paired_worlds": len(values),
        "bootstrap_replicates": replicates,
    }


def build_report(input_path: Path, protocol_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    rows = []
    for record in raw["rows"]:
        rows.append({"seed": record["seed"], "predictor": record["predictor"], **record["metrics"]})
    frame = pd.DataFrame(rows)
    expected = {
        "persistence",
        "idw",
        "histgb_spatiotemporal",
        "graph_squared",
        "airproof_predictive_residual",
    }
    if set(frame["predictor"]) != expected:
        raise ValueError("prediction endpoint result is missing a frozen method")
    worlds = frame.groupby("predictor")["seed"].nunique()
    if worlds.nunique() != 1 or worlds.iat[0] < 2:
        raise ValueError("prediction endpoint cells are incomplete")

    metrics = ["rmse", "mae", "bias", "r2", "worst_group_rmse", "worst_median_error_ratio"]
    means = frame.groupby("predictor", sort=True)[metrics].mean().reset_index()
    reference = frame[frame["predictor"] == "airproof_predictive_residual"].set_index("seed")
    contrasts = []
    bootstrap_seed = 20260902
    for predictor in sorted(expected - {"airproof_predictive_residual"}):
        baseline = frame[frame["predictor"] == predictor].set_index("seed")
        common = reference.index.intersection(baseline.index)
        row: dict[str, Any] = {"baseline": predictor}
        for metric in ("rmse", "mae", "worst_group_rmse"):
            ratio = (
                reference.loc[common, metric].to_numpy(dtype=float)
                / baseline.loc[common, metric].to_numpy(dtype=float)
            )
            row[f"airproof_to_baseline_{metric}_ratio"] = _bootstrap(
                ratio, bootstrap_seed
            )
            bootstrap_seed += 1
        contrasts.append(row)
    non_air = means[means["predictor"] != "airproof_predictive_residual"]
    strongest = str(non_air.loc[non_air["rmse"].idxmin(), "predictor"])
    primary = next(item for item in contrasts if item["baseline"] == strongest)[
        "airproof_to_baseline_rmse_ratio"
    ]
    report: dict[str, Any] = {
        "status": "completed-locked-validation-endpoint-comparison",
        "interpretation_boundary": (
            "All predictors replay the same synthetic worlds and receive the same citizen "
            "observations and current trusted-reference stream; scoring excludes reference "
            "grid cells. HistGB fits only the burn-in interval and is an executable strong "
            "tabular spatiotemporal baseline, not a claimed reproduction of a graph neural-"
            "network paper. This validation freezes the primary comparator family."
        ),
        "inputs": {
            str(input_path).replace("\\", "/"): _sha256(input_path),
            str(protocol_path).replace("\\", "/"): _sha256(protocol_path),
        },
        "source_tree_sha256": raw["source_tree_sha256"],
        "worlds": int(worlds.iat[0]),
        "rows": len(frame),
        "cell_means": means.to_dict(orient="records"),
        "paired_contrasts": contrasts,
        "strongest_non_airproof_predictor": strongest,
        "airproof_to_strongest_rmse_ratio": primary,
        "validation_gate": {
            "noninferiority_margin": 1.05,
            "point_estimate_pass": bool(primary["estimate"] <= 1.05),
            "confidence_interval_pass": bool(primary["bootstrap_95_ci"][1] <= 1.05),
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report, means


def write_outputs(report: dict[str, Any], means: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    means.to_csv(output.with_suffix(".csv"), index=False)
    order = [
        "persistence",
        "idw",
        "histgb_spatiotemporal",
        "graph_squared",
        "airproof_predictive_residual",
    ]
    labels = ["Persistence", "IDW", "HistGB-ST", "Graph squared", "AirProof v3"]
    plot = means.set_index("predictor").loc[order]
    colors = ["#8d99ae"] * 4 + ["#2a9d8f"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].bar(labels, plot["rmse"], color=colors)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Non-reference-cell prediction")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, plot["worst_group_rmse"], color=colors)
    axes[1].set_ylabel("Worst-group RMSE")
    axes[1].set_title("Outcome-equity endpoint")
    axes[1].tick_params(axis="x", rotation=25)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    lines = [
        "# Locked endpoint-specific prediction validation",
        "",
        f"Status: **{report['status']}**",
        "",
        report["interpretation_boundary"],
        "",
        "| Predictor | RMSE | MAE | Worst-group RMSE | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["cell_means"]:
        lines.append(
            f"| {row['predictor']} | {row['rmse']:.4f} | {row['mae']:.4f} | "
            f"{row['worst_group_rmse']:.4f} | {row['r2']:.4f} |"
        )
    primary = report["airproof_to_strongest_rmse_ratio"]
    lines += [
        "",
        "## Validation gate",
        "",
        (
            f"The strongest non-AirProof predictor was "
            f"**{report['strongest_non_airproof_predictor']}**. The paired AirProof/baseline "
            f"RMSE ratio was {primary['estimate']:.4f} (95% bootstrap CI "
            f"{primary['bootstrap_95_ci'][0]:.4f} to "
            f"{primary['bootstrap_95_ci'][1]:.4f})."
        ),
        "",
        f"Artifact semantic SHA-256: `{report['artifact_sha256']}`",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/v3/prediction_endpoints.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, means = build_report(args.input, args.protocol)
    write_outputs(report, means, args.output)


if __name__ == "__main__":
    main()
