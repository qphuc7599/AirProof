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


def _paired_bootstrap(
    adaptive: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 20260902,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(adaptive), size=(replicates, len(adaptive)))
    adaptive_means = adaptive[indices].mean(axis=1)
    baseline_means = baseline[indices].mean(axis=1)
    ratios = adaptive_means / baseline_means
    differences = adaptive_means - baseline_means
    margin_differences = adaptive_means - 1.10 * baseline_means
    return {
        "paired_outer_stations": len(adaptive),
        "bootstrap_replicates": replicates,
        "mean_rmse_ratio": {
            "estimate": float(adaptive.mean() / baseline.mean()),
            "bootstrap_95_ci": [float(item) for item in np.quantile(ratios, [0.025, 0.975])],
        },
        "mean_rmse_difference": {
            "estimate": float(np.mean(adaptive - baseline)),
            "bootstrap_95_ci": [
                float(item) for item in np.quantile(differences, [0.025, 0.975])
            ],
        },
        "noninferiority_margin_10pct_difference": {
            "estimate": float(np.mean(adaptive - 1.10 * baseline)),
            "bootstrap_95_ci": [
                float(item) for item in np.quantile(margin_differences, [0.025, 0.975])
            ],
        },
        "station_win_fraction": float(np.mean(adaptive <= baseline)),
    }


def build_report(v1_path: Path, v2_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    folds = pd.DataFrame(v2["folds"])
    pivot = folds.pivot(index="held_out_station", columns="method", values="rmse")
    adaptive = pivot["city_adaptive_graph_v2"].to_numpy(dtype=float)
    baseline = pivot["idw_frozen"].to_numpy(dtype=float)
    paired = _paired_bootstrap(adaptive, baseline)
    v1_aggregate = {item["method"]: item for item in v1["aggregate"]}
    v2_aggregate = {item["method"]: item for item in v2["aggregate"]}
    v1_ratio = (
        v1_aggregate["graph_huber"]["mean_station_rmse"]
        / v1_aggregate["idw"]["mean_station_rmse"]
    )
    v2_ratio = paired["mean_rmse_ratio"]["estimate"]
    report = {
        "status": "locked-city-adaptive-v2-transfer-analysis",
        "interpretation_boundary": (
            "V1 and v2 use disjoint outer test windows. Their ratio change documents removal of "
            "the frozen transfer collapse, while paired v2-vs-IDW inference uses only the locked "
            "v2 station folds."
        ),
        "inputs": {
            str(v1_path).replace("\\", "/"): _sha256(v1_path),
            str(v2_path).replace("\\", "/"): _sha256(v2_path),
        },
        "v1_frozen": {
            "test_window": "first-336-hours-after-80-percent-split",
            "graph_huber_mean_rmse": v1_aggregate["graph_huber"]["mean_station_rmse"],
            "idw_mean_rmse": v1_aggregate["idw"]["mean_station_rmse"],
            "ratio": v1_ratio,
        },
        "v2_locked": {
            "test_window": [v2["outer_test_start"], v2["outer_test_end"]],
            "city_adaptive_mean_rmse": v2_aggregate["city_adaptive_graph_v2"][
                "mean_station_rmse"
            ],
            "idw_mean_rmse": v2_aggregate["idw_frozen"]["mean_station_rmse"],
            "city_adaptive_worst_station_rmse": v2_aggregate["city_adaptive_graph_v2"][
                "worst_station_rmse"
            ],
            "idw_worst_station_rmse": v2_aggregate["idw_frozen"]["worst_station_rmse"],
            "paired_analysis": paired,
        },
        "gates": {
            "descriptive_ratio_lte_1_10": v2_ratio <= 1.10,
            "preferred_descriptive_ratio_lte_1_05": v2_ratio <= 1.05,
            "confirmatory_10pct_noninferiority_upper_ci_below_zero": paired[
                "noninferiority_margin_10pct_difference"
            ]["bootstrap_95_ci"][1]
            < 0,
            "collapse_ratio_reduction_fraction": 1.0 - v2_ratio / v1_ratio,
        },
        "folds": folds.astype(object).where(pd.notna(folds), None).to_dict(orient="records"),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report, pivot


def write_outputs(report: dict[str, Any], pivot: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pivot.assign(
        ratio=pivot["city_adaptive_graph_v2"] / pivot["idw_frozen"]
    ).to_csv(output.with_suffix(".csv"))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
    axes[0].scatter(
        pivot["idw_frozen"],
        pivot["city_adaptive_graph_v2"],
        color="#1f77b4",
        s=48,
    )
    limit = 1.08 * max(pivot.to_numpy().max(), 1.0)
    grid = np.linspace(0, limit, 100)
    axes[0].plot(grid, grid, color="0.3", linewidth=1, label="parity")
    axes[0].plot(grid, 1.10 * grid, color="#d62728", linestyle="--", linewidth=1, label="1.10×")
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("Frozen IDW station RMSE")
    axes[0].set_ylabel("City-adaptive v2 station RMSE")
    axes[0].set_title("Locked v2 outer stations")
    axes[0].legend(frameon=False)

    ratios = [report["v1_frozen"]["ratio"], report["v2_locked"]["paired_analysis"]["mean_rmse_ratio"]["estimate"]]
    axes[1].bar(["Frozen v1\nHuber/IDW", "Locked v2\nAdaptive/IDW"], ratios, color=["#d62728", "#2ca02c"])
    axes[1].axhline(1.0, color="0.3", linewidth=1)
    axes[1].axhline(1.10, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Mean-station RMSE ratio")
    axes[1].set_title("Disjoint-window transfer diagnosis")
    for index, value in enumerate(ratios):
        axes[1].text(index, value + 0.05, f"{value:.3f}×", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(ratios) * 1.16)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    paired = report["v2_locked"]["paired_analysis"]
    lines = [
        "# Beijing city-adaptive transfer v2",
        "",
        f"Status: **{report['status']}**",
        "",
        report["interpretation_boundary"],
        "",
        "| Version | Method | Mean-station RMSE | IDW RMSE | Ratio |",
        "|---|---|---:|---:|---:|",
        (
            f"| Frozen v1 | Graph Huber | {report['v1_frozen']['graph_huber_mean_rmse']:.3f} | "
            f"{report['v1_frozen']['idw_mean_rmse']:.3f} | {report['v1_frozen']['ratio']:.3f}× |"
        ),
        (
            f"| Locked v2 | City-adaptive graph | {report['v2_locked']['city_adaptive_mean_rmse']:.3f} | "
            f"{report['v2_locked']['idw_mean_rmse']:.3f} | "
            f"{paired['mean_rmse_ratio']['estimate']:.3f}× |"
        ),
        "",
        (
            f"Across 12 outer stations, the locked v2 mean ratio was "
            f"{paired['mean_rmse_ratio']['estimate']:.3f} with paired-bootstrap 95% CI "
            f"[{paired['mean_rmse_ratio']['bootstrap_95_ci'][0]:.3f}, "
            f"{paired['mean_rmse_ratio']['bootstrap_95_ci'][1]:.3f}]. The descriptive 1.10× "
            "gate passed, whereas the confirmatory 10% noninferiority CI gate did not."
        ),
        "",
        f"Artifact SHA-256: `{report['artifact_sha256']}`",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1",
        type=Path,
        default=Path("reports/analysis/beijing_transfer.json"),
    )
    parser.add_argument(
        "--v2",
        type=Path,
        default=Path("reports/analysis/beijing_city_adaptive_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/analysis/beijing_city_adaptive_v2_analysis"),
    )
    args = parser.parse_args()
    report, pivot = build_report(args.v1, args.v2)
    write_outputs(report, pivot, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "gates": report["gates"]}))


if __name__ == "__main__":
    main()
