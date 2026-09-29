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
        "n_paired_traces": len(values),
        "bootstrap_replicates": replicates,
    }


def build_report(input_path: Path, protocol_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(raw["rows"])
    expected = {
        "direct",
        "ttl_one_copy",
        "prophet_gtmx",
        "binary_spray_wait",
        "epidemic_cap",
        "airproof_deficit",
    }
    if set(frame["policy"]) != expected:
        raise ValueError("DTN result is missing a frozen policy")
    worlds = frame.groupby("policy")["seed"].nunique()
    if worlds.nunique() != 1 or worlds.iat[0] < 2:
        raise ValueError("DTN policy cells are incomplete")

    metrics = [
        "deadline_delivery_ratio",
        "restricted_mean_delivery_delay",
        "restricted_mean_aoi",
        "group_delivery_gap",
        "transmissions_per_delivered",
        "queue_rejections",
    ]
    means = frame.groupby("policy", sort=True)[metrics].mean().reset_index()
    airproof = frame[frame["policy"] == "airproof_deficit"].set_index("seed")
    contrasts = []
    bootstrap_seed = 20260902
    for policy in sorted(expected - {"airproof_deficit"}):
        baseline = frame[frame["policy"] == policy].set_index("seed")
        common = airproof.index.intersection(baseline.index)
        row: dict[str, Any] = {"baseline": policy}
        for metric in metrics[:5]:
            air = airproof.loc[common, metric].to_numpy(dtype=float)
            base = baseline.loc[common, metric].to_numpy(dtype=float)
            if metric in {
                "restricted_mean_delivery_delay",
                "restricted_mean_aoi",
                "group_delivery_gap",
                "transmissions_per_delivered",
            }:
                effect = (base - air) / np.maximum(np.abs(base), 1e-12)
            else:
                effect = (air - base) / np.maximum(np.abs(base), 1e-12)
            row[f"relative_improvement_{metric}"] = _bootstrap(effect, bootstrap_seed)
            bootstrap_seed += 1
        contrasts.append(row)

    non_air = means[means["policy"] != "airproof_deficit"]
    strongest_policy = str(
        non_air.loc[non_air["restricted_mean_delivery_delay"].idxmin(), "policy"]
    )
    strongest = next(item for item in contrasts if item["baseline"] == strongest_policy)
    primary = strongest["relative_improvement_restricted_mean_delivery_delay"]
    report: dict[str, Any] = {
        "status": "completed-confirmatory-capacity-constrained-dtn",
        "interpretation_boundary": (
            "Every policy replays the same per-seed messages, gateway states, peer contacts, "
            "bytes/contact, buffers, TTL, and maximum copy budget. PRoPHET uses RFC 6693 "
            "equations 1-3 with GTMX; epidemic is explicitly copy-budget capped. This is a "
            "controlled synthetic transport benchmark, not a wireless deployment trace."
        ),
        "inputs": {
            str(input_path).replace("\\", "/"): _sha256(input_path),
            str(protocol_path).replace("\\", "/"): _sha256(protocol_path),
        },
        "source_tree_sha256": raw["source_tree_sha256"],
        "traces": int(worlds.iat[0]),
        "rows": len(frame),
        "cell_means": means.to_dict(orient="records"),
        "paired_contrasts": contrasts,
        "primary_comparator": strongest_policy,
        "primary_delay_improvement": primary,
        "primary_gate": {
            "threshold": 0.15,
            "point_estimate_pass": bool(primary["estimate"] >= 0.15),
            "confidence_interval_pass": bool(primary["bootstrap_95_ci"][0] >= 0.15),
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
        "direct",
        "ttl_one_copy",
        "prophet_gtmx",
        "binary_spray_wait",
        "epidemic_cap",
        "airproof_deficit",
    ]
    plot = means.set_index("policy").loc[order]
    labels = ["Direct", "TTL-1", "PRoPHET", "Spray", "Epidemic", "AirProof"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    colors = ["#8d99ae"] * 5 + ["#2a9d8f"]
    axes[0].bar(labels, plot["restricted_mean_delivery_delay"], color=colors)
    axes[0].set_ylabel("Restricted mean delay (h)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Finite-capacity delivery delay")
    axes[1].scatter(
        plot["transmissions_per_delivered"],
        plot["deadline_delivery_ratio"],
        c=colors,
        s=64,
    )
    for label, x, y in zip(
        labels,
        plot["transmissions_per_delivered"],
        plot["deadline_delivery_ratio"],
    ):
        axes[1].annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Transmissions per delivered record")
    axes[1].set_ylabel("Deadline delivery ratio")
    axes[1].set_title("Delivery--overhead frontier")
    axes[1].grid(alpha=0.25)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    lines = [
        "# Capacity-constrained equal-resource DTN benchmark",
        "",
        f"Status: **{report['status']}**",
        "",
        report["interpretation_boundary"],
        "",
        "| Policy | Deadline delivery | Restricted mean delay | Restricted mean AoI | Group delivery gap | Tx/delivered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["cell_means"]:
        lines.append(
            f"| {row['policy']} | {row['deadline_delivery_ratio']:.4f} | "
            f"{row['restricted_mean_delivery_delay']:.4f} | "
            f"{row['restricted_mean_aoi']:.4f} | {row['group_delivery_gap']:.4f} | "
            f"{row['transmissions_per_delivered']:.4f} |"
        )
    primary = report["primary_delay_improvement"]
    lines += [
        "",
        "## Registered primary endpoint",
        "",
        (
            f"The strongest non-AirProof mean-delay comparator was "
            f"**{report['primary_comparator']}**. AirProof's paired "
            f"restricted-mean-delay improvement was {100 * primary['estimate']:.2f}% "
            f"(95% bootstrap CI {100 * primary['bootstrap_95_ci'][0]:.2f}% to "
            f"{100 * primary['bootstrap_95_ci'][1]:.2f}%)."
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
        "--protocol", type=Path, default=Path("configs/v3/dtn_equal_resource.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, means = build_report(args.input, args.protocol)
    write_outputs(report, means, args.output)


if __name__ == "__main__":
    main()
