from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#1f5a94"
ORANGE = "#d36b27"
GREEN = "#2f7d5b"
GRAY = "#5b6470"


def _error(interval: dict[str, object], *, scale: float = 1.0) -> tuple[float, list[float]]:
    estimate = float(interval["estimate"]) * scale
    low, high = [float(value) * scale for value in interval["bootstrap_95_ci"]]
    return estimate, [estimate - low, high - estimate]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=True)

    clean_labels = ["Anchor", "Outage", "Severe"]
    clean_keys = ["anchor", "outage", "severe_clean"]
    clean_values = [
        _error(report["clean_noninferiority"][key]["rmse"]) for key in clean_keys
    ]
    x = np.arange(len(clean_labels))
    axes[0].errorbar(
        x,
        [item[0] for item in clean_values],
        yerr=np.asarray([item[1] for item in clean_values]).T,
        fmt="o",
        color=BLUE,
        capsize=3,
        linewidth=1.2,
    )
    axes[0].axhline(1.0, color=GRAY, linewidth=0.8)
    axes[0].axhline(1.05, color=ORANGE, linestyle="--", linewidth=0.9)
    axes[0].set_xticks(x, clean_labels)
    axes[0].set_ylabel("AirProof / squared RMSE")
    axes[0].set_title("(a) Clean noninferiority")
    axes[0].grid(axis="y", alpha=0.22)

    attack_labels = ["White-box\ndrift", "Hotspot\nsuppression"]
    attack_keys = ["severe_drift", "severe_hotspot"]
    attack_values = [
        _error(report["attacks"][key]["rmse"]["attenuation_fraction"], scale=100.0)
        for key in attack_keys
    ]
    axes[1].errorbar(
        np.arange(2),
        [item[0] for item in attack_values],
        yerr=np.asarray([item[1] for item in attack_values]).T,
        fmt="o",
        color=ORANGE,
        capsize=3,
        linewidth=1.2,
    )
    axes[1].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[1].axhline(20.0, color=GREEN, linestyle="--", linewidth=0.9)
    axes[1].set_xticks(np.arange(2), attack_labels)
    axes[1].set_ylabel("RMSE-growth attenuation (%)")
    axes[1].set_title("(b) Attack response")
    axes[1].grid(axis="y", alpha=0.22)

    gap = _error(report["fairness"]["coverage_gap_reduction"], scale=100.0)
    ratio = report["fairness"]["rmse_ratio"]
    price = {
        "estimate": float(ratio["estimate"]) - 1.0,
        "bootstrap_95_ci": [
            float(ratio["bootstrap_95_ci"][0]) - 1.0,
            float(ratio["bootstrap_95_ci"][1]) - 1.0,
        ],
    }
    price_value = _error(price, scale=100.0)
    worst_group = _error(
        report["fairness"]["worst_group_rmse_reduction"], scale=100.0
    )
    axes[2].errorbar(
        np.arange(3),
        [gap[0], price_value[0], worst_group[0]],
        yerr=np.asarray([gap[1], price_value[1], worst_group[1]]).T,
        fmt="o",
        color=GREEN,
        capsize=3,
        linewidth=1.2,
    )
    axes[2].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[2].set_xticks(
        np.arange(3),
        ["Gap\neffect", "Overall\nRMSE", "Worst-group\nRMSE"],
    )
    axes[2].set_ylabel("Paired effect (%)")
    axes[2].set_title("(c) Severe-regime equity")
    axes[2].grid(axis="y", alpha=0.22)

    for path in (args.pdf, args.png):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.pdf, bbox_inches="tight")
    figure.savefig(args.png, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
