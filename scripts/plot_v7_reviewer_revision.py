#!/usr/bin/env python3
"""Plot the four reviewer-requested confirmation views from locked table sources."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "reports/v7/reviewer_revision/shared_resource_core_confirmation_v3/analysis_detail.json"
CONTROLS = ROOT / "reports/v7/reviewer_revision/shared_core_component_controls_v1/analysis.json"
EVENT = ROOT / "reports/v7/reviewer_revision/event_interval_confirmation_v1/analysis.json"
OUTPUT = ROOT / "overleaf/figs/v7_reviewer_revision.pdf"

COLORS = {
    "AP_SHARED_CORE": "#0B6E69",
    "SQ": "#D36B32",
    "HUBER": "#2D6A9F",
    "INPUT_CLIP_ONLY": "#7A5195",
    "OUTPUT_CAP_ONLY": "#B89C29",
    "PUBLIC": "#6B7280",
}
LABELS = {
    "AP_SHARED_CORE": "AirProof",
    "SQ": "Squared",
    "HUBER": "Huber",
    "INPUT_CLIP_ONLY": "Input clip",
    "OUTPUT_CAP_ONLY": "Output cap",
    "PUBLIC": "Public",
}


def _bar_panel(axis, names, values, *, ylabel, title, zero=False):
    positions = np.arange(len(names))
    axis.bar(positions, values, color=[COLORS[name] for name in names], width=0.72)
    axis.set_xticks(positions, [LABELS[name] for name in names], rotation=28, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    if zero:
        axis.axhline(0, color="#111827", linewidth=0.7)


def execute() -> Path:
    core = json.loads(CORE.read_text(encoding="utf-8"))
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    event = json.loads(EVENT.read_text(encoding="utf-8"))
    means = controls["means"]
    names = ["AP_SHARED_CORE", "SQ", "HUBER", "INPUT_CLIP_ONLY", "OUTPUT_CAP_ONLY", "PUBLIC"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.55), constrained_layout=True)

    _bar_panel(
        axes[0, 0],
        names,
        [
            core["cells"]["severe_clean"]["methods"].get(name, {}).get(
                "rmse", means.get(name, {}).get("clean_rmse")
            )
            for name in names
        ],
        ylabel="Reconstructed RMSE",
        title="(a) Common-resource clean utility",
    )

    attack_names = names[:-1]
    _bar_panel(
        axes[0, 1],
        attack_names,
        [means[name]["hotspot_growth"] for name in attack_names],
        ylabel="Hotspot RMSE growth",
        title="(b) One-interface-at-a-time attribution",
        zero=True,
    )
    axes[0, 1].text(
        0.98,
        0.96,
        f"AirProof attenuation: {100 * means['AP_SHARED_CORE']['attenuation_vs_sq_growth']:.1f}%",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
    )

    privacy = core["privacy_comparison"]["means"]
    privacy_names = ["Residual DP", "Raw [0,50] DP", "Public only"]
    privacy_values = [
        privacy["residual_release_rmse"],
        privacy["nonnegative_raw_release_rmse"],
        privacy["same_support_public_rmse"],
    ]
    axes[1, 0].bar(
        np.arange(3), privacy_values, color=["#0B6E69", "#D36B32", "#6B7280"], width=0.68
    )
    axes[1, 0].set_xticks(np.arange(3), privacy_names, rotation=20, ha="right")
    axes[1, 0].set_ylabel("Release RMSE")
    axes[1, 0].set_title("(c) Equal-budget release controls", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axes[1, 0].set_axisbelow(True)

    positions = np.arange(2)
    width = 0.34
    for offset, clock in ((-width / 2, "live"), (width / 2, "reconstructed")):
        entries = event["results"][clock]["event_coverage"]
        estimate = np.asarray([entry["estimate"] for entry in entries])
        lower = np.asarray([entry["ci95"][0] for entry in entries])
        upper = np.asarray([entry["ci95"][1] for entry in entries])
        axes[1, 1].bar(
            positions + offset,
            estimate,
            width,
            label="Live" if clock == "live" else "Reconstructed",
            color="#0B6E69" if clock == "live" else "#2D6A9F",
            yerr=np.vstack((estimate - lower, upper - estimate)),
            capsize=2.5,
            error_kw={"linewidth": 0.8},
        )
    axes[1, 1].scatter(positions, [0.9, 0.95], marker="_", s=420, linewidth=1.4, color="#D36B32", label="Nominal")
    axes[1, 1].set_xticks(positions, ["90%", "95%"])
    axes[1, 1].set_ylim(0.86, 1.01)
    axes[1, 1].set_ylabel("Event coverage")
    axes[1, 1].set_title("(d) Fresh high-pollution intervals", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axes[1, 1].set_axisbelow(True)
    axes[1, 1].legend(frameon=False, fontsize=7.2, loc="lower right")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": "AirProof reviewer-requested common-resource, component, privacy and interval evidence",
            "Creator": "scripts/plot_v7_reviewer_revision.py",
        },
    )
    plt.close(figure)
    return OUTPUT


if __name__ == "__main__":
    print(execute())
