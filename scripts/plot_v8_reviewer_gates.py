#!/usr/bin/env python3
"""Render the independently confirmed protected-release and wrapper gates."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    release = json.loads((
        ROOT / "reports/v8/m5_release_reference_stress_confirmation_v1/analysis.json"
    ).read_text())
    wrapper = json.loads((
        ROOT / "reports/v8/external_wrapper_gate_c_v1/confirmation_v1/summary.json"
    ).read_text())["candidate_summaries"][0]
    boot = wrapper["descriptive_bootstrap"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
    })
    colors = {"public": "#4C78A8", "protected": "#F58518", "raw": "#54A24B"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)

    ax = axes[0, 0]
    labels = ["Public only", "Protected pool"]
    values = [release["means"]["main_public_rmse"], release["means"]["main_protected_rmse"]]
    bars = ax.bar(labels, values, color=[colors["public"], colors["protected"]], width=.62)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)
    ax.set_ylabel("RMSE")
    ax.set_title("(a) Nominal reference regime")
    ax.set_ylim(0, max(values) * 1.22)

    ax = axes[0, 1]
    labels = ["Public only", "Protected pool", "Raw-DP pool"]
    values = [release["means"]["stress_public_rmse"], release["means"]["stress_protected_rmse"],
              release["means"]["stress_nonnegative_raw_pool_rmse"]]
    bars = ax.bar(labels, values, color=[colors["public"], colors["protected"], colors["raw"]], width=.62)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)
    ax.tick_params(axis="x", labelrotation=14)
    ax.set_ylabel("RMSE")
    ax.set_title(r"(b) Fixed $\pm4$ reference stress")
    ax.set_ylim(0, max(values) * 1.22)

    ax = axes[1, 0]
    regimes = ["Neutral", "+2 shift"]
    keys = ["neutral", "high2"]
    points = np.asarray([wrapper["clean_ratios"][key] for key in keys])
    lower = np.asarray([boot["clean_ratio"][key]["lower95"] for key in keys])
    upper = np.asarray([boot["clean_ratio"][key]["upper95"] for key in keys])
    x = np.arange(len(keys))
    ax.errorbar(x, points, yerr=np.vstack((points - lower, upper - points)), fmt="o",
                color=colors["protected"], capsize=3, lw=1.2)
    ax.axhline(1.05, color="#B22222", ls="--", lw=1, label="1.05 threshold")
    ax.set_xticks(x, regimes)
    ax.set_ylabel("Bounded / Huber clean RMSE")
    ax.set_ylim(.995, 1.06)
    ax.set_title("(c) Equal-information clean retention")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    attack_keys = list(wrapper["attacks"])
    short = ["N drift", "N negative", "N late", "+2 drift", "+2 negative"]
    points = np.asarray([wrapper["attacks"][key]["attenuation"] for key in attack_keys])
    lower = np.asarray([boot["attack_attenuation"][key]["lower95"] for key in attack_keys])
    upper = np.asarray([boot["attack_attenuation"][key]["upper95"] for key in attack_keys])
    x = np.arange(len(points))
    ax.errorbar(x, points, yerr=np.vstack((points - lower, upper - points)), fmt="o",
                color="#7A5195", capsize=3, lw=1.2)
    ax.axhline(.20, color="#B22222", ls="--", lw=1, label="20% target")
    ax.set_xticks(x, short, rotation=23, ha="right")
    ax.set_ylabel("Attack-growth attenuation")
    ax.set_title("(d) External corruption overlays")
    ax.set_ylim(0, max(1.42, float(upper.max()) * 1.05))
    ax.legend(frameon=False, loc="upper left")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", lw=.5, alpha=.65)
    output = ROOT / "overleaf/figs/reviewer_gate_confirmation.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
