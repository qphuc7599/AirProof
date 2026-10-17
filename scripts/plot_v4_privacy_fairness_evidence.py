"""Render completed v4 mechanism/selection evidence, without refitting anything."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


def save(figure, output, name, source, metadata):
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{name}.{suffix}", dpi=240,
                       bbox_inches="tight", pad_inches=.035)
    plt.close(figure)
    provenance = {
        "input": str(source),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **metadata,
    }
    (output / f"{name}.sources.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    print(output / f"{name}.pdf")


def privacy_figure(source, output):
    report = json.loads(source.read_text(encoding="utf-8"))
    if (report["rows"] != 32400 or report["physical_worlds"] != 150
            or report["independent_seed_worlds_per_cell"] != 30
            or not report["zero_accounting_violations"]):
        raise ValueError("complete registered privacy campaign required")
    rows = {(row["cell"], row["epsilon_user_max"], row["scheduled_epochs"],
             row["k_min"], row["private_eligibility"]): row
            for row in report["interaction_cells"]}
    if len(rows) != 540 or len(rows) != len(report["interaction_cells"]):
        raise ValueError("missing or duplicate interaction summary")
    epsilons, schedules = [1, 4, 8, 32, 128, 512], [28, 112, 672]
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.15), layout="constrained",
                               gridspec_kw={"width_ratios": [1.23, 1]})
    colors = ["#0072B2", "#D55E00", "#009E73"]
    exported = []
    for schedule, color in zip(schedules, colors, strict=True):
        selected = [rows["severe_clean", epsilon, schedule, 20, False]
                    for epsilon in epsilons]
        if any(row["worlds_with_releases"] != 30 for row in selected):
            raise ValueError("no-gate curve unexpectedly lacks world support")
        for metric, style in (("raw_mean_rmse", "--"), ("residual_mean_rmse", "-")):
            values = [row["paired_utility"][metric] for row in selected]
            axes[0].plot(epsilons, values, style, color=color, marker="o",
                         markersize=3, linewidth=1.15)
        exported.extend(selected)
    chosen = report["prespecified_secondary_contrast"]["setting"]
    axes[0].scatter(chosen["epsilon_user_max"],
                    chosen["paired_utility"]["residual_mean_rmse"],
                    marker="*", s=115, color="#E69F00", edgecolor="#292929",
                    linewidth=.65, zorder=5)
    axes[0].set(xscale="log", yscale="log", xlabel="Full-history budget (epsilon)",
                ylabel="Mean release RMSE", title="(a) No gate: utility and cadence")
    axes[0].set_xticks(epsilons, [str(value) for value in epsilons])
    axes[0].grid(alpha=.18, linewidth=.6, which="major")
    handles = [Line2D([0], [0], color=color, lw=1.4, label=f"{count} queries")
               for count, color in zip(schedules, colors, strict=True)]
    handles.extend([Line2D([0], [0], color="#444444", linestyle="--", label="Raw DP"),
                    Line2D([0], [0], color="#444444", label="Residual DP")])
    axes[0].legend(handles=handles, fontsize=6.6, loc="upper right",
                   framealpha=.94, handlelength=1.8, borderpad=.4)
    coverage = np.empty((len(epsilons), len(schedules)))
    support = np.empty_like(coverage, dtype=int)
    for i, epsilon in enumerate(epsilons):
        for j, schedule in enumerate(schedules):
            row = rows["severe_clean", epsilon, schedule, 20, True]
            # The stored fraction uses every scored acquisition epoch;
            # normalize by the public schedule, not by emitted-only support.
            coverage[i, j] = 100 * row["mean_all_epoch_release_fraction"] * 672 / schedule
            support[i, j] = row["worlds_with_releases"]
            exported.append(row)
    if np.any(coverage < -1e-8) or np.any(coverage > 100 + 1e-8):
        raise ValueError("scheduled-release coverage outside its probability range")
    heat = axes[1].imshow(coverage, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    for (i, j), value in np.ndenumerate(coverage):
        axes[1].text(j, i, f"{value:.1f}", ha="center", va="center",
                     color="white" if value >= 55 else "#20252a", fontsize=7.5)
    axes[1].set_xticks(range(3), [str(value) for value in schedules])
    axes[1].set_yticks(range(6), [str(value) for value in epsilons])
    axes[1].set(xlabel="Scheduled queries over 672 epochs", ylabel="Full-history budget",
                title="(b) Private gate: emitted queries (%)")
    colorbar = figure.colorbar(heat, ax=axes[1], fraction=.055, pad=.025)
    colorbar.ax.tick_params(labelsize=7)
    save(figure, output, "v4_privacy_interactions", source, {
        "scope": "severe-clean k=20; 30 paired physical worlds per setting",
        "star": "prespecified epsilon=8, 28-query, no-gate residual mechanism",
        "heatmap_denominator": "all scheduled scored group queries, including suppressed queries",
        "gated_worlds_with_releases": support.tolist(),
        "plotted_summary_rows": exported,
    })


def frontier_figure(source, output):
    report = json.loads(source.read_text(encoding="utf-8"))
    if report["rows"] != 168 or len(report["seeds"]) != 8 or len(report["candidates"]) != 20:
        raise ValueError("complete final-estimator selection frontier required")
    strengths, targets = [0, .25, .5, .75, 1], [5, 10, 20, 30]
    indexed = {(row["target"], row["strength"]): row for row in report["candidates"].values()}
    selected = report["candidates"][report["selected"]]
    selected_cell = (strengths.index(selected["strength"]), targets.index(selected["target"]))
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 4.3), layout="constrained")
    metrics = [
        ("rmse", "(a) Overall RMSE reduction (%)", 6),
        ("worst_group_rmse", "(b) Worst-group RMSE reduction (%)", 15),
        ("coverage_gap", "(c) Contributor-gap reduction (%)", 32),
        (None, "(d) Complete floor-vector feasibility (%)", 100),
    ]
    values_out = {}
    for axis, (metric, title, high) in zip(axes.flat, metrics, strict=True):
        values = np.array([
            [(-100 * indexed[target, strength]["paired_relative_changes"][metric]["mean"]
              if metric else 100 * indexed[target, strength]["unconditional_global_floor_feasibility"])
             for strength in strengths] for target in targets])
        if not np.isfinite(values).all():
            raise ValueError("nonfinite selection outcome")
        values_out[metric or "full_floor_feasibility"] = values.tolist()
        low = min(0., float(values.min()))
        high = max(high, float(values.max()))
        display = np.ma.array(values, mask=np.zeros_like(values, dtype=bool))
        if metric is None:
            display.mask[:, 0] = True
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#eeeeee")
        heat = axis.imshow(display, cmap=cmap, vmin=low, vmax=high, aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            label = "off" if metric is None and j == 0 else f"{0. if abs(value) < .05 else value:.1f}"
            axis.text(j, i, label, ha="center", va="center", fontsize=7.5,
                      color="white" if (value - low) / (high - low) >= .55 else "#20252a")
        axis.add_patch(Rectangle((selected_cell[0] - .48, selected_cell[1] - .48),
                                 .96, .96, fill=False, edgecolor="#E69F00", linewidth=2))
        axis.set_xticks(range(5), ["0", ".25", ".5", ".75", "1"])
        axis.set_yticks(range(4), [str(value) for value in targets])
        axis.set(xlabel="Fairness strength", ylabel="Contributor target q", title=title)
        bar = figure.colorbar(heat, ax=axis, fraction=.045, pad=.025)
        bar.ax.tick_params(labelsize=7)
    save(figure, output, "v4_fairness_frontier", source, {
        "scope": "eight selection worlds; means of paired relative changes, not primary estimates",
        "orange_outline": report["selected"],
        "row_targets": targets, "column_strengths": strengths,
        "zero_strength": "floor disabled; feasibility counter not evaluated and shown as off",
        "plotted_values": values_out,
    })


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "source_paper/AirProof_Elsevier/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    privacy_figure(root / "reports/v4_validation/privacy_full_grid_7500_7529/analysis.json", output)
    frontier_figure(root / "reports/v4_validation/fairness_frontier_7400_7407/selection.json", output)


if __name__ == "__main__":
    main()
