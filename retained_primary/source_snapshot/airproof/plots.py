from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_summary_figure(csv_path: str | Path, output: str | Path) -> Path:
    frame = pd.read_csv(csv_path)
    required = {"method", "worst_group_rmse", "aoi_p95", "coverage_gap", "delivery_ratio"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    summary = frame.groupby("method", sort=True)[
        ["worst_group_rmse", "aoi_p95", "coverage_gap", "delivery_ratio"]
    ].agg(["mean", "sem"])
    metrics = [
        ("worst_group_rmse", "Worst-group RMSE", "lower is better"),
        ("aoi_p95", "Censored AoI p95", "lower is better"),
        ("coverage_gap", "Coverage gap", "lower is better"),
        ("delivery_ratio", "Delivery ratio", "higher is better"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = plt.cm.viridis([index / max(1, len(summary) - 1) for index in range(len(summary))])
    for axis, (metric, title, direction) in zip(axes.ravel(), metrics):
        means = summary[(metric, "mean")]
        errors = summary[(metric, "sem")].fillna(0)
        axis.bar(range(len(means)), means.values, yerr=errors.values, color=colors, capsize=3)
        axis.set_xticks(range(len(means)), means.index, rotation=30, ha="right")
        axis.set_title(f"{title} ({direction})")
        axis.grid(axis="y", alpha=0.25)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle("AirProof measured simulation summary (mean +/- paired-world SEM)")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def create_privacy_figure(csv_path: str | Path, output: str | Path) -> Path:
    frame = pd.read_csv(csv_path)
    grouped = frame.groupby(["epsilon_mean", "method"], sort=True).agg(
        release_rmse=("release_rmse", "mean"),
        suppression=("suppressed_release_count", "mean"),
        declined=("declined_privacy_users", "mean"),
    ).reset_index()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for method, method_frame in grouped.groupby("method"):
        method_frame = method_frame.sort_values("epsilon_mean")
        axes[0].plot(method_frame["epsilon_mean"], method_frame["release_rmse"], marker="o", label=method)
        axes[1].plot(method_frame["epsilon_mean"], method_frame["suppression"], marker="o", label=method)
        axes[2].plot(method_frame["epsilon_mean"], method_frame["declined"], marker="o", label=method)
    axes[0].set_yscale("log")
    for axis, title, ylabel in zip(
        axes,
        ("Released-mean error", "Suppressed releases", "Budget-declined users"),
        ("RMSE (log scale)", "Mean count", "Mean count"),
    ):
        axis.set_xscale("log")
        axis.set_xlabel("epsilon per mean release")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle("AirProof measured privacy-utility/accountant curve")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def create_attack_figure(csv_path: str | Path, output: str | Path) -> Path:
    frame = pd.read_csv(csv_path)
    grouped = frame.groupby(["attack_kind", "attack_fraction", "method"], sort=True)["worst_group_rmse"].mean().reset_index()
    kinds = sorted(grouped["attack_kind"].unique())
    figure, axes = plt.subplots(1, len(kinds), figsize=(15, 4.5), sharey=True, constrained_layout=True)
    for axis, kind in zip(axes, kinds):
        subset = grouped[grouped["attack_kind"] == kind]
        for method, method_frame in subset.groupby("method"):
            axis.plot(method_frame["attack_fraction"] * 100, method_frame["worst_group_rmse"], marker="o", label=method)
        axis.set_title(kind.replace("_", " "))
        axis.set_xlabel("Corrupted agents (%)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Worst-group RMSE")
    axes[-1].legend(loc="best")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle("AirProof measured attack stress curve")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def create_scale_figure(csv_path: str | Path, output: str | Path, x_column: str) -> Path:
    frame = pd.read_csv(csv_path)
    grouped = frame.groupby(x_column, sort=True).agg(
        runtime=("runtime_seconds", "mean"),
        peak_memory=("peak_memory_mb", "mean"),
        rmse=("rmse", "mean"),
        failures=("solver_failure_rate", "mean"),
    ).reset_index()
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, column, title in zip(
        axes.ravel(),
        ("runtime", "peak_memory", "rmse", "failures"),
        ("Runtime (s)", "Peak traced memory (MB)", "RMSE", "Solver failure rate"),
    ):
        axis.plot(grouped[x_column], grouped[column], marker="o")
        axis.set_xlabel(x_column.replace("_", " "))
        axis.set_title(title)
        axis.grid(alpha=0.25)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle(f"AirProof measured one-axis scalability: {x_column}")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
