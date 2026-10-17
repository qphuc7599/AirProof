"""Publication transport figure from completed synthetic and empirical summaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    paths = {
        "synthetic": root / "reports/v4_validation/dtn_deadline_confirm_7950_7979/summary.json",
        "empirical": root / "reports/v4_validation/empirical_contacts_8050_8054/summary.json",
    }
    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    if reports["synthetic"]["rows"] != 210 or reports["synthetic"]["traces"] != 30:
        raise ValueError("complete synthetic confirmation required")
    if reports["empirical"]["jobs"] != 150 or len(reports["empirical"]["cells"]) != 6:
        raise ValueError("complete empirical collection/load matrix required")
    out = root / "source_paper/AirProof_Elsevier/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.35, 3.1), layout="constrained",
                                        gridspec_kw={"width_ratios": [1.05, 1]})
    styles = {
        "direct": ("Direct", (8, -8)),
        "ttl_one_copy": ("TTL", (-24, 17)),
        "prophet_gtmx": ("PRoPHET", (-70, 3)),
        "binary_spray_wait": ("Spray-and-Wait", (-88, -23)),
        "epidemic_cap": ("Capped epidemic", (-67, 44)),
        "airproof_deficit": ("Deficit-only relay", (-64, -19)),
        "airproof_deadline": ("AirProof", (-30, -26)),
    }
    for method, (label, offset) in styles.items():
        row = reports["synthetic"]["means"][method]
        point = (100 * row["deadline_delivery_ratio"], row["restricted_mean_delivery_delay"])
        color = "#0072B2" if method == "airproof_deadline" else "#626a73"
        marker = "D" if method == "airproof_deadline" else "o"
        left.scatter(*point, s=37 if method == "airproof_deadline" else 25,
                     color=color, marker=marker, zorder=3)
        arrow = {"arrowstyle": "-", "color": color, "lw": .6}
        if method == "epidemic_cap":
            arrow["connectionstyle"] = "angle,angleA=0,angleB=90,rad=2"
        left.annotate(label, point, xytext=offset, textcoords="offset points", fontsize=7,
            color=color, arrowprops=arrow,
            fontweight="bold" if method == "airproof_deadline" else "normal")
    left.set(xlim=(71, 102), ylim=(4.55, 12), xlabel="Deadline delivery (%)",
             ylabel="All-record restricted delay (slots)", title="(a) Independent synthetic traces")
    left.grid(alpha=.17, linewidth=.6)
    left.text(.02, .02, "30 paired traces; lower-right is better", transform=left.transAxes, fontsize=6.8)
    labels, values, lower, upper = [], [], [], []
    for key, row in reports["empirical"]["cells"].items():
        dataset, load = key.split(":")
        labels.append(dataset + (" / low" if float(load) == .002 else " / high"))
        effect = row["paired_vs_registered_spray"]["relative_delay_reduction"]
        values.append(100 * effect["mean"])
        lower.append(100 * effect["lower95"])
        upper.append(100 * effect["upper95"])
    values = np.array(values)
    y = np.arange(len(values))
    right.errorbar(values, y, xerr=[values - lower, upper - values], fmt="o",
                   color="#0072B2", capsize=3, markersize=4, lw=1.1)
    right.set_yticks(y, labels)
    right.invert_yaxis()
    right.axvline(0, color="#444444", linestyle="--", lw=.8)
    right.set(xlim=(-.12, 2.0), xlabel="Restricted-delay reduction vs Spray (%)",
              title="(b) Measured-contact replay")
    right.grid(axis="x", alpha=.17, linewidth=.6)
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"v4_transport_evidence.{suffix}", dpi=240,
                       bbox_inches="tight", pad_inches=.035)
    plt.close(figure)
    (out / "v4_transport_evidence.sources.json").write_text(json.dumps({
        "inputs": {key: {"path": str(path.relative_to(root)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for key, path in paths.items()},
        "synthetic_unit": "30 independent contact traces; policy means",
        "empirical_unit": "five conditional role/traffic replications per collection/load",
        "empirical_intervals": "descriptive paired-bootstrap 95 percent; not population-level replication",
    }, indent=2), encoding="utf-8")
    print(out / "v4_transport_evidence.pdf")


if __name__ == "__main__":
    main()
