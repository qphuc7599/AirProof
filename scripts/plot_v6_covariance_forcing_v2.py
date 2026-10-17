"""Create the publication plot for the frozen covariance/forcing development family."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reports/v6/innovation_covariance_forcing_development_v2/outcomes/analysis.json"
SCRIPT = Path(__file__).resolve()
FIGURE_DIR = ROOT / "reports/v6/figures"
PDF = FIGURE_DIR / "covariance_forcing_v2.pdf"
PNG = FIGURE_DIR / "covariance_forcing_v2.png"
MANIFEST = FIGURE_DIR / "covariance_forcing_v2.sources.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    if (analysis.get("gate_evaluation") is None
            or analysis["matrix_audit"].get("complete_rows") != 384
            or analysis["matrix_audit"].get("pass") is not True
            or analysis["invariant_audit"].get("pass") is not True):
        raise SystemExit("complete verified development analysis required")
    decisions = analysis["gate_evaluation"]["candidates"]
    labels = ["I-0", "I-F", "CF-0", "CF-F"]
    if [row["candidate"] for row in decisions] != [
            "independence__zero",
            "independence__producer_exact_causal",
            "world_cross_fitted_covariance__zero",
            "world_cross_fitted_covariance__producer_exact_causal",
    ]:
        raise ValueError("candidate order differs from frozen 2x2 registration")
    x = np.arange(4)
    clean = np.asarray([row["metrics"]["clean_ratio"] for row in decisions])
    drift = 100 * np.asarray([row["metrics"]["attenuation"]["severe_drift"]
                             for row in decisions])
    hotspot = 100 * np.asarray([row["metrics"]["attenuation"]["severe_hotspot"]
                               for row in decisions])
    event_drift = np.asarray([row["metrics"]["event_loss_percentage_points"]["severe_drift"]
                              for row in decisions])
    event_hotspot = np.asarray([row["metrics"]["event_loss_percentage_points"]["severe_hotspot"]
                                for row in decisions])

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "figure.dpi": 150})
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True)
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    axes[0].bar(x, clean, color=colors, width=.72)
    axes[0].axhline(1.05, color="black", linestyle="--", linewidth=.9, label="gate")
    axes[0].set_title("(a) Clean noninferiority")
    axes[0].set_ylabel("Candidate / SQ RMSE")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)

    width = .36
    axes[1].bar(x - width / 2, drift, width, label="drift", color="#4C78A8")
    axes[1].bar(x + width / 2, hotspot, width, label="hotspot", color="#F58518")
    axes[1].axhline(20, color="black", linestyle="--", linewidth=.9, label="gate")
    axes[1].set_title("(b) Attack-growth attenuation")
    axes[1].set_ylabel("Attenuation (%)")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False, ncols=3, loc="best")

    axes[2].scatter(x - width / 2, event_drift, marker="o", s=20,
                    label="drift", color="#4C78A8")
    axes[2].scatter(x + width / 2, event_hotspot, marker="s", s=20,
                    label="hotspot", color="#F58518")
    axes[2].axhline(5, color="black", linestyle="--", linewidth=.9, label="gate")
    axes[2].set_yscale("symlog", linthresh=.001)
    axes[2].set_ylim(0, 6.2)
    axes[2].set_title("(c) True-event preservation")
    axes[2].set_ylabel("Recall loss vs SQ (pp)")
    axes[2].set_xticks(x, labels)
    axes[2].legend(frameon=False, ncols=3, loc="best")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2, linewidth=.5)
    figure.suptitle("Exposed mechanistic development; eight paired worlds", fontsize=9)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF, bbox_inches="tight")
    figure.savefig(PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)
    manifest = {
        "schema_version": 1,
        "role": "publication figure from complete exposed development analysis",
        "analysis": str(ANALYSIS.relative_to(ROOT)).replace("\\", "/"),
        "analysis_sha256": sha256_file(ANALYSIS),
        "script": str(SCRIPT.relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(SCRIPT),
        "pdf_sha256": sha256_file(PDF),
        "png_sha256": sha256_file(PNG),
        "historical_v4_rerun": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
