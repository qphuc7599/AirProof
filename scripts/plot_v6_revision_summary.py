"""Build the source-bound v6 revision summary figure.

The figure reports historical-v4, changed-v6 validation, engineering, and
exposed-development evidence in separate panels.  It reads retained machine
artifacts and refuses missing, ambiguous, non-finite, or inconsistent inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v6" / "figures"
V4 = ROOT / "reports/v4_primary/core_final_7600_7629_20260903/analysis.json"
FAIR = ROOT / "reports/v6/fairness_validation_v1/analysis.json"
UNCERTAINTY_V2 = ROOT / "reports/v6/uncertainty_development_validation_v2/results.json"
UNCERTAINTY_V31 = ROOT / "reports/v6/uncertainty_event_v3_1/results.json"
UNCERTAINTY_V31_MANIFEST = ROOT / "reports/v6/uncertainty_event_v3_1/manifest.json"
STALE = ROOT / "reports/v6/integration_smoke_accountability_stale_gate_v2/outcome.json"
BATCH = ROOT / "reports/v6/integration_smoke_accountability_batch_v3/outcome.json"
DURABLE = ROOT / "reports/v6/durable_inclusion_async_return_v1/results.json"
PUBLIC_CFG = ROOT / "configs/v6_durable_public_checkpoint_v2.json"
ROLLING = ROOT / "reports/v6/epa_2025_public_rolling/selection_lock.json"
SPATIOTEMPORAL = ROOT / "reports/v6/epa_2025_public_spatiotemporal/selection_lock.json"
RICH = ROOT / "reports/v6/epa_2025_public_rich_forest/selection_lock.json"
CAP_CONTROL = ROOT / "reports/v6/epa_2025_spatiotemporal_controls/result.json"
PROJECTED = ROOT / "reports/v6/epa_2025_projected_output_cap/result.json"
PUBLIC_RESULTS = (
    ROOT / "reports/v6/durable_public_checkpoint_v2/result.json",
    ROOT / "reports/v6/durable_public_checkpoint_v2/results.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"required figure input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def need(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError("missing figure field: " + ".".join(map(str, keys)))
        value = value[key]
    return value


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} is NaN or infinite")
    return value


def selected_metrics(lock, label: str):
    selected = need(lock, "selected")
    matches = [row for row in need(lock, "development") if row.get("id") == selected]
    if len(matches) != 1:
        raise ValueError(f"{label} has {len(matches)} selected rows")
    metrics = need(matches[0], "metrics")
    return {
        "rmse": finite(need(metrics, "rmse"), f"{label}.rmse"),
        "floor": finite(need(metrics, "cap8_floor"), f"{label}.cap8_floor"),
    }


def collect():
    required = [V4, FAIR, UNCERTAINTY_V2, UNCERTAINTY_V31,
                UNCERTAINTY_V31_MANIFEST, STALE, BATCH, DURABLE, PUBLIC_CFG, ROLLING,
                SPATIOTEMPORAL, RICH, CAP_CONTROL, PROJECTED]
    data = {str(path.relative_to(ROOT)): load(path) for path in required}
    v4, fair = data[str(V4.relative_to(ROOT))], data[str(FAIR.relative_to(ROOT))]
    if need(v4, "worlds") != 30 or need(v4, "all_primary_hypotheses_rejected") is not True:
        raise ValueError("unexpected retained-v4 primary scope/disposition")
    v4_h = []
    for name in [f"H{i}" for i in range(1, 8)]:
        row = need(v4, "primary_hypotheses", name)
        if need(row, "reject_at_familywise_0.05") is not True:
            raise ValueError(f"retained v4 {name} is not Holm-rejected")
        v4_h.append(finite(need(row, "mean_world_contrast"), f"v4.{name}"))

    if need(fair, "complete_matrix") is not True or need(fair, "worlds") != 12 or need(fair, "evaluations") != 144:
        raise ValueError("unexpected fairness-validation matrix")
    h_means, h_low, h_high = [], [], []
    for name in ("H6", "H7"):
        row = need(fair, "hypotheses", name)
        if need(row, "descriptive_threshold_met") is not False:
            raise ValueError(f"changed-v6 {name} disposition unexpectedly changed")
        ci = need(row, "descriptive_95_ci")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError(f"{name} interval must contain two endpoints")
        h_means.append(finite(need(row, "mean"), f"fair.{name}.mean"))
        h_low.append(finite(ci[0], f"fair.{name}.low"))
        h_high.append(finite(ci[1], f"fair.{name}.high"))

    arms = need(fair, "arm_means")
    v6 = need(arms, "v6_minimax_R1_B1")
    utility = need(arms, "utility_only_R1_B1")
    service_v6 = finite(need(v6, "minimum_24epoch_capped_service", "mean"), "service_v6")
    service_utility = finite(need(utility, "minimum_24epoch_capped_service", "mean"), "service_utility")
    debt_v6 = finite(need(v6, "total_distinct_user_debt", "mean"), "debt_v6")
    debt_utility = finite(need(utility, "total_distinct_user_debt", "mean"), "debt_utility")
    gap_v6 = finite(need(v6, "coverage_gap", "mean"), "gap_v6")
    gap_utility = finite(need(utility, "coverage_gap", "mean"), "gap_utility")

    uncertainty_v2 = data[str(UNCERTAINTY_V2.relative_to(ROOT))]
    uncertainty_v31 = data[str(UNCERTAINTY_V31.relative_to(ROOT))]
    uncertainty_manifest = data[str(UNCERTAINTY_V31_MANIFEST.relative_to(ROOT))]
    if need(uncertainty_manifest, "post_v2_failure_design") is not True or need(uncertainty_manifest, "evaluation_residual_updates") != 0:
        raise ValueError("unexpected v3.1 uncertainty provenance")
    v2_event = need(uncertainty_v2, "anchor_clean", "live", "groups", "event", "coverage", "estimate")
    v31_live = need(uncertainty_v31, "live")
    v31_reconstructed = need(uncertainty_v31, "reconstructed")
    v31_event = need(v31_live, "groups", "event", "coverage", "estimate")
    v31_all = need(v31_live, "groups", "all", "coverage", "estimate")
    for label, pair in (("v2.event", v2_event), ("v3.1.event", v31_event), ("v3.1.all", v31_all)):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{label} must contain 90/95 values")
        pair[:] = [finite(value, f"{label}[{i}]") for i, value in enumerate(pair)]
    if need(v31_reconstructed, "groups", "event", "coverage", "estimate") != v31_event:
        raise ValueError("live/reconstructed v3.1 event results differ unexpectedly")
    if not all(new < old for new, old in zip(v31_event, v2_event)):
        raise ValueError("v3.1 adverse comparison to v2 unexpectedly changed")

    modes = []
    stale = data[str(STALE.relative_to(ROOT))]
    batch = data[str(BATCH.relative_to(ROOT))]
    durable = data[str(DURABLE.relative_to(ROOT))]
    modes.append(("Per-record\nstale gate", finite(need(stale, "selected"), "stale.selected"),
                  finite(need(stale, "transport", "audit_verified_return_pairs"), "stale.verified"), None))
    modes.append(("Terminal\nbatch v3", finite(need(batch, "selected"), "batch.selected"),
                  finite(need(batch, "origin_verified"), "batch.verified"), None))
    modes.append(("Durable async\nreturn", finite(need(durable, "selected"), "durable.selected"),
                  finite(need(durable, "origin_verified"), "durable.verified"),
                  finite(need(durable, "publicly_checkpointed"), "durable.public")))
    if any(finite(need(row, "observations"), "accountability.observations") != 529
           for row in (stale, batch, durable)):
        raise ValueError("accountability modes do not share the registered 529-observation fixture")

    existing_public = [path for path in PUBLIC_RESULTS if path.exists()]
    if len(existing_public) > 1:
        raise RuntimeError("ambiguous public-checkpoint result paths")
    public_status = "Registered public checkpoint v2\nresult pending"
    if existing_public:
        public = load(existing_public[0])
        data[str(existing_public[0].relative_to(ROOT))] = public
        modes.append(("Public\ncheckpoint v2",
                      finite(need(public, "selected"), "public.selected"),
                      finite(need(public, "origin_verified"), "public.verified"),
                      finite(need(public, "publicly_checkpointed"), "public.checkpointed")))
        public_status = "Public checkpoint v2: completed engineering fixture"
    else:
        cfg = data[str(PUBLIC_CFG.relative_to(ROOT))]
        if need(cfg, "registration") != "v6-durable-public-checkpoint-v2":
            raise ValueError("unexpected public-checkpoint registration")

    epa = [
        ("Rolling\nridge", selected_metrics(data[str(ROLLING.relative_to(ROOT))], "rolling")),
        ("Spatiotemporal\nridge", selected_metrics(data[str(SPATIOTEMPORAL.relative_to(ROOT))], "spatiotemporal")),
        ("Rich\nforest", selected_metrics(data[str(RICH.relative_to(ROOT))], "rich")),
    ]
    cap_ratio = finite(need(data[str(CAP_CONTROL.relative_to(ROOT))], "ratios_to_quadratic_unbounded", "quadratic_cap8"), "cap ratio")
    projected_ratio = finite(need(data[str(PROJECTED.relative_to(ROOT))], "clean_ratio"), "projected ratio")
    if need(data[str(CAP_CONTROL.relative_to(ROOT))], "cap8_compatible_with_clean_gate") is not False:
        raise ValueError("constrained cap disposition changed")
    if need(data[str(PROJECTED.relative_to(ROOT))], "passes_1_05") is not False:
        raise ValueError("projected cap disposition changed")
    return data, dict(v4=v4_h, fair=(h_means, h_low, h_high),
                      secondary=(service_v6, service_utility, debt_v6, debt_utility, gap_v6, gap_utility),
                      uncertainty=(v2_event, v31_event, v31_all),
                      modes=modes, public_status=public_status, epa=epa,
                      cap=(cap_ratio, projected_ratio))


def plot(values):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 10,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.4, "ytick.labelsize": 7.7,
        "legend.fontsize": 7.2, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    blue, orange, red, gray, green = "#2774AE", "#E67E22", "#C0392B", "#6C757D", "#2A9D8F"
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.15), constrained_layout=True)

    ax = axes[0, 0]
    y = np.arange(7)
    ax.barh(y, values["v4"], color=blue, height=.68)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, [f"H{i}" for i in range(1, 8)])
    ax.invert_yaxis()
    ax.set_xlabel("Registered mean contrast (threshold: < 0)")
    # Keep the evidence-role subtitle on its own baseline.  A trailing newline
    # in the title made Matplotlib reserve an implicit second title line that
    # overlapped this separately styled subtitle in the vector export.
    ax.set_title("A  Retained historical v4 primary", loc="left", fontweight="bold", pad=24)
    ax.text(0, 1.008, "30 worlds · all seven Holm-rejected · historical v4 only",
            transform=ax.transAxes, ha="left", va="bottom", color=gray, fontsize=7.0)
    ax.grid(axis="x", color="#E5E7EB", lw=.7)

    ax = axes[0, 1]
    means, low, high = map(np.asarray, values["fair"])
    err = np.vstack((means - low, high - means))
    ax.errorbar(means, [0, 1], xerr=err, fmt="o", color=red, ecolor=red,
                capsize=4, markersize=6, lw=1.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks([0, 1], ["H6 gap contrast", "H7 WG-RMSE contrast"])
    ax.invert_yaxis()
    ax.set_xlabel("Changed-v6 contrast (required < 0)")
    ax.set_title("B  Changed-v6 validation outcomes", loc="left", fontweight="bold")
    sv, su, dv, du, gv, gu = values["secondary"]
    text_box = (f"Both targets FAIL (12 worlds)\n"
                f"Debt: {du:.0f} → {dv:.0f} ({(dv/du-1)*100:+.1f}%)\n"
                f"Min 24-epoch service: {su:.3f} → {sv:.3f}\n"
                f"Coverage-gap change: {(gv/gu-1)*100:+.2f}%")
    ax.text(.98, .96, text_box, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=.35", fc="#FFF7ED", ec=orange, lw=.8), fontsize=7.2)
    v2_event, v31_event, v31_all = values["uncertainty"]
    uncertainty_box = ("Uncertainty v3.1 after v2 failure\n"
                       f"Event 90/95: {v2_event[0]*100:.2f}/{v2_event[1]*100:.2f}% → "
                       f"{v31_event[0]*100:.2f}/{v31_event[1]*100:.2f}%\n"
                       f"Overall: {v31_all[0]*100:.2f}/{v31_all[1]*100:.2f}% · family closed")
    ax.text(.52, .52, uncertainty_box, transform=ax.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=.35", fc="#FEF2F2", ec=red, lw=.8), fontsize=7.0)
    ax.grid(axis="x", color="#E5E7EB", lw=.7)

    ax = axes[1, 0]
    modes = values["modes"]
    x = np.arange(len(modes)); width = .25
    selected = [row[1] for row in modes]; verified = [row[2] for row in modes]
    public = [np.nan if row[3] is None else row[3] for row in modes]
    ax.bar(x - width, selected, width, label="Selected/included", color=blue)
    ax.bar(x, verified, width, label="Origin verified", color=green)
    if np.any(np.isfinite(public)):
        ax.bar(x + width, np.nan_to_num(public), width, label="Publicly checkpointed",
               color=orange, hatch="///")
    ax.set_xticks(x, [row[0] for row in modes])
    ax.set_ylabel("Records (same 529-observation fixture)")
    ax.set_title("C  Accountability throughput", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    if "pending" in values["public_status"].lower():
        ax.text(.98, .96, values["public_status"], transform=ax.transAxes, ha="right", va="top",
                color=gray, fontsize=7.2, bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#B8BDC4"))
    ax.grid(axis="y", color="#E5E7EB", lw=.7)

    ax = axes[1, 1]
    epa = values["epa"]; x = np.arange(len(epa)); width = .34
    rmses = [row[1]["rmse"] for row in epa]; floors = [row[1]["floor"] for row in epa]
    ax.bar(x - width/2, rmses, width, label="Public RMSE", color=blue)
    ax.bar(x + width/2, floors, width, label="Cap-8 oracle floor", color=orange)
    ax.set_xticks(x, [row[0] for row in epa])
    ax.set_ylabel("PM$_{2.5}$ RMSE (µg m$^{-3}$)")
    ax.set_title("D  EPA exposed development", loc="left", fontweight="bold")
    ax.set_ylim(0, 5.65)
    cr, pr = values["cap"]
    ax.text(.98, .96, f"Rich forest is worse; search stops\nCap diagnostics: {cr:.3f}, {pr:.3f} > 1.05 gate\nValidation/tests unread",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.1,
            bbox=dict(boxstyle="round,pad=.35", fc="#FEF2F2", ec=red, lw=.8))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#E5E7EB", lw=.7)

    fig.suptitle("AirProof evidence-preserving revision summary", fontsize=12.2, fontweight="bold")
    return fig


def main():
    data, values = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plot(values)
    pdf, png = OUT / "revision_summary.pdf", OUT / "revision_summary.png"
    metadata = {"Title": "AirProof evidence-preserving revision summary",
                "Author": "AirProof artifact pipeline", "Creator": "plot_v6_revision_summary.py"}
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.04, metadata=metadata)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=.04,
                metadata={"Title": metadata["Title"], "Software": metadata["Creator"]})
    plt.close(fig)
    if not pdf.is_file() or not png.is_file() or min(pdf.stat().st_size, png.stat().st_size) == 0:
        raise RuntimeError("figure export failed")
    inputs = {relative: sha256(ROOT / relative) for relative in sorted(data)}
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "publication figure from retained historical, validation, engineering, and exposed-development artifacts",
        "script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__))},
        "inputs": inputs,
        "outputs": {str(pdf.relative_to(ROOT)): sha256(pdf), str(png.relative_to(ROOT)): sha256(png)},
        "public_checkpoint_result_present": any(path.exists() for path in PUBLIC_RESULTS),
        "scope": {
            "panel_a": "historical v4 primary only",
            "panel_b": "changed-v6 descriptive fairness validation and post-v2 exposed uncertainty development",
            "panel_c": "single engineering fixture; no scientific/availability/finality claim",
            "panel_d": "sequential exposed EPA development; validation and test unread",
        },
    }
    (OUT / "revision_summary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
