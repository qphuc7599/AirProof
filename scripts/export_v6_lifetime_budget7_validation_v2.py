from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reports/v6/lifetime_budget7_validation_v2/outcomes/analysis.json"
REGISTRATION = ROOT / "configs/v6/lifetime_budget7_validation_v2.json"
DOC = ROOT / "docs/V6_LIFETIME_BUDGET7_VALIDATION_V2_RESULTS.md"
TEX = ROOT / "source_paper/AirProof_Elsevier/generated/v6_lifetime_budget7_validation_v2.tex"
FIGURE_PDF = ROOT / "reports/v6/figures/lifetime_budget7_validation_v2.pdf"
FIGURE_PNG = ROOT / "reports/v6/figures/lifetime_budget7_validation_v2.png"


ORDER = (
    "clean:anchor_clean",
    "clean:outage_clean",
    "clean:severe_clean",
    "attack:severe_drift",
    "event:severe_drift",
    "attack:severe_hotspot",
    "event:severe_hotspot",
    "citizen_value:severe_clean",
)
LABELS = {
    "clean:anchor_clean": "Clean ratio: anchor",
    "clean:outage_clean": "Clean ratio: outage",
    "clean:severe_clean": "Clean ratio: severe",
    "attack:severe_drift": "Drift attenuation",
    "event:severe_drift": "Drift event loss",
    "attack:severe_hotspot": "Hotspot attenuation",
    "event:severe_hotspot": "Hotspot event loss",
    "citizen_value:severe_clean": "Citizen AP/public ratio",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_bound(key: str, row: dict) -> tuple[float | None, str]:
    if key.startswith("attack:"):
        return (None if row["lower"] is None else float(row["lower"])), "lower"
    return (None if row["upper"] is None else float(row["upper"])), "upper"


def result_table(tests: dict) -> str:
    rows = []
    for key in ORDER:
        item = tests[key]
        bound, side = decision_bound(key, item)
        bound_text = "undefined (-inf)" if bound is None else f"{bound:.6f}"
        rows.append(
            f"| {LABELS[key]} | {item['estimate']:.6f} | {side} {bound_text} | "
            f"{item['threshold']:.6f} | {'PASS' if item['pass'] else 'FAIL'} |"
        )
    return "\n".join(rows)


def latex_table(tests: dict) -> str:
    rows = []
    for key in ORDER:
        item = tests[key]
        bound, side = decision_bound(key, item)
        bound_text = r"undefined ($-\infty$)" if bound is None else f"{bound:.6f}"
        label = LABELS[key].replace("%", r"\%")
        rows.append(
            f"{label} & {item['estimate']:.6f} & {side} {bound_text} & "
            f"{item['threshold']:.6f} & {'pass' if item['pass'] else 'fail'} \\\\" 
        )
    return "\n".join(rows)


def plot(tests: dict, analysis_digest: str) -> None:
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), constrained_layout=True)

    ratio_keys = ORDER[:3] + (ORDER[-1],)
    axis = axes[0]
    for index, key in enumerate(ratio_keys):
        item = tests[key]
        axis.errorbar(
            item["estimate"], index,
            xerr=[[item["estimate"] - item["lower"]], [item["upper"] - item["estimate"]]],
            fmt="o", color="#1f5a99" if item["pass"] else "#b0202a", capsize=3,
        )
    axis.axvline(1.05, color="#555555", linestyle="--", linewidth=1)
    axis.axvline(1.0, color="#2d7f4f", linestyle=":", linewidth=1)
    axis.set_yticks(range(len(ratio_keys)), [LABELS[key] for key in ratio_keys])
    axis.set_xlabel("AP/control RMSE ratio (simultaneous interval)")
    axis.invert_yaxis()

    attack_keys = ("attack:severe_drift", "attack:severe_hotspot")
    axis = axes[1]
    for index, key in enumerate(attack_keys):
        item = tests[key]
        if item["lower"] is None:
            axis.plot(100 * item["estimate"], index, marker="x", markersize=8,
                      color="#b0202a", linestyle="none")
            axis.annotate("LB undefined", (100 * item["estimate"], index),
                          xytext=(5, 7), textcoords="offset points", fontsize=8)
        else:
            axis.errorbar(
                100 * item["estimate"], index,
                xerr=[[100 * (item["estimate"] - item["lower"])],
                      [100 * (item["upper"] - item["estimate"])]],
                fmt="o", color="#1f5a99" if item["pass"] else "#b0202a", capsize=3,
            )
    axis.axvline(20, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(range(2), ["Drift", "Hotspot"])
    axis.set_xlabel("Excess-RMSE attenuation (percent)")
    axis.invert_yaxis()

    event_keys = ("event:severe_drift", "event:severe_hotspot")
    axis = axes[2]
    for index, key in enumerate(event_keys):
        item = tests[key]
        axis.errorbar(
            item["estimate"], index,
            xerr=[[item["estimate"] - item["lower"]], [item["upper"] - item["estimate"]]],
            fmt="o", color="#1f5a99" if item["pass"] else "#b0202a", capsize=3,
        )
    axis.axvline(5, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(range(2), ["Drift", "Hotspot"])
    axis.set_xlabel("Event-recall loss (percentage points)")
    axis.invert_yaxis()

    fig.suptitle("Locked 12-world lifetime-budget-7 validation")
    metadata = {"Title": "AirProof lifetime-budget-7 validation v2",
                "Subject": f"Analysis SHA-256 {analysis_digest}"}
    fig.savefig(FIGURE_PDF, metadata=metadata)
    fig.savefig(FIGURE_PNG, dpi=220, metadata={"Description": metadata["Subject"]})
    plt.close(fig)


def main() -> None:
    analysis = load_json(ANALYSIS)
    registration = load_json(REGISTRATION)
    matrix = analysis.get("matrix_audit", {})
    gates = analysis.get("gate_evaluation")
    registered_rows = (
        len(registration["worlds"]["seeds"])
        * len(registration["worlds"]["cells"])
        * len(registration["methods"])
    )
    if (
        registered_rows != 240
        or matrix.get("pass") is not True
        or matrix.get("complete_rows") != registered_rows
    ):
        raise RuntimeError("complete hash-verified 240-row analysis required")
    if not isinstance(gates, dict) or set(gates.get("tests", {})) != set(ORDER):
        raise RuntimeError("all eight registered gate results required")
    tests = gates["tests"]
    failed = [key for key in ORDER if tests[key]["pass"] is not True]
    digest = sha256(ANALYSIS)
    plot(tests, digest)
    figure_hashes = {path.name: sha256(path) for path in (FIGURE_PDF, FIGURE_PNG)}

    scope_sentence = (
        "The repaired budget-7 estimator is independently validated as component evidence. "
        "This does not by itself establish an integrated v6 primary or external confirmation."
        if gates["pass"] else
        "The repaired estimator remains unvalidated and does not replace the retained v4 core."
    )
    doc = f"""# V6 lifetime-budget-7 independent validation v2

The prospectively locked validation completed all
12 worlds x 5 cells x 4 methods = **{matrix['complete_rows']} rows**. The
analyzer recomputed every metric from hash-bound prediction and scoring arrays.
The prediction artifacts were written before scoring was loaded. The registered
paired-world bootstrap used {gates['bootstrap']['replicates']} draws and a
one-sided Bonferroni alpha of {gates['bootstrap']['one_sided_alpha_per_gate']:.6g}
for each of eight gates.

| Gate | Estimate | Simultaneous decision bound | Threshold | Decision |
|---|---:|---:|---:|:---:|
{result_table(tests)}

Overall decision: **{'PASS' if gates['pass'] else 'FAIL'}**. All invariants:
**{'PASS' if gates['all_invariants_pass'] else 'FAIL'}**. Failed scientific gates:
{', '.join(failed) if failed else 'none'}.

For drift, {tests['attack:severe_drift']['positive_denominator_draws']}/
{gates['bootstrap']['replicates']} bootstrap draws had positive SQ attack-excess.
The registered lower bound is {tests['attack:severe_drift']['lower']} and the
point estimate is {tests['attack:severe_drift']['estimate']:.6f}.

This is independent synthetic mechanism validation of the fixed
`AP_LIFETIME7` robust-twin component. It is not the five-pillar primary or an
external archive test. {scope_sentence} The historical v4 primary and its
H1--H7 claims retain their original scope.

Provenance:

- analysis: `{ANALYSIS.relative_to(ROOT).as_posix()}` (SHA-256 `{digest}`);
- registration: `{REGISTRATION.relative_to(ROOT).as_posix()}` (SHA-256 `{sha256(REGISTRATION)}`);
- source lock: SHA-256 `{matrix['source_lock_sha256']}`;
- input lock: SHA-256 `{matrix['input_lock_sha256']}`;
- figure hashes: `{json.dumps(figure_hashes, sort_keys=True)}`.
"""
    DOC.write_text(doc, encoding="utf-8")

    tex = rf"""% Generated by scripts/export_v6_lifetime_budget7_validation_v2.py
% Analysis SHA-256: {digest}
The locked independent synthetic validation completed 12 worlds, five physical
cells and four matched methods ({matrix['complete_rows']} rows).  Every reported metric was
recomputed from prediction and scoring artifacts; predictions were committed
before scoring truth was loaded.  The eight one-sided tests use paired-world
percentile bootstrap bounds with Bonferroni family control.

\begin{{center}}\small
\begin{{tabular}}{{lrrrr}}\toprule
Gate & Estimate & Decision bound & Target & Result\\\midrule
{latex_table(tests)}
\bottomrule\end{{tabular}}
\end{{center}}

The registered component decision is \textbf{{{'pass' if gates['pass'] else 'fail'}}};
all numerical, resource, cap, exposure, information and finite-output invariants
{'pass' if gates['all_invariants_pass'] else 'do not all pass'}.  This is
independent synthetic mechanism evidence for the fixed lifetime-budget-7 robust
twin.  It is neither the five-pillar primary nor an external archive test, and
it does not change the scope of the historical v4 results.

For drift, {tests['attack:severe_drift']['positive_denominator_draws']} of
{gates['bootstrap']['replicates']} paired bootstrap draws have positive SQ
attack-excess.  The registered drift estimate is
{tests['attack:severe_drift']['estimate']:.6f}, with lower decision bound
{tests['attack:severe_drift']['lower']}.
"""
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(tex, encoding="utf-8")
    print(json.dumps({
        "decision": gates["pass"], "failed_gates": failed,
        "doc": str(DOC.relative_to(ROOT)), "tex": str(TEX.relative_to(ROOT)),
        "figure_pdf": str(FIGURE_PDF.relative_to(ROOT)),
        "analysis_sha256": digest,
    }, indent=2))


if __name__ == "__main__":
    main()
