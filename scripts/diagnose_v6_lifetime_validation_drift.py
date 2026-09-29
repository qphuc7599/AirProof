"""Post-hoc diagnosis of the frozen v6 lifetime validation drift failure.

This script never changes the registered v1 decision.  It reads the completed,
hash-verified matrix and exposes why the ratio attenuation bound was undefined.
All alternative contrasts emitted here are descriptive design inputs only.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_v6_lifetime_validation_v1 import collect

from airproof.records import canonical_json
from airproof.v6_lifetime_validation_runner import load_registration

OUT = (
    ROOT
    / "reports/v6/causal_lifetime_exposure_validation_v1/drift_failure_diagnosis.json"
)
DOC = ROOT / "docs/V6_LIFETIME_VALIDATION_DRIFT_DIAGNOSIS.md"
ANALYSIS = (
    ROOT
    / "reports/v6/causal_lifetime_exposure_validation_v1/outcomes/analysis.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
        "sample_sd": float(np.std(values, ddof=1)),
    }


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _prediction_path(seed: int, cell: str, method: str) -> Path:
    return (
        ROOT
        / "reports/v6/causal_lifetime_exposure_validation_v1/outcomes/jobs"
        / str(seed)
        / cell
        / method
        / "prediction.npz"
    )


def _observation_path(seed: int, cell: str) -> Path:
    return (
        ROOT
        / "reports/v6/causal_lifetime_exposure_validation_v1/inputs/prediction"
        / str(seed)
        / f"observations_{cell}.npz"
    )


def _scoring_path(seed: int) -> Path:
    return (
        ROOT
        / "reports/v6/causal_lifetime_exposure_validation_v1/inputs/scoring"
        / str(seed)
        / "scoring_severe_clean.npz"
    )


def _load_prediction(seed: int, cell: str, method: str) -> np.ndarray:
    with np.load(_prediction_path(seed, cell, method), allow_pickle=False) as arrays:
        return arrays["reconstructed"].copy()


def _segment_diagnostics(seed: int, attack_start: int, burn_in: int) -> dict[str, Any]:
    with np.load(_scoring_path(seed), allow_pickle=False) as arrays:
        truth = arrays["evaluation_truth"].copy()
    onset = attack_start - burn_in
    if onset <= 0 or onset >= len(truth):
        raise RuntimeError("attack onset is outside the scoring horizon")
    result: dict[str, Any] = {
        "pre_attack_scoring_rows": onset,
        "post_attack_scoring_rows": len(truth) - onset,
        "methods": {},
    }
    for method in ("AP_LIFETIME28", "SQ"):
        clean = _load_prediction(seed, "severe_clean", method)[: len(truth)]
        drift = _load_prediction(seed, "severe_drift", method)[: len(truth)]
        clean_pre, clean_post = clean[:onset], clean[onset:]
        drift_pre, drift_post = drift[:onset], drift[onset:]
        truth_pre, truth_post = truth[:onset], truth[onset:]
        result["methods"][method] = {
            "clean_pre_rmse": _rmse(truth_pre, clean_pre),
            "drift_pre_rmse": _rmse(truth_pre, drift_pre),
            "clean_post_rmse": _rmse(truth_post, clean_post),
            "drift_post_rmse": _rmse(truth_post, drift_post),
            "pre_prediction_change_rmse": _rmse(clean_pre, drift_pre),
            "post_prediction_change_rmse": _rmse(clean_post, drift_post),
            "post_prediction_change_max_abs": float(
                np.max(np.abs(drift_post - clean_post))
            ),
        }
    return result


def _observation_diagnostics(
    seed: int, attack_start: int, burn_in: int,
) -> dict[str, Any]:
    with np.load(_observation_path(seed, "severe_clean"), allow_pickle=False) as clean_z:
        clean = {key: clean_z[key].copy() for key in clean_z.files}
    with np.load(_observation_path(seed, "severe_drift"), allow_pickle=False) as drift_z:
        drift = {key: drift_z[key].copy() for key in drift_z.files}
    identity_keys = ("user_id", "epoch", "cell", "group", "arrival", "nullifier")
    identities_equal = all(np.array_equal(clean[key], drift[key]) for key in identity_keys)
    value_delta = drift["value"] - clean["value"]
    changed = np.abs(value_delta) > 0
    absolute_epoch = clean["epoch"].astype(np.int64) + burn_in
    changed_after_onset = changed & (absolute_epoch >= attack_start)
    return {
        "row_count": len(value_delta),
        "identity_arrays_equal": bool(identities_equal),
        "changed_value_count": int(changed.sum()),
        "changed_value_fraction": float(changed.mean()),
        "stored_epoch_semantics": "relative_to_burn_in",
        "burn_in_added_for_attack_timing": burn_in,
        "changed_before_attack_start": int(
            np.sum(changed & (absolute_epoch < attack_start))
        ),
        "changed_at_or_after_attack_start": int(changed_after_onset.sum()),
        "changed_value_delta": (
            _summary(value_delta[changed]) if np.any(changed) else None
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    registration = load_registration(ROOT)
    rows, matrix = collect(registration, smoke=False)
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    if matrix["pass"] is not True or matrix["complete_rows"] != 240:
        raise RuntimeError("diagnosis requires the complete verified 240-row matrix")
    gate = analysis["gate_evaluation"]
    if (
        gate["pass"] is not False
        or gate["tests"]["attack:severe_drift"]["pass"] is not False
        or gate["tests"]["attack:severe_hotspot"]["pass"] is not True
    ):
        raise RuntimeError("the stored v1 decision is not the expected drift-only failure")

    seeds = [int(seed) for seed in registration["worlds"]["seeds"]]
    by_key = {
        (int(row["seed"]), row["scenario"], row["method"]): row for row in rows
    }

    def values(cell: str, method: str) -> np.ndarray:
        return np.asarray([by_key[seed, cell, method]["rmse"] for seed in seeds])

    clean_ap = values("severe_clean", "AP_LIFETIME28")
    drift_ap = values("severe_drift", "AP_LIFETIME28")
    clean_sq = values("severe_clean", "SQ")
    drift_sq = values("severe_drift", "SQ")
    ap_excess = drift_ap - clean_ap
    sq_excess = drift_sq - clean_sq
    target = float(
        registration["scientific_gates"][
            "drift_excess_RMSE_attenuation_lower_min"
        ]
    )
    direct = ap_excess - (1.0 - target) * sq_excess
    per_world_attenuation = np.where(sq_excess > 0, 1 - ap_excess / sq_excess, np.nan)

    gates = registration["scientific_gates"]
    rng = np.random.default_rng(int(gates["bootstrap_seed"]))
    draw_count = int(gates["bootstrap_replicates"])
    indices = rng.integers(0, len(seeds), size=(draw_count, len(seeds)))
    alpha = (1 - float(gates["confidence_level"])) / int(
        gates["simultaneous_family_size"]
    )
    ap_draw = ap_excess[indices].mean(axis=1)
    sq_draw = sq_excess[indices].mean(axis=1)
    direct_draw = direct[indices].mean(axis=1)
    finite_ratio = sq_draw > 0
    ratio_draw = 1 - ap_draw[finite_ratio] / sq_draw[finite_ratio]
    bad_draw_ids = np.flatnonzero(~finite_ratio)
    bad_draws = []
    for draw_id in bad_draw_ids:
        selected_indices = indices[draw_id]
        counts = Counter(seeds[int(index)] for index in selected_indices)
        bad_draws.append(
            {
                "draw_zero_based": int(draw_id),
                "world_seed_counts": {str(key): int(counts[key]) for key in sorted(counts)},
                "mean_sq_excess": float(sq_draw[draw_id]),
                "mean_ap_excess": float(ap_draw[draw_id]),
                "direct_target_contrast": float(direct_draw[draw_id]),
            }
        )

    per_world = []
    segment = {}
    observations = {}
    for index, seed in enumerate(seeds):
        per_world.append(
            {
                "seed": seed,
                "AP_clean_rmse": float(clean_ap[index]),
                "AP_drift_rmse": float(drift_ap[index]),
                "AP_excess_rmse": float(ap_excess[index]),
                "SQ_clean_rmse": float(clean_sq[index]),
                "SQ_drift_rmse": float(drift_sq[index]),
                "SQ_excess_rmse": float(sq_excess[index]),
                "attenuation_when_defined": (
                    float(per_world_attenuation[index])
                    if math.isfinite(per_world_attenuation[index])
                    else None
                ),
                "direct_20pct_target_contrast": float(direct[index]),
            }
        )
        segment[str(seed)] = _segment_diagnostics(
            seed,
            int(registration["attack_contract"]["severe_drift"]["start_epoch"]),
            int(registration["worlds"]["burn_in_epochs"]),
        )
        observations[str(seed)] = _observation_diagnostics(
            seed,
            int(registration["attack_contract"]["severe_drift"]["start_epoch"]),
            int(registration["worlds"]["burn_in_epochs"]),
        )

    loo = []
    for left_out in range(len(seeds)):
        keep = np.arange(len(seeds)) != left_out
        denominator = float(sq_excess[keep].mean())
        attenuation = (
            float(1 - ap_excess[keep].mean() / denominator)
            if denominator > 0
            else None
        )
        loo.append(
            {
                "left_out_seed": seeds[left_out],
                "mean_SQ_excess": denominator,
                "attenuation": attenuation,
                "direct_20pct_target_contrast": float(direct[keep].mean()),
            }
        )

    result = {
        "schema_version": 1,
        "role": "post-hoc failure diagnosis and future-design input only",
        "confirmatory_status": "descriptive; cannot change the registered v1 FAIL",
        "registered_v1_decision": {
            "pass": False,
            "failed_gate": "attack:severe_drift",
            "registered_point_attenuation": gate["tests"]["attack:severe_drift"][
                "estimate"
            ],
            "registered_lower_bound": None,
            "registered_positive_denominator_draws": gate["tests"]
            ["attack:severe_drift"]["positive_denominator_draws"],
        },
        "integrity": {
            "analysis": str(ANALYSIS.relative_to(ROOT)).replace("\\", "/"),
            "analysis_sha256": _sha256(ANALYSIS),
            "source_lock_sha256": matrix["source_lock_sha256"],
            "input_lock_sha256": matrix["input_lock_sha256"],
            "verified_matrix_rows": matrix["complete_rows"],
        },
        "per_world": per_world,
        "excess_summaries": {
            "AP_excess_rmse": _summary(ap_excess),
            "SQ_excess_rmse": _summary(sq_excess),
            "direct_20pct_target_contrast": _summary(direct),
            "worlds_with_nonpositive_SQ_excess": int(np.sum(sq_excess <= 0)),
            "worlds_meeting_direct_20pct_target": int(np.sum(direct <= 0)),
        },
        "registered_bootstrap_reproduction": {
            "seed": int(gates["bootstrap_seed"]),
            "replicates": draw_count,
            "alpha_per_gate": alpha,
            "positive_denominator_draws": int(np.sum(finite_ratio)),
            "nonpositive_denominator_draws": int(np.sum(~finite_ratio)),
            "bad_draws": bad_draws,
            "conditional_ratio_attenuation_quantiles_descriptive_only": {
                "lower": float(np.quantile(ratio_draw, alpha)),
                "median": float(np.median(ratio_draw)),
                "upper": float(np.quantile(ratio_draw, 1 - alpha)),
            },
        },
        "post_hoc_direct_contrast_descriptive_only": {
            "definition": "mean(AP_excess - 0.8*SQ_excess); target <= 0",
            "estimate": float(direct.mean()),
            "one_sided_bonferroni_upper": float(
                np.quantile(direct_draw, 1 - alpha)
            ),
            "target": 0.0,
            "descriptive_pass": bool(np.quantile(direct_draw, 1 - alpha) <= 0),
            "SQ_mean_excess": float(sq_excess.mean()),
            "SQ_mean_excess_one_sided_bonferroni_lower": float(
                np.quantile(sq_draw, alpha)
            ),
            "positive_signal_prerequisite_descriptive_pass": bool(
                np.quantile(sq_draw, alpha) > 0
            ),
        },
        "leave_one_world_out_descriptive_only": loo,
        "prediction_segment_diagnostics": segment,
        "observation_pairing_diagnostics": observations,
        "interpretation_rule": (
            "A future validation may use a direct target contrast only after a new "
            "protocol, sample size, seeds, source, and input boundary are frozen; "
            "this artifact must never be used to relabel validation v1 as passing."
        ),
    }
    safe = _json_safe(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(canonical_json(safe) + b"\n")

    direct_result = safe["post_hoc_direct_contrast_descriptive_only"]
    lines = [
        "# V6 lifetime validation v1: drift failure diagnosis",
        "",
        "This is a post-hoc diagnostic. It does not change the registered v1 FAIL.",
        "",
        "## Registered result",
        "",
        f"- Point attenuation: {100 * gate['tests']['attack:severe_drift']['estimate']:.3f}%.",
        f"- Positive denominator bootstrap draws: {int(np.sum(finite_ratio))}/{draw_count}.",
        f"- Nonpositive denominator draws: {int(np.sum(~finite_ratio))}.",
        "- Registered lower bound: undefined, so the 20% gate fails.",
        "",
        "## Direct 20% contrast, descriptive only",
        "",
        (
            "The equivalent target contrast is `mean(AP excess - 0.8 * SQ excess) "
            "<= 0`, provided the mean SQ attack excess is positive."
        ),
        "",
        f"- Contrast estimate: {direct_result['estimate']:.9g} RMSE.",
        f"- One-sided Bonferroni upper bound: {direct_result['one_sided_bonferroni_upper']:.9g} RMSE.",
        f"- SQ mean attack excess: {direct_result['SQ_mean_excess']:.9g} RMSE.",
        f"- SQ-excess one-sided lower bound: {direct_result['SQ_mean_excess_one_sided_bonferroni_lower']:.9g} RMSE.",
        "",
        "These quantities may inform a new independent protocol. They are not confirmatory evidence because the contrast was selected after v1 outcomes were available.",
        "",
        "## Reuse decision",
        "",
        "The seven passing gates and all invariants remain valid v1 evidence. A new run, if justified, should include only the unresolved drift contrast and its clean pairing on new worlds; it must not rerun the clean, hotspot, event, or citizen-value gates already supported here.",
        "",
        f"Machine-readable artifact: `{OUT.relative_to(ROOT).as_posix()}`",
        f"SHA-256: `{_sha256(OUT)}`",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": OUT.relative_to(ROOT).as_posix(),
                "artifact_sha256": _sha256(OUT),
                "document": DOC.relative_to(ROOT).as_posix(),
                "nonpositive_denominator_draws": int(np.sum(~finite_ratio)),
                "direct_contrast": direct_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
