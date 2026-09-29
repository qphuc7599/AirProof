"""Decompose the exposed v2 slow-drift failure without fitting a new estimator."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/v6/innovation_covariance_forcing_development_v2"
INPUTS = CAMPAIGN / "inputs"
OUTCOMES = CAMPAIGN / "outcomes"
ANALYSIS = OUTCOMES / "analysis.json"
DESTINATION = CAMPAIGN / "drift_failure_diagnosis.json"
REPORT = ROOT / "docs/V6_COVARIANCE_FORCING_DRIFT_DIAGNOSIS.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def prediction_decomposition(seed: int, candidate: str, method: str,
                             truth: np.ndarray) -> dict[str, float]:
    base = OUTCOMES / "jobs" / str(seed)
    clean = _npz(base / "severe_clean" / candidate / method / "prediction.npz")
    drift = _npz(base / "severe_drift" / candidate / method / "prediction.npz")
    steps = truth.shape[0]
    clean_pred = clean["reconstructed"][:steps]
    drift_pred = drift["reconstructed"][:steps]
    clean_error = clean_pred - truth
    attack_response = drift_pred - clean_pred
    clean_mse = float(np.mean(clean_error**2))
    cross = float(2 * np.mean(clean_error * attack_response))
    response_energy = float(np.mean(attack_response**2))
    drift_mse = float(np.mean((drift_pred - truth)**2))
    identity_error = abs((drift_mse - clean_mse) - (cross + response_energy))
    return {
        "clean_rmse": float(np.sqrt(clean_mse)),
        "drift_rmse": float(np.sqrt(drift_mse)),
        "rmse_growth": float(np.sqrt(drift_mse) - np.sqrt(clean_mse)),
        "attack_response_rms": float(np.sqrt(response_energy)),
        "mse_cross_term": cross,
        "mse_response_energy": response_energy,
        "mse_growth": drift_mse - clean_mse,
        "risk_identity_absolute_error": identity_error,
    }


def observation_diagnosis(seed: int, scale: float, delta: float) -> dict:
    directory = INPUTS / "prediction" / str(seed)
    context = _npz(directory / "context.npz")
    clean = _npz(directory / "observations_severe_clean.npz")
    drift = _npz(directory / "observations_severe_drift.npz")
    identity = ("user_id", "epoch", "cell", "group", "nullifier", "arrival")
    if any(not np.array_equal(clean[name], drift[name]) for name in identity):
        raise ValueError(f"scenario identity mismatch for world {seed}")
    changed = ~np.isclose(clean["value"], drift["value"], rtol=0, atol=1e-12)
    epoch = drift["epoch"].astype(int)
    cell = drift["cell"].astype(int)
    public = context["public_center"][epoch, cell]
    clean_residual = (clean["value"] - public) / scale
    drift_residual = (drift["value"] - public) / scale
    quality = drift["quality"].astype(float)
    quadratic_change = quality * (drift_residual - clean_residual) / scale
    huber_change = quality * (
        np.clip(drift_residual, -delta, delta)
        - np.clip(clean_residual, -delta, delta)
    ) / scale
    changed_counts = Counter(map(int, drift["user_id"][changed]))
    total_changed = int(changed.sum())
    top_ten = sum(value for _, value in changed_counts.most_common(10))
    return {
        "selected_records": len(epoch),
        "attacked_selected_records": total_changed,
        "attacked_selected_fraction": float(changed.mean()),
        "attacked_selected_users": len(changed_counts),
        "mean_records_per_attacked_selected_user": (
            float(fmean(changed_counts.values())) if changed_counts else 0.0),
        "maximum_records_one_attacked_user": max(changed_counts.values(), default=0),
        "top10_user_share_of_attacked_records": (
            top_ten / total_changed if total_changed else 0.0),
        "attack_first_shifted_epoch": int(epoch[changed].min()) if total_changed else None,
        "attack_last_shifted_epoch": int(epoch[changed].max()) if total_changed else None,
        "attacked_residual_huber_saturation_fraction": (
            float(np.mean(np.abs(drift_residual[changed]) > delta)) if total_changed else 0.0),
        "clean_same_record_huber_saturation_fraction": (
            float(np.mean(np.abs(clean_residual[changed]) > delta)) if total_changed else 0.0),
        "all_clean_record_huber_saturation_fraction": float(
            np.mean(np.abs(clean_residual) > delta)),
        "attack_gradient_change_l2_huber_to_quadratic_ratio": (
            float(np.linalg.norm(huber_change) / np.linalg.norm(quadratic_change))
            if np.linalg.norm(quadratic_change) else 0.0),
        "attack_gradient_change_mean_absolute_huber": float(np.mean(np.abs(huber_change[changed]))),
        "attack_gradient_change_mean_absolute_quadratic": float(
            np.mean(np.abs(quadratic_change[changed]))),
    }


def _aggregate(rows: list[dict]) -> dict[str, float]:
    keys = rows[0]
    return {key: float(fmean(float(row[key]) for row in rows))
            for key in keys if row_value_is_number(rows, key)}


def row_value_is_number(rows: list[dict], key: str) -> bool:
    return all(isinstance(row[key], (int, float)) and row[key] is not None for row in rows)


def render(result: dict) -> str:
    lines = [
        "# V6 slow-drift failure decomposition",
        "",
        ("This is a post-closure diagnosis on exposed mechanistic-synthetic development. "
         "It does not fit or authorize another candidate and does not open validation, EPA "
         "test, primary, or confirmation."),
        "",
        "## Observation mechanism",
        "",
        ("| Covariance scale | Attacked selected records | Attacked users | Records/user | "
         "Attack saturation | Clean saturation on same records | Huber/SQ gradient-change norm |"),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["observation_mechanism"].items():
        lines.append(
            f"| `{name}` | {row['attacked_selected_records']:.1f} "
            f"({100 * row['attacked_selected_fraction']:.2f}%) | "
            f"{row['attacked_selected_users']:.1f} | "
            f"{row['mean_records_per_attacked_selected_user']:.2f} | "
            f"{100 * row['attacked_residual_huber_saturation_fraction']:.2f}% | "
            f"{100 * row['clean_same_record_huber_saturation_fraction']:.2f}% | "
            f"{row['attack_gradient_change_l2_huber_to_quadratic_ratio']:.4f} |"
        )
    lines += ["", "## Prediction-risk identity", "",
        ("| Candidate | Method | Clean RMSE | Drift RMSE | RMSE growth | Response RMS | "
         "Cross term | Response energy |"),
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for candidate, methods in result["prediction_decomposition"].items():
        for method, row in methods.items():
            lines.append(
                f"| `{candidate}` | {method} | {row['clean_rmse']:.5f} | "
                f"{row['drift_rmse']:.5f} | {row['rmse_growth']:.5f} | "
                f"{row['attack_response_rms']:.5f} | {row['mse_cross_term']:.5f} | "
                f"{row['mse_response_energy']:.5f} |"
            )
    lines += ["", "## Mechanistic disposition", "",
        ("The correction-scale repair makes bounded Huber nearly identical to same-information "
         "SQ on clean data, so the earlier clean-margin failure is no longer the active cause. "
         "The remaining drift failure is consistent with repeated same-user evidence: ordinary "
         "Huber bounds each record's score but does not bound a user's accumulated score across "
         "the lag window or detect a coherent slow bias. Exact causal forcing improves the failed "
         "drift result only partially, and cross-fitted covariance improves it further, while all "
         "four variants remain below the unchanged 20% gate."), "",
        ("A next family would need a prospectively registered history-level or sequential drift "
         "mechanism, with the same clean/SQ, hotspot, event, cap and resource gates. This diagnostic "
         "does not authorize such a family or choose its parameters."), "",
        f"Analysis SHA-256: `{result['analysis_sha256']}`."]
    return "\n".join(lines) + "\n"


def main() -> None:
    analysis = _load(ANALYSIS)
    if (analysis.get("gate_evaluation", {}).get("family_closed") is not True
            or analysis["matrix_audit"].get("complete_rows") != 384):
        raise SystemExit("closed complete v2 analysis required")
    registration = _load(ROOT / analysis["registration"])
    calibration = _load(INPUTS / "global_calibration.json")
    seeds = list(map(int, registration["world_roles"]["development"]["seeds"]))
    candidates = registration["candidates"]
    delta = float(registration["fixed_estimator"]["huber_delta"])
    observation = {}
    for covariance, scale in calibration["innovation_scales"].items():
        observation[covariance] = _aggregate([
            observation_diagnosis(seed, float(scale), delta) for seed in seeds])
    prediction = {}
    max_identity_error = 0.0
    for candidate in candidates:
        name = candidate["id"]
        prediction[name] = {}
        for method in ("CANDIDATE", "SQ"):
            per_world = []
            for seed in seeds:
                scoring = _npz(INPUTS / "scoring" / str(seed) / "scoring_only.npz")
                row = prediction_decomposition(seed, name, method,
                                               scoring["evaluation_truth"])
                max_identity_error = max(max_identity_error,
                                         row["risk_identity_absolute_error"])
                per_world.append(row)
            prediction[name][method] = _aggregate(per_world)
    result = {
        "schema_version": 1,
        "role": "post-closure exposed-development diagnosis; no model fitting or authorization",
        "analysis": str(ANALYSIS.relative_to(ROOT)).replace("\\", "/"),
        "analysis_sha256": sha256_file(ANALYSIS),
        "input_manifest_sha256": sha256_file(INPUTS / "manifest.json"),
        "development_worlds": seeds,
        "observation_mechanism": observation,
        "prediction_decomposition": prediction,
        "maximum_risk_identity_absolute_error": max_identity_error,
        "family_extension_authorized": False,
        "validation_epa_test_primary_confirmation_opened": False,
        "historical_v4_rerun": False,
    }
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render(result), encoding="utf-8")
    print(json.dumps({"diagnosis": str(DESTINATION.relative_to(ROOT)),
        "report": str(REPORT.relative_to(ROOT)),
        "maximum_risk_identity_absolute_error": max_identity_error}, indent=2))


if __name__ == "__main__":
    main()
