"""Diagnose the closed v3 history-normalized family without fitting a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/v6/history_normalized_huber_development_v3"
ANALYSIS = CAMPAIGN / "outcomes/analysis.json"
INPUTS = ROOT / "reports/v6/innovation_covariance_forcing_development_v2/inputs"
OUTCOMES = CAMPAIGN / "outcomes/jobs"
DESTINATION = CAMPAIGN / "drift_failure_diagnosis.json"
REPORT = ROOT / "docs/V6_HISTORY_NORMALIZED_V3_RESULTS.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def decompose(seed: int, candidate: str) -> dict[str, float]:
    truth = load_npz(INPUTS / "scoring" / str(seed) / "scoring_only.npz")[
        "evaluation_truth"]
    clean = load_npz(OUTCOMES / str(seed) / "severe_clean" / candidate
                     / "prediction.npz")["reconstructed"][:len(truth)]
    drift = load_npz(OUTCOMES / str(seed) / "severe_drift" / candidate
                     / "prediction.npz")["reconstructed"][:len(truth)]
    error = clean - truth
    response = drift - clean
    clean_mse = float(np.mean(error**2))
    drift_mse = float(np.mean((drift - truth)**2))
    cross = float(2 * np.mean(error * response))
    energy = float(np.mean(response**2))
    return {
        "clean_rmse": float(np.sqrt(clean_mse)),
        "drift_rmse": float(np.sqrt(drift_mse)),
        "rmse_growth": float(np.sqrt(drift_mse) - np.sqrt(clean_mse)),
        "attack_response_rms": float(np.sqrt(energy)),
        "mse_cross_term": cross,
        "mse_response_energy": energy,
        "risk_identity_absolute_error": abs(
            (drift_mse - clean_mse) - (cross + energy)),
    }


def lifetime_exposure(seed: int, *, lag: int) -> dict[str, float]:
    directory = INPUTS / "prediction" / str(seed)
    clean = load_npz(directory / "observations_severe_clean.npz")
    drift = load_npz(directory / "observations_severe_drift.npz")
    identity = ("user_id", "epoch", "cell", "group", "nullifier", "arrival")
    if any(not np.array_equal(clean[name], drift[name]) for name in identity):
        raise ValueError(f"scenario identity mismatch for {seed}")
    attacked = ~np.isclose(clean["value"], drift["value"], rtol=0, atol=1e-12)
    users = clean["user_id"].astype(int)
    epoch = clean["epoch"].astype(int)
    arrival = clean["arrival"].astype(int)
    quality = clean["quality"].astype(float)
    steps = load_npz(directory / "context.npz")["public_center"].shape[0]
    uses = np.maximum(0, np.minimum(steps - 1, epoch + lag) - arrival + 1)
    exposure = quality * uses
    user_total = {int(user): float(exposure[users == user].sum())
                  for user in np.unique(users)}
    attacked_users = set(map(int, users[attacked]))
    return {
        "selected_users": len(user_total),
        "attacked_selected_users": len(attacked_users),
        "mean_lifetime_rolling_weight": float(fmean(user_total.values())),
        "maximum_lifetime_rolling_weight": max(user_total.values(), default=0.0),
        "fraction_users_above_7": float(np.mean(
            np.asarray(list(user_total.values())) > 7.0)),
        "fraction_attacked_users_above_7": float(np.mean([
            user_total[user] > 7.0 for user in attacked_users])) if attacked_users else 0.0,
        "attacked_share_of_rolling_weight": float(exposure[attacked].sum()
                                                   / exposure.sum()),
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {key: float(fmean(float(row[key]) for row in rows))
            for key in rows[0]}


def render(result: dict) -> str:
    gate = result["gate_evaluation"]
    lines = [
        "# V6 history-normalized Huber development v3",
        "",
        ("This is a complete, source-locked sequential development result on the exposed "
         "v2 worlds. It reruns no historical v4 evidence and opens no validation, EPA test, "
         "primary, or confirmation namespace."),
        "",
        "| Budget | Joint | Clean/SQ | Drift attenuation | Hotspot attenuation | Drift event loss | Hotspot event loss |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate["candidates"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['per_user_weight_budget']:.0f} | {row['pass']} | "
            f"{metrics['clean_ratio']:.5f} | "
            f"{100 * metrics['attenuation']['severe_drift']:.2f}% | "
            f"{100 * metrics['attenuation']['severe_hotspot']:.2f}% | "
            f"{metrics['event_loss_percentage_points']['severe_drift']:.3f} pp | "
            f"{metrics['event_loss_percentage_points']['severe_hotspot']:.3f} pp |")
    lines += [
        "",
        ("All numerical, cap, information-parity, user-window-budget and inherited-resource "
         "invariants pass. No budget passes the unchanged drift gate, so the family closes "
         "without a nomination."),
        "",
        "## Drift risk decomposition",
        "",
        "| Budget | Clean RMSE | Drift RMSE | Growth | Response RMS | Cross term | Response energy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, row in result["prediction_decomposition"].items():
        lines.append(
            f"| {candidate.removeprefix('history_budget')} | {row['clean_rmse']:.5f} | "
            f"{row['drift_rmse']:.5f} | {row['rmse_growth']:.6f} | "
            f"{row['attack_response_rms']:.5f} | {row['mse_cross_term']:.6f} | "
            f"{row['mse_response_energy']:.6f} |")
    exposure = result["lifetime_exposure"]
    lines += [
        "",
        "## Mechanistic disposition",
        "",
        (f"The budget-1 arm reduces the attack-response RMS but attains only "
         f"{100 * gate['candidates'][0]['metrics']['attenuation']['severe_drift']:.2f}% "
         "excess-RMSE attenuation. The input-only exposure audit finds mean rolling-use "
         f"weight {exposure['mean_lifetime_rolling_weight']:.2f} per selected user and "
         f"{100 * exposure['fraction_attacked_users_above_7']:.2f}% of attacked selected "
         "users above a seven-unit lifetime exposure. A fixed-window budget therefore does "
         "not bound cumulative influence through successive overlapping windows."),
        "",
        ("This diagnosis motivates a separately registered causal lifetime-exposure mechanism. "
         "It does not authorize a parameter extension, provide validation evidence, or alter "
         "the original clean, attack, event, cap, and resource targets."),
        "",
        f"Analysis SHA-256: `{result['analysis_sha256']}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    analysis = load_json(ANALYSIS)
    gate = analysis.get("gate_evaluation")
    if (analysis.get("matrix_audit", {}).get("pass") is not True
            or gate is None or gate.get("family_closed") is not True
            or analysis["matrix_audit"].get("complete_new_rows") != 72):
        raise SystemExit("complete closed v3 analysis required")
    registration = load_json(ROOT / analysis["registration"])
    seeds = list(map(int, registration["inputs"]["development_seeds"]))
    candidates = [row["id"] for row in registration["candidates"]]
    decomposition = {candidate: aggregate([
        decompose(seed, candidate) for seed in seeds]) for candidate in candidates}
    exposure = aggregate([lifetime_exposure(seed, lag=6) for seed in seeds])
    result = {
        "schema_version": 1,
        "role": "post-closure exposed-development diagnosis; no fitting or selection",
        "analysis": str(ANALYSIS.relative_to(ROOT)).replace("\\", "/"),
        "analysis_sha256": sha256_file(ANALYSIS),
        "gate_evaluation": gate,
        "prediction_decomposition": decomposition,
        "lifetime_exposure": exposure,
        "maximum_risk_identity_absolute_error": max(
            row["risk_identity_absolute_error"] for row in decomposition.values()),
        "family_extension_authorized": False,
        "historical_v4_rerun": False,
        "validation_epa_test_primary_confirmation_opened": False,
    }
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render(result), encoding="utf-8")
    print(json.dumps({
        "artifact": str(DESTINATION.relative_to(ROOT)),
        "report": str(REPORT.relative_to(ROOT)),
        "maximum_risk_identity_absolute_error": result[
            "maximum_risk_identity_absolute_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
