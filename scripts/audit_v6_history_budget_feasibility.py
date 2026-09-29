"""Audit whether the pre-existing history budgets bind on locked v2 inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v6/innovation_covariance_forcing_development_v2.json"
OLD_REGISTRATION = ROOT / "configs/v6/history_normalized_huber_development.json"
INPUTS = ROOT / "reports/v6/innovation_covariance_forcing_development_v2/inputs"
OUTPUT = ROOT / "reports/v6/innovation_covariance_forcing_development_v2/history_budget_input_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    old = json.loads(OLD_REGISTRATION.read_text(encoding="utf-8"))
    budgets = list(map(float, old["per_user_weight_budgets"]))
    lag = int(registration["fixed_estimator"]["lag"])
    horizon = (int(registration["scale"]["acquisition_epochs"])
               - int(registration["scale"]["burn_in_epochs"])
               + lag)
    result = {}
    for budget in budgets:
        bound = total = 0
        reduced = raw = 0.0
        maximum = 0.0
        for seed in registration["world_roles"]["development"]["seeds"]:
            path = INPUTS / "prediction" / str(seed) / "observations_severe_drift.npz"
            with np.load(path, allow_pickle=False) as arrays:
                arrival = arrays["arrival"].copy()
                epoch = arrays["epoch"].copy()
                user = arrays["user_id"].copy()
                quality = arrays["quality"].astype(float)
            for now in range(horizon):
                mask = ((arrival <= now) & (epoch >= max(0, now - lag))
                        & (epoch <= now))
                users, weights = user[mask], quality[mask]
                for identity in np.unique(users):
                    weight = float(weights[users == identity].sum())
                    maximum = max(maximum, weight)
                    total += 1
                    raw += weight
                    if weight > budget:
                        bound += 1
                        reduced += weight - budget
        result[str(int(budget))] = {
            "budget": budget,
            "user_windows": total,
            "binding_user_windows": bound,
            "binding_fraction": bound / total,
            "raw_weight": raw,
            "weight_removed": reduced,
            "weight_removed_fraction": reduced / raw,
            "maximum_user_window_weight": maximum,
        }
    payload = {
        "schema_version": 1,
        "role": "locked-input feasibility only; no new estimator or scoring outcome",
        "v2_registration_sha256": sha256_file(REGISTRATION),
        "v2_input_manifest_sha256": sha256_file(INPUTS / "manifest.json"),
        "prior_history_registration_sha256": sha256_file(OLD_REGISTRATION),
        "budgets_inherited_without_change": budgets,
        "development_scoring_read": False,
        "estimator_executed": False,
        "results": result,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
