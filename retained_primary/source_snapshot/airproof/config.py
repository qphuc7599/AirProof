from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            data = json.load(stream)
        else:
            data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return data


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise TypeError(f"Cannot assign nested key {dotted_key!r}")
    node[parts[-1]] = value


def with_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for key, value in overrides.items():
        set_nested(result, key, value)
    return result


def validate_config(config: dict[str, Any]) -> None:
    required = ["world", "network", "scheduler", "twin", "privacy", "attack"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    world = config["world"]
    if int(world["grid_side"]) < 2 or int(world["steps"]) < 2:
        raise ValueError("grid_side and steps must both be at least 2")
    if int(world["agents"]) < 1:
        raise ValueError("agents must be positive")
    burn_in = int(world.get("burn_in_steps", 0))
    if burn_in < 0 or burn_in >= int(world["steps"]):
        raise ValueError("world.burn_in_steps must lie in [0, world.steps)")
    twin = config["twin"]
    if float(twin.get("correction_delta", 3.0)) <= 0:
        raise ValueError("twin.correction_delta must be positive")
    if float(twin.get("lambda_correction", 0.5)) <= 0:
        raise ValueError("twin.lambda_correction must be positive")
    if float(twin.get("correction_clip", 8.0)) <= 0:
        raise ValueError("twin.correction_clip must be positive")
    clean_gain = float(twin.get("correction_clean_gain", 0.25))
    attack_gain = float(twin.get("correction_attack_gain", 3.0))
    if not 0 <= clean_gain <= attack_gain:
        raise ValueError("require 0 <= twin.correction_clean_gain <= correction_attack_gain")
    gate_start = float(twin.get("correction_gate_start", 0.15))
    gate_full = float(twin.get("correction_gate_full", 0.22))
    if not 0 <= gate_start < gate_full <= 1:
        raise ValueError("invalid twin correction-gate thresholds")
    if not 0 < float(twin.get("correction_gate_ewma", 0.25)) <= 1:
        raise ValueError("twin.correction_gate_ewma must lie in (0, 1]")
    if not 0 <= float(config["scheduler"]["reserve_fraction"]) <= 1:
        raise ValueError("scheduler.reserve_fraction must lie in [0, 1]")
    if float(config["scheduler"].get("fairness_strength", 1.0)) < 0:
        raise ValueError("scheduler.fairness_strength must be non-negative")
    if str(config["scheduler"].get("constraint_mode", "reserved")) not in {
        "reserved",
        "hard_if_feasible",
    }:
        raise ValueError("scheduler.constraint_mode must be reserved or hard_if_feasible")
    if float(config["privacy"]["epsilon_mean"]) <= 0:
        raise ValueError("privacy.epsilon_mean must be positive")
    if int(config["privacy"]["k_min"]) <= 0:
        raise ValueError("privacy.k_min must be positive")
    privacy = config["privacy"]
    release_mode = str(privacy.get("release_mode", "raw"))
    if release_mode not in {"raw", "residual"}:
        raise ValueError("privacy.release_mode must be raw or residual")
    if release_mode == "residual" and float(privacy.get("residual_clip", 0.0)) <= 0:
        raise ValueError("residual release requires a positive privacy.residual_clip")
    if bool(privacy.get("private_eligibility", False)) and float(
        privacy.get("epsilon_count", 0.0)
    ) <= 0:
        raise ValueError("private eligibility requires a positive epsilon_count")
    false_release_probability = float(
        privacy.get("eligibility_false_release_probability", 1e-6)
    )
    if not 0 < false_release_probability < 0.5:
        raise ValueError("eligibility_false_release_probability must lie in (0, 0.5)")
    target_mode = str(config["scheduler"].get("target_mode", "constant"))
    if target_mode not in {"constant", "population", "policy_weight"}:
        raise ValueError("scheduler.target_mode must be constant, population or policy_weight")
