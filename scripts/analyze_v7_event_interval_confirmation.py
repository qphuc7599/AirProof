#!/usr/bin/env python3
"""Fit the frozen v7 event radii and score the new shared-resource worlds."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from airproof.records import canonical_json
from airproof.simulator import generate_world
from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v7_event_intervals import (
    LEVELS,
    fit_world_robust_event_intervals,
    interval_scores,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v7/reviewer_event_interval_confirmation_v1.json"
OUTPUT = ROOT / "reports/v7/reviewer_revision/event_interval_confirmation_v1"


def _scoring_cell(cell: str) -> str:
    return "severe_clean" if cell.startswith("severe_") else cell


def _calibration_unit(root: Path, seed: int, cell: str, clock: str, stride: int):
    prediction_path = root / "outcomes/jobs" / str(seed) / cell / "AP_LIFETIME7/prediction.npz"
    scoring_path = (
        root
        / "inputs/scoring"
        / str(seed)
        / f"scoring_{_scoring_cell(cell)}.npz"
    )
    with np.load(prediction_path, allow_pickle=False) as prediction, np.load(
        scoring_path, allow_pickle=False
    ) as scoring:
        truth = scoring["evaluation_truth"][:, ::stride]
        event = scoring["event_mask"][:, ::stride]
        estimate = prediction[clock][: len(truth), ::stride]
    return estimate, truth, event, prediction_path, scoring_path


def _bootstrap_world_mean(values, *, seed: int, replicates: int):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = data[rng.integers(0, len(data), size=(replicates, len(data)))].mean(axis=1)
    return {
        "estimate": float(data.mean()),
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "worlds": len(data),
    }


def execute():
    registration_bytes = REGISTRATION.read_bytes()
    registration = json.loads(registration_bytes)
    if registration["status"] != "rule frozen before reading reviewer shared-resource v2 outcomes":
        raise ValueError("unexpected event-interval registration state")
    calibration_root = ROOT / registration["calibration"]["root"]
    evaluation_root = ROOT / registration["evaluation"]["root"]
    if not (evaluation_root / "analysis.json").is_file():
        raise FileNotFoundError("shared-resource confirmation is incomplete")
    stride = int(registration["calibration"]["spatial_stride"])
    models = {}
    calibration_paths = set()
    for clock in registration["clocks"]:
        units = []
        for seed in registration["calibration"]["seeds"]:
            for cell in registration["calibration"]["cells"]:
                estimate, truth, event, prediction_path, scoring_path = _calibration_unit(
                    calibration_root, seed, cell, clock, stride
                )
                units.append((estimate, truth, event))
                calibration_paths.update((prediction_path, scoring_path))
        models[clock] = fit_world_robust_event_intervals(
            units,
            clock=clock,
            calibration_split_id=registration["calibration"]["split_id"],
            evaluation_split_id=registration["evaluation"]["split_id"],
            minimum_event_support=int(
                registration["calibration"]["minimum_event_support_per_world_cell"]
            ),
        )
        del units

    # The model is fully determined before any evaluation truth is generated.
    model_identity = hashlib.sha256(
        canonical_json({clock: asdict(model) for clock, model in models.items()})
    ).hexdigest()
    try:
        from scripts.run_v7_shared_resource_confirmation import _config
    except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
        from run_v7_shared_resource_confirmation import _config

    rows = []
    evaluation_paths = set()
    shared_registration = json.loads(
        (ROOT / "configs/v7/reviewer_shared_resource_confirmation_v2.json").read_text()
    )
    for seed in registration["evaluation"]["seeds"]:
        for cell in registration["evaluation"]["cells"]:
            config = _config(shared_registration, cell)
            world = generate_world(config, int(seed))
            burn = int(config["world"]["burn_in_steps"])
            steps = int(config["world"]["steps"])
            truth = world.truth[burn:steps]
            threshold = float(np.quantile(world.truth[:burn], 0.95))
            event = truth >= threshold
            for clock, model in models.items():
                prediction_path = evaluation_root / "jobs" / str(seed) / cell / "AP_LIFETIME7/prediction.npz"
                with np.load(prediction_path, allow_pickle=False) as prediction:
                    estimate = prediction[clock][: len(truth)]
                lower, upper = model.predict(
                    estimate,
                    evaluation_split_id=registration["evaluation"]["split_id"],
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "cell": cell,
                        "clock": clock,
                        "event_threshold": threshold,
                        "model_identity": model_identity,
                        "prediction_sha256": sha256_file(prediction_path),
                        **interval_scores(truth, lower, upper, event_mask=event),
                    }
                )
                evaluation_paths.add(prediction_path)
    grouped = {}
    gates = {}
    bootstrap = registration["evaluation"]
    for clock in registration["clocks"]:
        clock_rows = [row for row in rows if row["clock"] == clock]
        world_rows = []
        for seed in registration["evaluation"]["seeds"]:
            selected = [row for row in clock_rows if row["seed"] == seed]
            world_rows.append(
                [
                    float(
                        np.average(
                            [row["coverage_event"][index] for row in selected],
                            weights=[row["support_event"] for row in selected],
                        )
                    )
                    for index in range(2)
                ]
            )
        world_rows = np.asarray(world_rows)
        event_coverage = [
            _bootstrap_world_mean(
                world_rows[:, index],
                seed=int(bootstrap["bootstrap_seed"]) + index,
                replicates=int(bootstrap["bootstrap_replicates"]),
            )
            for index in range(2)
        ]
        grouped[clock] = {
            "event_coverage": event_coverage,
            "mean_width_all": np.mean(
                [row["width_all"] for row in clock_rows], axis=0
            ).tolist(),
            "mean_width_event": np.mean(
                [row["width_event"] for row in clock_rows], axis=0
            ).tolist(),
            "mean_interval_score_all": np.mean(
                [row["interval_score_all"] for row in clock_rows], axis=0
            ).tolist(),
            "mean_interval_score_event": np.mean(
                [row["interval_score_event"] for row in clock_rows], axis=0
            ).tolist(),
            "event_support": sum(row["support_event"] for row in clock_rows),
        }
        gates[clock] = [
            event_coverage[index]["estimate"] >= LEVELS[index] for index in range(2)
        ]
    analysis = {
        "schema_version": 1,
        "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "model_identity": model_identity,
        "models": {clock: asdict(model) for clock, model in models.items()},
        "results": grouped,
        "gates": gates,
        "all_event_coverage_point_gates_pass": all(all(values) for values in gates.values()),
        "claim_scope": "empirical event-marginal coverage on independent worlds; no exact conditional or exchangeability guarantee",
        "input_hashes": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in sorted(calibration_paths | evaluation_paths)
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "registration.json").write_bytes(registration_bytes)
    write_json(OUTPUT / "models.json", {clock: asdict(model) for clock, model in models.items()})
    (OUTPUT / "per_world_cell.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    write_json(OUTPUT / "analysis.json", analysis)
    return analysis


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
