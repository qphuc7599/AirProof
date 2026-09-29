"""Contracts for candidate-only exposed lifetime-budget repair v5."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from airproof.v6_covariance_forcing_inputs import sha256_file
from airproof.v6_lifetime_budget_repair_registration import (
    CANDIDATES, CELLS, CONTROL_METHODS, SEEDS, assert_registration, evaluate)
from airproof.v6_lifetime_budget_repair_runner import planned_jobs
from scripts.run_v6_lifetime_budget_repair_v5 import require_ready

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v6/lifetime_budget_repair_development_v5.json"


def registration() -> dict:
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))


def rows(*, candidate_attack_excess: float = .5) -> list[dict]:
    result = []
    for seed in SEEDS:
        for candidate, budget in zip(CANDIDATES, (7., 14.)):
            for cell in CELLS:
                attacked = cell in ("severe_drift", "severe_hotspot")
                for method in ["CANDIDATE", *CONTROL_METHODS]:
                    rmse = (.9 + candidate_attack_excess if method == "CANDIDATE" and attacked
                            else .9 if method == "CANDIDATE"
                            else 2. if method == "SQ" and attacked
                            else 1.1 if method == "PUBLIC" else 1.)
                    result.append({"seed": seed, "candidate": candidate,
                        "scenario": cell, "method": method,
                        "information_fingerprint": f"same:{seed}:{cell}",
                        "rmse": rmse,
                        "event_recall": .90 if method == "CANDIDATE" else .92,
                        "wall_seconds": 1., "process_cpu_seconds": .5,
                        "peak_rss_bytes": 1024, "finite_outputs": True,
                        "resource_invariants_pass": True,
                        "solver_failure_rate": 0., "projected_gradient_inf": 0.,
                        "maximum_absolute_correction": 1.,
                        "maximum_user_lifetime_exposure": budget})
    return result


def test_exact_registration_and_candidate_only_job_contract():
    value = registration()
    assert_registration(value)
    assert planned_jobs(value, smoke=True) == [(SEEDS[0], cell) for cell in CELLS]
    assert len(planned_jobs(value, smoke=False)) == 60
    assert len(value["candidates"]) * len(planned_jobs(value, smoke=False)) == 120
    for mutation in (
        lambda x: x["input_reuse"].update(seeds=SEEDS[:-1]),
        lambda x: x["candidates"][0].update(per_user_exposure_budget=8.),
        lambda x: x["controls"].update(recompute=True),
        lambda x: x["development_gates"].update(attack_target_attenuation=.19),
    ):
        changed = copy.deepcopy(value); mutation(changed)
        with pytest.raises(ValueError):
            assert_registration(changed)


def test_direct_iut_preserves_all_gates_and_isolates_attack_failure():
    value = registration()
    passing = evaluate(value, rows(candidate_attack_excess=.5))
    assert passing["selected_candidate"] == CANDIDATES[0]
    assert passing["validation_v1_rescued"] is False
    for candidate in passing["candidates"]:
        assert candidate["metrics"]["attacks"]["severe_drift"]["direct_margin"] == pytest.approx(.3)
        assert candidate["pass"] is True
    failing = evaluate(value, rows(candidate_attack_excess=.9))
    assert failing["family_closed"] is True
    assert failing["selected_candidate"] is None
    assert all(not row["scientific_checks"]["attack:severe_drift"]
               for row in failing["candidates"])
    assert all(row["scientific_checks"]["clean:anchor_clean"]
               and row["scientific_checks"]["citizen"] for row in failing["candidates"])


def test_evaluator_rejects_missing_control_and_fingerprint_drift():
    value = registration()
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(value, rows()[:-1])
    changed = rows()
    changed[0]["information_fingerprint"] = "different"
    result = evaluate(value, changed)
    assert result["candidates"][0]["all_invariants_pass"] is False


def test_controller_requires_hash_bound_zero_outcome_readiness(tmp_path):
    value = registration()
    config = tmp_path / "configs/v6"; config.mkdir(parents=True)
    config.joinpath("lifetime_budget_repair_development_v5.json").write_text(
        json.dumps(value), encoding="utf-8")
    source = tmp_path / value["artifacts"]["source_lock"]
    input_lock = tmp_path / value["input_reuse"]["input_lock"]
    readiness = tmp_path / value["artifacts"]["readiness"]
    source.parent.mkdir(parents=True); input_lock.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source", encoding="utf-8"); input_lock.write_text("input", encoding="utf-8")
    ready = {"pass": True, "expected_jobs": 60, "expected_candidate_rows": 120,
        "reused_control_rows": 240, "scientific_outcomes_read_or_generated": 0,
        "controls_recomputed": False, "source_lock_sha256": sha256_file(source),
        "input_lock_sha256": sha256_file(input_lock)}
    readiness.write_text(json.dumps(ready), encoding="utf-8")
    assert require_ready(tmp_path) == ready
    for key, bad in (("expected_candidate_rows", 119),
                     ("scientific_outcomes_read_or_generated", 1),
                     ("controls_recomputed", True)):
        altered = dict(ready); altered[key] = bad
        readiness.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(RuntimeError, match="readiness"):
            require_ready(tmp_path)
