#!/usr/bin/env python3
"""Run the prospectively locked V8 Gate-B independent confirmation."""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.config import config_hash  # noqa: E402
from airproof.simulator import generate_world  # noqa: E402
from airproof.v7_privacy_controls import reviewer_release_evidence  # noqa: E402
from airproof.v8_release_assimilation import (  # noqa: E402
    ReleaseAssimilationConfig,
    published_release_carry,
    run_protected_assimilation,
    stress_public_reference_observations,
)
from scripts.run_v7_shared_resource_confirmation import (  # noqa: E402
    _config as shared_config,
    integrated_transport_reviewer,
    coupled_public_trace,
    prepare_public,
)

REGISTRATION = ROOT / "configs/v8/m5_release_reference_stress_confirmation_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _verify_lock(registration: dict, output: Path) -> tuple[dict, str]:
    lock_path = output / "source_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("run freeze_v8_m5_confirmation.py before outcomes")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["outcomes_before_lock"] != 0 or lock["registration_sha256"] != _sha(REGISTRATION):
        raise ValueError("invalid pre-outcome source lock")
    current = {
        relative: _sha(ROOT / relative) for relative in lock["source_inventory"]
    }
    if current != lock["source_inventory"]:
        raise ValueError("source/config changed after confirmation lock")
    identity = hashlib.sha256(
        json.dumps(current, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if identity != lock["source_identity"]:
        raise ValueError("source lock identity mismatch")
    return lock, _sha(lock_path)


def _consumer_predictions(
    *,
    public: np.ndarray,
    uncertainty: np.ndarray,
    residual: np.ndarray,
    residual_mask: np.ndarray,
    raw_nonnegative: np.ndarray,
    raw_mask: np.ndarray,
    schedule: tuple[int, ...],
    registration: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Privacy boundary: this function cannot receive truth, query or counts."""
    fixed = registration["selected_estimator"]
    privacy = registration["privacy_contract"]
    common = dict(
        groups=public.shape[1],
        scheduled_acquisition_epochs=schedule,
        deadline_epochs=int(privacy["deadline_epochs"]),
        epsilon_history=float(privacy["epsilon_history"]),
        k_min=int(privacy["k_min"]),
        prior_variance=float(fixed["prior_variance"]),
        public_uncertainty_gate=float(fixed["public_uncertainty_gate"]),
        innovation_clip=float(fixed["innovation_clip"]),
        correction_cap=float(fixed["correction_cap"]),
    )
    residual_config = ReleaseAssimilationConfig(
        **common, laplace_scale=float(privacy["residual_laplace_scale"])
    )
    raw_config = ReleaseAssimilationConfig(
        **common, laplace_scale=float(privacy["nonnegative_raw_laplace_scale"])
    )
    protected, protected_diagnostics = run_protected_assimilation(
        public, uncertainty, residual, residual_mask, residual_config
    )
    raw_pool, raw_diagnostics = run_protected_assimilation(
        public, uncertainty, raw_nonnegative, raw_mask, raw_config
    )
    predictions = {
        "PUBLIC": public.copy(),
        "PUBLIC_AVAILABILITY": public.copy(),
        "RESIDUAL_DIRECT": published_release_carry(
            public, residual, residual_mask, residual_config
        ),
        "RAW_NONNEGATIVE_DIRECT": published_release_carry(
            public, raw_nonnegative, raw_mask, raw_config
        ),
        "PROTECTED_POOL": protected,
        "RAW_NONNEGATIVE_POOL": raw_pool,
    }
    diagnostics = {
        "protected": protected_diagnostics,
        "raw_nonnegative_pool": raw_diagnostics,
        "residual_config": asdict(residual_config),
        "raw_nonnegative_config": asdict(raw_config),
        "consumer_signature_excludes": [
            "truth", "noiseless_query", "exact_contributor_counts", "raw_records"
        ],
    }
    return predictions, diagnostics


def _score(predictions: dict[str, np.ndarray], truth: np.ndarray, start: int, stop: int) -> dict:
    target = truth[start:stop]
    return {
        name: {
            "rmse": float(np.sqrt(np.mean((prediction[start:stop] - target) ** 2))),
            "bias": float(np.mean(prediction[start:stop] - target)),
            "support": int(target.size),
        }
        for name, prediction in predictions.items()
    }


def _mean_release_age(mask: np.ndarray, schedule: tuple[int, ...], *, deadline: int, start: int, stop: int) -> float:
    ages = []
    last = np.full(mask.shape[1], -1, dtype=int)
    scheduled = set(schedule)
    for epoch in range(stop):
        acquisition = epoch - deadline
        if acquisition in scheduled:
            take = mask[acquisition]
            last[take] = epoch
        if epoch >= start:
            ages.extend((epoch - last[last >= 0]).tolist())
    return float(np.mean(ages))


def _regime(
    *,
    world,
    public_grid: np.ndarray,
    public_provenance: dict,
    release_arrivals: dict,
    config: dict,
    registration: dict,
    stress_uncertainty: bool,
) -> tuple[dict, dict, dict, dict]:
    steps = int(config["world"]["steps"])
    release_summary, arrays = reviewer_release_evidence(
        world, public_grid[:steps], release_arrivals, config
    )
    public = arrays["baseline"]
    nominal = float(public_provenance["scale"]["pooled_scale"])
    uncertainty = np.full_like(public, nominal)
    if stress_uncertainty:
        magnitude = float(registration["stress_regime"]["magnitude"])
        uncertainty = np.sqrt(uncertainty**2 + magnitude**2)
    schedule = tuple(int(value) for value in release_summary["scheduled_epochs"])
    predictions, diagnostics = _consumer_predictions(
        public=public,
        uncertainty=uncertainty,
        residual=arrays["residual"],
        residual_mask=arrays["residual_mask"],
        raw_nonnegative=arrays["raw_nonnegative"],
        raw_mask=arrays["raw_nonnegative_mask"],
        schedule=schedule,
        registration=registration,
    )
    return predictions, diagnostics, release_summary, arrays


def _job(spec: tuple[dict, dict, int, str, str]) -> dict:
    registration, shared, seed, output_raw, source_lock_sha = spec
    output = Path(output_raw)
    target = output / "jobs" / str(seed)
    result_path = target / "result.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("source_lock_sha256") != source_lock_sha:
            raise ValueError("stale confirmation job")
        return prior
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = shared_config(shared, registration["cell"])
    base_world = generate_world(config, seed)
    trace, _ = coupled_public_trace(config, seed, base_world.observations)
    transport, collector = integrated_transport_reviewer(
        trace,
        base_world.observations,
        config,
        policy=shared["execution"]["relay_policy"],
        fairness=bool(shared["execution"]["fairness"]),
        allocation_policy=shared["execution"]["allocation_policy"],
    )
    release_arrivals = dict(transport.release_arrivals)

    main_public, _, _, main_provenance = prepare_public(base_world, config)
    main_predictions, main_diagnostics, main_release, main_arrays = _regime(
        world=base_world,
        public_grid=main_public,
        public_provenance=main_provenance,
        release_arrivals=release_arrivals,
        config=config,
        registration=registration,
        stress_uncertainty=False,
    )

    bias = np.asarray(registration["stress_regime"]["group_bias"], dtype=float)
    stressed_world = replace(
        base_world,
        reference_observations=stress_public_reference_observations(
            base_world.reference_observations, bias
        ),
    )
    stress_public, _, _, stress_provenance = prepare_public(stressed_world, config)
    stress_predictions, stress_diagnostics, stress_release, stress_arrays = _regime(
        world=stressed_world,
        public_grid=stress_public,
        public_provenance=stress_provenance,
        release_arrivals=release_arrivals,
        config=config,
        registration=registration,
        stress_uncertainty=True,
    )
    start, stop = registration["evaluation"]["score_half_open"]
    main_scores = _score(main_predictions, base_world.truth[:, :0] if False else main_arrays["truth"], start, stop)
    stress_scores = _score(stress_predictions, stress_arrays["truth"], start, stop)

    masks_equal = (
        np.array_equal(stress_arrays["residual_mask"], stress_arrays["raw_nonnegative_mask"])
        and np.array_equal(main_arrays["residual_mask"], main_arrays["raw_nonnegative_mask"])
    )
    support_equal = len({row["support"] for row in stress_scores.values()}) == 1
    consumer_safe = all(
        not diagnostics["protected"][key]
        for diagnostics in (main_diagnostics, stress_diagnostics)
        for key in ("exact_private_counts_used", "raw_records_used", "noiseless_query_used")
    )
    invariants = {
        "epsilon_history_exact_8": main_release["epsilon_history"] == stress_release["epsilon_history"] == 8.0,
        "release_count_exact_28": len(main_release["scheduled_epochs"]) == len(stress_release["scheduled_epochs"]) == 28,
        "deadline_exact_24": main_release["deadline"] == stress_release["deadline"] == 24,
        "k_min_exact_20": int(config["privacy"]["k_min"]) == 20,
        "postprocessing_epsilon_zero": main_diagnostics["protected"]["privacy_spend_added_by_postprocessing"] == stress_diagnostics["protected"]["privacy_spend_added_by_postprocessing"] == 0.0,
        "consumer_information_safe": consumer_safe,
        "same_release_masks": masks_equal,
        "same_scoring_support": support_equal,
        "same_transport_and_admission": True,
        "privacy_budget_violations_zero": main_release["privacy_budget_violations"] == stress_release["privacy_budget_violations"] == 0,
        "transport_token_violations_zero": int(transport.metrics["token_violations"]) == 0,
    }
    arrays_path = target / "consumer_inputs.npz"
    np.savez_compressed(
        arrays_path,
        main_public=main_arrays["baseline"],
        main_uncertainty=np.full_like(main_arrays["baseline"], float(main_provenance["scale"]["pooled_scale"])),
        main_residual=main_arrays["residual"],
        main_residual_mask=main_arrays["residual_mask"],
        stress_public=stress_arrays["baseline"],
        stress_uncertainty=np.full_like(stress_arrays["baseline"], np.sqrt(float(stress_provenance["scale"]["pooled_scale"])**2 + float(registration["stress_regime"]["magnitude"])**2)),
        stress_residual=stress_arrays["residual"],
        stress_residual_mask=stress_arrays["residual_mask"],
        stress_raw_nonnegative=stress_arrays["raw_nonnegative"],
        stress_raw_nonnegative_mask=stress_arrays["raw_nonnegative_mask"],
    )
    scoring_path = target / "scoring_outputs.npz"
    np.savez_compressed(
        scoring_path,
        truth=stress_arrays["truth"],
        **{f"main_{key}": value for key, value in main_predictions.items()},
        **{f"stress_{key}": value for key, value in stress_predictions.items()},
    )
    result = {
        "schema_version": 1,
        "role": registration["role"],
        "status": "complete",
        "seed": seed,
        "cell": registration["cell"],
        "source_lock_sha256": source_lock_sha,
        "config_hash": config_hash(config),
        "trace_hash": trace.trace_hash,
        "stress_transform": {
            "public_observations_shifted": len(stressed_world.reference_observations),
            "group_bias": bias.tolist(),
            "truth_changed": False,
            "citizen_observations_changed": False,
            "applied_before_public_fit_and_release_query": True,
        },
        "main_scores": main_scores,
        "stress_scores": stress_scores,
        "main_release": main_release,
        "stress_release": stress_release,
        "decomposition": {
            "main_public_reference_bias": float(np.mean(main_arrays["baseline"][start:stop] - main_arrays["truth"][start:stop])),
            "stress_public_reference_bias": float(np.mean(stress_arrays["baseline"][start:stop] - stress_arrays["truth"][start:stop])),
            "residual_deterministic_query_bias": float(np.mean(stress_arrays["residual_query"][start:stop][stress_arrays["residual_mask"][start:stop]] - stress_arrays["truth"][start:stop][stress_arrays["residual_mask"][start:stop]])),
            "nonnegative_raw_deterministic_query_bias": float(np.mean(stress_arrays["raw_nonnegative_query"][start:stop][stress_arrays["raw_nonnegative_mask"][start:stop]] - stress_arrays["truth"][start:stop][stress_arrays["raw_nonnegative_mask"][start:stop]])),
            "residual_laplace_variance": stress_release["residual"]["laplace_variance"],
            "nonnegative_raw_laplace_variance": stress_release["raw_nonnegative"]["laplace_variance"],
            "released_group_queries": stress_release["residual"]["released_group_queries"],
            "mean_release_age_epochs": _mean_release_age(stress_arrays["residual_mask"], tuple(stress_release["scheduled_epochs"]), deadline=24, start=start, stop=stop),
            "clipping_sampling_stabilization_scope": "deterministic query bias is joint; no raw private state was opened to isolate clipping alone",
        },
        "consumer_diagnostics": {
            "main": main_diagnostics,
            "stress": stress_diagnostics,
        },
        "invariants": invariants,
        "invariant_failures": [key for key, value in invariants.items() if not value],
        "shared_resource": {
            "raw_timely_delivered": int(transport.metrics["raw_timely_delivered"]),
            "release_delivered": int(transport.metrics["release_delivered"]),
            "release_buffer_drops": int(transport.metrics["release_buffer_drops"]),
            "total_wire_bytes": int(transport.metrics["total_wire_bytes"]),
            "max_contact_direction_bytes": int(transport.metrics["max_contact_direction_bytes"]),
            "max_release_buffer_bytes": int(transport.metrics["max_release_buffer_bytes"]),
        },
        "consumer_inputs_sha256": _sha(arrays_path),
        "scoring_outputs_sha256": _sha(scoring_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(result_path, result)
    return result


def _upper_bound(values: np.ndarray, *, seed: int, replicates: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    distribution = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "upper95": float(np.quantile(distribution, 0.95)),
        "worlds": len(values),
    }


def _analyze(registration: dict, results: list[dict], output: Path, lock_sha: str) -> dict:
    rows = []
    for result in sorted(results, key=lambda item: item["seed"]):
        main = result["main_scores"]
        stress = result["stress_scores"]
        rows.append(
            {
                "seed": result["seed"],
                "trace_hash": result["trace_hash"],
                "main_public_rmse": main["PUBLIC"]["rmse"],
                "main_protected_rmse": main["PROTECTED_POOL"]["rmse"],
                "main_protected_minus_public": main["PROTECTED_POOL"]["rmse"] - main["PUBLIC"]["rmse"],
                "stress_public_rmse": stress["PUBLIC"]["rmse"],
                "stress_direct_residual_rmse": stress["RESIDUAL_DIRECT"]["rmse"],
                "stress_direct_nonnegative_raw_rmse": stress["RAW_NONNEGATIVE_DIRECT"]["rmse"],
                "stress_protected_rmse": stress["PROTECTED_POOL"]["rmse"],
                "stress_nonnegative_raw_pool_rmse": stress["RAW_NONNEGATIVE_POOL"]["rmse"],
                "stress_protected_minus_public": stress["PROTECTED_POOL"]["rmse"] - stress["PUBLIC"]["rmse"],
                "stress_protected_minus_direct_nonnegative_raw": stress["PROTECTED_POOL"]["rmse"] - stress["RAW_NONNEGATIVE_DIRECT"]["rmse"],
                "stress_protected_minus_nonnegative_raw_pool": stress["PROTECTED_POOL"]["rmse"] - stress["RAW_NONNEGATIVE_POOL"]["rmse"],
                "invariant_failures": result["invariant_failures"],
            }
        )
    replicates = int(registration["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(registration["evaluation"]["bootstrap_seed"])
    main = _upper_bound(np.asarray([row["main_protected_minus_public"] for row in rows]), seed=bootstrap_seed, replicates=replicates)
    protected_public = _upper_bound(np.asarray([row["stress_protected_minus_public"] for row in rows]), seed=bootstrap_seed + 1, replicates=replicates)
    protected_raw = _upper_bound(np.asarray([row["stress_protected_minus_direct_nonnegative_raw"] for row in rows]), seed=bootstrap_seed + 2, replicates=replicates)
    protected_raw_pool = _upper_bound(np.asarray([row["stress_protected_minus_nonnegative_raw_pool"] for row in rows]), seed=bootstrap_seed + 3, replicates=replicates)
    invariants_pass = all(not row["invariant_failures"] for row in rows)
    gates = {
        "protected_below_public_reference_stress": protected_public["upper95"] < 0,
        "protected_below_direct_nonnegative_raw": protected_raw["upper95"] < 0,
        "privacy_information_resource_invariants": invariants_pass,
    }
    result = {
        "schema_version": 1,
        "role": registration["role"],
        "source_lock_sha256": lock_sha,
        "worlds": len(rows),
        "stress_regime": registration["stress_regime"],
        "selected_estimator": registration["selected_estimator"],
        "paired_bounds": {
            "main_protected_minus_public": main,
            "stress_protected_minus_public": protected_public,
            "stress_protected_minus_direct_nonnegative_raw": protected_raw,
            "stress_protected_minus_nonnegative_raw_pool": protected_raw_pool,
        },
        "means": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "main_public_rmse",
                "main_protected_rmse",
                "stress_public_rmse",
                "stress_direct_residual_rmse",
                "stress_direct_nonnegative_raw_rmse",
                "stress_protected_rmse",
                "stress_nonnegative_raw_pool_rmse",
            )
        },
        "gates": gates,
        "main_regime_incremental_value": main["upper95"] < 0,
        "main_regime_adverse_result_retained": main["upper95"] >= 0,
        "claim_authorized": all(gates.values()),
        "claim_scope": "incremental protected-release value under the registered persistent public-reference calibration stress only",
        "observed_proxy_confirmation": "not evaluated in this synthetic runner; eligibility audit is a separate artifact",
    }
    (output / "paired_worlds.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    _write_json(output / "analysis.json", result)
    _write_json(
        output / "adverse_outcome_ledger.json",
        {
            "schema_version": 1,
            "main_regime": {
                "reported": True,
                "result": "incremental value passes" if result["main_regime_incremental_value"] else "adverse/no incremental value",
                "bound": main,
                "claim_scope_changed": False,
            },
            "failed_gates": [key for key, value in gates.items() if not value],
        },
    )
    lines = [
        "# V8 M5 protected-release confirmation",
        "",
        f"Independent worlds: {len(rows)}. Claim authorized: {result['claim_authorized']}.",
        "",
        f"Reference stress: protected RMSE {result['means']['stress_protected_rmse']:.6f}, public-only {result['means']['stress_public_rmse']:.6f}, direct nonnegative raw {result['means']['stress_direct_nonnegative_raw_rmse']:.6f}.",
        f"Protected minus public mean {protected_public['mean']:.6f}, one-sided 95% upper {protected_public['upper95']:.6f}.",
        f"Protected minus direct nonnegative raw mean {protected_raw['mean']:.6f}, one-sided 95% upper {protected_raw['upper95']:.6f}.",
        "",
        f"Main regime: protected RMSE {result['means']['main_protected_rmse']:.6f}, public-only {result['means']['main_public_rmse']:.6f}; protected minus public upper {main['upper95']:.6f}. The main-regime adverse result is retained whenever this upper bound is nonnegative.",
        "",
        "The positive claim, if authorized, is limited to the prospectively fixed reference-sensor calibration stress. Every consumer used only public inputs and already protected releases; post-processing epsilon is zero.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def execute(*, workers: int = 4, smoke: bool = False) -> dict:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    output = ROOT / registration["artifacts"]["directory"]
    _, lock_sha = _verify_lock(registration, output)
    shared = json.loads((ROOT / registration["shared_resource_configuration"]).read_text(encoding="utf-8"))
    seeds = registration["seeds"][:1] if smoke else registration["seeds"]
    specs = [(registration, shared, seed, str(output), lock_sha) for seed in seeds]
    results = []
    if workers == 1:
        for spec in specs:
            results.append(_job(spec))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_job, spec): spec[2] for spec in specs}
            for future in as_completed(futures):
                results.append(future.result())
                print(f"completed {futures[future]}", flush=True)
    if smoke:
        smoke_result = {"role": "engineering smoke", "seed": seeds[0], "result": results[0]}
        _write_json(output / "smoke.json", smoke_result)
        return smoke_result
    return _analyze(registration, results, output, lock_sha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(workers=args.workers, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
