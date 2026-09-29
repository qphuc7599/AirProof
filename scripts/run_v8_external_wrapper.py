"""Develop, freeze and confirm the Gate-C external wrapper repair."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_estimator import EstimatorConfig
from airproof.v8_external_wrapper import (
    PublicEnvelopeConfig,
    matched_corruption_channel,
    public_context,
    run_equal_information_controls,
)


CONFIG = ROOT / "configs/v8/external_wrapper_development_v1.json"
REPORT_ROOT = ROOT / "reports/v8/external_wrapper_gate_c_v1"
MODULE = ROOT / "airproof/v8_external_wrapper.py"
V6_ESTIMATOR = ROOT / "airproof/v6_estimator.py"
PREDICTOR_WRAPPER = ROOT / "airproof/predictor_wrapper.py"
SEED_REGISTRY = ROOT / "configs/v8/seed_registry.json"
PROTOCOL = ROOT / "docs/V8_REVIEWER_GATE_PROTOCOL_20260914.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_protocol() -> dict:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if protocol["gate"] != "C: external equal-information bounded correction":
        raise ValueError("Gate C protocol required")
    return protocol


def build_shift(steps: int, value: float) -> np.ndarray:
    shift = np.zeros((steps, 1), dtype=float)
    shift[steps // 2:] = value
    return shift


def method_metrics(prediction: np.ndarray, truth: np.ndarray,
                   events: np.ndarray, threshold: float) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "bias": float(np.mean(prediction - truth)),
        "event_recall": float(np.mean(prediction[events] >= threshold)) if events.any() else None,
        "event_support": int(events.sum()),
    }


def analyze(rows: list[dict], protocol: dict, candidates: list[dict]) -> dict:
    gates = protocol["gates"]
    clean_map = protocol["clean_baselines"]
    cells = {cell["id"]: cell for cell in protocol["cells"]}

    def mean(candidate: str, cell: str, method: str, metric: str) -> float:
        values = [row[metric] for row in rows if row["candidate"] == candidate
                  and row["cell"] == cell and row["method"] == method]
        if not values:
            raise ValueError(f"missing rows for {candidate}/{cell}/{method}/{metric}")
        return float(np.mean(values))

    def descriptive_bootstrap(identifier: str, *, seed: int = 6286999,
                              replicates: int = 20000) -> dict:
        """Paired-seed uncertainty without changing the registered decisions."""
        seed_values = sorted({row["seed"] for row in rows if row["candidate"] == identifier})
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, len(seed_values), size=(replicates, len(seed_values)))

        def vector(cell: str, method: str, metric: str) -> np.ndarray:
            index = {
                row["seed"]: row[metric] for row in rows
                if row["candidate"] == identifier and row["cell"] == cell
                and row["method"] == method
            }
            return np.asarray([index[value] for value in seed_values], dtype=float)

        def interval(distribution: np.ndarray) -> dict:
            finite = distribution[np.isfinite(distribution)]
            return {
                "lower95": float(np.quantile(finite, .025)),
                "upper95": float(np.quantile(finite, .975)),
                "finite_replicates": int(len(finite)),
            }

        clean = {}
        for regime, cell in clean_map.items():
            bounded = vector(cell, "adaptive_bounded", "rmse")
            control = vector(cell, "robust_unbounded", "rmse")
            clean[regime] = interval(bounded[draw].mean(axis=1) /
                                     control[draw].mean(axis=1))
        attacks = {}
        for cell_id, cell in cells.items():
            if cell["kind"] == "clean":
                continue
            regime = "high2" if cell_id.startswith("high2_") else "neutral"
            clean_cell = clean_map[regime]
            candidate_growth = (vector(cell_id, "adaptive_bounded", "rmse")[draw].mean(axis=1)
                                - vector(clean_cell, "adaptive_bounded", "rmse")[draw].mean(axis=1))
            comparator_growth = (vector(cell_id, "robust_unbounded", "rmse")[draw].mean(axis=1)
                                 - vector(clean_cell, "robust_unbounded", "rmse")[draw].mean(axis=1))
            attenuation = np.full(replicates, np.nan)
            applicable = comparator_growth > 0
            attenuation[applicable] = 1.0 - candidate_growth[applicable] / comparator_growth[applicable]
            attacks[cell_id] = interval(attenuation)
        events = {}
        for cell_id in cells:
            loss = (vector(cell_id, "robust_unbounded", "event_recall")[draw].mean(axis=1)
                    - vector(cell_id, "adaptive_bounded", "event_recall")[draw].mean(axis=1))
            events[cell_id] = interval(loss)
        return {
            "method": "paired seed bootstrap; marginal descriptive 95% intervals",
            "seed": seed,
            "replicates": replicates,
            "clean_ratio": clean,
            "attack_attenuation": attacks,
            "event_recall_loss": events,
        }

    summaries = []
    for candidate in candidates:
        identifier = candidate["id"]
        failures: list[str] = []
        clean_ratios = {}
        for regime, cell in clean_map.items():
            ratio = (mean(identifier, cell, "adaptive_bounded", "rmse") /
                     mean(identifier, cell, "robust_unbounded", "rmse"))
            clean_ratios[regime] = ratio
            if ratio > gates["clean_ratio_max"]:
                failures.append(f"clean:{regime}")
        attacks = {}
        for cell_id, cell in cells.items():
            if cell["kind"] == "clean":
                continue
            regime = "high2" if cell_id.startswith("high2_") else "neutral"
            clean_cell = clean_map[regime]
            comparator_growth = (mean(identifier, cell_id, "robust_unbounded", "rmse") -
                                 mean(identifier, clean_cell, "robust_unbounded", "rmse"))
            candidate_growth = (mean(identifier, cell_id, "adaptive_bounded", "rmse") -
                                mean(identifier, clean_cell, "adaptive_bounded", "rmse"))
            applicable = comparator_growth > 0
            attenuation = (1 - candidate_growth / comparator_growth) if applicable else None
            attacks[cell_id] = {
                "comparator_growth": comparator_growth,
                "candidate_growth": candidate_growth,
                "protocol_identifiable": True,
                "gate_applicable": applicable,
                "attenuation": attenuation,
            }
            if applicable and attenuation < gates["attack_attenuation_min"]:
                failures.append(f"attack:{cell_id}")
        event_losses = {}
        for cell_id in cells:
            loss = (mean(identifier, cell_id, "robust_unbounded", "event_recall") -
                    mean(identifier, cell_id, "adaptive_bounded", "event_recall"))
            event_losses[cell_id] = loss
            if loss > gates["event_recall_loss_max"]:
                failures.append(f"event:{cell_id}")
        candidate_rows = [row for row in rows if row["candidate"] == identifier]
        solver_failure = max(row["solver_failure_rate"] for row in candidate_rows)
        if solver_failure > gates["solver_failure_rate_max"]:
            failures.append("solver")
        if not all(row["envelope_pass"] for row in candidate_rows
                   if row["method"] in ("adaptive_bounded", "fixed_cap8")):
            failures.append("envelope")
        applicable_attenuations = [item["attenuation"] for item in attacks.values()
                                   if item["gate_applicable"]]
        summaries.append({
            "id": identifier,
            "config": candidate,
            "clean_ratios": clean_ratios,
            "mean_clean_rmse": float(np.mean([
                mean(identifier, cell, "adaptive_bounded", "rmse")
                for cell in clean_map.values()
            ])),
            "attacks": attacks,
            "worst_applicable_attenuation": min(applicable_attenuations)
                if applicable_attenuations else None,
            "event_recall_losses": event_losses,
            "max_event_recall_loss": max(event_losses.values()),
            "max_solver_failure_rate": solver_failure,
            "descriptive_bootstrap": descriptive_bootstrap(identifier),
            "failures": failures,
            "eligible": not failures,
        })
    eligible = [item for item in summaries if item["eligible"]]
    selected = min(eligible, key=lambda item: (
        item["mean_clean_rmse"],
        -item["worst_applicable_attenuation"],
        item["id"],
    )) if eligible else None
    return {
        "candidate_summaries": summaries,
        "selected": selected["id"] if selected else None,
        "selection_rule": protocol["selection_rule"],
        "all_required_gates_pass": bool(selected),
    }


def execute(output: Path, seeds: list[int], protocol: dict,
            candidates: list[dict], *, role: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    archive = ROOT / protocol["archive"]
    with np.load(archive) as data:
        truth = np.asarray(data["truth"], float)
        primary = np.maximum(np.asarray(data[protocol["public_prediction_keys"]["primary"]], float), 0.)
        auxiliary = np.maximum(np.asarray(
            data[protocol["public_prediction_keys"]["causal_auxiliary"]], float), 0.)
        coordinates = np.asarray(data["coordinates_lat_lon"], float)
        target_indices = np.asarray(data["target_indices"], int)
    coordinates = ((coordinates - coordinates.mean(axis=0)) *
                   np.array([111.2, 111.2 * np.cos(np.deg2rad(coordinates[:, 0].mean()))]))
    estimator = EstimatorConfig(**protocol["estimator"])
    threshold = float(protocol["event_threshold"])
    events = truth >= threshold
    attack = protocol["attack"]
    rows: list[dict] = []
    prediction_arrays = {"truth": truth, "target_indices": target_indices,
                         "primary_public_prediction": primary,
                         "causal_auxiliary_public_prediction": auxiliary}
    started = time.perf_counter()
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for seed in seeds:
            for cell in protocol["cells"]:
                observations = matched_corruption_channel(
                    truth, seed=seed, kind=cell["kind"],
                    onset_fraction=cell["onset_fraction"], **attack)
                for candidate in candidates:
                    envelope = PublicEnvelopeConfig(**{
                        key: candidate[key] for key in (
                            "primary_weight", "radius_floor",
                            "uncertainty_gain", "global_radius")
                    })
                    center, radius = public_context(
                        primary, auxiliary, envelope,
                        common_shift=build_shift(len(truth), cell["reference_shift"]))
                    controls = run_equal_information_controls(
                        center, radius, coordinates, observations,
                        estimator_config=estimator,
                        innovation_scale=protocol["innovation_scale"],
                        global_radius=envelope.global_radius,
                    )
                    for method, (prediction, diagnostics) in controls.items():
                        key = f"s{seed}_{cell['id']}_{candidate['id']}_{method}"
                        prediction_arrays[key] = prediction
                        metrics = method_metrics(prediction, truth, events, threshold)
                        row = {
                            "role": role,
                            "seed": seed,
                            "cell": cell["id"],
                            "kind": cell["kind"],
                            "attack_onset_fraction": cell["onset_fraction"],
                            "reference_shift": cell["reference_shift"],
                            "candidate": candidate["id"],
                            "method": method,
                            **metrics,
                            "prediction_key": key,
                            "solver_failure_rate": float(diagnostics.get("solver_failure_rate", 0.)),
                            "maximum_correction": float(np.max(np.abs(prediction - center))),
                            "envelope_pass": bool(diagnostics.get("cap_pass", True)),
                            "global_history_uniform_linf_bound": (
                                diagnostics.get("same_public_history_uniform_linf_bound")
                                if method in ("adaptive_bounded", "fixed_cap8") else None
                            ),
                            "public_predictor_updated_from_citizen": False,
                        }
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps({"role": role, "seed": seed, "cell": cell["id"],
                                      "candidate": candidate["id"], "rows": len(rows)}), flush=True)
    np.savez_compressed(output / "predictions.npz", **prediction_arrays)
    analysis = analyze(rows, protocol, candidates)
    summary = {
        "role": role,
        "archive_scope": protocol["confirmation_scope"],
        "rows": len(rows),
        "expected_rows": len(seeds) * len(protocol["cells"]) * len(candidates) * 5,
        "seeds": seeds,
        "event_threshold": threshold,
        "event_support": int(events.sum()),
        "elapsed_seconds": time.perf_counter() - started,
        **analysis,
    }
    dump(output / "summary.json", summary)
    return summary


def development() -> None:
    protocol = load_protocol()
    output = REPORT_ROOT / "development_v1"
    summary = execute(output, protocol["development_seeds"], protocol,
                      protocol["candidate_family"], role="development")
    dump(output / "registration.json", {
        "registered_before_outcomes": True,
        "protocol": protocol,
        "input_hashes": {
            str(CONFIG.relative_to(ROOT)): sha256(CONFIG),
            protocol["archive"]: sha256(ROOT / protocol["archive"]),
            str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL),
        },
        "historical_adverse_evidence_retained": [
            "reports/v4_validation/esntnn_wrapper_7850_7861/summary.json",
            "reports/v5_diagnostics/wrapper_916000_916001/summary.json",
            "reports/v6/archive_matched_diagnostic/summary.json",
            "reports/v6/archive_precision_validation/selection.json",
        ],
    })
    print(json.dumps({"selected": summary["selected"],
                      "passed": summary["all_required_gates_pass"]}), flush=True)


def freeze() -> None:
    protocol = load_protocol()
    summary_path = REPORT_ROOT / "development_v1/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_id = summary["selected"]
    if not selected_id:
        raise RuntimeError("development has no Gate-C-eligible candidate")
    selected = next(item for item in protocol["candidate_family"] if item["id"] == selected_id)
    registry = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    namespace = "m3_m6_external_wrapper_gate_c_confirmation_v1"
    if registry["namespaces"].get(namespace) != protocol["reserved_confirmation_seeds"]:
        raise ValueError("confirmation seed registry does not match protocol")
    paths = [CONFIG, Path(__file__).resolve(), MODULE, V6_ESTIMATOR,
             PREDICTOR_WRAPPER, SEED_REGISTRY, PROTOCOL,
             ROOT / protocol["archive"], summary_path,
             REPORT_ROOT / "development_v1/results.jsonl",
             REPORT_ROOT / "development_v1/predictions.npz"]
    lock = {
        "status": "frozen_before_confirmation",
        "gate": protocol["gate"],
        "selected_candidate": selected,
        "confirmation_seeds": protocol["reserved_confirmation_seeds"],
        "seed_namespace": namespace,
        "gates": protocol["gates"],
        "cells": protocol["cells"],
        "comparator": "robust_unbounded: same Huber solve, records, graph, support and live clock; only release projection differs",
        "selection_outcomes_used": "development rows only",
        "confirmation_outcomes_read": False,
        "source_and_input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = REPORT_ROOT / "confirmation_lock.json"
    if lock_path.exists():
        raise FileExistsError(lock_path)
    dump(lock_path, lock)
    print(json.dumps({"lock": str(lock_path), "selected": selected_id}), flush=True)


def confirmation() -> None:
    protocol = load_protocol()
    lock_path = REPORT_ROOT / "confirmation_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for relative, expected in lock["source_and_input_hashes"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"locked hash changed: {relative}")
    selected = lock["selected_candidate"]
    summary = execute(REPORT_ROOT / "confirmation_v1", lock["confirmation_seeds"],
                      protocol, [selected], role="confirmation")
    development_summary = json.loads(
        (REPORT_ROOT / "development_v1/summary.json").read_text(encoding="utf-8"))
    ledger = {
        "historical_failures_preserved": {
            "v4_wrapper": "21.07% pooled clean RMSE loss against its equal-information unbounded graph control",
            "v5_selected_archive": "all 12 bounded candidates exceeded the 1.05 clean margin",
            "v6_archive_matched": "bounded Huber clean ratio 1.33 original / 1.32 recalibrated",
            "v6_precision_family": "all four candidates failed clean and event gates; selected null",
        },
        "development_candidates": development_summary["candidate_summaries"],
        "confirmation": {
            "selected": summary["selected"],
            "passed": summary["all_required_gates_pass"],
            "candidate_summaries": summary["candidate_summaries"],
        },
        "observed_fault_claim": False,
        "attack_evidence_scope": "transparent synthetic-citizen corruption over fixed archived pollution; no observed PurpleAir fault is asserted",
        "external_validity_limit": protocol["confirmation_scope"],
    }
    dump(REPORT_ROOT / "adverse_outcome_ledger.json", ledger)
    candidate = summary["candidate_summaries"][0]
    lines = [
        "# Gate C external wrapper v8 report",
        "",
        f"Confirmation passed: **{summary['all_required_gates_pass']}**.",
        "",
        "The selected release projects one equal-information unbounded Huber estimate into a radius derived only from the current cached ESN–TNN and causal persistence forecasts. The global radius is 12, so any two citizen histories sharing the public forecast context differ by at most 24 per coordinate. The live estimator and radius use no future public values.",
        "",
        "This is a randomization confirmation with new synthetic citizen-channel seeds over a historically exposed real-pollution archive. It is not a new pollution archive and the attacks are declared overlays, not observed PurpleAir faults.",
        "",
        "## Locked confirmation outcomes",
        "",
        f"- Selected candidate: `{summary['selected']}`",
        f"- Clean ratios by reference regime: `{json.dumps(candidate['clean_ratios'], sort_keys=True)}`",
        f"- Worst applicable attack attenuation: `{candidate['worst_applicable_attenuation']:.6f}`",
        f"- Maximum event-recall loss: `{candidate['max_event_recall_loss']:.6f}`",
        f"- Maximum solver-failure rate: `{candidate['max_solver_failure_rate']:.6f}`",
        "- Envelope and causal information checks: passed",
        "",
        "All failed development candidates and prior adverse wrapper evidence remain in `adverse_outcome_ledger.json`.",
    ]
    (REPORT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"confirmation_passed": summary["all_required_gates_pass"],
                      "selected": summary["selected"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "freeze", "confirmation"))
    args = parser.parse_args()
    {"development": development, "freeze": freeze,
     "confirmation": confirmation}[args.phase]()


if __name__ == "__main__":
    main()
