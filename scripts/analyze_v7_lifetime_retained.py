#!/usr/bin/env python3
"""Audit retained v6 lifetime evidence without rerunning the frozen estimator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from airproof.v6_covariance_forcing_inputs import restore_records, sha256_file, write_json
from airproof.v6_lifetime_budget7_validation_v2_inputs import (
    load_scoring_view,
    verify_bundle,
)
from airproof.v7_lifetime_review import (
    lifetime_exposure_profile,
    registered_lifetime_certificate,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = Path("configs/v6/lifetime_budget7_validation_v2.json")
OUTPUT = Path("reports/v7/reviewer_revision/lifetime_retained_diagnostic")
METHODS = ("AP_LIFETIME7", "PUBLIC", "SQ")


def _load_prediction(path: Path, manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("prediction_sha256") != sha256_file(path):
        raise ValueError(f"prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"live", "reconstructed"}:
            raise ValueError(f"prediction schema mismatch: {path}")
        return arrays["live"].copy(), arrays["reconstructed"].copy()


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _phase_rows(
    registration: dict,
    bundle: Path,
    outcomes: Path,
) -> list[dict]:
    start, stop = map(int, registration["evaluation_clock"]["scored_acquisition_half_open"])
    length = stop - start
    thirds = length // 3
    phases = {
        "early_third": (0, thirds),
        "middle_third": (thirds, 2 * thirds),
        "late_third": (2 * thirds, length),
    }
    attack_start = int(registration["attack_contract"]["severe_drift"]["start_epoch"])
    attack_relative = attack_start - start
    remaining = length - attack_relative
    phases.update(
        {
            "post_attack_early": (attack_relative, attack_relative + remaining // 2),
            "post_attack_late": (attack_relative + remaining // 2, length),
        }
    )
    rows = []
    for seed in registration["worlds"]["seeds"]:
        for cell in registration["worlds"]["cells"]:
            scoring = load_scoring_view(bundle, int(seed), cell)
            truth = scoring["evaluation_truth"]
            events = scoring["event_mask"]
            if len(truth) != length:
                raise ValueError("scoring clock differs from registered horizon")
            for method in METHODS:
                arm = outcomes / "jobs" / str(seed) / cell / method
                live, reconstructed = _load_prediction(
                    arm / "prediction.npz", arm / "prediction_manifest.json"
                )
                live = live[:length]
                reconstructed = reconstructed[:length]
                for phase, (left, right) in phases.items():
                    if phase.startswith("post_attack") and cell not in (
                        "severe_drift",
                        "severe_hotspot",
                    ):
                        continue
                    mask = events[left:right]
                    rows.append(
                        {
                            "seed": int(seed),
                            "scenario": cell,
                            "method": method,
                            "phase": phase,
                            "absolute_half_open": [start + left, start + right],
                            "rmse": _rmse(truth[left:right], reconstructed[left:right]),
                            "live_rmse": _rmse(truth[left:right], live[left:right]),
                            "event_support": int(mask.sum()),
                            "event_recall": (
                                float(
                                    np.mean(
                                        live[left:right][mask]
                                        >= float(scoring["event_threshold"])
                                    )
                                )
                                if mask.any()
                                else None
                            ),
                        }
                    )
    return rows


def _profile_rows(registration: dict, bundle: Path) -> tuple[list[dict], list[dict]]:
    budget = float(registration["fixed_estimator"]["per_user_exposure_budget"])
    lag = int(registration["fixed_estimator"]["lag"])
    acquisition = int(registration["worlds"]["acquisition_epochs"])
    burn = int(registration["worlds"]["burn_in_epochs"])
    drain = int(registration["worlds"]["transport_drain_epochs"])
    horizon = acquisition - burn + drain
    rows = []
    audits = []
    for seed in registration["worlds"]["seeds"]:
        observation_path = bundle / "prediction" / str(seed) / "observations_severe_clean.npz"
        observations, arrivals = restore_records(observation_path)
        _, profile, audit = lifetime_exposure_profile(
            observations,
            arrivals,
            lag=lag,
            per_user_exposure_budget=budget,
            horizon=horizon,
        )
        audits.append({"seed": int(seed), **audit})
        rows.extend({"seed": int(seed), "absolute_epoch": burn + row["epoch"], **row} for row in profile)
    return rows, audits


def _aggregate_profiles(rows: list[dict]) -> list[dict]:
    groups: defaultdict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["epoch"])].append(row)
    fields = (
        "input_records",
        "effective_records",
        "dropped_exhausted_records",
        "active_users",
        "granted_active_users",
        "active_user_fraction",
        "mean_effective_quality",
        "users_seen",
        "mean_remaining_budget",
        "median_remaining_budget",
        "exhausted_user_fraction",
        "cumulative_input_records",
        "cumulative_effective_records",
        "cumulative_dropped_exhausted_records",
        "cumulative_requested_exposure",
        "cumulative_granted_exposure",
    )
    result = []
    for epoch, values in sorted(groups.items()):
        record = {"epoch": epoch, "absolute_epoch": values[0]["absolute_epoch"]}
        for field in fields:
            numeric = np.asarray([float(row[field]) for row in values], dtype=float)
            finite = numeric[np.isfinite(numeric)]
            record[f"mean_{field}"] = float(np.mean(finite)) if len(finite) else None
            record[f"min_{field}"] = float(np.min(finite)) if len(finite) else None
            record[f"max_{field}"] = float(np.max(finite)) if len(finite) else None
        result.append(record)
    return result


def _paired_summary(rows: list[dict]) -> list[dict]:
    indexed = {
        (row["seed"], row["scenario"], row["phase"], row["method"]): row
        for row in rows
    }
    keys = sorted({(row["scenario"], row["phase"]) for row in rows})
    result = []
    seeds = sorted({int(row["seed"]) for row in rows})
    for scenario, phase in keys:
        ap = np.asarray(
            [indexed[(seed, scenario, phase, "AP_LIFETIME7")]["rmse"] for seed in seeds]
        )
        public = np.asarray(
            [indexed[(seed, scenario, phase, "PUBLIC")]["rmse"] for seed in seeds]
        )
        sq = np.asarray([indexed[(seed, scenario, phase, "SQ")]["rmse"] for seed in seeds])
        result.append(
            {
                "scenario": scenario,
                "phase": phase,
                "worlds": len(seeds),
                "ap_rmse_mean": float(ap.mean()),
                "public_rmse_mean": float(public.mean()),
                "sq_rmse_mean": float(sq.mean()),
                "ap_minus_public_mean": float(np.mean(ap - public)),
                "ap_to_public_ratio_of_means": float(ap.mean() / public.mean()),
                "ap_minus_sq_mean": float(np.mean(ap - sq)),
                "ap_to_sq_ratio_of_means": float(ap.mean() / sq.mean()),
                "ap_better_than_public_worlds": int(np.sum(ap < public)),
                "ap_better_than_sq_worlds": int(np.sum(ap < sq)),
            }
        )
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(output: Path, *, verify_inputs: bool) -> dict:
    registration_file = ROOT / REGISTRATION
    registration = json.loads(registration_file.read_text(encoding="utf-8"))
    bundle = ROOT / registration["input_producer"]["bundle"]
    outcomes = ROOT / registration["artifacts"]["outcomes"]
    input_lock = json.loads(
        (ROOT / registration["input_producer"]["input_lock"]).read_text(encoding="utf-8")
    )
    if verify_inputs:
        verify_bundle(
            ROOT,
            bundle,
            expected_manifest_sha256=input_lock["manifest_sha256"],
        )
    output.mkdir(parents=True, exist_ok=True)
    certificate = registered_lifetime_certificate(ROOT, REGISTRATION)
    phase_rows = _phase_rows(registration, bundle, outcomes)
    profile_rows, exposure_audits = _profile_rows(registration, bundle)
    aggregate_profiles = _aggregate_profiles(profile_rows)
    paired = _paired_summary(phase_rows)
    _write_jsonl(output / "per_world_phase_metrics.jsonl", phase_rows)
    _write_csv(output / "per_world_epoch_exposure.csv", profile_rows)
    _write_csv(output / "aggregate_epoch_exposure.csv", aggregate_profiles)
    write_json(output / "certificate.json", certificate)
    write_json(output / "exposure_audits.json", {"rows": exposure_audits})
    write_json(output / "paired_phase_summary.json", {"rows": paired})
    analysis = {
        "schema_version": 1,
        "role": "post-hoc retained-evidence diagnostic; no estimator rerun or model selection",
        "input_bundle_verified": bool(verify_inputs),
        "worlds": len(registration["worlds"]["seeds"]),
        "scenarios": registration["worlds"]["cells"],
        "phase_metric_rows": len(phase_rows),
        "profile_rows": len(profile_rows),
        "paired_phase_summary": paired,
        "exposure_final": exposure_audits,
        "certificate": certificate,
        "source": {
            "script": "scripts/analyze_v7_lifetime_retained.py",
            "script_sha256": sha256_file(Path(__file__)),
            "registration": REGISTRATION.as_posix(),
            "registration_sha256": sha256_file(registration_file),
            "input_manifest_sha256": input_lock["manifest_sha256"],
        },
    }
    analysis["analysis_identity"] = hashlib.sha256(
        json.dumps(analysis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(output / "analysis.json", analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--skip-input-rehash",
        action="store_true",
        help="trust the existing immutable input lock for a faster local diagnostic",
    )
    args = parser.parse_args()
    analysis = execute(ROOT / args.output, verify_inputs=not args.skip_input_rehash)
    print(json.dumps({
        "output": str((ROOT / args.output).resolve()),
        "worlds": analysis["worlds"],
        "phase_metric_rows": analysis["phase_metric_rows"],
        "profile_rows": analysis["profile_rows"],
        "analysis_identity": analysis["analysis_identity"],
    }, indent=2))


if __name__ == "__main__":
    main()
