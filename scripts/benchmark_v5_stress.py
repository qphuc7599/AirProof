"""Register compact OAT stress and run transformation preflight; scoring is opt-in."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from airproof.config import with_overrides
from airproof.simulator import generate_world
from airproof.v5_estimator import EstimatorConfig
from airproof.v5_stress import (STRESS_SEEDS, configuration_for_stress, prepare_stress_public,
                               score_case, stress_matrix, transform_world, validate_selected)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/v5/stress")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--selected-config", type=Path)
    args = parser.parse_args()
    if args.score and args.selected_config is None:
        parser.error("--score requires an explicit locked --selected-config; no default selection")
    protocol_path = ROOT / "configs/v5/protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    base_path = ROOT / protocol["historical_configuration"]
    base = yaml.safe_load(base_path.read_text())
    base["transport"] = protocol["transport"]
    base["scale"] = protocol["scale"]
    cases = stress_matrix()
    selected = None
    if args.score:
        selected = EstimatorConfig(**json.loads(args.selected_config.read_text()))
        validate_selected(selected)
    inventory = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted((ROOT / "airproof").glob("*.py"))}
    inventory[str(Path(__file__).relative_to(ROOT)).replace("\\", "/")] = sha(Path(__file__))
    manifest = {"version": "v5-oat-stress-1", "stage": "scoring" if args.score else "registered-preflight-only",
        "seeds": STRESS_SEEDS, "cases": [asdict(c) for c in cases], "methods": ["AP", "SQ", "PUBLIC", "HUBER"],
        "expected_full_rows": len(cases)*len(STRESS_SEEDS)*4, "source_sha256": inventory,
        "protocol_sha256": sha(protocol_path), "base_sha256": sha(base_path),
        "selected_config": asdict(selected) if selected else None,
        "selection_file_sha256": sha(args.selected_config) if selected else None,
        "not_selection_data": True, "no_extra_search": True,
        "reference_count_order": "nested sorted reference-cell subset",
        "event": {"amplitude": 8, "duration_max": 24, "start": "max(burn,steps//2)",
                  "cells": "nearest max(1,cells//64) non-reference cells to farthest-from-reference center"},
        "attacks": {"drift": "fixed-positive ramp amplitude12; no truth-directed sign", "hotspot": "minus amplitude12 only group0", "inlier": "plus min(amplitude,1.25*sigma)"},
        "controls": "same public model and actual arrived/selected citizen/regulatory support; PUBLIC has no extra citizen inputs",
        "bandwidth512": "128-byte raw lane after fixed reservations cannot fit a512-byte raw packet; retain this structural starvation endpoint"}
    destination = args.output / ("scoring" if args.score else "preflight")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not args.score:
        toy = with_overrides(base, {"world.agents": 12, "world.grid_side": 8, "world.steps": 48,
            "world.burn_in_steps": 8, "twin.reference_calibration_epochs": 8, "attack.start_epoch": 16})
        toy["scale"] = {**base["scale"], "acquisition_epochs": 48}
        world = generate_world(configuration_for_stress(toy, cases[0]), 910009)
        original = repr(world.observations), repr(world.reference_observations), world.truth.copy()
        checks = []
        for case in cases:
            cfg = configuration_for_stress(toy, case)
            changed, event = transform_world(world, cfg, case)
            public, _, diagnostic = prepare_stress_public(changed, cfg, case)
            assert np.isfinite(public).all()
            assert original[0] == repr(world.observations) and original[1] == repr(world.reference_observations)
            assert np.array_equal(original[2], world.truth)
            checks.append({"case": case.name, "finite_public": True, "reference_count": len({r.cell for r in changed.reference_observations}),
                "first_available_model": diagnostic["first_available_selected_model_epoch"], "event_entries": int(event.sum())})
        (destination / "checks.json").write_text(json.dumps({"scored": False, "toy_seed": 910009, "full_stress_seeds_used": False,
                                                            "input_immutable": True, "cases": checks}, indent=2))
        print(json.dumps({"registered_cases": len(cases), "planned_rows": manifest["expected_full_rows"], "preflight_pass": True, "scored": False}))
        return
    completed_rows = 0
    with (destination / "results.jsonl").open("w") as handle:
        for seed in STRESS_SEEDS:
            world = generate_world(configuration_for_stress(base, cases[0]), seed)
            for case in cases:
                cfg = configuration_for_stress(base, case)
                changed, event = transform_world(world, cfg, case)
                public, operators, diagnostic = prepare_stress_public(changed, cfg, case)
                for row in score_case(changed, public, operators, cfg, selected, event):
                    handle.write(json.dumps({**row, "public_diagnostics": diagnostic}, allow_nan=False)+"\n")
                    completed_rows += 1
                handle.flush()
                print(json.dumps({"seed": seed, "case": case.name, "completed": True}), flush=True)
    if completed_rows != manifest["expected_full_rows"]:
        raise ValueError("incomplete stress matrix")
    (destination / "completion.json").write_text(json.dumps({"rows": completed_rows, "complete": True,
                                                            "results_sha256": sha(destination / "results.jsonl")}, indent=2))


if __name__ == "__main__":
    main()
