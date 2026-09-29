"""Final-configuration validation and subsequently locked independent confirmation."""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"


def activate():
    source = os.environ["AIRPROOF_CORE_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


def run_job(job):
    activate()
    from airproof.config import with_overrides
    from airproof.experiment import run_method, _v4_public_components, _fixed_release_metrics, METHODS
    from airproof.simulator import generate_world
    cfg, seed, method, cell = job
    world = generate_world(cfg, seed)
    result = run_method(world, cfg, method)
    result["physical_cell"] = cell
    if method == "airproof_v4":
        baseline, _, _ = _v4_public_components(world, cfg)
        raw = _fixed_release_metrics(world, with_overrides(cfg, {"privacy.release_mode": "raw"}), METHODS[method], baseline)
        result["metrics"]["same_budget_raw_dp_rmse"] = raw["release_rmse"]
        result["metrics"]["residual_dp_rmse_reduction"] = 1 - result["metrics"]["release_rmse"] / raw["release_rmse"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "primary"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/v4/core_campaign.yaml"))
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--reserve-memory-mb", type=int, default=2000)
    args = parser.parse_args()
    import yaml
    import psutil
    if not 1 <= args.workers <= 12:
        raise ValueError("invalid local worker count")
    if args.stage == "primary" and args.validation_dir is None:
        raise ValueError("primary requires completed validation and prospective power lock")
    root = Path(__file__).resolve().parents[1]
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    frontier = (root / protocol["frontier"]).resolve()
    selected = json.loads((frontier / "selection.json").read_text(encoding="utf-8"))
    if selected["selected"] is None:
        raise ValueError("no final-estimator fairness operating point was selected")
    if not selected["candidates"][selected["selected"]]["eligible"]:
        raise ValueError("selected frontier point violates registered feasibility conditions")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    power = None
    if args.stage == "validation":
        (snapshot / "airproof").mkdir(parents=True)
        (snapshot / "configs").mkdir()
        for path in (root / "airproof").glob("*.py"):
            shutil.copy2(path, snapshot / "airproof" / path.name)
        for name in ("claims_registry.yaml", "experiment_registry.yaml", "baseline_registry.yaml"):
            shutil.copy2(root / "configs" / name, snapshot / "configs" / name)
        seeds = protocol["validation_seeds"]
    else:
        validation = args.validation_dir.resolve()
        registration = json.loads((validation / "manifest.json").read_text(encoding="utf-8"))
        completion = json.loads((validation / "completion.json").read_text(encoding="utf-8"))
        power = json.loads((validation / "power_lock.json").read_text(encoding="utf-8"))
        if completion["completed"] != registration["total_jobs"] or registration["stage"] != "validation":
            raise ValueError("complete final-configuration validation required")
        if power["protocol"] != protocol or power["source_hash"] != registration["source_hash"]:
            raise ValueError("power/protocol/source mismatch")
        if power["validation_results_sha256"] != hashlib.sha256((validation / "results.jsonl").read_bytes()).hexdigest():
            raise ValueError("validation results changed after prospective power lock")
        shutil.copytree(validation / "source_snapshot", snapshot)
        seeds = power["primary_seeds"]
        if set(seeds) & set(protocol["validation_seeds"]):
            raise ValueError("primary/validation world overlap")
        if not protocol["primary_min_worlds"] <= len(seeds) <= protocol["primary_max_worlds"]:
            raise ValueError("primary sample size outside prospective resource envelope")
        shutil.copy2(validation / "power_lock.json", output / "power_lock.json")
    os.environ["AIRPROOF_CORE_SOURCE"] = str(snapshot)
    activate()
    from airproof.config import load_config, with_overrides, config_hash
    from airproof.experiment import source_tree_digest
    from airproof.continual_release import FixedReleasePlan
    base = with_overrides(load_config(frontier / "selected_configuration.yaml"), protocol["predeclared_operational_overrides"])
    if base["world"]["agents"] != protocol["agents"] or base["world"]["steps"] != protocol["steps"]:
        raise ValueError("primary scale must not shrink")
    if base["twin"]["predictive_adaptive_delta"] or base["scheduler"]["allocation_clock"] != "arrival":
        raise ValueError("final causal public-reference single-radius architecture required")
    plan = FixedReleasePlan(base["world"]["steps"], base["world"]["groups"], base["privacy"]["k_min"],
        base["privacy"]["epsilon_mean"], base["privacy"]["epsilon_count"], base["privacy"]["epsilon_user_max"])
    if len(plan.scheduled_epochs) != 28 or plan.composed_epsilon > 8 + 1e-10:
        raise ValueError("main integrated privacy budget/cadence differs from prospective protocol")
    if power and (config_hash(base) != power["base_configuration_hash"] or source_tree_digest() != power["source_hash"]):
        raise ValueError("final configuration changed after validation/power lock")
    jobs = []
    for seed in seeds:
        for cell, overrides in protocol["cells"].items():
            cfg = with_overrides(base, overrides)
            methods = list(protocol["common_methods"])
            if cell == "severe_clean":
                methods.extend(protocol["additional_severe_clean_methods"])
            jobs.extend((cfg, seed, method, cell) for method in methods)
    manifest = {"stage": args.stage, "protocol": protocol, "source_hash": source_tree_digest(),
        "base_configuration_hash": config_hash(base), "seeds": seeds, "total_jobs": len(jobs),
        "selected_fairness": selected["candidates"][selected["selected"]],
        "selection_sha256": hashlib.sha256((frontier / "selection.json").read_bytes()).hexdigest(),
        "job_plan": [{"seed": seed, "cell": cell, "method": method, "config_hash": config_hash(cfg)}
                     for cfg, seed, method, cell in jobs], "workers_max": args.workers,
        "created_unix": time.time(), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    (output / "base_configuration.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    shutil.copy2(args.protocol, output / "protocol.yaml")
    shutil.copy2(__file__, output / "runner.py")
    started, index, completed, active = time.perf_counter(), 0, 0, {}
    with (output / "results.jsonl").open("x", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while index < len(jobs) or active:
                available = psutil.virtual_memory().available / 1024**2
                target = min(args.workers, len(active) + max(0, int((available - args.reserve_memory_mb) / 800)))
                while index < len(jobs) and len(active) < target:
                    job = jobs[index]
                    active[pool.submit(run_job, job)] = job
                    index += 1
                if not active:
                    raise RuntimeError("insufficient memory for one core worker")
                ready, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in ready:
                    job = active.pop(future)
                    try:
                        row = future.result()
                    except Exception as error:
                        (output / "failure.json").write_text(json.dumps({"seed": job[1], "method": job[2], "cell": job[3],
                            "error_type": type(error).__name__, "error": str(error)}), encoding="utf-8")
                        raise
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    completed += 1
                    print(json.dumps({"event": "complete", "stage": args.stage, "completed": completed,
                        "total": len(jobs), "seed": row["seed"], "cell": row["physical_cell"], "method": row["method"],
                        "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    (output / "completion.json").write_text(json.dumps({"completed": completed,
        "elapsed_seconds": time.perf_counter() - started, "finished_unix": time.time()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
