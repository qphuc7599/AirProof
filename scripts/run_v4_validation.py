"""Run the complete 12-world v4 selection matrix from an isolated source copy."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"


def activate_snapshot():
    source = os.environ["AIRPROOF_VALIDATION_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


def run_job(job):
    activate_snapshot()
    from airproof.config import with_overrides
    from airproof.experiment import _fixed_release_metrics, _v4_public_components, METHODS, run_method
    from airproof.simulator import generate_world

    config, seed, method = job
    world = generate_world(config, seed)
    result = run_method(world, config, method)
    # Same input, public baseline, epsilon and random stream: counterfactual raw
    # release utility. This is not an additional free real-data publication.
    if method == "airproof_v4":
        baseline, _, _ = _v4_public_components(world, config)
        raw = _fixed_release_metrics(world, with_overrides(config, {"privacy.release_mode": "raw"}),
                                     METHODS[method], baseline)
        result["metrics"]["same_budget_raw_dp_rmse"] = raw["release_rmse"]
        result["metrics"]["residual_dp_rmse_reduction"] = (
            1 - result["metrics"]["release_rmse"] / raw["release_rmse"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v4/public_reference_validation.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", default=",".join(map(str, range(7300, 7312))))
    parser.add_argument("--reserve-memory-mb", type=int, default=1200)
    parser.add_argument("--estimated-job-mb", type=int, default=700)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("workers must lie in [1,12]")
    seeds = list(map(int, args.seeds.split(",")))
    if len(set(seeds)) != len(seeds):
        raise SystemExit("duplicate validation seeds")
    root = Path(__file__).resolve().parents[1]
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / "source_snapshot"
    (snapshot / "airproof").mkdir(parents=True)
    (snapshot / "configs").mkdir()
    for source in (root / "airproof").glob("*.py"):
        shutil.copy2(source, snapshot / "airproof" / source.name)
    for name in ("claims_registry.yaml", "experiment_registry.yaml", "baseline_registry.yaml"):
        shutil.copy2(root / "configs" / name, snapshot / "configs" / name)
    shutil.copy2(args.config, destination / "configuration.yaml")
    shutil.copy2(__file__, destination / "runner.py")
    os.environ["AIRPROOF_VALIDATION_SOURCE"] = str(snapshot)
    activate_snapshot()
    import psutil
    from airproof.config import config_hash, load_config, with_overrides
    from airproof.experiment import source_tree_digest

    base = load_config(destination / "configuration.yaml")
    jobs = []
    # Interleave clean and matched attack worlds to detect implementation failures
    # promptly. All cells are still run; early results do not change parameters.
    for seed in seeds:
        for kind in ("clean", "adversarial_drift", "hotspot_suppression"):
            config = with_overrides(base, {"attack.kind": kind})
            for method in ("airproof_v4", "squared_loss_v4"):
                jobs.append((config, seed, method))
    manifest = {
        "protocol": "12-world-full-horizon-validation-not-final-confirmation",
        "source_hash": source_tree_digest(), "config_hash": config_hash(base),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds": seeds, "total_jobs": len(jobs), "workers_max": args.workers,
        "created_unix": time.time(), "python": sys.version,
        "memory_reserve_mb": args.reserve_memory_mb, "estimated_job_mb": args.estimated_job_mb,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started, index, completed = time.perf_counter(), 0, 0
    pending = {}
    with (destination / "results.jsonl").open("x", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while index < len(jobs) or pending:
                available_mb = psutil.virtual_memory().available / 1024**2
                # Account for tasks submitted but not yet resident. Keep memory
                # bounded rather than forcing 12 simultaneous jobs into 16 GB.
                additional = max(0, int((available_mb - args.reserve_memory_mb) / args.estimated_job_mb))
                target = min(args.workers, len(pending) + additional)
                submitted = 0
                while index < len(jobs) and len(pending) < target:
                    job = jobs[index]
                    pending[pool.submit(run_job, job)] = job
                    index += 1
                    submitted += 1
                if submitted:
                    print(json.dumps({"event": "submitted", "active": len(pending),
                                      "submitted_total": index, "available_mb": round(available_mb)}), flush=True)
                if not pending:
                    raise RuntimeError("insufficient free memory for one validation worker")
                ready, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in ready:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        failure = {"seed": job[1], "method": job[2], "kind": job[0]["attack"]["kind"],
                                   "error_type": type(error).__name__, "error": str(error)}
                        with (destination / "failures.jsonl").open("a", encoding="utf-8") as failures:
                            failures.write(json.dumps(failure) + "\n")
                        raise
                    stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    completed += 1
                    print(json.dumps({"event": "complete", "completed": completed, "total": len(jobs),
                                      "seed": result["seed"], "method": result["method"],
                                      "kind": result["scenario"]["attack_kind"],
                                      "rmse": result["metrics"]["rmse"],
                                      "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    (destination / "completion.json").write_text(json.dumps({
        "completed": completed, "elapsed_seconds": time.perf_counter() - started,
        "finished_unix": time.time(),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
