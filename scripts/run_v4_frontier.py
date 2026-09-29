"""Run the complete final-estimator 8-world x (20 settings + control) frontier."""
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
    source = os.environ["AIRPROOF_FRONTIER_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


def run_job(job):
    activate()
    from airproof.experiment import run_method
    from airproof.simulator import generate_world
    config, seed, method, label = job
    result = run_method(generate_world(config, seed), config, method)
    result["frontier_candidate"] = label
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/v4/fairness_frontier.yaml"))
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--reserve-memory-mb", type=int, default=2000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("worker count outside local resource envelope")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    (snapshot / "airproof").mkdir(parents=True)
    (snapshot / "configs").mkdir()
    for path in (root / "airproof").glob("*.py"):
        shutil.copy2(path, snapshot / "airproof" / path.name)
    for name in ("claims_registry.yaml", "experiment_registry.yaml", "baseline_registry.yaml"):
        shutil.copy2(root / "configs" / name, snapshot / "configs" / name)
    shutil.copy2(__file__, output / "runner.py")
    os.environ["AIRPROOF_FRONTIER_SOURCE"] = str(snapshot)
    activate()
    import psutil
    import yaml
    from airproof.config import config_hash, load_config, with_overrides
    from airproof.experiment import source_tree_digest
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    base = load_config(args.protocol.parent / protocol["base_config"])
    if base["world"]["agents"] != protocol["agents"] or base["world"]["steps"] != protocol["steps"]:
        raise ValueError("frontier must use the registered full population/horizon")
    if base["scheduler"]["allocation_clock"] != "arrival" or base["attack"]["kind"] != "clean":
        raise ValueError("frontier requires causal allocation and matched clean worlds")
    shutil.copy2(args.protocol, output / "protocol.yaml")
    (output / "base_configuration.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    candidates = {}
    for q in protocol["targets"]:
        for strength in protocol["fairness_strengths"]:
            label = f"q{q}_lambda{strength:g}"
            candidates[label] = {"target": q, "strength": strength,
                                 "config": with_overrides(base, {"scheduler.target_contributors": q,
                                                                "scheduler.fairness_strength": strength})}
    jobs = []
    for seed in protocol["seeds"]:
        jobs.append((base, seed, protocol["control_method"], "no_fairness"))
        jobs.extend((item["config"], seed, protocol["candidate_method"], label) for label, item in candidates.items())
    manifest = {"role": "final-estimator-validation-frontier-not-primary-inference",
                "protocol": protocol, "source_hash": source_tree_digest(), "seeds": protocol["seeds"],
                "candidates": {label: {"target": item["target"], "strength": item["strength"],
                                       "config_hash": config_hash(item["config"])} for label, item in candidates.items()},
                "control_config_hash": config_hash(base), "workers_max": args.workers,
                "job_plan": [{"seed": seed, "method": method, "label": label, "config_hash": config_hash(cfg)}
                             for cfg, seed, method, label in jobs],
                "total_jobs": len(jobs), "created_unix": time.time(),
                "reserve_memory_mb": args.reserve_memory_mb,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started, index, completed, pending = time.perf_counter(), 0, 0, {}
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while index < len(jobs) or pending:
                available = psutil.virtual_memory().available / 1024**2
                additional = max(0, int((available - args.reserve_memory_mb) / 800))
                target = min(args.workers, len(pending) + additional)
                while index < len(jobs) and len(pending) < target:
                    job = jobs[index]
                    pending[pool.submit(run_job, job)] = job
                    index += 1
                if not pending:
                    raise RuntimeError("not enough memory for one frontier job")
                ready, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in ready:
                    job = pending.pop(future)
                    try:
                        row = future.result()
                    except Exception as error:
                        (output / "failure.json").write_text(json.dumps({"seed": job[1], "label": job[3],
                            "error_type": type(error).__name__, "error": str(error)}), encoding="utf-8")
                        raise
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    completed += 1
                    print(json.dumps({"event": "complete", "completed": completed, "total": len(jobs),
                                      "seed": row["seed"], "candidate": row["frontier_candidate"],
                                      "rmse": row["metrics"]["rmse"],
                                      "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    (output / "completion.json").write_text(json.dumps({"completed": completed,
        "elapsed_seconds": time.perf_counter() - started, "finished_unix": time.time()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
