"""Full fixed-reference DP interactions, with optional live-process dependency."""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"


def activate():
    source = os.environ["AIRPROOF_PRIVACY_GRID_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


def run_job(job):
    activate()
    import numpy as np
    from airproof.config import config_hash
    from airproof.experiment import _v4_public_components
    from airproof.privacy_grid import prepare_statistics, evaluate_setting, grid_settings
    from airproof.simulator import generate_world
    cfg, protocol, seed, cell = job
    started = time.perf_counter()
    world = generate_world(cfg, seed)
    fields, _, public_diagnostics = _v4_public_components(world, cfg)
    groups = int(cfg["world"]["groups"])
    baseline = np.column_stack([fields[:, world.cell_groups == group].mean(axis=1) for group in range(groups)])
    truth = np.column_stack([world.truth[:, world.cell_groups == group].mean(axis=1) for group in range(groups)])
    statistics = prepare_statistics(list(world.observations), public_baselines=baseline, groups=groups,
        raw_clip=protocol["raw_clip"], residual_clip=protocol["residual_clip"],
        deadline_steps=protocol["public_collection_deadline_steps"])
    rows = [evaluate_setting(statistics, truth, setting, seed=seed,
        burn_in=cfg["world"]["burn_in_steps"], deadline_steps=protocol["public_collection_deadline_steps"],
        false_release_probability=protocol["false_release_probability"]) for setting in grid_settings(protocol)]
    return {"seed": seed, "cell": cell, "config_hash": config_hash(cfg), "world": world.metadata,
            "public_calibration": public_diagnostics, "contributor_epochs": statistics.contributor_epochs,
            "rows": rows, "runtime_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/v4/privacy_interactions.yaml"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--reserve-memory-mb", type=int, default=2000)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--dependency-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("worker count outside local resource envelope")
    if bool(args.wait_for_pid) != bool(args.dependency_output):
        raise ValueError("dependency requires both exact PID and output directory")
    import psutil
    dependency = None
    if args.wait_for_pid:
        process = psutil.Process(args.wait_for_pid)
        if "run_v4_frontier.py" not in " ".join(process.cmdline()):
            raise ValueError("dependency PID is not the expected live frontier")
        dependency = {"pid": process.pid, "created": process.create_time(), "output": str(args.dependency_output.resolve())}
    root, output = Path(__file__).resolve().parents[1], args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    (snapshot / "airproof").mkdir(parents=True)
    (snapshot / "configs").mkdir()
    for path in (root / "airproof").glob("*.py"):
        shutil.copy2(path, snapshot / "airproof" / path.name)
    for name in ("claims_registry.yaml", "experiment_registry.yaml", "baseline_registry.yaml"):
        shutil.copy2(root / "configs" / name, snapshot / "configs" / name)
    shutil.copy2(args.protocol, output / "protocol.yaml")
    shutil.copy2(__file__, output / "runner.py")
    os.environ["AIRPROOF_PRIVACY_GRID_SOURCE"] = str(snapshot)
    activate()
    import yaml
    from airproof.config import config_hash, load_config, with_overrides
    from airproof.data import sha256_file
    from airproof.experiment import source_tree_digest
    from airproof.privacy_grid import grid_settings
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    base = load_config(args.protocol.parent / protocol["base_config"])
    if base["world"]["agents"] != protocol["agents"] or base["world"]["steps"] != protocol["steps"]:
        raise ValueError("registered full population/horizon required")
    (output / "base_configuration.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    jobs = [(with_overrides(base, overrides), protocol, seed, cell)
            for seed in protocol["seeds"] for cell, overrides in protocol["cells"].items()]
    manifest = {"protocol": protocol, "source_hash": source_tree_digest(), "dependency": dependency,
        "runner_hash": sha256_file(Path(__file__)), "created_unix": time.time(),
        "physical_jobs": len(jobs), "settings_per_world": len(grid_settings(protocol)),
        "total_rows": len(jobs) * len(grid_settings(protocol)), "workers_max": args.workers,
        "job_plan": [{"seed": seed, "cell": cell, "config_hash": config_hash(cfg)} for cfg, _, seed, cell in jobs],
        "expected_settings": grid_settings(protocol)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if dependency:
        print(json.dumps({"event": "waiting_for_verified_frontier", **dependency}), flush=True)
        while psutil.pid_exists(dependency["pid"]):
            try:
                if psutil.Process(dependency["pid"]).create_time() != dependency["created"]:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(30)
        dependency_root = Path(dependency["output"])
        completion = json.loads((dependency_root / "completion.json").read_text(encoding="utf-8"))
        registration = json.loads((dependency_root / "manifest.json").read_text(encoding="utf-8"))
        if completion["completed"] != registration["total_jobs"]:
            raise RuntimeError("frontier dependency terminated without full completion")
    started, index, completed, row_count, pending = time.perf_counter(), 0, 0, 0, {}
    (output / "started.json").write_text(json.dumps({"started_unix": time.time()}), encoding="utf-8")
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle, (output / "worlds.jsonl").open("w", encoding="utf-8") as worlds:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while index < len(jobs) or pending:
                available = psutil.virtual_memory().available / 1024**2
                additional = max(0, int((available - args.reserve_memory_mb) / 900))
                target = min(args.workers, len(pending) + additional)
                while index < len(jobs) and len(pending) < target:
                    job = jobs[index]
                    pending[pool.submit(run_job, job)] = job
                    index += 1
                if not pending:
                    raise RuntimeError("insufficient memory for one privacy world")
                ready, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in ready:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        (output / "failure.json").write_text(json.dumps({"seed": job[2], "cell": job[3],
                            "type": type(error).__name__, "error": str(error)}), encoding="utf-8")
                        raise
                    rows = result.pop("rows")
                    worlds.write(json.dumps(result, allow_nan=False) + "\n")
                    worlds.flush()
                    for row in rows:
                        handle.write(json.dumps({"seed": result["seed"], "cell": result["cell"],
                            "config_hash": result["config_hash"], "source_hash": manifest["source_hash"], **row}, allow_nan=False) + "\n")
                    handle.flush()
                    completed += 1
                    row_count += len(rows)
                    print(json.dumps({"event": "complete_world", "completed": completed, "total": len(jobs),
                        "rows": row_count, "seed": result["seed"], "cell": result["cell"],
                        "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    (output / "completion.json").write_text(json.dumps({"physical_worlds": completed, "rows": row_count,
        "elapsed_seconds": time.perf_counter() - started, "finished_unix": time.time()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
