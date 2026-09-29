"""Preserve a completed selection checkpoint and scale remaining frozen folds.

Only orchestration changes. The original source snapshot, protocol, completed
folds and untouched test partition remain unchanged. Resource dependencies keep
one active fold while the larger frontier/privacy jobs occupy the machine.
"""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"


def activate():
    source = os.environ["AIRPROOF_BEIJING_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


def select_job(job):
    activate()
    from airproof.beijing_v4 import load_beijing_archive, select_outer_fold
    dataset, index, protocol = job
    return select_outer_fold(load_beijing_archive(dataset), index, protocol)


def test_job(job):
    activate()
    from airproof.beijing_v4 import evaluate_outer_fold, load_beijing_archive
    dataset, selection, protocol = job
    return evaluate_outer_fold(load_beijing_archive(dataset), selection, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--replace-controller-pid", type=int, required=True)
    parser.add_argument("--resource-pids", required=True)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    import psutil
    import numpy as np
    if not 1 <= args.workers <= 12:
        raise ValueError("invalid local worker budget")
    directory = args.campaign.resolve()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    log = directory / "selections.jsonl"
    def read_selections():
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if len({row["outer_target_index"] for row in rows}) != len(rows):
            raise ValueError("duplicate completed checkpoint")
        return rows
    selections = read_selections()
    if len(selections) >= 11 or (directory / "selection_lock.json").exists():
        raise ValueError("near-complete/locked campaign should finish without a handoff")
    original = psutil.Process(args.replace_controller_pid)
    command = " ".join(original.cmdline())
    if "benchmark_beijing_v4.py" not in command or directory.name not in command or Path(original.cwd()).resolve() != root:
        raise ValueError("refusing to replace an unrelated process")
    original_birth = original.create_time()
    resources = []
    for pid in map(int, args.resource_pids.split(",")):
        process = psutil.Process(pid)
        command = " ".join(process.cmdline())
        if not any(name in command for name in ("run_v4_frontier.py", "benchmark_privacy_grid_v4.py")):
            raise ValueError("unrecognized resource dependency")
        resources.append({"pid": pid, "created": process.create_time()})
    plan = {"purpose": "parallelize-remaining-frozen-folds-after-current-checkpoint",
            "original_controller": original.pid, "original_created": original_birth,
            "initial_completed_folds": len(selections), "handoff_after_completed_folds": len(selections) + 1,
            "resource_dependencies": resources, "maximum_workers_after_dependencies": args.workers,
            "source_hash_preserved": manifest["source_hash"], "created_unix": time.time()}
    continuation_plan = directory / "continuation_plan.json"
    if continuation_plan.exists():
        raise ValueError("a handoff is already planned")
    continuation_plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"event": "waiting_for_selection_checkpoint", "target": plan["handoff_after_completed_folds"]}), flush=True)
    while True:
        if not original.is_running() or original.create_time() != original_birth:
            raise RuntimeError("original controller ended before the coordinated handoff")
        try:
            current = read_selections()
        except json.JSONDecodeError:
            time.sleep(.25)
            continue
        if len(current) >= plan["handoff_after_completed_folds"]:
            break
        time.sleep(1)
    # The parent flushes each complete JSON record before its progress message.
    # Stop only the verified controller and its current pool descendants; the
    # preserved checkpoint contains every completed numerical fold.
    descendants = original.children(recursive=True)
    original.terminate()
    for process in descendants:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([original, *descendants], timeout=10)
    if alive:
        raise RuntimeError("old controller or worker did not terminate")
    selections = read_selections()
    handoff = {**plan, "handoff_unix": time.time(), "preserved_completed_folds": len(selections),
               "stopped_process_ids": [original.pid, *(process.pid for process in descendants)],
               "data_or_results_deleted": False}
    (directory / "continuation_handoff.json").write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    os.environ["AIRPROOF_BEIJING_SOURCE"] = str(directory / "source_snapshot")
    activate()
    from airproof.experiment import source_tree_digest
    if source_tree_digest() != manifest["source_hash"]:
        raise ValueError("frozen numerical source changed during handoff")
    protocol = manifest["protocol"]
    dataset = str((root / protocol["dataset"]).resolve())
    completed = {row["outer_target_index"] for row in selections}
    pending_jobs = [(dataset, index, protocol) for index in range(12) if index not in completed]
    def capacity():
        for item in resources:
            try:
                process = psutil.Process(item["pid"])
                if process.is_running() and process.create_time() == item["created"]:
                    return 1
            except psutil.NoSuchProcess:
                pass
        return args.workers
    start = time.perf_counter()
    print(json.dumps({"event": "continued_from_checkpoint", "preserved": len(selections)}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        active, index = {}, 0
        with log.open("a", encoding="utf-8") as handle:
            while index < len(pending_jobs) or active:
                available = psutil.virtual_memory().available / 1024**2
                target = min(capacity(), len(active) + max(0, int((available - 1800) / 500)))
                while index < len(pending_jobs) and len(active) < target:
                    job = pending_jobs[index]
                    active[pool.submit(select_job, job)] = job[1]
                    index += 1
                if not active:
                    raise RuntimeError("no memory for one continuation fold")
                ready, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                for future in ready:
                    active.pop(future)
                    selection = future.result()
                    selections.append(selection)
                    handle.write(json.dumps(selection, allow_nan=False) + "\n")
                    handle.flush()
                    print(json.dumps({"event": "outer_selection_complete", "completed": len(selections),
                        "total": 12, "station": selection["outer_target"], "active_workers": len(active)}), flush=True)
        selections.sort(key=lambda row: row["outer_target_index"])
        if [row["outer_target_index"] for row in selections] != list(range(12)):
            raise ValueError("all twelve selections required before any test fold")
        lock = {"selections": selections, "source_hash": manifest["source_hash"], "protocol": protocol,
                "locked_unix": time.time(), "test_rows_evaluated_before_lock": 0}
        canonical = json.dumps(lock, sort_keys=True, allow_nan=False)
        (directory / "selection_lock.json").write_text(canonical, encoding="utf-8")
        results = []
        # Resource-dependent selection cannot finish while a heavy dependency is
        # live unless the entire 12-fold sequence was already completed serially.
        # Keep the test phase bounded by the same current capacity.
        jobs = [(dataset, selection, protocol) for selection in selections]
        active, index = {}, 0
        with (directory / "results.jsonl").open("x", encoding="utf-8") as handle:
            while index < len(jobs) or active:
                while index < len(jobs) and len(active) < capacity():
                    active[pool.submit(test_job, jobs[index])] = index
                    index += 1
                ready, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                for future in ready:
                    active.pop(future)
                    result, arrays = future.result()
                    np.savez_compressed(directory / (result["station"] + ".npz"), **arrays)
                    handle.write(json.dumps(result, allow_nan=False) + "\n")
                    handle.flush()
                    results.append(result)
                    print(json.dumps({"event": "outer_test_complete", "completed": len(results), "total": 12}), flush=True)
    import hashlib
    (directory / "completion.json").write_text(json.dumps({"selections": 12, "test_folds": len(results),
        "selection_lock_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "elapsed_seconds": time.time() - manifest["created_unix"], "continuation_seconds": time.perf_counter() - start,
        "finished_unix": time.time()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
