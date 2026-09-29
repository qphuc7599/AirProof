"""Freeze seasonal calibration, then score a fresh seven-month Beijing partition."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"


def activate():
    source = os.environ["AIRPROOF_BEIJING_SEASONAL_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


def run_job(job):
    activate()
    from airproof.beijing_v4 import load_beijing_archive
    from airproof.beijing_seasonal import evaluate_seasonal_fold, select_seasonal_fold
    stage, dataset, item, protocol = job
    archive = load_beijing_archive(dataset)
    return (select_seasonal_fold(archive, item, protocol) if stage == "selection"
            else evaluate_seasonal_fold(archive, item, protocol))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/transfer/beijing_seasonal_v4.yaml"))
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--resource-core-pid", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("worker count outside local allocation")
    import numpy as np
    import psutil
    import yaml
    dependency = None
    if args.resource_core_pid:
        process = psutil.Process(args.resource_core_pid)
        command = " ".join(process.cmdline())
        if "run_v4_core_campaign.py" not in command or "validation" not in command:
            raise ValueError("resource dependency is not the existing core validation")
        dependency = {"pid": process.pid, "created": process.create_time()}
    root, output = Path(__file__).resolve().parents[1], args.output_dir.resolve()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
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
    shutil.copy2(root / "scripts/analyze_beijing_v4.py", output / "analyzer.py")
    os.environ["AIRPROOF_BEIJING_SEASONAL_SOURCE"] = str(snapshot)
    activate()
    from airproof.beijing_v4 import load_beijing_archive
    from airproof.beijing_seasonal import seasonal_spans
    from airproof.data import sha256_file
    from airproof.experiment import source_tree_digest
    dataset = (root / protocol["dataset"]).resolve()
    archive = load_beijing_archive(dataset)
    spans = seasonal_spans(archive, protocol)

    def time_window(span):
        return {"start_inclusive": span.start, "stop_exclusive": span.stop,
                "first_local": archive.times[span.start].isoformat(),
                "last_local": archive.times[span.stop - 1].isoformat()}

    manifest = {"role": "locked-seasonal-calibration-fresh-contiguous-transfer",
        "protocol": protocol, "source_hash": source_tree_digest(), "dataset_hash": sha256_file(dataset),
        "runner_hash": sha256_file(Path(__file__)), "analyzer_hash": sha256_file(output / "analyzer.py"),
        "stations": archive.stations, "workers_max": args.workers,
        "resource_dependency": dependency, "created_unix": time.time(),
        "time_windows": {"test": time_window(spans["test"]),
            "deployment_fit": time_window(spans["deployment_fit"]),
            "seasons": [{key: time_window(span) for key, span in item.items()} for item in spans["seasons"]]},
        "all_selection_precedes_any_test": True,
        "earlier_result_preserved": protocol["preserved_diagnostic"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del archive

    def capacity(active_count):
        available = psutil.virtual_memory().available / 1024**2
        limit = args.workers
        if dependency:
            try:
                process = psutil.Process(dependency["pid"])
                if process.is_running() and process.create_time() == dependency["created"]:
                    limit = 1
            except psutil.NoSuchProcess:
                pass
        return min(limit, active_count + max(0, int((available - 1800) / 700)))

    start = time.perf_counter()
    print(json.dumps({"event": "registered", "test_hours": protocol["test_steps"],
                      "test_evaluated": False, "resource_dependency": dependency}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def collect(stage, items, filename):
            index, active, results = 0, {}, []
            with (output / filename).open("w", encoding="utf-8") as handle:
                while index < len(items) or active:
                    target = capacity(len(active))
                    while index < len(items) and len(active) < target:
                        future = pool.submit(run_job, (stage, str(dataset), items[index], protocol))
                        active[future] = index
                        index += 1
                    if not active:
                        raise RuntimeError("insufficient memory for one seasonal fold")
                    ready, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
                    for future in ready:
                        active.pop(future)
                        result = future.result()
                        if stage == "test":
                            result, arrays = result
                            np.savez_compressed(output / (result["station"] + ".npz"), **arrays)
                        results.append(result)
                        handle.write(json.dumps(result, allow_nan=False) + "\n")
                        handle.flush()
                        print(json.dumps({"event": stage + "_fold_complete", "completed": len(results),
                                          "total": len(items), "active_workers": len(active)}), flush=True)
            return results

        selections = collect("selection", list(range(12)), "selections.jsonl")
        selections.sort(key=lambda row: row["outer_target_index"])
        if [row["outer_target_index"] for row in selections] != list(range(12)):
            raise ValueError("all selections required before accessing the test")
        lock = {"selections": selections, "source_hash": manifest["source_hash"], "protocol": protocol,
                "locked_unix": time.time(), "test_rows_evaluated_before_lock": 0}
        canonical = json.dumps(lock, sort_keys=True, allow_nan=False)
        (output / "selection_lock.json").write_text(canonical, encoding="utf-8")
        print(json.dumps({"event": "all_selections_locked", "test_evaluated": False}), flush=True)
        results = collect("test", selections, "results.jsonl")
    (output / "completion.json").write_text(json.dumps({"selections": len(selections),
        "test_folds": len(results), "selection_lock_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - start, "finished_unix": time.time()}, indent=2), encoding="utf-8")
    analyzed = subprocess.run([sys.executable, str(output / "analyzer.py"), str(output)], check=False)
    if analyzed.returncode:
        raise RuntimeError("completed seasonal results require analysis failure review")


if __name__ == "__main__":
    main()
