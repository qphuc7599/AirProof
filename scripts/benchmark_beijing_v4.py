"""Isolated nested Beijing selection; all twelve folds locked before test access."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    source = os.environ["AIRPROOF_BEIJING_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


def progress(event):
    print(json.dumps(event), flush=True)


def select_job(job):
    activate()
    from airproof.beijing_v4 import load_beijing_archive, select_outer_fold
    dataset, index, protocol = job
    return select_outer_fold(load_beijing_archive(dataset), index, protocol, progress=progress)


def test_job(job):
    activate()
    from airproof.beijing_v4 import evaluate_outer_fold, load_beijing_archive
    dataset, selection, protocol = job
    return evaluate_outer_fold(load_beijing_archive(dataset), selection, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/transfer/beijing_public_weather_v4.yaml"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("worker count outside local envelope")
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
    os.environ["AIRPROOF_BEIJING_SOURCE"] = str(snapshot)
    activate()
    import numpy as np
    import yaml
    from airproof.beijing_v4 import archive_slices, load_beijing_archive
    from airproof.data import sha256_file
    from airproof.experiment import source_tree_digest
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    dataset = (root / protocol["dataset"]).resolve()
    archive = load_beijing_archive(dataset)
    spans = archive_slices(len(archive.times), protocol)
    manifest = {"role": "nested-archived-public-station-transfer", "protocol": protocol,
        "source_hash": source_tree_digest(), "dataset_hash": sha256_file(dataset),
        "runner_hash": sha256_file(Path(__file__)), "workers": args.workers,
        "source_doi": protocol["source_doi"], "stations": archive.stations,
        "time_windows": {key: {"start_inclusive": span.start, "stop_exclusive": span.stop,
            "first_local": archive.times[span.start].isoformat(),
            "last_local": archive.times[span.stop - 1].isoformat()} for key, span in spans.items()},
        "created_unix": time.time(), "all_selection_precedes_any_test": True,
        "selection_only": args.selection_only}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del archive
    started = time.perf_counter()
    selections = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(select_job, (str(dataset), index, protocol)): index for index in range(12)}
        with (output / "selections.jsonl").open("w", encoding="utf-8") as handle:
            for future in as_completed(futures):
                selection = future.result()
                selections.append(selection)
                handle.write(json.dumps(selection, allow_nan=False) + "\n")
                handle.flush()
                progress({"stage": "outer_selection_complete", "completed": len(selections), "total": 12,
                    "station": selection["outer_target"], "elapsed_seconds": round(time.perf_counter() - started, 2)})
        selections.sort(key=lambda row: row["outer_target_index"])
        if [row["outer_target_index"] for row in selections] != list(range(12)):
            raise ValueError("all twelve station selections required before test")
        lock = {"selections": selections, "source_hash": manifest["source_hash"],
                "protocol": protocol, "locked_unix": time.time(), "test_rows_evaluated_before_lock": 0}
        canonical = json.dumps(lock, sort_keys=True, allow_nan=False)
        (output / "selection_lock.json").write_text(canonical, encoding="utf-8")
        if args.selection_only:
            progress({"stage": "selection_complete", "test_evaluated": False})
            return
        results = []
        futures = [pool.submit(test_job, (str(dataset), selection, protocol)) for selection in selections]
        with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
            for future in as_completed(futures):
                result, arrays = future.result()
                np.savez_compressed(output / (result["station"] + ".npz"), **arrays)
                handle.write(json.dumps(result, allow_nan=False) + "\n")
                handle.flush()
                results.append(result)
                progress({"stage": "outer_test_complete", "completed": len(results), "total": 12,
                          "station": result["station"]})
    (output / "completion.json").write_text(json.dumps({"selections": len(selections), "test_folds": len(results),
        "selection_lock_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started, "finished_unix": time.time()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
