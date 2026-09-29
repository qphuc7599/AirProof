"""Finish the reviewed v4 campaign after the live seasonal transfer closes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("worker allocation outside local envelope")
    root = Path(__file__).resolve().parents[1]
    validation = root / "reports/v4_validation/core_selected_7300_7311"
    transfer = root / "reports/v4_validation/beijing_seasonal_20260903"
    primary = root / "reports/v4_primary/core_final_7600_7629_20260903"
    publication = root / "source_paper/AirProof_Elsevier/generated/v4_primary"
    power, validated = read(validation / "power_lock.json"), read(validation / "analysis.json")
    if (power["selected_n"] != 30 or not power["all_estimated_powers_at_least_target"]
            or power["primary_worlds_observed"] != 0 or not validated["all_primary_hypotheses_rejected"]):
        raise ValueError("validation or power differs from the reviewed 30-world plan")
    if power["source_hash"] != validated["source_hash"] or power["validation_results_sha256"] != validated["results_sha256"]:
        raise ValueError("validation/power source mismatch")
    if primary.exists():
        raise ValueError("primary output exists; inspect its owner instead of restarting")
    process = psutil.Process(args.transfer_pid)
    if "benchmark_beijing_seasonal_v4.py" not in " ".join(process.cmdline()):
        raise ValueError("unexpected transfer dependency")
    identity = {"pid": process.pid, "created": process.create_time()}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"dependency": identity, "transfer": str(transfer), "primary": str(primary),
        "validation": str(validation), "workers": args.workers, "worlds": 30, "jobs": 360,
        "source_hash": power["source_hash"], "power_lock_sha256": hashlib.sha256((validation / "power_lock.json").read_bytes()).hexdigest(),
        "transfer_gate": "complete-12-fold-locked-test-and-all-reported-upper-bounds-below-1.10",
        "numerical_source_rule": "primary-uses-exact-core-validation-snapshot-not-current-working-tree",
        "created_unix": time.time(), "publication": str(publication),
        "after_completion": "export-measured-tables-and-figures; final-manuscript-integration-and-visual-QA-still-required"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def event(name, **details):
        value = {"event": name, "time_unix": time.time(), **details}
        with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value, allow_nan=False), flush=True)

    def live():
        try:
            current = psutil.Process(identity["pid"])
            return current.is_running() and current.create_time() == identity["created"]
        except psutil.NoSuchProcess:
            return False

    def execute(stage, script, *arguments):
        event("stage_started", stage=stage)
        with (output / (stage + ".log")).open("x", encoding="utf-8") as handle:
            result = subprocess.run([sys.executable, str(root / "scripts" / script), *map(str, arguments)],
                cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{stage} returned {result.returncode}; inspect the saved log")
        event("stage_complete", stage=stage)

    try:
        event("waiting_for_verified_transfer", pid=identity["pid"], primary_started=False)
        while live():
            time.sleep(45)
        completion = read(transfer / "completion.json")
        result = read(transfer / "analysis.json")
        manifest = read(transfer / "manifest.json")
        lock = (transfer / "selection_lock.json").read_bytes()
        if (completion["test_folds"] != 12 or completion["selections"] != 12
                or completion["selection_lock_sha256"] != hashlib.sha256(lock).hexdigest()
                or result["source_hash"] != manifest["source_hash"]
                or json.loads(lock)["test_rows_evaluated_before_lock"] != 0):
            raise ValueError("transfer completion/selection mismatch")
        if not result["gates"]["all_reported_upper_bounds_below_1.10"]:
            event("review_required", reason="seasonal-transfer-noninferiority-not-established", primary_started=False)
            return
        if plan["power_lock_sha256"] != hashlib.sha256((validation / "power_lock.json").read_bytes()).hexdigest():
            raise ValueError("prospective power lock changed while waiting")
        event("transfer_gate_passed", upper_gate=1.10)
        execute("primary", "run_v4_core_campaign.py", "--stage", "primary", "--validation-dir", validation,
                "--output-dir", primary, "--workers", args.workers)
        execute("primary_analysis", "analyze_v4_core_campaign.py", primary)
        execute("primary_publication_export", "export_v4_core_paper.py", primary, "--output-dir", publication)
        event("ready_for_final_manuscript", primary=str(primary), publication=str(publication))
    except Exception as error:
        event("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
