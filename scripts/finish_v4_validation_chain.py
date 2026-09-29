"""Drain the registered live v4 prerequisites, then validate; never start primary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def same_live_process(identity):
    try:
        process = psutil.Process(identity["pid"])
        return process.is_running() and process.create_time() == identity["created"]
    except psutil.NoSuchProcess:
        return False


def identify(pid, expected_script):
    process = psutil.Process(pid)
    if expected_script not in " ".join(process.cmdline()):
        raise ValueError(f"PID {pid} is not the expected {expected_script}")
    return {"pid": pid, "created": process.create_time(), "script": expected_script}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-pid", type=int, required=True)
    parser.add_argument("--privacy-pid", type=int, required=True)
    parser.add_argument("--beijing-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-output", type=Path,
                        default=Path("reports/v4_validation/core_selected_7300_7311"))
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frontier = root / "reports/v4_validation/fairness_frontier_7400_7407"
    privacy = root / "reports/v4_validation/privacy_full_grid_7500_7529"
    beijing = root / "reports/v4_validation/beijing_public_weather_20260903"
    dependencies = {
        "frontier": identify(args.frontier_pid, "scripts/run_v4_frontier.py"),
        "privacy": identify(args.privacy_pid, "scripts/benchmark_privacy_grid_v4.py"),
        "beijing": identify(args.beijing_pid, "scripts/continue_beijing_v4.py"),
    }
    if not 1 <= args.workers <= 12:
        raise ValueError("worker count outside declared local envelope")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    core = args.core_output.resolve()
    if core.exists():
        raise ValueError("core output already exists; inspect instead of restarting")
    (output / "plan.json").write_text(json.dumps({
        "dependencies": dependencies, "created_unix": time.time(),
        "core_output": str(core), "workers": args.workers,
        "scope": "analyze-complete-prerequisites-then-final-config-validation",
        "primary_auto_start": False,
        "fairness_gate": "registered-selection-target-pass-required",
        "wait_policy": "exact-live-process-identity; no restart on observation timeout",
    }, indent=2), encoding="utf-8")

    def event(name, **details):
        value = {"event": name, "time_unix": time.time(), **details}
        with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value, allow_nan=False), flush=True)

    def wait_dependency(name):
        event("waiting", dependency=name, pid=dependencies[name]["pid"])
        while same_live_process(dependencies[name]):
            time.sleep(30)
        event("dependency_process_ended", dependency=name)

    def execute(name, script, *arguments):
        command = [sys.executable, str(root / "scripts" / script), *map(str, arguments)]
        event("stage_started", stage=name)
        with (output / f"{name}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} returned {result.returncode}; inspect its saved log")
        event("stage_complete", stage=name)

    try:
        wait_dependency("frontier")
        execute("frontier_analysis", "analyze_v4_frontier.py", frontier)
        selection = json.loads((frontier / "selection.json").read_text(encoding="utf-8"))
        if not selection["selection_target_pass"]:
            event("review_required", reason="registered-frontier-target-not-met",
                  selected=selection["selected"], primary_started=False)
            return
        wait_dependency("privacy")
        execute("privacy_analysis", "analyze_privacy_grid_v4.py", privacy)
        wait_dependency("beijing")
        execute("beijing_analysis", "analyze_beijing_v4.py", beijing)
        execute("core_validation", "run_v4_core_campaign.py", "--stage", "validation",
                "--output-dir", core, "--workers", args.workers)
        execute("core_analysis", "analyze_v4_core_campaign.py", core)
        execute("prospective_power", "lock_v4_primary_power.py", core)
        event("ready_for_primary_review", primary_started=False,
              validation=str(core), power_lock=str(core / "power_lock.json"))
    except Exception as error:
        event("failed", error_type=type(error).__name__, error=str(error), primary_started=False)
        raise


if __name__ == "__main__":
    main()
