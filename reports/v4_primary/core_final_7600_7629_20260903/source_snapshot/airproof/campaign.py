from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from .archived import run_beijing_station_transfer, run_epa_station_transfer
from .benchmarks import run_audit_benchmark, run_availability_benchmark, write_benchmark
from .config import load_config, with_overrides
from .experiment import (
    read_results,
    run_experiment,
    run_privacy_replay,
    source_tree_digest,
    write_results,
)
from .plots import (
    create_attack_figure,
    create_privacy_figure,
    create_scale_figure,
    create_summary_figure,
)
from .simulator import generate_world
from .statistics import (
    analyze_attack_induced_effects,
    analyze_factorial_results,
    analyze_results,
    write_analysis,
)

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class CampaignJob:
    study: str
    job_id: str
    base_config: str
    seed: int
    methods: tuple[str, ...]
    variants: tuple[dict[str, Any], ...]
    execution: str
    shard_path: str


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(5):
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(temporary, path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.25 * (attempt + 1))


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(5):
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            os.replace(temporary, path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.25 * (attempt + 1))


def _expected_rows(job: CampaignJob) -> int:
    return len(job.methods) * len(job.variants)


def _complete_shard(job: CampaignJob) -> bool:
    path = Path(job.shard_path)
    if not path.exists():
        return False
    try:
        rows = read_results([path])
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return len(rows) == _expected_rows(job) and len({row["run_id"] for row in rows}) == len(rows)


def _run_job(job: CampaignJob) -> dict[str, Any]:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    base = load_config(job.base_config)
    try:
        if job.execution == "simulation":
            if len(job.variants) != 1:
                raise ValueError("simulation jobs require exactly one physical-world variant")
            variant = job.variants[0]
            config = with_overrides(base, variant["overrides"])
            rows = run_experiment(config, [job.seed], job.methods)
            for row in rows:
                row["study"] = job.study
                row["scenario_id"] = variant["id"]
                row["job_id"] = job.job_id
                row["registered_overrides"] = variant["overrides"]
        elif job.execution == "privacy_replay":
            physical_overrides = job.variants[0].get("physical_overrides", {})
            world_config = with_overrides(base, physical_overrides)
            world = generate_world(world_config, job.seed)
            for variant in job.variants:
                config = with_overrides(base, variant["overrides"])
                for method in job.methods:
                    row = run_privacy_replay(world, config, method)
                    row["study"] = job.study
                    row["scenario_id"] = variant["id"]
                    row["job_id"] = job.job_id
                    row["registered_overrides"] = variant["overrides"]
                    rows.append(row)
        else:
            raise ValueError(f"unknown execution mode: {job.execution}")
        _atomic_jsonl(Path(job.shard_path), rows)
        return {
            "status": "complete",
            "study": job.study,
            "job_id": job.job_id,
            "rows": len(rows),
            "elapsed_seconds": time.perf_counter() - started,
            "shard": job.shard_path,
        }
    except Exception as error:  # noqa: BLE001 - worker failures are recorded as outcomes
        failure = {
            "status": "failed",
            "study": job.study,
            "job_id": job.job_id,
            "seed": job.seed,
            "execution": job.execution,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(Path(job.shard_path).with_suffix(".error.json"), failure)
        return failure


def _factor_variants(
    *,
    scenarios: list[dict[str, Any]],
    factors: dict[str, list[Any]],
    base_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    names = list(factors)
    values = [factors[name] for name in names]
    combinations = list(itertools.product(*values)) if names else [()]
    variants: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_overrides = dict(scenario.get("overrides", {}))
        physical_overrides = {
            key: value
            for key, value in scenario_overrides.items()
            if not key.startswith("privacy.")
        }
        for index, combination in enumerate(combinations):
            overrides = dict(base_overrides)
            overrides.update(scenario_overrides)
            overrides.update(dict(zip(names, combination)))
            suffix = "_".join(str(value).replace(".", "p") for value in combination)
            variant_id = scenario["id"] if not suffix else f"{scenario['id']}_{index:03d}_{suffix}"
            variants.append(
                {
                    "id": variant_id,
                    "overrides": overrides,
                    "physical_overrides": physical_overrides,
                }
            )
    return variants


def _registered_e5_jobs(
    base_config: Path, shard_root: Path, spec: dict[str, Any]
) -> list[CampaignJob]:
    jobs: list[CampaignJob] = []
    shared = dict(spec.get("base_overrides", {}))
    for case in spec["cases"]:
        case_id = str(case["id"])
        overrides = dict(shared)
        overrides.update(case.get("overrides", {}))
        seeds = [int(seed) for seed in case["seeds"]]
        for replicate, seed in enumerate(seeds):
            job_id = f"{case_id}_r{replicate}"
            jobs.append(
                CampaignJob(
                    study="e5",
                    job_id=job_id,
                    base_config=str(base_config),
                    seed=seed,
                    methods=tuple(case.get("methods", spec.get("methods", ["airproof"]))),
                    variants=({"id": job_id, "overrides": overrides},),
                    execution="simulation",
                    shard_path=str(shard_root / "e5" / f"{job_id}.jsonl"),
                )
            )
    return jobs


def _e5_jobs(
    base_config: Path, shard_root: Path, spec: dict[str, Any]
) -> list[CampaignJob]:
    if spec.get("cases"):
        return _registered_e5_jobs(base_config, shard_root, spec)
    jobs: list[CampaignJob] = []
    base = {
        "world.steps": 48,
        "world.burn_in_steps": 8,
        "attack.start_epoch": 24,
        "network.ttl_steps": 12,
        "world.participation_skew": 5.0,
        "network.availability": 0.75,
        "network.outage_median_hours": 6,
        "network.outage_p95_hours": 24,
    }

    def add(job_id: str, seed: int, overrides: dict[str, Any]) -> None:
        merged = dict(base)
        merged.update(overrides)
        jobs.append(
            CampaignJob(
                study="e5",
                job_id=job_id,
                base_config=str(base_config),
                seed=seed,
                methods=("airproof",),
                variants=({"id": job_id, "overrides": merged},),
                execution="simulation",
                shard_path=str(shard_root / "e5" / f"{job_id}.jsonl"),
            )
        )

    for agents in (100, 500, 1000, 2500, 5000):
        for replicate in range(3):
            add(
                f"agents_{agents}_r{replicate}",
                12000 + replicate,
                {"world.agents": agents, "world.grid_side": 16},
            )
    for side in (8, 16, 32, 64):
        for replicate in range(3):
            add(
                f"grid_{side}_r{replicate}",
                12100 + replicate,
                {"world.agents": 500, "world.grid_side": side},
            )
    for agents in (500, 1000, 5000):
        for mode in ("fixed", "per_agent"):
            for replicate in range(2):
                budget = 32768 if mode == "fixed" else agents * 128
                add(
                    f"budget_{mode}_{agents}_r{replicate}",
                    12200 + replicate,
                    {
                        "world.agents": agents,
                        "world.grid_side": 16,
                        "scheduler.budget_bytes_per_epoch": budget,
                    },
                )
    add(
        "sentinel_10000_grid64_steps12",
        12300,
        {
            "world.agents": 10000,
            "world.grid_side": 64,
            "world.steps": 12,
            "world.burn_in_steps": 0,
            "attack.start_epoch": 6,
            "network.ttl_steps": 6,
        },
    )
    return jobs


def build_jobs(config_path: Path, output_root: Path) -> dict[str, list[CampaignJob]]:
    campaign = load_config(config_path)
    base_config = (config_path.parent / campaign["base_config"]).resolve()
    shard_root = output_root / "shards"
    result: dict[str, list[CampaignJob]] = {}
    for study, spec in campaign["studies"].items():
        if spec.get("generated_scalability_design"):
            result[study] = _e5_jobs(base_config, shard_root, spec)
            continue
        execution = str(spec["execution"])
        scenarios = list(spec.get("scenarios", [{"id": "anchor", "overrides": {}}]))
        factors = dict(spec.get("factors", {}))
        base_overrides = dict(spec.get("base_overrides", {}))
        methods = tuple(spec["methods"])
        jobs: list[CampaignJob] = []
        if execution == "simulation":
            variants = _factor_variants(
                scenarios=scenarios, factors=factors, base_overrides=base_overrides
            )
            for variant in variants:
                for seed in spec["seeds"]:
                    job_id = f"{variant['id']}_seed{seed}"
                    jobs.append(
                        CampaignJob(
                            study=study,
                            job_id=job_id,
                            base_config=str(base_config),
                            seed=int(seed),
                            methods=methods,
                            variants=(variant,),
                            execution=execution,
                            shard_path=str(shard_root / study / f"{job_id}.jsonl"),
                        )
                    )
        elif execution == "privacy_replay":
            # A job owns one physical world and reuses it for every privacy-only
            # variant. Non-privacy factors must create separate physical worlds.
            physical_names = [name for name in factors if not name.startswith("privacy.")]
            privacy_factors = {
                name: values for name, values in factors.items() if name.startswith("privacy.")
            }
            physical_values = [factors[name] for name in physical_names]
            physical_combinations = (
                list(itertools.product(*physical_values)) if physical_names else [()]
            )
            for scenario in scenarios:
                for physical_index, physical_combination in enumerate(physical_combinations):
                    physical_overrides = dict(scenario.get("overrides", {}))
                    physical_overrides.update(
                        dict(zip(physical_names, physical_combination))
                    )
                    physical_id = scenario["id"]
                    if physical_combination:
                        suffix = "_".join(
                            str(value).replace(".", "p") for value in physical_combination
                        )
                        physical_id = f"{physical_id}_p{physical_index:02d}_{suffix}"
                    variants = _factor_variants(
                        scenarios=[{"id": physical_id, "overrides": physical_overrides}],
                        factors=privacy_factors,
                        base_overrides=base_overrides,
                    )
                    for seed in spec["seeds"]:
                        job_id = f"{physical_id}_seed{seed}"
                        jobs.append(
                            CampaignJob(
                                study=study,
                                job_id=job_id,
                                base_config=str(base_config),
                                seed=int(seed),
                                methods=methods,
                                variants=tuple(variants),
                                execution=execution,
                                shard_path=str(shard_root / study / f"{job_id}.jsonl"),
                            )
                        )
        else:
            raise ValueError(f"unsupported execution mode {execution!r}")
        result[study] = jobs
    return result


def _resource_snapshot(output_root: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    # Query the volume root. Windows can briefly report a deep directory as
    # missing while concurrent workers atomically replace shard files.
    disk_target = Path(output_root.anchor) if output_root.anchor else output_root
    disk = shutil.disk_usage(disk_target)
    return {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
    }


def _write_status(
    output_root: Path,
    *,
    state: str,
    study: str | None,
    total_jobs: int,
    completed_jobs: int,
    failed_jobs: int,
    skipped_jobs: int,
    active_jobs: int,
    started: float,
) -> None:
    payload = {
        "state": state,
        "study": study,
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "active_jobs": active_jobs,
        "elapsed_seconds": time.monotonic() - started,
        "resource": _resource_snapshot(output_root),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(output_root / "campaign_status.json", payload)


def _consolidate(study: str, jobs: list[CampaignJob], output_root: Path) -> Path:
    shard_paths = [job.shard_path for job in jobs if _complete_shard(job)]
    rows = read_results(shard_paths)
    output = output_root / "results" / f"{study}.jsonl"
    write_results(rows, output)
    return output


def run_campaign(
    config_path: Path,
    output_root: Path,
    *,
    requested_workers: int,
    selected_studies: list[str] | None = None,
    include_auxiliary: bool = True,
) -> int:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    jobs_by_study = build_jobs(config_path, output_root)
    order = selected_studies or list(jobs_by_study)
    unknown = sorted(set(order) - set(jobs_by_study))
    if unknown:
        raise ValueError(f"unknown studies: {unknown}")
    workers = max(1, min(int(requested_workers), 12))
    hard_limit = float(config.get("hard_limit_hours", 23)) * 3600
    minimum_free_disk = float(config.get("minimum_free_disk_gb", 5))
    started = time.monotonic()
    campaign_manifest = {
        "campaign_version": config["campaign_version"],
        "campaign_config": str(config_path.resolve()),
        "campaign_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_tree_sha256": source_tree_digest(),
        "requested_workers": requested_workers,
        "effective_worker_ceiling": workers,
        "thread_environment": THREAD_ENV,
        "host": platform.platform(),
        "python": sys.version,
        "studies": order,
        "jobs": {study: len(jobs_by_study[study]) for study in order},
        "expected_rows": {
            study: sum(_expected_rows(job) for job in jobs_by_study[study]) for study in order
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(output_root / "campaign_manifest.json", campaign_manifest)
    total_jobs = sum(len(jobs_by_study[study]) for study in order)
    completed_jobs = failed_jobs = skipped_jobs = 0
    event_log = output_root / "campaign_events.jsonl"

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for study in order:
            jobs = jobs_by_study[study]
            pending = []
            for job in jobs:
                if _complete_shard(job):
                    skipped_jobs += 1
                else:
                    pending.append(job)
            active: dict[Any, CampaignJob] = {}
            last_monitor = 0.0
            while pending or active:
                elapsed = time.monotonic() - started
                snapshot = _resource_snapshot(output_root)
                if elapsed >= hard_limit:
                    for job in pending:
                        _atomic_json(
                            Path(job.shard_path).with_suffix(".error.json"),
                            {
                                "status": "not-started-hard-campaign-timeout",
                                "study": job.study,
                                "job_id": job.job_id,
                            },
                        )
                        failed_jobs += 1
                    pending.clear()
                if snapshot["disk_free_gb"] < minimum_free_disk:
                    raise RuntimeError(
                        f"campaign disk guard fired at {snapshot['disk_free_gb']:.2f} GB free"
                    )
                # Twelve workers are allowed, but paging is not: reduce new submissions
                # when available RAM falls below a safe operating floor.
                if snapshot["memory_percent"] >= 92:
                    current_cap = max(1, workers // 4)
                elif snapshot["memory_percent"] >= 85:
                    current_cap = max(1, workers // 2)
                else:
                    current_cap = workers
                while pending and len(active) < current_cap:
                    job = pending.pop(0)
                    active[executor.submit(_run_job, job)] = job
                done, _ = wait(active, timeout=5.0, return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    result = future.result()
                    if result["status"] == "complete":
                        completed_jobs += 1
                    else:
                        failed_jobs += 1
                    with event_log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, sort_keys=True) + "\n")
                if time.monotonic() - last_monitor >= 15:
                    _write_status(
                        output_root,
                        state="running",
                        study=study,
                        total_jobs=total_jobs,
                        completed_jobs=completed_jobs,
                        failed_jobs=failed_jobs,
                        skipped_jobs=skipped_jobs,
                        active_jobs=len(active),
                        started=started,
                    )
                    last_monitor = time.monotonic()
            _consolidate(study, jobs, output_root)

    auxiliary = output_root / "results"
    e6: Path | None = None
    e7: Path | None = None
    transfer: dict[str, Any] | None = None
    epa_transfer: dict[str, Any] | None = None
    if include_auxiliary:
        auxiliary_config = dict(config.get("auxiliary", {}))
        e6 = write_benchmark(
            run_audit_benchmark(
                batch_sizes=tuple(
                    int(value)
                    for value in auxiliary_config.get(
                        "e6_batch_sizes", (32, 128, 512, 2048)
                    )
                ),
                replicates=int(auxiliary_config.get("e6_replicates", 20)),
            ),
            auxiliary / "e6_audit.json",
        )
        e7 = write_benchmark(
            run_availability_benchmark(
                replicas=tuple(
                    int(value) for value in auxiliary_config.get("e7_replicas", (1, 2, 3))
                ),
                loss_probabilities=tuple(
                    float(value)
                    for value in auxiliary_config.get(
                        "e7_loss_probabilities", (0.0, 0.25, 0.5, 0.75)
                    )
                ),
                trials=int(auxiliary_config.get("e7_trials", 20_000)),
                seed=int(auxiliary_config.get("e7_seed", 20260902)),
            ),
            auxiliary / "e7_availability.json",
        )
        transfer = run_beijing_station_transfer(
            Path(
                auxiliary_config.get(
                    "transfer_input", "data/processed/beijing_hourly.parquet"
                )
            ),
            auxiliary / "beijing_transfer.json",
            max_test_steps=int(auxiliary_config.get("transfer_max_test_steps", 336)),
        )
        epa_input = auxiliary_config.get("epa_transfer_input")
        if epa_input:
            epa_transfer = run_epa_station_transfer(
                Path(epa_input),
                auxiliary / "epa_transfer.json",
                max_test_steps=int(auxiliary_config.get("epa_transfer_max_test_steps", 336)),
            )
    references = {"e1": "airproof", "e3": "factorial_111", "e4": "airproof"}
    for study, reference in references.items():
        result_path = auxiliary / f"{study}.jsonl"
        if result_path.exists():
            study_rows = read_results([result_path])
            report = analyze_results(study_rows, reference=reference)
            write_analysis(report, output_root / "analysis" / f"{study}.json")
            for scenario in sorted(
                {str(row.get("scenario_id", "unspecified")) for row in study_rows}
            ):
                scenario_rows = [
                    row
                    for row in study_rows
                    if str(row.get("scenario_id", "unspecified")) == scenario
                ]
                scenario_report = analyze_results(scenario_rows, reference=reference)
                write_analysis(
                    scenario_report,
                    output_root / "analysis" / f"{study}_{scenario}.json",
                )
    e3_path = auxiliary / "e3.jsonl"
    if e3_path.exists():
        _atomic_json(
            output_root / "analysis" / "e3_factorial.json",
            analyze_factorial_results(read_results([e3_path])),
        )
    e4_path = auxiliary / "e4.jsonl"
    if e4_path.exists():
        _atomic_json(
            output_root / "analysis" / "e4_attack_induced.json",
            analyze_attack_induced_effects(read_results([e4_path])),
        )
    figures = output_root / "figures"
    if (auxiliary / "e1.csv").exists():
        create_summary_figure(auxiliary / "e1.csv", figures / "e1_summary.png")
    if (auxiliary / "e2a.csv").exists():
        create_privacy_figure(auxiliary / "e2a.csv", figures / "e2_privacy.png")
    if (auxiliary / "e4.csv").exists():
        create_attack_figure(auxiliary / "e4.csv", figures / "e4_attack.png")
    if (auxiliary / "e5.csv").exists():
        create_scale_figure(auxiliary / "e5.csv", figures / "e5_agents.png", "agents")
        create_scale_figure(auxiliary / "e5.csv", figures / "e5_grid.png", "grid_side")
    final = {
        **campaign_manifest,
        "state": "complete" if failed_jobs == 0 else "complete-with-failures",
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "skipped_jobs": skipped_jobs,
        "elapsed_seconds": time.monotonic() - started,
        "e6": str(e6.resolve()) if e6 else None,
        "e7": str(e7.resolve()) if e7 else None,
        "transfer_folds": len(transfer["folds"]) if transfer else 0,
        "epa_transfer_folds": len(epa_transfer["folds"]) if epa_transfer else 0,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(output_root / "campaign_final.json", final)
    _write_status(
        output_root,
        state=final["state"],
        study=None,
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        skipped_jobs=skipped_jobs,
        active_jobs=0,
        started=started,
    )
    return 0 if failed_jobs == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the resumable AirProof-24H local campaign")
    parser.add_argument("--config", default="configs/airproof24h/campaign.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--studies", help="Comma-separated subset for calibration/recovery")
    parser.add_argument("--list", action="store_true", help="Print registered job/row counts")
    parser.add_argument("--skip-auxiliary", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    studies = [item for item in (args.studies or "").split(",") if item] or None
    if args.list:
        jobs = build_jobs(config_path, output)
        payload = {
            study: {
                "jobs": len(items),
                "rows": sum(_expected_rows(item) for item in items),
            }
            for study, items in jobs.items()
            if studies is None or study in studies
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    raise SystemExit(
        run_campaign(
            config_path,
            output,
            requested_workers=args.workers,
            selected_studies=studies,
            include_auxiliary=not args.skip_auxiliary,
        )
    )


if __name__ == "__main__":
    main()
