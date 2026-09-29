from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .archived import run_beijing_city_adaptive_transfer, run_beijing_station_transfer
from .benchmarks import (
    run_audit_benchmark,
    run_availability_benchmark,
    run_receipt_fault_benchmark,
    write_benchmark,
)
from .config import load_config, with_overrides
from .data import download_beijing, preprocess_beijing, preprocess_epa_bay_area_pm25
from .dtn import POLICIES, DTNConfig, run_dtn_benchmark, write_dtn_benchmark
from .experiment import METHODS, read_results, run_experiment, write_results
from .plots import (
    create_attack_figure,
    create_privacy_figure,
    create_scale_figure,
    create_summary_figure,
)
from .prediction import (
    PREDICTORS,
    run_prediction_benchmark,
    write_prediction_benchmark,
)
from .q1_reporting import build_q1_small_report, write_q1_small_report
from .reporting import build_measured_report, write_measured_report
from .statistics import analyze_results, pilot_power_analysis, write_analysis
from .validation import run_p0_fixtures


def _parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = map(int, value.split(":", 1))
        return list(range(start, stop))
    return [int(item) for item in value.split(",") if item]


def _run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    methods = [item for item in args.methods.split(",") if item]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    results = run_experiment(config, _parse_seeds(args.seeds), methods)
    write_results(results, args.output)
    print(json.dumps({"runs": len(results), "output": str(Path(args.output).resolve())}))


def _suite(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    matrix = load_config(args.matrix)
    base_path = Path(args.matrix).parent / matrix["base_config"]
    base = load_config(base_path)
    factor_names = list(matrix.get("factors", {}))
    factor_values = [matrix["factors"][name] for name in factor_names]
    jobs: list[tuple[dict[str, Any], int, tuple[str, ...]]] = []
    for combination in itertools.product(*factor_values):
        config = with_overrides(base, dict(zip(factor_names, combination)))
        for seed in matrix["seeds"]:
            jobs.append((config, int(seed), tuple(matrix["methods"])))
    if args.workers == 1:
        batches = [_run_suite_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            batches = list(pool.map(_run_suite_job, jobs))
    all_results = [row for batch in batches for row in batch]
    write_results(all_results, args.output)
    print(json.dumps({"runs": len(all_results), "output": str(Path(args.output).resolve())}))


def _run_suite_job(job: tuple[dict[str, Any], int, tuple[str, ...]]) -> list[dict[str, Any]]:
    """Execute one paired seed/config batch; top-level placement keeps it picklable."""
    config, seed, methods = job
    return run_experiment(config, [seed], methods)


def _fixtures(_: argparse.Namespace) -> None:
    report = run_p0_fixtures()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all"]:
        raise SystemExit(1)


def _analyze(args: argparse.Namespace) -> None:
    report = analyze_results(read_results(args.inputs), reference=args.reference)
    write_analysis(report, args.output)
    print(json.dumps({"comparisons": len(report["comparisons"]), "output": str(Path(args.output).resolve())}))


def _data(args: argparse.Namespace) -> None:
    manifest = download_beijing(args.destination)
    output = preprocess_beijing(Path(args.destination) / "extracted", args.output)
    print(json.dumps({"manifest": str(manifest.resolve()), "processed": str(output.resolve())}))


def _data_epa(args: argparse.Namespace) -> None:
    output = preprocess_epa_bay_area_pm25(
        args.archive, args.output, max_stations=args.max_stations
    )
    print(json.dumps({"processed": str(output.resolve())}))


def _plot(args: argparse.Namespace) -> None:
    output = create_summary_figure(args.input, args.output)
    print(json.dumps({"figure": str(output.resolve())}))


def _plot_privacy(args: argparse.Namespace) -> None:
    output = create_privacy_figure(args.input, args.output)
    print(json.dumps({"figure": str(output.resolve())}))


def _plot_attack(args: argparse.Namespace) -> None:
    output = create_attack_figure(args.input, args.output)
    print(json.dumps({"figure": str(output.resolve())}))


def _plot_scale(args: argparse.Namespace) -> None:
    output = create_scale_figure(args.input, args.output, args.x)
    print(json.dumps({"figure": str(output.resolve())}))


def _archived(args: argparse.Namespace) -> None:
    report = run_beijing_station_transfer(
        args.input,
        args.output,
        pollutant=args.pollutant,
        max_test_steps=args.max_test_steps,
    )
    print(json.dumps({"folds": len(report["folds"]), "output": str(Path(args.output).resolve())}))


def _archived_v2(args: argparse.Namespace) -> None:
    report = run_beijing_city_adaptive_transfer(
        args.input,
        args.output,
        pollutant=args.pollutant,
        max_test_steps=args.max_test_steps,
        inner_fit_steps=args.inner_fit_steps,
        inner_validation_steps=args.inner_validation_steps,
        shortlist=args.shortlist,
        selection_only=args.selection_only,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "folds": len(report["folds"]),
                "output": str(Path(args.output).resolve()),
                "runtime_seconds": report["runtime_seconds"],
            }
        )
    )


def _report(args: argparse.Namespace) -> None:
    report = build_measured_report(args.repo)
    write_measured_report(report, args.output)
    print(json.dumps({"runs": report["run_rows"], "output": str(Path(args.output).resolve())}))


def _audit_benchmark(args: argparse.Namespace) -> None:
    sizes = tuple(int(item) for item in args.batch_sizes.split(",") if item)
    rows = run_audit_benchmark(batch_sizes=sizes, replicates=args.replicates)
    output = write_benchmark(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(output.resolve())}))


def _availability_benchmark(args: argparse.Namespace) -> None:
    replicas = tuple(int(item) for item in args.replicas.split(",") if item)
    losses = tuple(float(item) for item in args.loss_probabilities.split(",") if item)
    rows = run_availability_benchmark(
        replicas=replicas,
        loss_probabilities=losses,
        trials=args.trials,
        seed=args.seed,
    )
    output = write_benchmark(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(output.resolve())}))


def _receipt_fault_benchmark(args: argparse.Namespace) -> None:
    rows = run_receipt_fault_benchmark(replicates=args.replicates)
    output = write_benchmark(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(output.resolve())}))


def _dtn_benchmark(args: argparse.Namespace) -> None:
    config = DTNConfig(
        nodes=args.nodes,
        groups=args.groups,
        steps=args.steps,
        message_probability=args.message_probability,
        record_size_bytes=args.record_size_bytes,
        ttl_steps=args.ttl_steps,
        bytes_per_contact=args.bytes_per_contact,
        queue_records=args.queue_records,
        copy_budget=args.copy_budget,
        peer_contact_rate=args.peer_contact_rate,
        regular_gateway_availability=args.regular_gateway_availability,
        underserved_gateway_availability=args.underserved_gateway_availability,
    )
    policies = tuple(item for item in args.policies.split(",") if item)
    report = run_dtn_benchmark(config, _parse_seeds(args.seeds), policies)
    output = write_dtn_benchmark(report, args.output)
    print(json.dumps({"rows": len(report["rows"]), "output": str(output.resolve())}))


def _prediction_benchmark(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    predictors = tuple(item for item in args.predictors.split(",") if item)
    report = run_prediction_benchmark(
        config,
        _parse_seeds(args.seeds),
        predictors,
        workers=args.workers,
    )
    output = write_prediction_benchmark(report, args.output)
    print(json.dumps({"rows": len(report["rows"]), "output": str(output.resolve())}))


def _power(args: argparse.Namespace) -> None:
    report = pilot_power_analysis(
        read_results(args.inputs),
        metric=args.metric,
        reference=args.reference,
        comparator=args.comparator,
        practical_delta=args.practical_delta,
        alpha=args.alpha,
        target_power=args.target_power,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"recommended_worlds": report["recommended_worlds"], "output": str(output.resolve())}))


def _q1_report(args: argparse.Namespace) -> None:
    report = build_q1_small_report(args.repo)
    write_q1_small_report(report, args.output)
    print(json.dumps({"rows": report["simulation_rows"], "output": str(Path(args.output).resolve())}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airproof")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixtures = subparsers.add_parser("fixtures", help="Run mathematical and integrity P0 gates")
    fixtures.set_defaults(handler=_fixtures)
    run = subparsers.add_parser("run", help="Run paired methods on one immutable configuration")
    run.add_argument("--config", required=True)
    run.add_argument("--seeds", default="0:3", help="Comma list or half-open range, e.g. 0:30")
    run.add_argument("--methods", default="airproof,centralized,air_quality_dt,qpta_adaptation,opportunistic_ttl,blockchain_dt")
    run.add_argument("--output", required=True)
    run.set_defaults(handler=_run)
    suite = subparsers.add_parser("suite", help="Expand a pre-registered factor matrix")
    suite.add_argument("--matrix", required=True)
    suite.add_argument("--output", required=True)
    suite.add_argument("--workers", type=int, default=1)
    suite.set_defaults(handler=_suite)
    analyze = subparsers.add_parser("analyze", help="Paired bootstrap/Wilcoxon/Holm analysis")
    analyze.add_argument("inputs", nargs="+")
    analyze.add_argument("--reference", default="airproof")
    analyze.add_argument("--output", required=True)
    analyze.set_defaults(handler=_analyze)
    data = subparsers.add_parser("data", help="Download and normalize the official UCI Beijing dataset")
    data.add_argument("--destination", default="data/raw/beijing")
    data.add_argument("--output", default="data/processed/beijing_hourly.parquet")
    data.set_defaults(handler=_data)
    data_epa = subparsers.add_parser(
        "data-epa", help="Normalize official EPA AQS hourly PM2.5 for Bay Area transfer"
    )
    data_epa.add_argument("--archive", required=True)
    data_epa.add_argument(
        "--output",
        default="data/external/epa/epa_bay_area_pm25_2024.parquet",
    )
    data_epa.add_argument("--max-stations", type=int, default=16)
    data_epa.set_defaults(handler=_data_epa)
    plot = subparsers.add_parser("plot", help="Create a measured summary figure from result CSV")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    plot.set_defaults(handler=_plot)
    privacy_plot = subparsers.add_parser("plot-privacy", help="Create privacy-utility/accountant curves")
    privacy_plot.add_argument("--input", required=True)
    privacy_plot.add_argument("--output", required=True)
    privacy_plot.set_defaults(handler=_plot_privacy)
    attack_plot = subparsers.add_parser("plot-attack", help="Create attack stress curves")
    attack_plot.add_argument("--input", required=True)
    attack_plot.add_argument("--output", required=True)
    attack_plot.set_defaults(handler=_plot_attack)
    scale_plot = subparsers.add_parser("plot-scale", help="Create one-axis scalability curves")
    scale_plot.add_argument("--input", required=True)
    scale_plot.add_argument("--output", required=True)
    scale_plot.add_argument("--x", required=True)
    scale_plot.set_defaults(handler=_plot_scale)
    archived = subparsers.add_parser("archived", help="Run leakage-safe Beijing station-held-out transfer")
    archived.add_argument("--input", default="data/processed/beijing_hourly.parquet")
    archived.add_argument("--output", default="reports/analysis/beijing_transfer.json")
    archived.add_argument("--pollutant", default="PM2.5")
    archived.add_argument("--max-test-steps", type=int, default=336)
    archived.set_defaults(handler=_archived)
    archived_v2 = subparsers.add_parser(
        "archived-v2",
        help="Run nested city-adaptive Beijing transfer on a locked tail test window",
    )
    archived_v2.add_argument("--input", default="data/processed/beijing_hourly.parquet")
    archived_v2.add_argument(
        "--output",
        default="reports/analysis/beijing_city_adaptive_v2.json",
    )
    archived_v2.add_argument("--pollutant", default="PM2.5")
    archived_v2.add_argument("--max-test-steps", type=int, default=336)
    archived_v2.add_argument("--inner-fit-steps", type=int, default=2016)
    archived_v2.add_argument("--inner-validation-steps", type=int, default=336)
    archived_v2.add_argument("--shortlist", type=int, default=6)
    archived_v2.add_argument("--selection-only", action="store_true")
    archived_v2.set_defaults(handler=_archived_v2)
    report = subparsers.add_parser("report", help="Build the immutable measured implementation report")
    report.add_argument("--repo", default=".")
    report.add_argument("--output", default="reports/MEASURED_REPORT")
    report.set_defaults(handler=_report)
    audit = subparsers.add_parser("audit-benchmark", help="Run local E6 Merkle/log benchmark")
    audit.add_argument("--batch-sizes", default="32,128,512,2048")
    audit.add_argument("--replicates", type=int, default=5)
    audit.add_argument("--output", default="reports/raw/e6_audit_small.json")
    audit.set_defaults(handler=_audit_benchmark)
    availability = subparsers.add_parser(
        "availability-benchmark", help="Run small E7 replicated-availability fault study"
    )
    availability.add_argument("--replicas", default="1,2,3")
    availability.add_argument("--loss-probabilities", default="0,0.25,0.5,0.75,1")
    availability.add_argument("--trials", type=int, default=20000)
    availability.add_argument("--seed", type=int, default=20260901)
    availability.add_argument("--output", default="reports/raw/e7_availability_small.json")
    availability.set_defaults(handler=_availability_benchmark)
    receipt_fault = subparsers.add_parser(
        "receipt-fault-benchmark",
        help="Inject post-acceptance omission, proof, replay, fork and revocation faults",
    )
    receipt_fault.add_argument("--replicates", type=int, default=100)
    receipt_fault.add_argument(
        "--output",
        default="reports/raw/e6_receipt_faults.json",
    )
    receipt_fault.set_defaults(handler=_receipt_fault_benchmark)
    dtn = subparsers.add_parser(
        "dtn-benchmark",
        help="Run equal-resource capacity-constrained DTN forwarding baselines",
    )
    dtn.add_argument("--seeds", default="5200:5212")
    dtn.add_argument("--policies", default=",".join(POLICIES))
    dtn.add_argument("--nodes", type=int, default=120)
    dtn.add_argument("--groups", type=int, default=4)
    dtn.add_argument("--steps", type=int, default=168)
    dtn.add_argument("--message-probability", type=float, default=0.50)
    dtn.add_argument("--record-size-bytes", type=int, default=512)
    dtn.add_argument("--ttl-steps", type=int, default=24)
    dtn.add_argument("--bytes-per-contact", type=int, default=1024)
    dtn.add_argument("--queue-records", type=int, default=48)
    dtn.add_argument("--copy-budget", type=int, default=4)
    dtn.add_argument("--peer-contact-rate", type=float, default=0.35)
    dtn.add_argument("--regular-gateway-availability", type=float, default=0.45)
    dtn.add_argument("--underserved-gateway-availability", type=float, default=0.18)
    dtn.add_argument("--output", default="reports/v3_validation/dtn_equal_resource.json")
    dtn.set_defaults(handler=_dtn_benchmark)
    prediction = subparsers.add_parser(
        "prediction-benchmark",
        help="Run isolated persistence, IDW, HistGB, graph and AirProof prediction endpoints",
    )
    prediction.add_argument("--config", required=True)
    prediction.add_argument("--seeds", default="5400:5412")
    prediction.add_argument("--predictors", default=",".join(PREDICTORS))
    prediction.add_argument("--workers", type=int, default=1)
    prediction.add_argument(
        "--output", default="reports/v3_validation/prediction_endpoints.json"
    )
    prediction.set_defaults(handler=_prediction_benchmark)
    power = subparsers.add_parser("power", help="Size primary worlds from validation-only pilot")
    power.add_argument("inputs", nargs="+")
    power.add_argument("--metric", default="worst_group_rmse")
    power.add_argument("--reference", default="airproof")
    power.add_argument("--comparator")
    power.add_argument("--practical-delta", type=float, required=True)
    power.add_argument("--alpha", type=float, default=0.05)
    power.add_argument("--target-power", type=float, default=0.8)
    power.add_argument("--output", default="reports/analysis/pilot_power.json")
    power.set_defaults(handler=_power)
    q1_report = subparsers.add_parser("q1-report", help="Build the Q1 validation small-study report")
    q1_report.add_argument("--repo", default=".")
    q1_report.add_argument("--output", default="reports/Q1_SMALL_REPORT")
    q1_report.set_defaults(handler=_q1_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
