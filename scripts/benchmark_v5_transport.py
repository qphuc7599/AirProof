"""Frozen native-contact v5 replay plan; execution is a separate explicit action."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import t as student_t

from airproof.contact_replay import build_contact_replay, load_contacts
from airproof.dtn import DTNConfig
from airproof.records import Observation
from airproof.statistics import holm_adjust
from airproof.v5_transport import TransportTrace, simulate_transport

DATASETS = {
    "InVS13": ("workplace_InVS_tij.dat.zip", "bf818f7e819864e134c90ed82f2cda6ca9d9548d43fe4d05002ac068c91258c0"),
    "InVS15": ("workplace_InVS15_tij.dat.gz", "339eb090d506d750f38a2c1ac6aca76c00c55b462f9dea325933d8b7bc260871"),
    "Hypertext09": ("ht2009_contact_list.dat.gz", "43014e65bb8f6aa7d75d36a8ab61279b617bee88197e04a330c4386305ae1738"),
}
POLICIES = ("direct", "airproof_deadline", "binary_spray_wait", "epidemic_cap")
SOURCES = ("scripts/benchmark_v5_transport.py", "airproof/v5_transport.py", "airproof/contact_replay.py",
           "airproof/dtn.py", "airproof/records.py", "airproof/statistics.py")
RESOURCES = {"capacity_bytes_per_direction": 1024, "raw_packet_bytes": 512,
    "release_packet_bytes": 256, "raw_buffer_bytes": 18432, "release_buffer_bytes": 6144,
    "copy_tokens": 4, "release_gateway_reservation_bytes": 256, "control_budget_bytes": 128}


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def cell_id(dataset, ttl_hours, collector_fraction, probability):
    return f"{dataset}:ttl{ttl_hours}:collectors{collector_fraction:g}:load{probability:g}"


def planned_manifest():
    archives = {}
    for name, (filename, digest) in DATASETS.items():
        path = ROOT / "data/external/contacts/sociopatterns" / filename
        dataset = load_contacts(path, name=name, expected_sha256=digest)
        archives[name] = dict(filename=filename, sha256=digest, native_slot_seconds=dataset.step_seconds,
            steps=dataset.steps, participants=len(dataset.node_ids), canonical_contact_rows=len(dataset.events))
    cells = [dict(cell=cell_id(name, ttl, collectors, load), dataset=name, ttl_hours=ttl,
                  collector_fraction=collectors, probability_per_native_slot=load)
             for name, ttl, collectors, load in itertools.product(DATASETS, (6, 24), (.05, .10), (.002, .01))]
    return dict(role="prespecified-conditional-empirical-transport-diagnostic", status="planned-not-executed",
        created_unix=time.time(), source_sha256=source_hashes(), archives=archives,
        seeds=list(range(917000, 917005)), seed_namespace="transport_confirmation: subset917000..917004 reserved for conditional empirical roles/traffic",
        seed_reuse_scope="already-viewed contact archives; new traffic seeds are not new physical contact collections",
        seed_audit={"before_any_v5_transport_matrix_outcomes": True,
                    "command": "rg exact seed field917000..917004 in reports JSON/JSONL excluding source snapshots",
                    "result": "no previous exact seed-field matches at plan creation"},
        cells=cells, policies=list(POLICIES), comparator={"D1":"epidemic_cap","D2":"binary_spray_wait","D3":"binary_spray_wait"}, groups=4,
        warmup_fraction=.2, resources=RESOURCES, total_jobs=len(cells)*5*len(POLICIES), workers=1,
        ttl_units="hours converted exactly to native20-second slots; all generated deadlines fully observed",
        workload_units="Bernoulli reports per native slot: .002/.01 = mean2.778/.556hours per active source",
        release="enabled for all policies, own256-byte lane and24-packet FIFO; never borrowed by raw",
        extraction="existing load_contacts/build_contact_replay; full cached contacts, prefix-only groups, synthetic collectors",
        analysis={"D1": "delivery_epidemic_cap - .02 - delivery_AP",
                  "D2": "restricted_delay_AP - .85*restricted_delay_spray",
                  "D3": "group_gap_AP - .80*group_gap_spray",
                  "alternative": "mean paired contrast <0", "familywise_alpha": .05,
                  "adjustment": "Holm across D1,D2,D3", "unit": "seed after equal-cell averaging",
                  "support": "all frozen cells/seeds/policies required; no complete-case dropping",
                  "scope": "conditional role/traffic inference only; three archives, two at same workplace; no population transport confirmation"},
        execution_gate="Root must authorize compute slots separately; this manifest does not launch jobs")


def adapt_replay(dataset, *, seed, cell, manifest):
    ttl_seconds = int(cell["ttl_hours"] * 3600)
    trace, base, metadata = build_contact_replay(dataset, seed=seed,
        base=DTNConfig(groups=manifest["groups"]),
        message_probability=cell["probability_per_native_slot"], ttl_seconds=ttl_seconds,
        collector_fraction=cell["collector_fraction"], warmup_fraction=manifest["warmup_fraction"])
    digest = hashlib.sha256(json.dumps(asdict(trace), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    acquisition = base.steps - base.ttl_steps
    converted = TransportTrace(seed, base.nodes, acquisition, base.ttl_steps, trace.node_groups,
                               trace.gateway_contacts, trace.peer_contacts, digest)
    observations = tuple(Observation(m.source, m.created, m.group, m.group, 10., 1., 1., 512,
                                    f"empirical:{dataset.name}:{seed}:{m.identifier}", None, None)
                         for m in trace.messages)
    config = {"transport": {**manifest["resources"], "ttl_epochs": base.ttl_steps}}
    return converted, observations, config, metadata


def delivery_metrics(observations, result, *, groups, ttl_slots, slot_seconds):
    generated = np.zeros(groups, dtype=int)
    delivered = np.zeros(groups, dtype=int)
    delays = []
    restricted = []
    for record in observations:
        generated[record.group] += 1
        arrival = result.raw_arrivals.get(record.nullifier)
        if arrival is not None:
            delay = arrival - record.epoch
            if not 0 <= delay <= ttl_slots:
                raise AssertionError("delivery outside inclusive TTL")
            delivered[record.group] += 1
            delays.append(delay * slot_seconds)
            restricted.append(delay * slot_seconds)
        else:
            restricted.append(ttl_slots * slot_seconds)
    rates = delivered / np.maximum(generated, 1)
    return dict(deadline_delivery_ratio=float(delivered.sum()/generated.sum()) if generated.sum() else None,
                restricted_mean_delay_seconds=float(np.mean(restricted)) if restricted else None,
                delivered_mean_delay_seconds=float(np.mean(delays)) if delays else None,
                group_delivery_gap=float(rates.max()-rates.min()) if np.all(generated > 0) else None,
                generated_by_group=generated.tolist(), delivered_by_group=delivered.tolist(),
                group_delivery_rates=[float(v) if n else None for v, n in zip(rates, generated)],
                generated=int(generated.sum()), delivered=int(delivered.sum()),
                undelivered_delay_assignment_seconds=ttl_slots*slot_seconds)


def analyze_transport_family(rows, *, seeds, cells, independent_worlds=False):
    """Prespecified one-sided paired family; repeated cells never inflate N."""
    wanted = {"airproof_deadline", "binary_spray_wait", "epidemic_cap"}
    indexed = {}
    for row in rows:
        if row["policy"] not in wanted:
            continue
        key = (row["seed"], row["cell"], row["policy"])
        if key in indexed:
            raise ValueError("duplicate paired transport row")
        indexed[key] = row
    required = set(itertools.product(seeds, cells, wanted))
    if set(indexed) != required or len(seeds) < 2 or len(set(seeds)) != len(seeds) or not cells:
        raise ValueError("complete prespecified paired seed/cell support required")
    contrasts = {name: [] for name in ("D1", "D2", "D3")}
    for seed in seeds:
        per_cell = {name: [] for name in contrasts}
        for cell in cells:
            ap = indexed[seed, cell, "airproof_deadline"]["metrics"]
            comparator = indexed[seed, cell, "binary_spray_wait"]["metrics"]
            epidemic = indexed[seed, cell, "epidemic_cap"]["metrics"]
            if epidemic.get("deadline_delivery_ratio") is None or not np.isfinite(epidemic["deadline_delivery_ratio"]):
                raise ValueError("unsupported epidemic delivery metric")
            fields = ("deadline_delivery_ratio", "restricted_mean_delay_seconds", "group_delivery_gap")
            if any(m.get(field) is None or not np.isfinite(m[field]) for m in (ap, comparator) for field in fields):
                raise ValueError("unsupported/nonfinite paired metric; no posthoc support deletion")
            per_cell["D1"].append(epidemic[fields[0]] - .02 - ap[fields[0]])
            per_cell["D2"].append(ap[fields[1]] - .85*comparator[fields[1]])
            per_cell["D3"].append(ap[fields[2]] - .80*comparator[fields[2]])
        for name in contrasts:
            contrasts[name].append(float(np.mean(per_cell[name])))
    tests = {}
    for name, values in contrasts.items():
        values = np.asarray(values)
        mean = float(values.mean())
        sem = float(values.std(ddof=1)/np.sqrt(len(values)))
        valid = bool(np.isfinite(values).all() and sem > 0)
        pvalue = float(student_t.cdf(mean/sem, len(values)-1)) if valid else None
        tests[name] = dict(valid=valid, mean_contrast=mean, standard_error=sem, one_sided_p=pvalue,
            upper_one_sided95=mean + float(student_t.ppf(.95, len(values)-1))*sem if valid else None,
            upper_bonferroni_family95=mean + float(student_t.ppf(1-.05/3, len(values)-1))*sem if valid else None,
            paired_seed_contrasts=values.tolist())
    family_valid = all(test["valid"] for test in tests.values())
    adjusted = holm_adjust({name: test["one_sided_p"] if test["valid"] else 1. for name, test in tests.items()})
    for name, test in tests.items():
        test["holm_p"] = adjusted[name]
        test["conditional_threshold_met"] = family_valid and adjusted[name] < .05 and test["mean_contrast"] < 0
    return dict(comparator={"D1":"epidemic_cap","D2":"binary_spray_wait","D3":"binary_spray_wait"}, unit="independent_world" if independent_worlds else "conditional_role_traffic_seed",
        scope="paired-world confirmation" if independent_worlds else "descriptive conditional replay; not population confirmation",
        paired_units=len(seeds), repeated_cells_per_unit=len(cells), familywise_alpha=.05, family_valid=family_valid, tests=tests,
        family_pass=bool(independent_worlds and all(test["conditional_threshold_met"] for test in tests.values())))


def execute(manifest_path: Path, *, max_jobs=None, job_index=None):
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("source changed after manifest; preserve plan and register a new version before outcomes")
    output = manifest_path.parent
    result_path = output / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines() if line] if result_path.exists() else []
    completed = {(r["seed"], r["cell"], r["policy"]) for r in rows}
    planned = set(itertools.product(manifest["seeds"], [c["cell"] for c in manifest["cells"]], manifest["policies"]))
    execution_order = [(seed, cell["cell"], policy) for cell in manifest["cells"]
                       for seed in manifest["seeds"] for policy in manifest["policies"]]
    if job_index is not None and not 0 <= job_index < len(execution_order):
        raise ValueError("job index outside frozen matrix")
    selected_job = execution_order[job_index] if job_index is not None else None
    if len(completed) != len(rows) or not completed <= planned:
        raise ValueError("duplicate or out-of-plan result rows")
    executed = 0
    with result_path.open("a", encoding="utf-8") as handle:
        for name, archive in manifest["archives"].items():
            dataset = load_contacts(ROOT / "data/external/contacts/sociopatterns" / archive["filename"],
                                    name=name, expected_sha256=archive["sha256"])
            for cell in (c for c in manifest["cells"] if c["dataset"] == name):
                for seed in manifest["seeds"]:
                    missing = [policy for policy in manifest["policies"] if (seed, cell["cell"], policy) not in completed
                               and (selected_job is None or (seed, cell["cell"], policy) == selected_job)]
                    if not missing:
                        continue
                    trace, observations, config, metadata = adapt_replay(dataset, seed=seed, cell=cell, manifest=manifest)
                    for policy in missing:
                        if max_jobs is not None and executed >= max_jobs:
                            return
                        started = time.perf_counter()
                        result = simulate_transport(trace, observations, config, policy=policy, release=True)
                        row = dict(seed=seed, cell=cell["cell"], dataset=name, policy=policy,
                            trace_hash=trace.trace_hash, config=config,
                            metrics=delivery_metrics(observations, result, groups=manifest["groups"],
                                ttl_slots=config["transport"]["ttl_epochs"], slot_seconds=dataset.step_seconds),
                            release_metrics=delivery_metrics(observations, SimpleNamespace(raw_arrivals={
                                r.nullifier: result.release_arrivals[r.user_id, r.epoch] for r in observations
                                if (r.user_id, r.epoch) in result.release_arrivals}), groups=manifest["groups"],
                                ttl_slots=config["transport"]["ttl_epochs"], slot_seconds=dataset.step_seconds),
                            transport=result.metrics, extraction=metadata, elapsed_seconds=time.perf_counter()-started)
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
                        rows.append(row)
                        executed += 1
                        print(json.dumps(dict(event="transport_job_complete", seed=seed, cell=cell["cell"], policy=policy,
                                              jobs=len(rows), elapsed_seconds=row["elapsed_seconds"])), flush=True)
    if len(rows) == manifest["total_jobs"]:
        analysis = analyze_transport_family(rows, seeds=manifest["seeds"], cells=[c["cell"] for c in manifest["cells"]])
        (output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports/v5/preflight/empirical_transport_v1")
    parser.add_argument("--max-jobs", type=int, help="bounded compute chunk only; analysis still requires complete frozen matrix")
    parser.add_argument("--job-index", type=int, help="compute one frozen job out of order; never changes final required matrix")
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("max-jobs must be positive")
    manifest_path = args.output / "manifest.json"
    if args.action == "plan":
        args.output.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(planned_manifest(), indent=2))
        print(manifest_path)
    elif args.action == "run":
        execute(manifest_path, max_jobs=args.max_jobs, job_index=args.job_index)
    else:
        manifest = json.loads(manifest_path.read_text())
        rows = [json.loads(line) for line in (args.output / "results.jsonl").read_text().splitlines() if line]
        if len(rows) != manifest["total_jobs"]:
            raise ValueError("full frozen matrix required before final analysis")
        result = analyze_transport_family(rows, seeds=manifest["seeds"], cells=[c["cell"] for c in manifest["cells"]])
        (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
