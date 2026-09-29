"""Full native-resolution replay of three observed contact datasets, five policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

DATASETS = {
    "InVS13": ("workplace_InVS_tij.dat.zip", "bf818f7e819864e134c90ed82f2cda6ca9d9548d43fe4d05002ac068c91258c0"),
    "InVS15": ("workplace_InVS15_tij.dat.gz", "339eb090d506d750f38a2c1ac6aca76c00c55b462f9dea325933d8b7bc260871"),
    "Hypertext09": ("ht2009_contact_list.dat.gz", "43014e65bb8f6aa7d75d36a8ab61279b617bee88197e04a330c4386305ae1738"),
}


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260903)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lower95": float(np.quantile(samples, .025)),
            "upper95": float(np.quantile(samples, .975)), "paired_role_workload_replications": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("reports/v4_validation/dtn_deadline_7900_7907/selection.json"))
    parser.add_argument("--seeds", default=",".join(map(str, range(8050, 8055))))
    parser.add_argument("--datasets", default="Hypertext09,InVS13,InVS15")
    parser.add_argument("--probabilities", default=".002,.01")
    args = parser.parse_args()
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
    sys.path.insert(0, str(snapshot))
    from airproof.contact_replay import build_contact_replay, load_contacts
    from airproof.dtn import DTNConfig, simulate_dtn_policy
    from airproof.experiment import source_tree_digest
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    base = DTNConfig(**selection["selected_config"])
    seeds = list(map(int, args.seeds.split(",")))
    datasets = args.datasets.split(",")
    probabilities = list(map(float, args.probabilities.split(",")))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(name not in DATASETS for name in datasets):
        raise ValueError("unique paired seeds and known primary-source datasets required")
    if any(not 0 < probability <= 1 for probability in probabilities):
        raise ValueError("invalid workload probability")
    policies = ("ttl_one_copy", "prophet_gtmx", "binary_spray_wait", "epidemic_cap", "airproof_deadline")
    manifest = {"role": "empirical-contact-replay-with-synthetic-roles-and-traffic", "source_hash": source_tree_digest(),
                "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                "datasets": datasets, "seeds": seeds, "probabilities_per_native_slot": probabilities,
                "policies": policies, "ttl_seconds": 21600, "warmup_fraction": .2,
                "collector_fraction": .05, "groups": base.groups, "base_resources": asdict(base),
                "parameter_selection": "relay weights frozen on synthetic validation, no tuning on these empirical replay outcomes",
                "registered_descriptive_comparator": "binary_spray_wait",
                "statistical_scope": "three physical contact collections, two at the same workplace; five conditional role/traffic replications are not independent real traces or cities",
                "traffic_units": "native 20-second slots; workload probabilities .002 and .01 imply mean report intervals 2.778 and .556 hours per observed source",
                "acknowledgments": "ideal common delivery acknowledgment and copy bookkeeping, equal across all five policies",
                "total_jobs": len(datasets) * len(seeds) * len(probabilities) * len(policies),
                "created_unix": time.time()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started, rows, traces = time.perf_counter(), [], {}
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for name in datasets:
            filename, digest = DATASETS[name]
            dataset = load_contacts(root / "data/external/contacts/sociopatterns" / filename,
                                    name=name, expected_sha256=digest)
            for probability in probabilities:
                for seed in seeds:
                    trace, config, metadata = build_contact_replay(dataset, seed=seed, base=base,
                                                                   message_probability=probability)
                    key = f"{name}:{probability}:{seed}"
                    metadata["trace_sha256"] = hashlib.sha256(json.dumps(asdict(trace), sort_keys=True).encode()).hexdigest()
                    traces[key] = metadata
                    (output / "trace_metadata.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
                    for policy in policies:
                        row = simulate_dtn_policy(trace, config, policy)
                        row.update(dataset=name, probability=probability,
                                   restricted_mean_delay_seconds=row["restricted_mean_delivery_delay"] * dataset.step_seconds,
                                   restricted_mean_aoi_seconds=row["restricted_mean_aoi"] * dataset.step_seconds)
                        if row["maximum_realized_copies"] > config.copy_budget:
                            raise AssertionError("copy-budget violation")
                        rows.append(row)
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
                        print(json.dumps({"event": "policy_complete", "dataset": name, "probability": probability,
                                          "seed": seed, "policy": policy, "rows": len(rows),
                                          "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    cells = {}
    fields = ("deadline_delivery_ratio", "restricted_mean_delay_seconds", "restricted_mean_aoi_seconds",
              "group_delivery_gap", "transmissions_per_delivered")
    for name in datasets:
        for probability in probabilities:
            indexed = {(row["seed"], row["policy"]): row for row in rows
                       if row["dataset"] == name and row["probability"] == probability}
            def vector(policy, field):
                return np.array([indexed[seed, policy][field] for seed in seeds])
            ap, control = "airproof_deadline", "binary_spray_wait"
            cells[f"{name}:{probability}"] = {
                "means": {p: {field: float(vector(p, field).mean()) for field in fields} for p in policies},
                "paired_vs_registered_spray": {
                    "delivery_difference_pp": bootstrap(100 * (vector(ap, "deadline_delivery_ratio") - vector(control, "deadline_delivery_ratio"))),
                    "relative_delay_reduction": bootstrap(1 - vector(ap, "restricted_mean_delay_seconds") / vector(control, "restricted_mean_delay_seconds")),
                    "group_gap_reduction_pp": bootstrap(100 * (vector(control, "group_delivery_gap") - vector(ap, "group_delivery_gap"))),
                },
            }
    summary = {"role": manifest["role"], "jobs": len(rows), "cells": cells,
               "statistical_scope": manifest["statistical_scope"],
               "intervals": "descriptive paired bootstrap conditional on each observed trace, not confirmatory population-level inference",
               "elapsed_seconds": time.perf_counter() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "complete", "jobs": len(rows), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
