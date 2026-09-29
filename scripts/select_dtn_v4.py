"""Validation-only selection of deadline/deficit relay weights on paired traces."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, range(7900, 7908))))
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
    from airproof.dtn import DTNConfig, generate_dtn_trace, simulate_dtn_policy
    from airproof.experiment import source_tree_digest
    seeds = list(map(int, args.seeds.split(",")))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique validation seeds required")
    base = DTNConfig()
    comparators = ("direct", "ttl_one_copy", "prophet_gtmx", "binary_spray_wait", "epidemic_cap", "airproof_deficit")
    candidates = [replace(base, deadline_deficit_weight=deficit, deadline_slack_weight=slack)
                  for deficit, slack in itertools.product((1., 3., 6.), (.25, 1., 2.))]
    manifest = {"role": "validation-selection-only", "seeds": seeds, "base": asdict(base),
                "comparators": comparators, "candidates": [asdict(cfg) for cfg in candidates],
                "source_hash": source_tree_digest(), "created_unix": time.time(),
                "selection_rule": "eligible if mean delivery >= strongest comparator minus .02 and mean gap <= strongest mean-delay comparator; among eligible minimize mean restricted delay, then gap; if none choose maximum delivery then minimum delay and label gate failure",
                "prospective_confirmatory_seeds": list(range(7950, 7980)),
                "same_trace_for_every_candidate_and_comparator": True,
                "total_jobs": len(seeds) * (len(comparators) + len(candidates))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, started = [], time.perf_counter()
    trace_hashes = {}
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for seed in seeds:
            trace = generate_dtn_trace(base, seed)
            trace_hashes[str(seed)] = hashlib.sha256(json.dumps(asdict(trace), sort_keys=True).encode()).hexdigest()
            jobs = [(policy, base, policy) for policy in comparators]
            jobs += [("airproof_deadline", cfg, f"deadline_{index}") for index, cfg in enumerate(candidates)]
            for policy, cfg, label in jobs:
                result = simulate_dtn_policy(trace, cfg, policy)
                result["candidate"] = label
                rows.append(result)
                handle.write(json.dumps(result, allow_nan=False) + "\n")
                handle.flush()
            print(json.dumps({"event": "world_complete", "seed": seed, "rows": len(rows),
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    metrics = ("deadline_delivery_ratio", "restricted_mean_delivery_delay", "group_delivery_gap", "transmissions_per_delivered")
    labels = [row["candidate"] for row in rows if row["seed"] == seeds[0]]
    means = {label: {metric: float(np.mean([r[metric] for r in rows if r["candidate"] == label]))
                     for metric in metrics} for label in labels}
    best_delivery = max(means[p]["deadline_delivery_ratio"] for p in comparators)
    best_delay_policy = min(comparators, key=lambda p: means[p]["restricted_mean_delivery_delay"])
    eligible = [label for label in labels if label.startswith("deadline_")
                and means[label]["deadline_delivery_ratio"] >= best_delivery - .02
                and means[label]["group_delivery_gap"] <= means[best_delay_policy]["group_delivery_gap"]]
    if eligible:
        selected = min(eligible, key=lambda label: (means[label]["restricted_mean_delivery_delay"],
                                                   means[label]["group_delivery_gap"]))
    else:
        selected = min((label for label in labels if label.startswith("deadline_")),
                       key=lambda label: (-means[label]["deadline_delivery_ratio"],
                                          means[label]["restricted_mean_delivery_delay"]))
    summary = {"role": "validation-selection-only", "rows": len(rows), "means": means,
               "selected": selected, "selected_config": asdict(candidates[int(selected.split("_")[-1])]),
               "selection_gate_pass": bool(eligible), "eligible": eligible,
               "best_delay_comparator": best_delay_policy, "trace_sha256": trace_hashes,
               "delivery_deficit_percentage_points": 100 * (best_delivery - means[selected]["deadline_delivery_ratio"]),
               "elapsed_seconds": time.perf_counter() - started}
    (output / "selection.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "complete", "selected": selected, "gate": bool(eligible),
                      "means": means[selected], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
