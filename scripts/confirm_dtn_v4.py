"""Confirm the locked DTN selection on its independently reserved 30 traces."""
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
from scipy import stats


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260903)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lower95": float(np.quantile(samples, .025)),
            "upper95": float(np.quantile(samples, .975)), "paired_traces": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection_path = args.selection_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validation = json.loads((args.selection_dir / "manifest.json").read_text(encoding="utf-8"))
    if not selection["selection_gate_pass"]:
        raise ValueError("validation selection did not satisfy its registered constraints")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Run precisely the selected implementation, not an evolving working tree.
    source = (args.selection_dir / "source_snapshot").resolve()
    sys.path.insert(0, str(source))
    from airproof.dtn import DTNConfig, generate_dtn_trace, simulate_dtn_policy
    from airproof.experiment import source_tree_digest
    if source_tree_digest() != validation["source_hash"]:
        raise ValueError("selected implementation no longer matches its manifest")
    config = DTNConfig(**selection["selected_config"])
    seeds = validation["prospective_confirmatory_seeds"]
    if set(seeds) & set(validation["seeds"]) or len(set(seeds)) != 30:
        raise ValueError("30 independent reserved confirmation seeds required")
    policies = tuple(validation["comparators"]) + ("airproof_deadline",)
    external_baselines = [p for p in policies if not p.startswith("airproof_")]
    delivery_baseline = max(external_baselines, key=lambda p: selection["means"][p]["deadline_delivery_ratio"])
    delay_baseline = min(external_baselines, key=lambda p: selection["means"][p]["restricted_mean_delivery_delay"])
    hypotheses = [
        {"id": "D1", "comparator": delivery_baseline, "contrast": "AP delivery - comparator delivery + .02", "alternative": ">0"},
        {"id": "D2", "comparator": delay_baseline, "contrast": ".85 comparator restricted delay - AP restricted delay", "alternative": ">0"},
        {"id": "D3", "comparator": delay_baseline, "contrast": ".8 comparator group gap - AP group gap", "alternative": ">0"},
    ]
    manifest = {"role": "independent-confirmatory-DTN-secondary-family", "config": asdict(config),
                "source_snapshot": str(source), "source_hash": source_tree_digest(), "seeds": seeds,
                "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                "policies": policies, "hypotheses": hypotheses,
                "multiplicity": "Holm FWER .05 within the three-endpoint transport family; not a global all-paper FWER assertion",
                "comparator_selection": "strongest external mean delivery/delay on validation only",
                "created_unix": time.time(), "total_jobs": len(seeds) * len(policies)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(__file__, output / "runner.py")
    started, rows, trace_hashes = time.perf_counter(), [], {}
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for seed in seeds:
            trace = generate_dtn_trace(config, seed)
            trace_hashes[str(seed)] = hashlib.sha256(json.dumps(asdict(trace), sort_keys=True).encode()).hexdigest()
            for policy in policies:
                result = simulate_dtn_policy(trace, config, policy)
                rows.append(result)
                handle.write(json.dumps(result, allow_nan=False) + "\n")
                handle.flush()
            print(json.dumps({"event": "world_complete", "seed": seed, "rows": len(rows),
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    indexed = {(row["seed"], row["policy"]): row for row in rows}
    def metric(policy, name):
        return np.array([indexed[seed, policy][name] for seed in seeds])
    ap = "airproof_deadline"
    hypothesis_values = [metric(ap, "deadline_delivery_ratio") - metric(delivery_baseline, "deadline_delivery_ratio") + .02,
                         .85 * metric(delay_baseline, "restricted_mean_delivery_delay") - metric(ap, "restricted_mean_delivery_delay"),
                         .8 * metric(delay_baseline, "group_delivery_gap") - metric(ap, "group_delivery_gap")]
    tests = [{**description, "effect": bootstrap(values),
              "p_raw": float(stats.ttest_1samp(values, 0, alternative="greater").pvalue)}
             for description, values in zip(hypotheses, hypothesis_values, strict=True)]
    running = 0.
    for rank, test in enumerate(sorted(tests, key=lambda value: value["p_raw"])):
        running = max(running, min(1., (len(tests) - rank) * test["p_raw"]))
        test["p_holm"] = running
        test["reject_at_05"] = running < .05
    comparisons = {}
    for policy in policies:
        if policy == ap:
            continue
        comparisons[policy] = {
            "delivery_difference_pp": bootstrap(100 * (metric(ap, "deadline_delivery_ratio") - metric(policy, "deadline_delivery_ratio"))),
            "relative_delay_reduction": bootstrap(1 - metric(ap, "restricted_mean_delivery_delay") / metric(policy, "restricted_mean_delivery_delay")),
            "relative_gap_reduction": bootstrap(1 - metric(ap, "group_delivery_gap") / np.maximum(metric(policy, "group_delivery_gap"), 1e-12)),
        }
    fields = ("deadline_delivery_ratio", "restricted_mean_delivery_delay", "group_delivery_gap", "transmissions_per_delivered")
    summary = {"role": manifest["role"], "rows": len(rows), "traces": len(seeds),
               "means": {p: {name: float(metric(p, name).mean()) for name in fields} for p in policies},
               "paired_comparisons": comparisons, "hypothesis_tests": tests,
               "all_transport_gates_pass_holm": all(t["reject_at_05"] for t in tests),
               "trace_sha256": trace_hashes, "elapsed_seconds": time.perf_counter() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "complete", "gates": summary["all_transport_gates_pass_holm"],
                      "means": summary["means"][ap], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
