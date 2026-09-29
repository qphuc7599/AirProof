"""Deterministic synthetic development comparison; never primary evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airproof.records import Observation
from airproof.scheduler import select_evidence
from airproof.v5_fairness import select_minimax_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/v5/development/fairness")
    parser.add_argument("--epochs", type=int, default=168)
    args = parser.parse_args()
    if args.epochs < 24:
        parser.error("at least 24 epochs required for rolling service")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed = 560901
    rng = random.Random(seed)
    targets = dict.fromkeys(range(13), 30)
    rows = []
    timings = defaultdict(list)
    violations = []
    for epoch in range(args.epochs):
        pool = []
        for g in targets:
            n = 0 if (epoch + g) % 29 == 0 else rng.randint(5 + g, 45 + g)
            for i in range(n):
                pool.append(Observation(g * 100 + i, epoch, g, g, 10., rng.uniform(.5, 3),
                                        rng.uniform(.1, 1), rng.choice([128, 256, 512, 1024]),
                                        f"{epoch}-{g}-{i}", epoch, epoch))
        for budget in (0, 20000, 83200, 400000):
            for method in ("v5_minimax", "v4_fallback", "utility_per_byte"):
                start = time.perf_counter()
                if method == "v4_fallback":
                    result = select_evidence(pool, budget_bytes=budget, reserve_fraction=1., targets=targets,
                                             fairness=True, constraint_mode="hard_if_feasible", allocation_epoch=epoch)
                else:
                    result = select_minimax_evidence(pool, budget_bytes=budget, targets=targets,
                                                     allocation_epoch=epoch, fairness=method == "v5_minimax")
                    violations.extend(result.violations)
                timings[method].append(time.perf_counter() - start)
                available = {g: min(30, sum(x.group == g for x in pool)) for g in targets}
                rows.append(dict(epoch=epoch, budget=budget, method=method, spent_bytes=result.spent_bytes,
                                 counts=result.counts, available=available,
                                 max_avoidable=max(max(0, available[g] - result.counts[g]) / 30 for g in targets),
                                 max_total=max(result.deficits.values()), globally_feasible=result.globally_feasible))
    summaries = []
    for budget in (0, 20000, 83200, 400000):
        for method in timings:
            subset = [r for r in rows if r["budget"] == budget and r["method"] == method]
            longest = 0
            rolling = []
            for g in targets:
                run = 0
                for r in subset:
                    run = run + 1 if r["available"][g] > 0 and r["counts"][g] == 0 else 0
                    longest = max(longest, run)
                for t in range(23, len(subset)):
                    rolling.append(sum(min(30, r["counts"][g]) for r in subset[t-23:t+1]) / (24 * 30))
            summaries.append(dict(budget=budget, method=method,
                                  mean_max_avoidable=statistics.mean(r["max_avoidable"] for r in subset),
                                  mean_max_total=statistics.mean(r["max_total"] for r in subset),
                                  mean_spent=statistics.mean(r["spent_bytes"] for r in subset),
                                  feasible_fraction=statistics.mean(r["globally_feasible"] for r in subset),
                                  longest_available_zero_service_run=longest,
                                  worst_24h_capped_target_service=min(rolling)))
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                     ["airproof/v5_fairness.py", "airproof/scheduler.py", "scripts/benchmark_v5_fairness.py"]}
    analysis = dict(stage="development", seed=seed, epochs=args.epochs, targets=targets,
                    source_sha256=source_hashes, violations=violations, summaries=summaries,
                    mean_runtime_seconds={k: statistics.mean(v) for k, v in timings.items()},
                    caveat="Synthetic representative allocation only; 13 allocation groups, not 39 outcome strata. No air-quality or independent confirmation claim.")
    (out / "outcomes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
