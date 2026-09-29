"""Finite exhaustive component audit; no H6/H7 or confirmation claim."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airproof.records import Observation
from airproof.scheduler import select_evidence
from airproof.v6_fairness import CumulativeServiceState, causal_representatives, select_cumulative_evidence


def score(selected, pool, targets):
    return max((Fraction(max(0, min(q, sum(x.group == g for x in pool))
                                   - sum(x.group == g for x in selected)), q)
                for g, q in targets.items()), default=Fraction(0))


def oracle(pool, targets, budget):
    return min(score(subset, pool, targets)
               for mask in product((False, True), repeat=len(pool))
               if sum(x.size_bytes for x, yes in zip(pool, mask) if yes) <= budget
               for subset in [[x for x, yes in zip(pool, mask) if yes]])


def record(user, group, cost, *, quality=1., nullifier=None):
    return Observation(user, 0, group, group, 1., 1., quality, cost,
                       str(user) if nullifier is None else nullifier, None, None)


def adverse_counterexamples():
    cheap = record(0, 0, 1, quality=0., nullifier='cheap')
    costly = record(0, 0, 3, quality=1., nullifier='costly')
    raw_pool = [cheap, costly]
    representatives = causal_representatives(raw_pool, 0)
    reduced = select_cumulative_evidence(raw_pool, budget_bytes=1,
        targets={0: 1}, allocation_epoch=0)
    assert [x.nullifier for x in representatives] == ['costly']
    assert score([cheap], raw_pool, {0: 1}) == 0
    assert Fraction(reduced.optimum_numerator, reduced.optimum_denominator) == 1

    state = CumulativeServiceState()
    repeated_pool = [record(0, 0, 1), record(1, 0, 2)]
    selections = []
    for epoch in range(4):
        result = state.allocate(repeated_pool, budget_bytes=2,
            targets={0: 1}, allocation_epoch=epoch)
        selections.append([x.user_id for x in result.selected])
    assert selections == [[0], [0], [0], [0]]
    assert state.distinct_debt({0: 2}) == {0: 1}
    return {
        'representative_reduction': {
            'raw_costs': {'cheap': 1, 'higher_utility': 3},
            'budget_bytes': 1,
            'raw_pool_floor_feasible': True,
            'chosen_representative': 'costly',
            'representative_pool_optimum_avoidable_deficit': 1.0,
            'conclusion': 'The theorem does not imply raw-pool optimality: utility reduction can discard a cheaper feasible alternative.',
        },
        'cumulative_preference': {
            'costs_by_user': {'0': 1, '1': 2},
            'budget_bytes': 2,
            'target': 1,
            'selected_users_by_epoch': selections,
            'final_distinct_debt': {'0': 1},
            'conclusion': 'Equal-cost certificate preference cannot prevent repetition when the cheapest served user is required by the byte budget.',
        },
    }


def audit():
    comparisons = {m: dict(cases=0, worse_than_oracle=0, maximum_gap=0.)
                   for m in ('v6', 'utility_only', 'v4_fallback')}
    history_cases = allocation_instances = zero_budget_instances = 0
    for layout in product((0, 1), repeat=3):
        for costs in product((0, 1, 3), repeat=3):
            pool = [record(i, group, cost) for i, (group, cost) in enumerate(zip(layout, costs))]
            for qs in product((1, 2), repeat=3):
                targets = dict(enumerate(qs))  # group 2 is deliberately absent
                for budget in range(sum(costs) + 2):
                    allocation_instances += 1
                    zero_budget_instances += int(budget == 0)
                    expected = oracle(pool, targets, budget)
                    for served in (frozenset(), frozenset({(0, 0), (2, 1)})):
                        result = select_cumulative_evidence(pool, budget_bytes=budget,
                            targets=targets, allocation_epoch=0, served_users=served)
                        assert Fraction(result.optimum_numerator, result.optimum_denominator) == expected
                        assert score(result.selected, pool, targets) == expected
                        assert not result.violations and result.spent_bytes <= budget
                        assert result.certificate_bytes <= result.spent_bytes
                        # Independently reconstruct the prefix lower bound.
                        assert result.certificate_bytes == sum(sum(sorted(x.size_bytes for x in pool
                            if x.group == g)[:k]) for g, k in result.certificate_counts.items())
                        for g, q in targets.items():
                            total = Fraction(max(0, q-result.counts[g]), q)
                            opportunity = Fraction(q-result.available_targets[g], q)
                            allocation = Fraction(max(0, result.available_targets[g]-result.counts[g]), q)
                            assert total == opportunity + allocation
                            assert abs(result.deficits[g]-float(total)) < 1e-12
                            assert abs(result.unavoidable_deficits[g]-float(opportunity)) < 1e-12
                            assert abs(result.avoidable_deficits[g]-float(allocation)) < 1e-12
                        history_cases += 1
                    arms = {'v6': result,
                        'utility_only': select_cumulative_evidence(pool, budget_bytes=budget,
                            targets=targets, allocation_epoch=0, fairness=False)}
                    # Historical fallback divides by cost; retain that unsupported domain explicitly.
                    if all(costs):
                        arms['v4_fallback'] = select_evidence(pool, budget_bytes=budget,
                            targets=targets, reserve_fraction=1., constraint_mode='hard_if_feasible', allocation_epoch=0)
                    for name, arm in arms.items():
                        gap = score(arm.selected, pool, targets)-expected
                        assert gap >= 0
                        row = comparisons[name]
                        row['cases'] += 1
                        row['worse_than_oracle'] += int(gap > 0)
                        row['maximum_gap'] = max(row['maximum_gap'], float(gap))
    return dict(allocation_instances=allocation_instances,
        served_history_cases=history_cases, zero_budget_instances=zero_budget_instances,
        violations=0, comparisons=comparisons,
        scope='All group layouts in {0,1}^3, costs in {0,1,3}^3, targets in {1,2}^3 for groups 0,1,2, budgets 0..sum(costs)+1, and two served histories. Group 2 is always absent. Every feasible subset is enumerated.',
        exclusions='v4 comparison excludes zero-cost pools because historical fallback divides by cost. This is a finite component check, not actual-arrival or H6/H7 evidence.')


if __name__ == '__main__':
    result = audit()
    result['source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ('airproof/v6_fairness.py', 'airproof/scheduler.py', 'scripts/verify_v6_fairness.py', 'tests/test_v6_fairness.py')}
    path = Path('reports/v6/fairness_closure/exhaustive.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    counterexamples = adverse_counterexamples()
    counterexamples['source_sha256'] = result['source_sha256']
    counterexample_path = path.parent/'counterexamples.json'
    counterexample_path.write_text(
        json.dumps(counterexamples, indent=2)+'\n', encoding='utf-8')
    document = Path('docs/V6_FAIRNESS_CLOSURE.md')
    manifest = {
        'role': 'finite_component_proof_and_exhaustive_audit_not_H6_H7_confirmation',
        'configured_scientific_target_q': 30,
        'theorem_scope': 'fixed causal representative partition; additive nonnegative integer byte costs; positive integer targets; nonnegative budget',
        'files_sha256': {str(p).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (document, path, counterexample_path)},
    }
    (path.parent/'manifest.json').write_text(
        json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
