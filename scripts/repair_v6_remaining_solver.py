"""Repair only recorded failed arms; never replace frozen development results."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json
import hashlib
import time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path('reports/v6/development_v1')
OUT = Path('reports/v6/remaining_solver_repair')


def repair(item):
    from airproof.simulator import generate_world
    from airproof.v6_mobility import coupled_public_trace
    from airproof.v6_experiment import integrated_transport
    from airproof.v6_numerical_experiment import prepare_public
    from airproof.v6_estimator import EstimatorConfig, estimate_public_field
    from airproof.meteorology import grid_coordinates
    from airproof.field import grid_laplacian
    job, methods = item
    started = time.perf_counter()
    old = ROOT / job
    historical = json.loads((old / 'result.json').read_text())
    cfg = historical['config']
    world = generate_world(cfg, historical['seed'])
    trace, _ = coupled_public_trace(cfg, historical['seed'], world.observations)
    transport, collector = integrated_transport(trace, world.observations, cfg)
    assert transport.metrics['trace_hash'] == historical['transport']['trace_hash']
    public, ops, scales, _ = prepare_public(world, cfg)
    burn = cfg['world']['burn_in_steps']
    horizon = len(world.truth) - burn
    records = [replace(r, epoch=r.epoch-burn) for r in collector.selected if r.epoch >= burn]
    arrivals = {r.nullifier: collector.selection_times[r.nullifier]-burn for r in records}
    target = OUT / job
    target.mkdir(exist_ok=False)
    rows = []
    for method in methods:
        robust = method.startswith('HUBER')
        config = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05, lambda_spatial=.02,
            loss='huber' if robust else 'quadratic', output_cap=robust, numerical_refinements=2)
        result = estimate_public_field(public[burn:], grid_coordinates(cfg['world']['grid_side']),
            records, config, innovation_scales=scales[burn:],
            transitions=None if ops is None else ops[burn:],
            spatial_laplacian=grid_laplacian(cfg['world']['grid_side']), arrival_map=arrivals)
        live, reconstructed = result.live[:horizon], result.reconstructed[:horizon]
        with np.load(old / (method+'.npz')) as cached:
            deltas = {name: float(np.max(np.abs(value-cached[name])))
                      for name, value in [('live', live), ('reconstructed', reconstructed)]}
        np.savez_compressed(target / (method+'.npz'), live=live, reconstructed=reconstructed)
        rows.append({'method': method, 'max_prediction_changes': deltas,
            'rmse': float(np.sqrt(np.mean((reconstructed-world.truth[burn:])**2))),
            'solver_failure_rate': result.diagnostics['solver_failure_rate'],
            'refined_epochs': [t for t in result.diagnostics['epoch_terms'] if len(t['refinement_history']) > 1]})
    outcome = {'job': job, 'rows': rows, 'elapsed_seconds': time.perf_counter()-started}
    (target/'result.json').write_text(json.dumps(outcome, indent=2))
    return outcome


def main():
    failures = json.loads((ROOT/'numerical_failure_audit.json').read_text())['failed_arms']
    done = json.loads(Path('reports/v6/targeted_solver_repair/result.json').read_text())
    completed = {('6201001_severe_hotspot', r['method']) for r in done['rows']}
    jobs = {}
    for row in failures:
        if (row['job'], row['method']) not in completed:
            assert row['method'] in ('HUBER_reg0.1_cap8', 'SQ_reg0.1')
            jobs.setdefault(row['job'], []).append(row['method'])
    OUT.mkdir(exist_ok=False)
    paths = [Path(__file__), *Path('airproof').glob('*.py')]
    registration = {'role': 'exposed development numerical repair only', 'jobs': jobs,
        'original_contribution': 'robust digital twin numerical correctness',
        'reuse': 'all nonfailed arms and two previously repaired arms retained',
        'change': 'two same-objective restarts only; no seed, cap, objective or KKT threshold change',
        'exit_criterion': 'report every failure and prediction delta; scientific gates remain unchanged',
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'input_hashes': {str(ROOT/job/'result.json'): hashlib.sha256((ROOT/job/'result.json').read_bytes()).hexdigest() for job in jobs}}
    (OUT/'registration.json').write_text(json.dumps(registration, indent=2))
    outcomes = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for outcome in pool.map(repair, jobs.items()):
            outcomes.append(outcome)
            print(json.dumps({'job': outcome['job'], 'failures': [r['solver_failure_rate'] for r in outcome['rows']]}), flush=True)
    (OUT/'result.json').write_text(json.dumps({'outcomes': outcomes, 'confirmation': False}, indent=2))


if __name__ == '__main__':
    main()
