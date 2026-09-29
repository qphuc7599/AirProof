"""Descriptive scale extension of the retained primary; no estimator retuning."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, sys, time, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / 'retained_primary'
sys.path.insert(0, str(PRIMARY / 'source_snapshot'))
import numpy as np
import psutil
from airproof.config import load_config, with_overrides, config_hash
from airproof.simulator import generate_world
import airproof.experiment as experiment
import airproof.twin as twin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, required=True)
    parser.add_argument('--side', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    out = ROOT / 'reports/submission_checks_20260927/scalability'
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'n{args.agents}_g{args.side}_s{args.seed}.json'
    if target.exists():
        raise RuntimeError('Refusing to overwrite completed measurement')
    cfg = with_overrides(load_config(PRIMARY / 'base_configuration.yaml'), {
        'world.agents': args.agents, 'world.grid_side': args.side,
        'attack.kind': 'clean', 'execution.trace_python_allocations': False,
        'privacy.reserved_budget_bytes_per_epoch': 256 * args.agents})
    cg_counts, allocations = [], []
    original_cg, original_select = twin.cg, experiment.select_evidence
    def counted_cg(*a, **kw):
        count = 0
        prior = kw.get('callback')
        def callback(x):
            nonlocal count
            count += 1
            if prior is not None: prior(x)
        kw['callback'] = callback
        result = original_cg(*a, **kw)
        cg_counts.append(count)
        return result
    def timed_select(*a, **kw):
        start = time.perf_counter()
        result = original_select(*a, **kw)
        allocations.append(time.perf_counter() - start)
        return result
    twin.cg, experiment.select_evidence = counted_cg, timed_select
    process = psutil.Process()
    peak = [process.memory_info().rss]
    done = threading.Event()
    def monitor():
        while not done.wait(.01): peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    world = generate_world(cfg, args.seed)
    generated = time.perf_counter()
    result = experiment.run_method(world, cfg, 'airproof_v4')
    elapsed = time.perf_counter() - started
    done.set(); thread.join()
    result['scale_measurement'] = {
        'agents': args.agents, 'grid_side': args.side, 'epochs': cfg['world']['steps'],
        'world_generation_seconds': generated-started, 'end_to_end_seconds': elapsed,
        'end_to_end_peak_rss_mb': peak[0]/1024**2,
        'cg_iterations_mean': float(np.mean(cg_counts)),
        'cg_iterations_p95': float(np.quantile(cg_counts,.95)), 'cg_solves': len(cg_counts),
        'allocation_seconds_total': sum(allocations),
        'allocation_seconds_p95': float(np.quantile(allocations,.95)),
        'config_hash': config_hash(cfg), 'role': 'descriptive scalability, retained estimator',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['scale_measurement']), flush=True)

if __name__ == '__main__': main()
