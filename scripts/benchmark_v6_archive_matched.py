"""One registered archive diagnostic; no backbone training or candidate search."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import time
import argparse
import numpy as np
from airproof.predictor_wrapper import synthetic_citizen_replay
from airproof.v6_estimator import EstimatorConfig, estimate_matched_controls


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--public-dynamics',action='store_true');args=parser.parse_args()
    out = Path('reports/v6/archive_public_dynamics_matched' if args.public_dynamics else 'reports/v6/archive_matched_diagnostic')
    out.mkdir(exist_ok=False)
    original = Path('reports/v4_validation/esntnn_purpleair_7800/heldout_predictions.npz')
    center = Path('reports/v6/archive_center_diagnostic/predictions.npz')
    calibration = Path('reports/v6/backbone_validation_recovery/public_validation_predictions.npz')
    cfg = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05, lambda_spatial=.02,
                          lag=1, time_step=4., numerical_refinements=2)
    paths = [Path(__file__), original, center, calibration, Path('airproof/v6_estimator.py'),
             Path('airproof/predictor_wrapper.py'), Path('airproof/v6_solver_refinement.py')]
    selected=None
    if args.public_dynamics:
        lock=Path('reports/v6/public_dynamics_development/selection_lock.json')
        selected=json.loads(lock.read_text())['selected']
        if selected is None:raise ValueError('no validation-selected public model')
        selected_path=lock.parent/(selected+'_test.npz')
        paths.extend([lock,selected_path])
    manifest = {'role': 'exposed archive interface diagnostic; synthetic citizen channel; no independent confirmation',
        'original_pillar': 'robust uncertainty-aware public-referenced digital twin',
        'reuse': 'frozen backbone predictions and existing public-center recalibration; same v5 replay seeds916000/916001',
        'weakness': 'missing actual v6 matched archive control and physical-time lag check',
        'seeds': [916000, 916001], 'kinds': ['clean', 'adversarial_drift', 'negative_bias'],
        'centers': [selected] if selected else ['original', 'recalibrated'], 'selection': 'no wrapper selection; public dynamics selected on its registered validation only' if selected else 'none; one fixed estimator plus mechanistic controls, all reported',
        'config': asdict(cfg), 'source_interval_hours': 4, 'lag_hours_target': 6,
        'lag_discretization': 'only acquisition ages0/4h are admissible;8h exceeds6h; no invented intermediate data',
        'drain_epochs': 6, 'drain_public': 'last available center carried forward; scoring excludes drain',
        'scale': {'value': 3., 'source': 'public synthetic replay measurement-noise design',
                  'interpretation': 'likelihood residual at the latent state, not citizen-minus-public marginal innovation',
                  'private_record_sigma_used': False, 'real_sensor_calibration_established': False},
        'thresholds': {'clean_ratio': 1.05, 'attack_attenuation': .2, 'event_recall_loss': .05},
        'event_threshold': '95th percentile of pretest exposed public selection labels',
        'historical_comparison': 'new within-run SQ only; changed clock/dynamics cannot be pooled with old wrapper outcomes',
        'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/'registration.json').write_text(json.dumps(manifest, indent=2))
    with np.load(original) as z:
        xy = z['coordinates_lat_lon']
    xy = (xy-xy.mean(0))*np.array([111.2, 111.2*np.cos(np.deg2rad(xy[:,0].mean()))])
    with np.load(calibration) as z:
        threshold = float(np.quantile(z['truth'], .95))
    with np.load(center) as z:
        truth = z['truth'].copy()
        centers = {} if selected else {name: z[name].copy() for name in manifest['centers']}
    if selected:
        with np.load(selected_path) as z:centers[selected]=z['prediction'].copy()
    steps = len(truth)
    event = truth >= threshold
    rows = []
    started = time.perf_counter()
    with (out/'results.jsonl').open('w') as stream:
        for seed in manifest['seeds']:
            for kind in manifest['kinds']:
                records = synthetic_citizen_replay(truth, seed=seed, kind=kind)
                for name, public in centers.items():
                    extended = np.concatenate((public, np.repeat(public[-1:], 6, axis=0)))
                    results = estimate_matched_controls(extended, xy, records, cfg, innovation_scales=3.)
                    for method, result in results.items():
                        live, recon = result.live[:steps], result.reconstructed[:steps]
                        row = {'seed': seed, 'kind': kind, 'center': name, 'method': method,
                            'live_rmse': float(np.sqrt(np.mean((live-truth)**2))),
                            'reconstructed_rmse': float(np.sqrt(np.mean((recon-truth)**2))),
                            'event_recall': float(np.mean(live[event]>=threshold)) if event.any() else None,
                            'event_support': int(event.sum()), 'event_threshold': threshold,
                            'diagnostics': {k:v for k,v in result.diagnostics.items() if k!='epoch_terms'}}
                        rows.append(row)
                        np.savez_compressed(out/f'{seed}_{kind}_{name}_{method}.npz', live=live, reconstructed=recon)
                        stream.write(json.dumps(row)+'\n'); stream.flush()
                    print(json.dumps({'seed':seed,'kind':kind,'center':name,'rows':len(rows)}), flush=True)
    summary = {'role': manifest['role'], 'rows':len(rows), 'expected_rows':len(centers)*30,
               'elapsed_seconds':time.perf_counter()-started,'contrasts':[]}
    for name in centers:
        means = {(kind,method): float(np.mean([r['reconstructed_rmse'] for r in rows if r['center']==name and r['kind']==kind and r['method']==method]))
                 for kind in manifest['kinds'] for method in ('bounded_huber','quadratic','public_only')}
        attacks = {}
        for kind in manifest['kinds'][1:]:
            ap=means[kind,'bounded_huber']-means['clean','bounded_huber']
            sq=means[kind,'quadratic']-means['clean','quadratic']
            attacks[kind]={'ap_growth':ap,'sq_growth':sq,'attenuation':1-ap/sq if sq>0 else None,
                           'passes':sq>0 and ap<=.8*sq}
        summary['contrasts'].append({'center':name,'clean_ratio':means['clean','bounded_huber']/means['clean','quadratic'],
            'clean_rmse':{m:means['clean',m] for m in ('bounded_huber','quadratic','public_only')}, 'attacks':attacks,
            'max_solver_failure_rate':max(r['diagnostics'].get('solver_failure_rate',0) for r in rows if r['center']==name)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
