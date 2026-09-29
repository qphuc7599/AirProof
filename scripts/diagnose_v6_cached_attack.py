"""Descriptive error decomposition from frozen predictions; no fitting or reruns."""
from pathlib import Path
import json
import numpy as np


def main():
    root = Path('reports/v6/development_v1')
    rows = []
    for seed in range(6201000, 6201008):
        for cell in ('severe_clean', 'severe_drift', 'severe_hotspot'):
            for method in ('HUBER_reg0.1_cap8', 'SQ_reg0.1'):
                path = root / f'{seed}_{cell}' / (method+'.npz')
                with np.load(path) as z:
                    truth, public, estimate = z['truth'], z['public'], z['reconstructed']
                    error = public-truth
                    correction = estimate-public
                    public_mse = float(np.mean(error**2))
                    cross = float(2*np.mean(error*correction))
                    energy = float(np.mean(correction**2))
                    mse = float(np.mean((estimate-truth)**2))
                    assert np.isclose(mse, public_mse+cross+energy, atol=1e-12)
                    rows.append({'seed': seed, 'cell': cell, 'method': method,
                        'artifact': str(path), 'mse': mse, 'public_mse': public_mse,
                        'public_error_correction_cross_term': cross, 'correction_energy': energy,
                        'signed_error_mean': float(np.mean(estimate-truth)),
                        'correction_abs_mean': float(np.mean(np.abs(correction))),
                        'cap_touch_fraction': float(np.mean(np.abs(correction) >= 8-1e-6)),
                        'scale_quantiles': np.quantile(z['scales'], [.05, .5, .95]).tolist()})
    index = {(r['seed'], r['cell'], r['method']): r for r in rows}
    decompositions = []
    fields = ('mse', 'public_mse', 'public_error_correction_cross_term', 'correction_energy')
    for attack in ('severe_drift', 'severe_hotspot'):
        for method in ('HUBER_reg0.1_cap8', 'SQ_reg0.1'):
            deltas = {field: [index[s, attack, method][field]-index[s, 'severe_clean', method][field]
                              for s in range(6201000, 6201008)] for field in fields}
            decompositions.append({'attack': attack, 'method': method,
                'mean_paired_changes': {k: float(np.mean(v)) for k, v in deltas.items()},
                'world_changes': deltas})
    output = {'role': 'post hoc descriptive diagnosis of exposed development predictions',
        'scope': 'reg0.1 only; reconstructed clock; MSE decomposition does not replace RMSE gate',
        'identity': 'MSE(public+correction)=MSE(public)+2 E[public_error*correction]+E[correction^2]',
        'causal_limit': 'Cross-term and energy are algebraic attribution, not identification of attack causality or new validation.',
        'rows': rows, 'decompositions': decompositions}
    (root/'cached_attack_decomposition.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(decompositions, indent=2))


if __name__ == '__main__':
    main()
