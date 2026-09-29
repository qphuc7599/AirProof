"""Recheck retained numerical evidence without rerunning scientific outcomes."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airproof.v7_lifetime_inexact import inexact_replacement_bounds


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'reports/submission_checks_20260927'
    cert_path = ROOT / 'reports/v7/reviewer_revision/lifetime_retained_diagnostic/inexact_solver_certificate.json'
    cert = json.loads(cert_path.read_text())
    registration = ROOT / cert['provenance']['registration']
    assert sha(registration) == cert['provenance']['registration_sha256']
    reg = json.loads(registration.read_text())
    start, stop = reg['evaluation_clock']['prediction_half_open']
    assert stop - start == 630
    manifests = cert['solver_outcome_audit']['solver_manifest_sha256']
    predictions = cert['solver_outcome_audit']['prediction_sha256']
    assert len(manifests) == len(predictions) == 60
    residuals = []
    for name, expected in manifests.items():
        path = ROOT / name
        assert sha(path) == expected, name
        manifest = json.loads(path.read_text())
        assert manifest['solver_failure_rate'] == 0
        residuals.append(manifest['projected_gradient_inf'])
    for name, expected in predictions.items():
        path = ROOT / name
        assert sha(path) == expected, name
        with np.load(path) as data:
            assert data['live'].shape == data['reconstructed'].shape == (630, 1024)
    exact = cert['exact_minimizer_certificate']
    smooth = cert['smoothness_certificate']
    old = cert['inexact_solver_certificate']
    recalculated = inexact_replacement_bounds(
        strong_convexity_mu=smooth['strong_convexity_mu'],
        gradient_lipschitz_upper=smooth['gradient_lipschitz_upper'],
        projected_gradient_step=old['projected_gradient_step'],
        projected_gradient_inf=max(residuals),
        maximum_dimension=old['maximum_window_dimension'],
        solve_count=stop-start,
        recurrence_rho=exact['normalized_boundary_contraction'],
        exact_uniform_bound=exact['uniform_active_window_l2_bound'],
        exact_summed_bound=exact['sum_active_window_l2_bound'])
    for key in recalculated:
        assert np.isclose(recalculated[key], old[key], rtol=1e-13, atol=1e-13), key
    deltas = []
    for job in sorted((ROOT/'reports/v8/m5_release_reference_stress_confirmation_v1/jobs').iterdir()):
        result = json.loads((job/'result.json').read_text())
        path = job/'scoring_outputs.npz'
        assert sha(path) == result['scoring_outputs_sha256']
        with np.load(path) as data:
            truth = data['truth'][48:672]
            values = {}
            for arm in ('PROTECTED_POOL', 'RAW_NONNEGATIVE_DIRECT'):
                values[arm] = float(np.sqrt(np.mean((data['stress_'+arm][48:672]-truth)**2)))
            deltas.append(values['PROTECTED_POOL']-values['RAW_NONNEGATIVE_DIRECT'])
    release_reg = json.loads((ROOT/'configs/v8/m5_release_reference_stress_confirmation_v1.json').read_text())
    evaluation = release_reg['evaluation']
    rng = np.random.default_rng(evaluation['bootstrap_seed']+2)
    indices = rng.integers(0,len(deltas),size=(evaluation['bootstrap_replicates'],len(deltas)))
    upper = float(np.quantile(np.asarray(deltas)[indices].mean(axis=1),.95))
    assert round(upper,3) == -14.647
    report = {'T1': {'paired_worlds': len(deltas), 'protected_minus_raw_upper_95': upper},
        'T2': {'prediction_interval': [start,stop], 'N':stop-start,
        'verified_manifest_hashes':len(manifests), 'verified_prediction_hashes':len(predictions),
        'recalculated':recalculated, 'certificate_sha256':sha(cert_path),
        'scope':'Rehash saved outputs and recompute bound; does not regenerate source-locked inputs. Current transport source differs from historical source lock, which is preserved unchanged.'}}
    (output/'retained_evidence_recheck.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
