"""Same-clock six-control B3 evaluation; fitting and scoring are separate APIs."""
from dataclasses import dataclass, replace
import numpy as np
from .continual_release import FixedReleasePlan
from .v5_release_twin import ReleaseOnlyTwin as V5Twin
from .v6_release_twin import (ReleaseOnlyTwin, PublicCalibratedTwin,
    PublicReleaseObservation, fit_sparse_release_calibration)


@dataclass(frozen=True)
class ReleaseControlsCalibration:
    model: object
    oracle_coefficients: np.ndarray
    calibration_id: str
    groups: int


def _schedule(steps, groups):
    schedule=FixedReleasePlan(steps, groups, 20, 8/28, 0., 8.).scheduled_epochs
    if len(schedule)!=28:
        raise ValueError('v6 release controls require exactly 28 public query epochs')
    return schedule


def _validate(public, query, values, mask):
    p, q, n = [np.asarray(x,float) for x in (public, query, values)]
    m = np.asarray(mask)
    if p.ndim != 2 or p.shape != q.shape or p.shape != n.shape or m.shape != p.shape or m.dtype != bool:
        raise ValueError('matched time/group arrays and boolean release mask required')
    scheduled = np.zeros(len(p),bool); scheduled[list(_schedule(len(p),p.shape[1]))] = True
    if not np.isfinite(p).all() or not np.isfinite(q[scheduled]).all() or not np.isnan(q[~scheduled]).all():
        raise ValueError('actual noiseless query must follow exact FixedReleasePlan schedule')
    if np.any(m[~scheduled]) or not np.isfinite(n[m]).all() or not np.isnan(n[~m]).all():
        raise ValueError('noisy values must follow exact protected release mask')
    return p,q,n,m,scheduled


def from_release_evidence(outputs, *, kind='residual'):
    """Use v5_experiment.release_evidence outputs without rebuilding admission.

    Returned kwargs exclude its offline truth array. Truth is passed separately
    only to calibration or score_controls, never to run_release_controls.
    """
    if kind not in ('residual','raw'): raise ValueError('unknown query kind')
    return dict(public=outputs['baseline'], actual_query=outputs[kind+'_query'],
                noisy_values=outputs[kind], released_mask=outputs[kind+'_mask'])


def _features(public, noisy, mask, mean):
    # Causal newest published query per group; zero innovation before publication.
    g = public.shape[1]; latest = np.zeros(g); available = np.zeros(g)
    rows=[]
    for epoch in range(len(public)):
        acquisition=epoch-24
        if acquisition >= 0:
            take=mask[acquisition]
            latest[take]=noisy[acquisition,take]-public[acquisition,take]
            available[take]=1.
        rows.append(np.r_[1., public[epoch]+mean[:g], latest, available])
    return np.asarray(rows)


def fit_release_controls(*, public, actual_query, noisy_values, released_mask,
                         calibration_truth, calibration_id, ridge=.1):
    """Independent calibration only, using actual admitted/scheduled queries.

    P3 is a linear oracle-moment diagnostic fitted to independent truth labels.
    Noise is already present in its input features, never added a second time.
    """
    p,q,n,m,scheduled=_validate(public,actual_query,noisy_values,released_mask)
    t=np.asarray(calibration_truth,float)
    if not calibration_id or t.shape != p.shape or not np.isfinite(t).all():
        raise ValueError('independent calibration identity and finite matching truth required')
    if not np.isfinite(ridge) or ridge <= 0: raise ValueError('positive ridge required')
    model=fit_sparse_release_calibration(t,p,q,scheduled,ridge=ridge)
    x=_features(p,n,m,model.mean)
    coefficients=np.linalg.solve(x.T@x/len(x)+ridge*np.eye(x.shape[1]),x.T@t/len(x))
    return ReleaseControlsCalibration(model,coefficients,str(calibration_id),p.shape[1])


def run_release_controls(calibration, *, public, actual_query, noisy_values,
                         released_mask, evaluation_id, laplace_scale=7., public_drain=None, v5_calibration=None):
    """Return predictions; no test truth or exact contributor count argument.

    P1 is isolated diagnostic raw-query access. Only public/noisy protected
    observations enter P0/P3/P4/P5. P1/P2 use identical delayed query carry model.
    Caller must supply causal public forecasts for optional 24-hour drain.
    """
    if not evaluation_id or evaluation_id == calibration.calibration_id:
        raise ValueError('evaluation must have a different exposure identity')
    if not np.isfinite(laplace_scale) or laplace_scale <= 0: raise ValueError('positive noise scale required')
    p,q,n,m,scheduled=_validate(public,actual_query,noisy_values,released_mask)
    if p.shape[1] != calibration.groups: raise ValueError('group mismatch')
    steps=len(p)
    if public_drain is not None:
        drain=np.asarray(public_drain,float)
        if drain.shape != (24,calibration.groups) or not np.isfinite(drain).all():
            raise ValueError('24 public forecast drain epochs required')
        p=np.concatenate((p,drain))
    predictions={k:[] for k in ('P0','P1','P2','P3','P4','P5')}
    c=calibration.model
    # Matched V5 mechanics with diagonalized dynamics; not historical frozen V5.
    old=v5_calibration if v5_calibration is not None else replace(c,transition=np.diag(np.diag(c.transition)))
    if old.groups != calibration.groups: raise ValueError('V5 calibration group mismatch')
    twins={'P0':PublicCalibratedTwin(c),'P4':V5Twin(old),'P5':ReleaseOnlyTwin(c)}
    last_q=np.zeros(calibration.groups); last_n=last_q.copy(); seen=np.zeros(calibration.groups,bool)
    for epoch in range(len(p)):
        acquisition=epoch-24; releases=[]
        if 0 <= acquisition < steps and scheduled[acquisition]:
            for group in range(calibration.groups):
                releases.append(PublicReleaseObservation(f'{acquisition}:{group}',acquisition,epoch,group,
                    float(n[acquisition,group]) if m[acquisition,group] else None,laplace_scale))
            take=m[acquisition]
            last_q[take]=q[acquisition,take]-p[acquisition,take]
            last_n[take]=n[acquisition,take]-p[acquisition,take]
            seen |= take
        base=twins['P0'].update(epoch,p[epoch])
        predictions['P0'].append(base)
        predictions['P1'].append(np.maximum(p[epoch]+np.where(seen,last_q,c.mean[:calibration.groups]),0))
        predictions['P2'].append(np.maximum(p[epoch]+np.where(seen,last_n,c.mean[:calibration.groups]),0))
        features=np.r_[1.,p[epoch]+c.mean[:calibration.groups],last_n,seen.astype(float)]
        predictions['P3'].append(np.maximum(features@calibration.oracle_coefficients,0))
        for arm in ('P4','P5'): predictions[arm].append(twins[arm].update(epoch,p[epoch],releases))
    return {'predictions':{k:np.asarray(v) for k,v in predictions.items()},
            'publication_epochs':[int(e+24) for e in np.flatnonzero(scheduled)],
            'metadata':{'calibration_id':calibration.calibration_id,'evaluation_id':evaluation_id,
                        'diagnostic_oracles':['P1','P3'], 'P4_scope':('caller-supplied frozen V5 calibration' if v5_calibration is not None else 'matched sparse calibration; V5 diagonal dynamics adapter, not historical frozen arm'),
                        'P1_P2_scope':'same delayed carry estimator; actual release mask; no clairvoyant query access',
                        'deadline':24,'query_count':len(np.flatnonzero(scheduled)),
                        'noise_scale':laplace_scale,'privacy_mechanism_changed':False,
                        'calibration_provenance_verification':'required from caller; distinct IDs alone do not prove independence'}}


def score_controls(result, evaluation_truth, *, burn=48):
    """Offline identical-support scoring, separate from all consumer updates."""
    t=np.asarray(evaluation_truth,float)
    predictions=result['predictions']
    if t.ndim != 2 or not np.isfinite(t).all() or not 0 <= burn < len(t):
        raise ValueError('finite scored truth and valid burn required')
    scores={}
    for key,p in predictions.items():
        if p.shape[1:] != t.shape[1:] or len(p)<len(t): raise ValueError('prediction support mismatch')
        error=p[burn:len(t)]-t[burn:]
        scores[key]={'rmse':float(np.sqrt(np.mean(error**2))),'scored_group_hours':int(error.size)}
    return scores

