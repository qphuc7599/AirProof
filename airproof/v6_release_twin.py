"""V6 protected-release VAR moment estimator; calibration is offline only."""
from dataclasses import dataclass
import numpy as np
from .v5_release_twin import ReleaseCalibration, PublicReleaseObservation, ReleaseOnlyTwin as _V5Twin, _covariance


def fit_release_calibration(public_truth, public_baselines, noiseless_queries, *,
                            ridge=0.1, covariance_shrinkage=0.1,
                            query_count=28, deadline=24,
                            source="independent-actual-admission-calibration"):
    """Fit hourly latent VAR from independent matched actual-query calibration.

    Missing sparse queries are not imputed: use independent simulator calibration
    exposing the actual stabilized query each hour, or a separately verified model.
    Query-count metadata does not establish sparse calibration transfer by itself.
    """
    arrays = [np.asarray(a, float) for a in (public_truth, public_baselines, noiseless_queries)]
    if arrays[0].ndim == 2:
        arrays = [a[None] for a in arrays]
    t, p, q = arrays
    if t.ndim != 3 or t.shape != p.shape or t.shape != q.shape or t.shape[1] < 4 or t.shape[2] < 1:
        raise ValueError("matched world/time/group arrays with four epochs required")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("finite calibration required")
    if not np.isfinite(ridge) or ridge <= 0 or not 0 <= covariance_shrinkage <= 1:
        raise ValueError("positive ridge and valid shrinkage required")
    if query_count != 28 or deadline != 24 or not source:
        raise ValueError("registered query count28 deadline24 and provenance required")
    latent = np.concatenate((t-p, q-t), axis=-1)
    mean = latent.mean(axis=(0, 1))
    centered = latent-mean
    x = centered[:, :-1].reshape(-1, len(mean))
    y = centered[:, 1:].reshape(-1, len(mean))
    transition = np.linalg.solve(x.T@x/len(x)+ridge*np.eye(len(mean)), x.T@y/len(x)).T
    # Operator-norm contraction gives stability even for nonnormal VAR matrices.
    transition *= min(1., .98/max(np.linalg.norm(transition, 2), 1e-12))
    return ReleaseCalibration(mean, transition, _covariance(y-x@transition.T, covariance_shrinkage),
                              _covariance(centered.reshape(-1, len(mean)), covariance_shrinkage),
                              t.shape[2], source, covariance_shrinkage)


class ReleaseOnlyTwin(_V5Twin):
    """Online API accepts public baselines and protected observations only."""
    def __init__(self, calibration, *, lag=48):
        if lag != 48:
            raise ValueError("v6 registered lag is 48")
        for a in (calibration.initial_covariance, calibration.process_covariance):
            if not np.allclose(a, np.asarray(a).T, rtol=0, atol=1e-10):
                raise ValueError("covariance must be symmetric")
        if np.linalg.norm(calibration.transition, 2) > .98000001:
            raise ValueError("transition must be a stable contraction")
        super().__init__(calibration, lag=lag)
        self.query_slots = set()

    def update(self, epoch, public_baseline, releases=()):
        incoming = list(releases)
        new_slots = set()
        for r in incoming:
            if not isinstance(r, PublicReleaseObservation):
                raise TypeError("protected PublicReleaseObservation required")
            if r.publication_epoch-r.acquisition_epoch != 24:
                raise ValueError("registered release deadline is 24")
            slot = (r.acquisition_epoch, r.group)
            if r.release_id not in self.seen:
                if slot in self.query_slots or (slot in new_slots and sum(x.release_id != r.release_id and (x.acquisition_epoch, x.group) == slot for x in incoming)):
                    raise ValueError("multiple identifiers cannot influence one query slot")
                new_slots.add(slot)
        result = super().update(epoch, public_baseline, incoming)
        self.query_slots.update(new_slots)
        return result

    def diagnostics(self):
        d = super().diagnostics()
        d.update(state="full regularized contractive VAR truth residual/query bias",
                 covariance_shrinkage=self.calibration.covariance_shrinkage,
                 sparse_calibration_transfer_verified=False)
        return d


class PublicCalibratedTwin(ReleaseOnlyTwin):
    """Identical calibration and dynamics, no citizen observations."""
    def update(self, epoch, public_baseline):
        return super().update(epoch, public_baseline)


def information_diagnostics(truth, public_calibrated, noiseless_actual_query, noisy_query):
    """Offline P0/P1/P2 same-support query diagnostic, never a deployable consumer.

    Direct query error does not measure delayed filter gain or establish P3-P5.
    """
    arrays = [np.asarray(a, float) for a in (truth, public_calibrated, noiseless_actual_query, noisy_query)]
    if not arrays[0].size or any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("finite identical nonempty support required")
    t, p, q, n = arrays
    residuals = np.column_stack([(a-t).ravel() for a in (p, q, n)])
    return {"scope": "offline diagnostic only; actual admission and delayed alignment required",
            "support": int(t.size), "p0_public_mse": float(np.mean((p-t)**2)),
            "p1_noiseless_actual_query_mse": float(np.mean((q-t)**2)),
            "p2_noisy_query_mse": float(np.mean((n-t)**2)),
            "error_second_moment": (residuals.T@residuals/len(residuals)).tolist()}

@dataclass(frozen=True)
class SparseReleaseCalibration(ReleaseCalibration):
    scheduled_samples: int = 0
    bias_dynamics_assumption: str = "temporally independent query bias; hourly dynamics unidentified"


def fit_sparse_release_calibration(public_truth, public_baselines, noiseless_queries,
                                   scheduled_mask, *, ridge=.1, covariance_shrinkage=.1):
    """Hourly residual VAR plus bias/cross moments at actual public scheduled slots.

    Mask is [world,time] (or [time]); every group is observed at scheduled slots.
    Query bias is modeled as temporally independent: sparse data does not identify
    a unique hourly bias transition. This restriction is explicit, not interpolation.
    """
    t, p, q = [np.asarray(a, float) for a in (public_truth, public_baselines, noiseless_queries)]
    mask = np.asarray(scheduled_mask)
    if t.ndim == 2:
        t, p, q, mask = t[None], p[None], q[None], mask[None]
    if t.ndim != 3 or t.shape != p.shape or t.shape != q.shape or t.shape[1] < 4 or t.shape[2] < 1:
        raise ValueError("matched world/time/group arrays required")
    if mask.dtype != bool or mask.shape != t.shape[:2] or not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError("boolean scheduled mask and finite public/truth required")
    if not np.isfinite(q[mask]).all() or not np.isnan(q[~mask]).all():
        raise ValueError("queries must be finite exactly on mask, NaN elsewhere")
    if not np.isfinite(ridge) or ridge <= 0 or not 0 <= covariance_shrinkage <= 1:
        raise ValueError("valid regularization required")
    for world_mask in mask:
        slots = np.flatnonzero(world_mask)
        if len(slots) > 28:
            raise ValueError("at most 28 scheduled acquisitions per history")
    if mask.sum() < 4 or mask[:, 1:].sum() < 4:
        raise ValueError("at least four matched scheduled innovation slices required")
    g = t.shape[-1]
    residual = t-p
    truth_mean = residual.mean(axis=(0,1))
    z = residual-truth_mean
    x, y = z[:, :-1].reshape(-1,g), z[:, 1:].reshape(-1,g)
    a = np.linalg.solve(x.T@x/len(x)+ridge*np.eye(g), x.T@y/len(x)).T
    a *= min(1., .98/max(np.linalg.norm(a,2), 1e-12))
    bias_mean = (q[mask]-t[mask]).mean(axis=0)
    bias = q-t-bias_mean
    innovations = z[:,1:]-z[:,:-1]@a.T
    paired = mask[:,1:]
    # Regress bias on matched truth innovations, then assemble PSD covariance
    # using the hourly truth innovation covariance, not unmatched pairwise moments.
    def joint_cov(hourly, matched, b):
        v = _covariance(hourly, covariance_shrinkage)
        mx, mb = matched-matched.mean(axis=0), b-b.mean(axis=0)
        beta = np.linalg.solve(mx.T@mx/len(mx)+ridge*np.eye(g), mx.T@mb/len(mx)).T
        noise = _covariance(mb-mx@beta.T, covariance_shrinkage)
        return np.block([[v, v@beta.T], [beta@v, beta@v@beta.T+noise]])
    process = joint_cov(innovations.reshape(-1,g), innovations[paired], bias[:,1:][paired])
    initial = joint_cov(z.reshape(-1,g), z[mask], bias[mask])
    transition = np.zeros((2*g,2*g)); transition[:g,:g] = a
    return SparseReleaseCalibration(np.concatenate((truth_mean,bias_mean)), transition,
                                    process, initial, g, "independent-sparse-actual-admission-calibration",
                                    covariance_shrinkage, int(mask.sum()))


def scalar_laplace_reference(prior_mean, prior_variance, observed_value, laplace_scale,
                             *, epsabs=1e-11, epsrel=1e-10):
    """Numerical Gaussian-prior × exact Laplace-likelihood posterior moments.

    Scalar reference only; a sequence of Gaussian moment projections is not an
    exact multivariate/non-Gaussian filtering posterior. Returns log evidence and
    quadrature error estimates, not a claim of rigorous floating-point bounds.
    """
    from scipy.integrate import quad
    vals = (prior_mean, prior_variance, observed_value, laplace_scale, epsabs, epsrel)
    if not all(np.isfinite(v) for v in vals) or min(prior_variance, laplace_scale, epsabs, epsrel) <= 0:
        raise ValueError("finite arguments and positive variance/scale/tolerances required")
    sd = np.sqrt(prior_variance)
    d = (observed_value-prior_mean)/sd
    lam = sd/laplace_scale
    mode = np.clip(d, -lam, lam)
    peak = -.5*mode**2-lam*abs(mode-d)
    # Scale coordinates to resolve a narrow cusp for very small noise.
    width = min(1., 1/lam) if mode == d else 1.
    cusp = (d-mode)/width
    def kernel(u, power):
        z = mode+width*u
        return u**power*np.exp(-.5*z*z-lam*abs(z-d)-peak)
    moments, errors = [], []
    for power in range(3):
        cuts = sorted(set([-np.inf, -12., 0., 12., float(cusp), np.inf]))
        pieces = [quad(kernel, lo, hi, args=(power,), epsabs=epsabs, epsrel=epsrel)
                  for lo, hi in zip(cuts[:-1], cuts[1:])]
        moments.append(sum(v for v, e in pieces)); errors.append(sum(e for v, e in pieces))
    norm = moments[0]
    if not np.isfinite(norm) or norm <= 0:
        raise ArithmeticError("quadrature normalization failed")
    um = moments[1]/norm
    variance = prior_variance*width**2*(moments[2]/norm-um**2)
    return {"mean": float(prior_mean+sd*(mode+width*um)),
            "variance": float(max(0.,variance)),
            "log_normalization": float(peak+np.log(width*norm)-.5*np.log(2*np.pi)-np.log(2*laplace_scale)),
            "scaled_normalization": float(norm), "quadrature_absolute_errors": errors,
            "normalization_relative_error_estimate": float(errors[0]/norm)}

