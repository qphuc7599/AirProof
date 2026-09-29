"""Explicit observation-error covariance composition, never test-fitted."""
import numpy as np

def innovation_variance(public_error_variance,sensor_error_variance,error_covariance=0.):
    """Var(y-m)=Var(sensor error)+Var(public error)-2Cov(errors).

    Inputs must be frozen from compatible independent public/sensor calibration.
    This function supplies algebra and PSD checks, not provenance or calibration.
    """
    p,s,c=np.broadcast_arrays(*[np.asarray(v,float) for v in (public_error_variance,sensor_error_variance,error_covariance)])
    if not all(np.isfinite(x).all() for x in (p,s,c)) or (p<0).any() or (s<0).any():
        raise ValueError('finite nonnegative marginal variances required')
    if (c*c>p*s+1e-12).any():raise ValueError('error covariance violates positive semidefiniteness')
    return np.maximum(p+s-2*c,0.)

def conservative_innovation_variance(public_error_variance,sensor_error_variance):
    """Sharp upper variance bound over unknown valid correlation, not a coverage guarantee."""
    p,s=np.broadcast_arrays(np.asarray(public_error_variance,float),np.asarray(sensor_error_variance,float))
    innovation_variance(p,s)
    return (np.sqrt(p)+np.sqrt(s))**2
