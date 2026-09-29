"""Causal features isolating protected numerical value from public availability."""
import numpy as np


def release_age_features(values,mask,scheduled,*,deadline=24,age_edges=(6,12,24),centers=None):
    """Return availability and centered-value features at publication time.

    Rows are time-major then group-major. A release acquired at a becomes
    available at a+deadline. Missing/suppressed releases do not update state.
    """
    v=np.asarray(values,float);m=np.asarray(mask);s=np.asarray(scheduled)
    if (v.ndim!=2 or m.shape!=v.shape or m.dtype!=bool or s.shape!=(len(v),)
            or s.dtype!=bool or deadline<0 or tuple(age_edges)!=(6,12,24)
            or not np.isfinite(v[m]).all() or not np.isnan(v[~m]).all() or np.any(m[~s])):
        raise ValueError('valid sparse protected release arrays required')
    groups=v.shape[1]
    if centers is None:
        centers=np.array([np.mean(v[m[:,g],g]) if m[:,g].any() else 0. for g in range(groups)])
    centers=np.asarray(centers,float)
    if centers.shape!=(groups,) or not np.isfinite(centers).all():raise ValueError('finite group centers required')
    availability=np.zeros((len(v)*groups,groups*3));numeric=np.zeros_like(availability)
    last=np.zeros(groups);published=np.full(groups,-1);seen=np.zeros(groups,bool)
    for epoch in range(len(v)):
        acquisition=epoch-deadline
        if acquisition>=0 and s[acquisition]:
            take=m[acquisition];last[take]=v[acquisition,take];published[take]=epoch;seen[take]=True
        for group in range(groups):
            if not seen[group]:continue
            age=epoch-published[group];bucket=0 if age<6 else 1 if age<12 else 2
            column=group*3+bucket;row=epoch*groups+group
            availability[row,column]=1.;numeric[row,column]=last[group]-centers[group]
    return availability,numeric,centers


def ridge_correction(target,features,*,ridge=.1):
    y=np.asarray(target,float).ravel();x=np.asarray(features,float)
    if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or ridge<=0:
        raise ValueError('finite matched correction design required')
    return np.linalg.solve(x.T@x/len(x)+ridge*np.eye(x.shape[1]),x.T@y/len(x))
