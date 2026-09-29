"""Public-only causal increment regression; no citizen or evaluation labels."""
from dataclasses import dataclass
import numpy as np


def increment_features(history, neighbors):
    h = np.asarray(history, float)
    n = np.asarray(neighbors, int)
    if h.ndim != 2 or len(h)<6 or not np.isfinite(h).all() or (h<0).any():
        raise ValueError('six finite nonnegative past public fields required')
    if n.ndim!=2 or len(n)!=h.shape[1] or (n<0).any() or (n>=h.shape[1]).any():
        raise ValueError('valid fixed public neighbor indices required')
    last=h[-1]; nearby=h[-1,n].mean(1); oldnear=h[-2,n].mean(1)
    return np.column_stack((last,last-h[-2],h[-2]-h[-3],last-h[-6],
                            nearby-last,nearby-oldnear,np.full(len(last),np.median(last))))


@dataclass(frozen=True)
class PublicIncrementModel:
    neighbors: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    training_epochs: int

    def predict_next(self, public_history):
        h=np.asarray(public_history,float)
        features=increment_features(h,self.neighbors)
        x=np.column_stack((np.ones(len(features)),(features-self.mean)/self.scale))
        return np.maximum(h[-1]+x@self.coefficients,0)


def fit_public_increment(prefix, neighbors, *, ridge=.1, event_weight=1.):
    """Fit only the supplied historical prefix; weights defined by prefix q95."""
    h=np.asarray(prefix,float)
    if h.ndim!=2 or len(h)<12 or not np.isfinite((ridge,event_weight)).all() or ridge<=0 or event_weight<1:
        raise ValueError('valid public prefix and fixed positive regularization required')
    features=np.concatenate([increment_features(h[:t],neighbors) for t in range(6,len(h))])
    y=(h[6:]-h[5:-1]).ravel()
    mean=features.mean(0);scale=np.maximum(features.std(0),1e-6)
    x=np.column_stack((np.ones(len(features)),(features-mean)/scale))
    weights=np.where(h[6:].ravel()>=np.quantile(h,.95),event_weight,1.)
    penalty=ridge*np.eye(x.shape[1]);penalty[0,0]=0
    coef=np.linalg.solve(x.T@(weights[:,None]*x)/weights.sum()+penalty,x.T@(weights*y)/weights.sum())
    return PublicIncrementModel(np.array(neighbors,copy=True),mean,scale,coef,len(h))
