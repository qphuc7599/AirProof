"""Optimistic bounds, never deployable schedules or claims of achieved service."""
import numpy as np


def cap_rmse_floor(truth,public,cap):
    y,p=np.asarray(truth,float),np.asarray(public,float)
    if y.shape!=p.shape or y.size==0 or not np.isfinite(y).all() or not np.isfinite(p).all() or cap<0:
        raise ValueError('finite matched arrays and nonnegative cap required')
    return float(np.sqrt(np.mean(np.maximum(np.abs(y-p)-cap,0)**2)))


def timely_reachability_upper_bound(trace, observations, lag=6):
    """Relax buffer/copy/contact competition; retain contact timing and next-slot hops.

    Backward reachability per acquisition gives a necessary opportunity bound.
    It is not a feasible flow, does not upper-bound RMSE gain, and cannot certify
    achievable fairness. One record can use unlimited replicas in this relaxation.
    """
    if type(lag) is not int or lag<0:raise ValueError('nonnegative lag required')
    horizon=len(trace.gateway_contacts);memo={};eligible=[]
    for r in observations:
        if r.source_class!='citizen':continue
        if not 0<=r.epoch<trace.acquisition_epochs or not 0<=r.user_id<trace.nodes:
            raise ValueError('record outside trace')
        if r.epoch not in memo:
            last=min(r.epoch+lag,horizon-1)
            possible=np.zeros(trace.nodes,dtype=bool)
            for t in range(last,r.epoch-1,-1):
                future=possible.copy()
                for a,b in trace.peer_contacts[t]:
                    if future[b]:possible[a]=True
                    if future[a]:possible[b]=True
                possible[list(trace.gateway_contacts[t])]=True
            memo[r.epoch]=possible
        if memo[r.epoch][r.user_id]:eligible.append(r)
    groups=sorted(set(r.group for r in observations if r.source_class=='citizen'))
    return {'relaxation':'unlimited-copy-and-capacity causal reachability; necessary not sufficient',
            'timely_records_upper':len(eligible),
            'distinct_users_upper':{g:len({r.user_id for r in eligible if r.group==g}) for g in groups},
            'eligible_nullifiers':tuple(r.nullifier for r in eligible),'lag':lag}


def delivery_required_for_delay(spray_restricted_delay,ttl,reduction=.15):
    if ttl<=0 or not 0<=reduction<1 or not 0<=spray_restricted_delay<=ttl:
        raise ValueError('invalid fixed-TTL cell')
    return max(0.,1-(1-reduction)*spray_restricted_delay/ttl)
