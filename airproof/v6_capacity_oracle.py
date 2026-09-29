"""Sparse offline time-expanded multicommodity relaxation of timely service.

An optimistic upper bound, not a causal scheduler. See V6_CAPACITY_ORACLE.md.
"""
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def coupled_service_bound(trace, observations, config, *, processing_budget_bytes=83200,
                          useful_lag=6, max_variables=200000, max_constraints=200000,
                          time_limit_seconds=30.):
    """Maximize fractional cumulative distinct (user,group) selected service.

    Per-epoch processing budget may be an integer or mapping epoch -> bytes.
    Returns a diagnostic dictionary; resource/solver failure never supplies a bound.
    """
    records=tuple(observations)
    cfg=config.get('transport',{})
    n=trace.nodes; horizon=trace.acquisition_epochs+trace.drain_epochs
    ttl=int(cfg.get('ttl_epochs',24)); raw=int(cfg.get('raw_packet_bytes',512))
    capacity=int(cfg.get('capacity_bytes_per_direction',1024))
    release=int(cfg.get('release_gateway_reservation_bytes',256))
    control=int(cfg.get('control_budget_bytes',128))
    buffer=int(cfg.get('raw_buffer_bytes',18432))
    if min(n,horizon,raw,max_variables,max_constraints)<1 or min(useful_lag,ttl,buffer)<0:
        raise ValueError('invalid dimensions or finite limits')
    if min(capacity,release,control)<0 or release+control>capacity:
        raise ValueError('invalid contact partitions')
    if not math.isfinite(time_limit_seconds) or time_limit_seconds<=0:
        raise ValueError('invalid time limit')
    if len(trace.gateway_contacts)!=horizon or len(trace.peer_contacts)!=horizon:
        raise ValueError('trace horizon mismatch')
    for r in records:
        if not 0<=r.user_id<n or not 0<=r.epoch<trace.acquisition_epochs or r.group<0:
            raise ValueError('record outside population/horizon')
        if type(r.size_bytes) is not int or r.size_bytes<0:
            raise ValueError('processing sizes must be nonnegative integer bytes')
    if len({r.nullifier for r in records})!=len(records):
        raise ValueError('duplicate raw nullifier')
    budgets={t: processing_budget_bytes if isinstance(processing_budget_bytes,int)
             else processing_budget_bytes.get(t,0) for t in range(horizon)}
    if any(type(b) is not int or b<0 for b in budgets.values()):
        raise ValueError('invalid processing budget')
    gateway_slots=max(0,min((capacity-release-control)//raw,(control-32)//32))
    # Optimistically omit optional history/feedback frames, but retain header+ACK.
    peer_slots=max(0,min((capacity-control)//raw,control//32))
    buffer_slots=buffer//raw
    inequalities={}; limits={}; equalities={}; costs=[]; service=defaultdict(list)

    def ub(key,limit):
        limits[key]=limit
        return inequalities.setdefault(key,{})

    def put(row,col,value):
        row[col]=row.get(col,0.)+value

    def variable(cost=0.):
        if len(costs)>=max_variables:
            raise OverflowError
        costs.append(cost)
        return len(costs)-1

    def edge(mid,start,end):
        col=variable()
        if start is not None:
            put(equalities.setdefault((mid,start),{}),col,1.)
        if end is not None:
            put(equalities.setdefault((mid,end),{}),col,-1.)
        return col

    try:
        for mid,r in enumerate(records):
            end=min(horizon-1,r.epoch+useful_lag,r.epoch+ttl)
            col=edge(mid,None,(r.user_id,r.epoch)) # admission, at most one unit
            for t in range(r.epoch,end+1):
                for node in range(n):
                    occupancy=ub(('buffer',node,t),buffer_slots)
                    if t<end:
                        col=edge(mid,(node,t),(node,t+1));put(occupancy,col,1.)
                for encounter,node in enumerate(trace.gateway_contacts[t]):
                    if not 0<=node<n: raise ValueError('invalid gateway node')
                    col=edge(mid,(node,t),('collector',t))
                    put(ub(('gateway',t,encounter),gateway_slots),col,1.)
                    put(ub(('buffer',node,t),buffer_slots),col,1.)
                if t<end:
                    for encounter,(a,b) in enumerate(trace.peer_contacts[t]):
                        if not 0<=a<n or not 0<=b<n or a==b: raise ValueError('invalid peer')
                        for sender,receiver in ((a,b),(b,a)):
                            col=edge(mid,(sender,t),(receiver,t+1))
                            put(ub(('peer',t,encounter,sender),peer_slots),col,1.)
                            put(ub(('buffer',sender,t),buffer_slots),col,1.)
                    edge(mid,('collector',t),('collector',t+1))
                col=edge(mid,('collector',t),None)
                service[r.user_id,r.group].append(col)
                put(ub(('processing',t),budgets[t]),col,r.size_bytes)
                put(ub(('representative',r.user_id,r.group,t),1.),col,1.)
            if len(equalities)+len(inequalities)>max_constraints: raise OverflowError
        for key,columns in service.items():
            z=variable(-1.)
            row=ub(('distinct',*key),0.)
            put(row,z,1.)
            for col in columns: put(row,col,-1.)
        if len(equalities)+len(inequalities)>max_constraints: raise OverflowError
    except OverflowError:
        return dict(status='resource_limit',upper_bound=None,variables=len(costs),
                    constraints=len(equalities)+len(inequalities),diagnostic_oracle=True)
    if not costs:
        return dict(status='optimal',upper_bound=0.,objective=0.,variables=0,
                    constraints=0,diagnostic_oracle=True)

    def matrix(rows):
        rr=[];cc=[];vv=[]
        for i,row in enumerate(rows.values()):
            for j,value in row.items():
                rr.append(i);cc.append(j);vv.append(value)
        return coo_matrix((vv,(rr,cc)),shape=(len(rows),len(costs))).tocsr()

    aeq=matrix(equalities);aub=matrix(inequalities)
    beq=np.zeros(len(equalities));bub=np.array([limits[k] for k in inequalities],dtype=float)
    c=np.array(costs)
    fit=linprog(c,A_ub=aub,b_ub=bub,A_eq=aeq,b_eq=beq,bounds=(0.,1.),
                method='highs',options={'time_limit':time_limit_seconds})
    result=dict(status='optimal' if fit.success else 'solver_incomplete',upper_bound=None,
        solver_status=int(fit.status),solver_message=fit.message,variables=len(costs),
        constraints=len(equalities)+len(inequalities),diagnostic_oracle=True,
        method='sparse_time_expanded_multicommodity_lp',gateway_slots_per_contact=gateway_slots,
        peer_slots_per_direction=peer_slots,buffer_slots=buffer_slots,
        processing_budget_bytes=budgets,trace_hash=trace.trace_hash)
    if fit.success:
        # Weak-duality bound even with imperfect dual stationarity: x_j in [0,1].
        y=np.minimum(np.asarray(fit.ineqlin.marginals),0.)
        v=np.asarray(fit.eqlin.marginals)
        reduced=c-aub.T@y-aeq.T@v
        dual_lower=float(bub@y+beq@v+np.minimum(reduced,0.).sum())
        numerical_margin=1e-7*(1+abs(dual_lower))
        upper=min(float(len(service)),-dual_lower+numerical_margin)
        result.update(upper_bound=upper,objective=float(-fit.fun),
            dual_upper_bound=float(-dual_lower),numerical_margin=numerical_margin,
            max_equality_residual=float(np.max(np.abs(aeq@fit.x),initial=0.)),
            max_capacity_violation=float(np.max(aub@fit.x-bub,initial=0.)),
            selected_fractional_by_identity={f'{u}:{g}':float(sum(fit.x[j] for j in cols))
                                             for (u,g),cols in service.items()})
    return result
