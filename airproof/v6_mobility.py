"""Public sensing/peer geometry shares the simulator mobility stream.

Gateway access is an exogenous renewal process, not geographic coverage. The
renewal RNG draw order depends on horizon: extending a trace preserves mobility
and peer prefixes, but need not preserve gateway-contact prefixes. Freeze the
full acquisition+drain trace before paired comparisons.
"""
import hashlib
import json
import numpy as np
from .simulator import _random_walks, _connectivity
from .rng import rng_streams
from .field import policy_groups
from .v5_transport import TransportTrace


def coupled_public_trace(config, seed, observations=()):
    w=config['world'];n=int(w['agents']);steps=int(w['steps']);side=int(w['grid_side'])
    drain=int(config.get('transport',{}).get('drain_epochs',24));net=config['network']
    if min(n,steps,side)<1 or drain<0:
        raise ValueError('invalid public trajectory dimensions')
    streams=rng_streams(seed)
    paths=_random_walks(side,steps+drain,n,streams['mobility'])
    for r in observations:
        if r.source_class=='citizen':
            if not (0<=r.epoch<steps and 0<=r.user_id<n):
                raise ValueError('citizen outside acquisition population/horizon')
            if paths[r.epoch,r.user_id]!=r.cell:
                raise ValueError('observation path differs from public trajectory')
    groups=policy_groups(side,int(w.get('groups',4)))[paths[0]]
    online=_connectivity(groups,steps+drain,float(net.get('availability',.4)),
        float(net.get('group_correlation',.5)),streams['outage'],
        float(net.get('outage_median_hours',24)),float(net.get('outage_p95_hours',72)))
    rng=np.random.default_rng(np.random.SeedSequence([seed,6006]))
    probability=float(net.get('contact_probability',.15));peers=[]
    if not 0<=probability<=1:raise ValueError('invalid contact probability')
    for positions in paths:
        pairs=[]
        for cell in np.unique(positions):
            nodes=np.flatnonzero(positions==cell);rng.shuffle(nodes)
            pairs.extend((int(nodes[i]),int(nodes[i+1])) for i in range(0,len(nodes)-1,2) if rng.random()<probability)
        peers.append(tuple(pairs))
    gateways=tuple(tuple(map(int,np.flatnonzero(row))) for row in online)
    identity={'seed':seed,'paths_sha256':hashlib.sha256(paths.tobytes()).hexdigest(),
              'groups':groups.tolist(),'gateways':gateways,'peers':peers,'scope':'fixed-public-sensing-trajectory'}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return TransportTrace(seed,n,steps,drain,tuple(map(int,groups)),gateways,tuple(peers),digest),paths
