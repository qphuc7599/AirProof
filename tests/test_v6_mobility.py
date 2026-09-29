from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from airproof.simulator import generate_world
from airproof.v6_mobility import coupled_public_trace


def config():
    return dict(world=dict(agents=12,steps=15,grid_side=4,groups=4,
                participation_rate=1.,participation_skew=1.),
                network=dict(availability=.5,group_correlation=.5,
                outage_median_hours=2,outage_p95_hours=5,contact_probability=1.),
                transport=dict(drain_epochs=3),attack=dict(fraction=0),privacy=dict(clip=50))


def test_sensing_and_peer_geometry_share_exact_paths():
    cfg=config()
    world=generate_world(cfg,17)
    trace,paths=coupled_public_trace(cfg,17,world.observations)
    assert len(world.observations)==12*15
    assert all(paths[r.epoch,r.user_id]==r.cell for r in world.observations)
    for epoch,pairs in enumerate(trace.peer_contacts):
        endpoints=[node for pair in pairs for node in pair]
        assert len(endpoints)==len(set(endpoints))
        assert all(paths[epoch,a]==paths[epoch,b] for a,b in pairs)
    again,other=coupled_public_trace(cfg,17,world.observations)
    assert again==trace and np.array_equal(paths,other)


def test_future_extension_preserves_geometry_but_not_gateway_contract():
    cfg=config()
    short,paths=coupled_public_trace(cfg,17)
    extended=deepcopy(cfg);extended['transport']['drain_epochs']=30
    long,long_paths=coupled_public_trace(extended,17)
    assert np.array_equal(paths,long_paths[:len(paths)])
    assert short.peer_contacts==long.peer_contacts[:len(paths)]
    # Explicit regression documenting horizon-dependent renewal RNG allocation.
    assert short.gateway_contacts!=long.gateway_contacts[:len(paths)]


def test_wrong_path_or_population_is_rejected():
    cfg=config();world=generate_world(cfg,17);r=world.observations[0]
    with pytest.raises(ValueError,match='path differs'):
        coupled_public_trace(cfg,17,[replace(r,cell=(r.cell+1)%16)])
    with pytest.raises(ValueError,match='outside'):
        coupled_public_trace(cfg,17,[replace(r,user_id=-1)])
    with pytest.raises(ValueError,match='outside'):
        coupled_public_trace(cfg,17,[replace(r,epoch=15)])
