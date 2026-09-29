import numpy as np
import pytest
from airproof.records import Observation
from airproof.v6_transport import TransportTrace
from airproof.v6_capacity_oracle import coupled_service_bound
from airproof.v6_uncertainty import fit_frozen_intervals,evaluate_intervals


def test_capacity_competition_and_processing():
    records=[Observation(i,0,0,0,10.,1.,1.,512,str(i),None,None) for i in range(2)]
    t=TransportTrace(1,2,1,0,(0,0),((0,1),),((),),'tiny')
    one=coupled_service_bound(t,records,{},processing_budget_bytes=512)
    zero=coupled_service_bound(t,records,{},processing_budget_bytes=0)
    assert one['status']=='optimal'
    assert 1<=one['upper_bound']<1.00001
    assert zero['upper_bound']<1e-6


def test_frozen_uncertainty_does_not_refit_test():
    p=np.ones((24,2))*10
    c=fit_frozen_intervals(p,p+2,clock='live',split_id='independent-fixture',
        training_end=24,target_available_at=24,selection_id='selected',selection_end=23,
        evaluation_split_id='eval',evaluation_start=25)
    with pytest.raises(ValueError):c.predict(p,epochs=24,evaluation_split_id='eval')
    low,high=c.predict(p,epochs=25,evaluation_split_id='eval')
    out=evaluate_intervals(p+3,low,high,event_mask=np.zeros_like(p,bool),bootstrap_replicates=10)
    assert out['groups']['all']['coverage']['estimate']==[0.,0.]
    assert out['groups']['event']['status']=='unavailable'
    assert c.pooled_radii==(2.,2.)
