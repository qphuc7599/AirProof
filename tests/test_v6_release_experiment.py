import unittest
import numpy as np
from airproof.v6_release_experiment import *
from airproof.v6_release_experiment import _schedule
from airproof.continual_release import FixedReleasePlan
from airproof.records import Observation
from airproof.v6_transport import TransportTrace,simulate_transport

class ReleaseExperimentTests(unittest.TestCase):
    def fixture(self):
        # Deterministic engineering fixture, not development scientific world.
        time=np.arange(112.)
        p=(10+.1*np.sin(time))[:,None]
        truth=p+.2*np.cos(time[:,None]/4)
        q=np.full_like(p,np.nan)
        slots=FixedReleasePlan(112,1,20,8/28,0,8).scheduled_epochs
        q[list(slots)]=truth[list(slots)]
        mask=np.isfinite(q); noisy=q.copy(); noisy[mask]+=.3
        return dict(public=p,actual_query=q,noisy_values=noisy,released_mask=mask),truth

    def test_six_controls_same_clock_and_isolation(self):
        data,t=self.fixture()
        c=fit_release_controls(**data,calibration_truth=t,calibration_id='engineering-train')
        result=run_release_controls(c,**data,evaluation_id='engineering-test',public_drain=np.repeat(data['public'][-1:],24,axis=0))
        self.assertEqual(set(result['predictions']),{'P0','P1','P2','P3','P4','P5'})
        for p in result['predictions'].values(): self.assertEqual(p.shape,(136,1))
        np.testing.assert_allclose(result['predictions']['P1'][:24],result['predictions']['P2'][:24])
        scores=score_controls(result,t)
        self.assertEqual(len({s['scored_group_hours'] for s in scores.values()}),1)
        with self.assertRaises(ValueError): run_release_controls(c,**data,evaluation_id='engineering-train')
        altered=dict(data); altered['actual_query']=data['actual_query']+100
        changed=run_release_controls(c,**altered,evaluation_id='engineering-test')
        for arm in ('P0','P2','P3','P4','P5'):
            np.testing.assert_array_equal(result['predictions'][arm][:112],changed['predictions'][arm])
        self.assertFalse(np.array_equal(result['predictions']['P1'][:112],changed['predictions']['P1']))

    def test_exact_schedule_and_suppression(self):
        data,t=self.fixture()
        bad=dict(data); bad['actual_query']=data['actual_query'].copy(); bad['actual_query'][1]=2
        with self.assertRaises(ValueError): fit_release_controls(**bad,calibration_truth=t,calibration_id='a')
        data['released_mask'][0]=False; data['noisy_values'][0]=np.nan
        c=fit_release_controls(**data,calibration_truth=t,calibration_id='a')
        r=run_release_controls(c,**data,evaluation_id='b')
        np.testing.assert_array_equal(r['predictions']['P1'][:28],r['predictions']['P2'][:28])
        self.assertEqual(r['publication_epochs'][0],24)
        self.assertEqual(len(r['publication_epochs']),28)
        with self.assertRaises(ValueError):
            _schedule(27,1)

    def test_reserved_release_delivery_is_per_credential_and_group_switch_invariant(self):
        def observation(user,group):
            return Observation(user,0,0,group,10.,1.,1.,512,f'{user}:{group}',None,None)
        trace=TransportTrace(1,2,1,0,(0,1),((0,1),),((),),'reserved-release')
        original=simulate_transport(trace,[observation(0,0),observation(1,1)],{})
        switched=simulate_transport(trace,[observation(0,1),observation(1,1)],{})
        self.assertEqual(original.release_arrivals[(1,0)],0)
        self.assertEqual(switched.release_arrivals[(1,0)],0)
        self.assertEqual(original.metrics['gateway_reserved_bytes'],512)
        self.assertEqual(switched.metrics['gateway_reserved_bytes'],512)
