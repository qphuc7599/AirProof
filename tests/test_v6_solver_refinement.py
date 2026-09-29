import numpy as np
from airproof.v6_solver_refinement import refine_to_kkt

def test_refinement_matches_box_optimum_without_tolerance_change():
    h=np.diag([.01,1.,20.]);b=np.array([.1,2.,-4.])
    def objective(x):return .5*x@h@x-b@x,h@x-b
    x,history=refine_to_kkt(objective,np.zeros(3),-np.ones(3),np.ones(3))
    assert np.allclose(x,[1.,1.,-.2],atol=1e-6)
    assert history[-1]['kkt']<=1e-6
    assert history[-1]['objective']<=history[0]['objective']
