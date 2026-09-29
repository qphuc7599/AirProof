import importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
from airproof.v5_estimator import EstimatorConfig

path=Path(__file__).resolve().parents[1]/'reports/v5/tools/diagnose_selected_archive_factorial.py'
spec=importlib.util.spec_from_file_location('energy_observer',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_energy_observer_exact_return_and_capped_boundary():
    config=EstimatorConfig(regularization_multiplier=.1,lag=0,cap=8.)
    public=np.full((2,2),20.)
    solutions=[(np.array([12.,2.]),0),(np.array([3.,4.]),0)]
    calls=[]
    def solver(*args,**kwargs):
        result=solutions[len(calls)];calls.append(result);return result
    lap=sparse.csr_matrix([[1.,-1.],[-1.,1.]])
    observer=module.EnergyObserver(public,lap,config,solver)
    assert observer(None,None) is solutions[0]
    assert observer(None,None) is solutions[1]
    row=observer.rows[1]
    # The closed correction was capped to [8,2], not raw [12,2].
    assert np.isclose(row['temporal_including_fixed_boundary'],.025*((3-8)**2+(4-2)**2))
    assert np.isclose(row['zero_prior'],.025*(3**2+4**2))
    assert np.isclose(row['spatial'],.01*(3-4)**2)
    np.testing.assert_array_equal(solutions[0][0],[12.,2.])
