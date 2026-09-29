import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'reports/v5/tools/certify_archive_cap_accuracy.py'
spec=importlib.util.spec_from_file_location('cap_certificate',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_cap_floor_and_clipped_truth_oracle():
    floor,oracle=module.cap_floor(np.array([20.]),np.array([0.]),8.)
    assert floor==12. and oracle[0]==8.
    truth=np.array([0.,1.,20.,30.]);public=np.array([10.,0.,3.,29.])
    floor,oracle=module.cap_floor(truth,public,8.)
    assert np.all(oracle>=0) and np.all(abs(oracle-public)<=8)
    np.testing.assert_allclose(floor,np.sqrt(np.mean((oracle-truth)**2)))
