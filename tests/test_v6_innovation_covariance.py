import numpy as np
import pytest
from airproof.v6_innovation_covariance import innovation_variance,conservative_innovation_variance

def test_innovation_covariance_limits():
    assert innovation_variance(4,9,0)==13
    assert innovation_variance(4,9,6)==1
    assert innovation_variance(4,9,-6)==25
    assert conservative_innovation_variance(4,9)==25
    with pytest.raises(ValueError):innovation_variance(4,9,7)
    assert np.all(innovation_variance([1,4],9,0)==[10,13])
