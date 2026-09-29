import importlib.util
from pathlib import Path
import numpy as np
import pytest


def test_negative_channel_proxy_never_changes_observed_scoring_targets():
    path = Path(__file__).resolve().parents[1]/"reports/v5/tools/run_epa_domain_adapter.py"
    spec = importlib.util.spec_from_file_location("epa_domain_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    targets = np.array([[-2., 0., 4.]])
    proxy = module.channel_proxy(targets)
    np.testing.assert_array_equal(proxy, [[0., 0., 4.]])
    np.testing.assert_array_equal(targets, [[-2., 0., 4.]])
    assert not np.shares_memory(proxy, targets)
    with pytest.raises(ValueError):
        module.channel_proxy([[np.nan]])
