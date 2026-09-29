from __future__ import annotations

import numpy as np

from scripts.analyze_v7_shared_privacy import _bootstrap_ratio


def test_paired_bootstrap_ratio_is_deterministic_and_ordered():
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([2.0, 4.0, 6.0])
    first = _bootstrap_ratio(left, right, seed=4, replicates=1000)
    second = _bootstrap_ratio(left, right, seed=4, replicates=1000)
    assert first == second
    assert first == [0.5, 0.5]
