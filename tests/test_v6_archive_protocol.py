import numpy as np
import pytest

from airproof.v6_archive_protocol import latest_disjoint_windows


def test_latest_windows_depend_only_on_missingness_and_are_separated():
    mask = np.ones((40, 5), bool)
    chosen = latest_disjoint_windows(mask, candidate_start=10, width=6, gap=4,
                                     minimum_support=.8, minimum_stations=4)
    assert chosen["windows"] == [[24, 30], [34, 40]]
    assert chosen["gap"] == 4
    changed_values = np.random.default_rng(8).normal(size=mask.shape)
    assert chosen == latest_disjoint_windows(np.isfinite(changed_values), candidate_start=10,
                                             width=6, gap=4, minimum_support=.8,
                                             minimum_stations=4)


def test_latest_windows_respect_frozen_station_pool_and_support():
    mask = np.ones((40, 6), bool)
    mask[34:, 0] = False
    chosen = latest_disjoint_windows(mask, candidate_start=10, width=6, gap=4,
                                     minimum_support=.8, minimum_stations=4,
                                     station_pool=[0, 1, 2, 3, 4])
    assert chosen["station_indices"] == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        latest_disjoint_windows(mask[:, :3], candidate_start=10, width=6, gap=4,
                                minimum_stations=4)
