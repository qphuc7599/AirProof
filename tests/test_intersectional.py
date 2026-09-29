from dataclasses import replace

import numpy as np

from airproof.intersectional import audit_intersections, intersection_masks
from airproof.records import Observation
from airproof.simulator import SyntheticWorld


def fixture():
    truth = np.arange(64., dtype=float)[None, :] + np.arange(10.)[:, None]
    records = tuple(Observation(i, e, i, i // 16, float(truth[e, i]), 2., 1., 512,
                                f"{e}:{i}", e, e) for e in range(10) for i in range(64) if i % 3)
    return SyntheticWorld(0, truth, np.repeat(np.arange(4), 16), records, (), {})


def test_masks_ignore_future_values_and_arrivals_and_cover_partition_atoms():
    world = fixture()
    fields = world.truth.copy()
    a, meta_a = intersection_masks(world, fields, calibration_epochs=4, fixed_lag=2)
    fields[4:] += 1000
    changed = replace(world, observations=tuple(item for item in world.observations if item.epoch < 4))
    b, meta_b = intersection_masks(changed, fields, calibration_epochs=4, fixed_lag=2)
    assert meta_a == meta_b
    assert len(a) == 39
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    atoms = [mask for name, mask in a.items() if name.startswith("zone=")]
    np.testing.assert_array_equal(np.sum(atoms, axis=0), np.ones(64))


def test_audit_separates_live_and_reconstructed_error_and_empty_support():
    world = fixture()
    audit = audit_intersections(world, world.truth + 2, world.truth + 3,
        list(world.observations), world.truth, burn_in=4, fixed_lag=2, minimum_cells=8)
    assert audit["worst_supported_reconstruction_rmse"] == 2
    assert audit["worst_supported_live_rmse"] == 3
    for row in audit["strata"].values():
        if row["cells"]:
            assert row["reconstruction_rmse"] == 2
            assert row["live_rmse"] == 3
            assert 0 <= row["citizen_covered_cell_epoch_fraction"] <= 1
        else:
            assert row["live_rmse"] is None
            assert row["supported_for_worst_stratum"] is False
