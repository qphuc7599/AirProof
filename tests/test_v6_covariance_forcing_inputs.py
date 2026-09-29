
from pathlib import Path

import numpy as np

from airproof.field import synthetic_field, synthetic_field_with_mechanism
from airproof.records import Observation
from airproof.v6_covariance_forcing_inputs import (
    _moments,
    _record_arrays,
    combine_moments,
    covariance_from_moments,
    expected_physical_operator,
    load_csr,
    restore_records,
    save_csr,
)


def observation(index, epoch, arrival):
    return Observation(index, epoch, index % 4, index % 2, 10.0 + index, 2.0,
                       .85, 512, f"{index:064x}", arrival, arrival)


def test_sparse_operator_roundtrip_is_linear_grid_storage(tmp_path):
    matrix = expected_physical_operator(32)
    path = tmp_path / "operator.npz"
    save_csr(path, matrix)
    loaded = load_csr(path)
    assert loaded.shape == (1024, 1024)
    assert loaded.nnz <= 5 * 1024
    assert (loaded != matrix).nnz == 0
    with np.load(path, allow_pickle=False) as arrays:
        assert "transitions" not in arrays.files


def test_exact_observation_pack_roundtrip_keeps_identity_and_arrival(tmp_path):
    records = (observation(1, 48, 50), observation(2, 51, 51))
    arrivals = {row.nullifier: row.relay_arrival for row in records}
    path = tmp_path / "records.npz"
    np.savez_compressed(path, **_record_arrays(records, arrivals, burn=48))
    restored, restored_arrivals = restore_records(path)
    assert [(row.user_id, row.epoch, row.cell, row.value, row.nullifier)
            for row in restored] == [(1, 0, 1, 11.0, f"{1:064x}"),
                                     (2, 3, 2, 12.0, f"{2:064x}")]
    assert restored_arrivals == {f"{1:064x}": 2, f"{2:064x}": 3}


def test_streamed_covariance_matches_numpy_and_is_psd():
    sensor = np.asarray([-1.0, 0.0, 2.0, 1.0])
    public = np.asarray([-.5, .5, 1.0, 1.5])
    result = covariance_from_moments(combine_moments([
        _moments(sensor[:2], public[:2]), _moments(sensor[2:], public[2:])]),
        tolerance=1e-10)
    expected = np.cov(np.column_stack((sensor, public)), rowvar=False, ddof=1)
    assert np.allclose(result["covariance_matrix"], expected)
    assert result["psd_audit_pass"] is True


def test_mechanism_tail_does_not_change_historical_truth_draws():
    first = synthetic_field(4, 12, np.random.default_rng(91))
    second, mechanism = synthetic_field_with_mechanism(
        4, 12, np.random.default_rng(91), mechanism_horizon=18)
    assert np.array_equal(first, second)
    assert mechanism.exogenous_forcing.shape == (18, 16)
    assert not mechanism.exogenous_forcing.flags.writeable


def test_campaign_loader_and_builder_use_the_same_operator_name():
    source = (Path(__file__).resolve().parents[1]
              / "airproof" / "v6_covariance_forcing_inputs.py").read_text()
    builder = (Path(__file__).resolve().parents[1]
               / "scripts" / "build_v6_covariance_forcing_v2_inputs.py").read_text()
    assert '"physical_operator.npz"' in source
    assert '"physical_operator.npz"' in builder
    assert "physical_operator_csr.npz" not in source + builder
