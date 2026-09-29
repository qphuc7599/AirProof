import numpy as np

from airproof.v6_archive_attacks import matched_archive_channel


def test_archive_attacks_preserve_schedule_and_honest_payloads():
    truth = np.full((48, 4), 10.)
    public = np.full_like(truth, 9.)
    clean = matched_archive_channel(truth, public, seed=44, kind="clean")
    drift = matched_archive_channel(truth, public, seed=44, kind="drift")
    assert [(x.user_id, x.epoch, x.cell, x.relay_arrival, x.nullifier) for x in clean] == [
        (x.user_id, x.epoch, x.cell, x.relay_arrival, x.nullifier) for x in drift]
    for left, right in zip(clean, drift, strict=True):
        if not right.corrupted:
            assert left.value == right.value


def test_coordinated_inlier_is_inside_declared_cutoff_and_hotspot_is_local():
    truth = np.full((48, 4), 10.)
    public = np.full_like(truth, 9.)
    inlier = matched_archive_channel(truth, public, seed=51, kind="coordinated_inlier",
                                     attack_fraction=1., huber_delta=2., sigma=3.)
    malicious = [x for x in inlier if x.corrupted]
    assert malicious and all(abs(x.value-public[x.epoch, x.cell]) < 2*3 for x in malicious)
    hotspot = matched_archive_channel(truth, public, seed=51, kind="hotspot_suppression",
                                      attack_fraction=1., hotspot_cells=[0])
    changed = [x for x in hotspot if x.corrupted and x.cell == 0]
    outside = [x for x in hotspot if x.corrupted and x.cell != 0]
    # Corrupted identity is recorded even outside the spatial target, but its payload is unchanged.
    clean = matched_archive_channel(truth, public, seed=51, kind="clean", attack_fraction=1.)
    clean_map = {x.nullifier: x.value for x in clean}
    assert changed and all(x.value <= clean_map[x.nullifier] for x in changed)
    assert all(x.value == clean_map[x.nullifier] for x in outside)
