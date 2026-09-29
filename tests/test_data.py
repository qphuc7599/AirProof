import pandas as pd

from airproof.data import chronological_split


def test_chronological_split_has_purge_gaps_and_no_overlap():
    frame = pd.DataFrame({"timestamp_utc": pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")})
    split = chronological_split(frame, purge_steps=3)
    sets = {key: set(value["timestamp_utc"]) for key, value in split.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert max(sets["train"]) < min(sets["validation"])
    assert max(sets["validation"]) < min(sets["test"])
