import pandas as pd

from airproof.reporting import _factor_effect


def test_factor_effect_uses_paired_seed_means():
    frame = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "method": ["factorial_000", "factorial_100", "factorial_000", "factorial_100"],
            "metric": [1.0, 3.0, 2.0, 6.0],
        }
    )
    assert _factor_effect(frame, 0, "metric") == 3.0
