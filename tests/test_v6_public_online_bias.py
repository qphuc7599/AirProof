import numpy as np

from airproof.v6_public_online_bias import causal_ewma_bias


def test_online_bias_is_causal_and_adapts_to_persistent_offset():
    base = np.full((100, 2), 10.)
    truth = np.full_like(base, 14.)
    first, history = causal_ewma_bias(base, truth, start_epoch=10,
                                      initial_bias=np.zeros(2), half_life=4)
    changed = truth.copy(); changed[60:] = 1e6
    second, _ = causal_ewma_bias(base, changed, start_epoch=10,
                                 initial_bias=np.zeros(2), half_life=4)
    assert np.array_equal(first[:60], second[:60])
    assert np.all(first[10] == 10.)
    assert np.all(first[40] > 13.9)
    assert np.all(history[10] == 0.)


def test_online_bias_skips_missing_updates():
    base = np.full((8, 1), 10.)
    truth = np.full_like(base, 14.); truth[2:6] = np.nan
    prediction, history = causal_ewma_bias(base, truth, start_epoch=1,
                                           initial_bias=np.zeros(1), half_life=1)
    assert history[2, 0] == history[5, 0]
    assert np.isfinite(prediction).all()
