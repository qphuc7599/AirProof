import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1]/'reports/v5/tools/export_stress_event_intervals.py'
spec = importlib.util.spec_from_file_location('conditional_stress_export', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_empty_event_support_stays_missing_and_conditional_score_is_exact():
    truth = np.array([0., 10.])
    lower = np.array([[0., 0.], [6., 6.]])
    upper = np.array([[2., 2.], [8., 8.]])
    missing = module.conditional_metrics(truth, lower, upper, np.array([False, False]))
    assert missing['0.9']['n'] == 0 and missing['0.9']['coverage'] is None
    result = module.conditional_metrics(truth, lower, upper, np.array([False, True]))
    assert result['0.9']['n'] == 1 and result['0.9']['coverage'] == 0
    assert np.isclose(result['0.9']['mean_interval_score'], 42.)
    assert np.isclose(result['0.95']['mean_interval_score'], 82.)
