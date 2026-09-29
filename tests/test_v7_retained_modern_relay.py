import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/analyze_v7_retained_modern_relay.py"
SPEC = importlib.util.spec_from_file_location("modern_relay", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_analysis_keeps_pairing_and_direction():
    rows = []
    for seed in (1, 2):
        for policy, delivery, delay, gap, transmissions in (
            ("airproof_deadline", 0.9, 4.0, 0.1, 2.0),
            ("prophet_gtmx", 0.8, 5.0, 0.2, 3.0),
        ):
            rows.append(
                {
                    "dataset": "trace",
                    "probability": 0.1,
                    "seed": seed,
                    "policy": policy,
                    "deadline_delivery_ratio": delivery,
                    "restricted_mean_delay_seconds": delay,
                    "group_delivery_gap": gap,
                    "transmissions_per_delivered": transmissions,
                }
            )
    result = MODULE.analyze(rows)
    assert result["favorable_point_direction_cells_of_six"] == {
        "delivery": 1,
        "delay": 1,
        "gap": 1,
        "transmissions": 1,
    }
