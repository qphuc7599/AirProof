import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from airproof.v5_estimator import EstimatorConfig

ROOT = Path(__file__).resolve().parents[1]


def test_staged_huber_ceiling_preserves_selected_and_squared_control():
    path = ROOT/"reports/v5/staged_numerical_repair/airproof/v5_experiment.py"
    spec = importlib.util.spec_from_file_location("airproof.staged_v5_experiment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    selected = EstimatorConfig()
    methods = {r["method"]:r["estimator"] for r in module.method_specs(selected, "anchor_clean")}
    assert methods["AP"] is selected and selected.max_irls == 20
    assert asdict(methods["SQ"]) == {**asdict(selected), "input_clip":False, "output_cap":False}
    assert asdict(methods["HUBER"]) == {**asdict(selected), "input_clip":False, "output_cap":False, "loss":"huber", "max_irls":200}
