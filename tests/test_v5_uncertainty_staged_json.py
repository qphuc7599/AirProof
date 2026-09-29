from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import yaml
from airproof.v5_estimator import EstimatorConfig


def test_staged_selected_json_scientific_notation_matches_yaml(tmp_path):
    path = Path(__file__).resolve().parents[1]/"reports/v5/staged_integrity_fix/scripts/analyze_v5_uncertainty.py"
    spec = importlib.util.spec_from_file_location("staged_uncertainty", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = asdict(EstimatorConfig(regularization_multiplier=.1))
    json_path = tmp_path/"selected.json"
    yaml_path = tmp_path/"selected.yaml"
    json_path.write_text(json.dumps(expected))
    yaml_path.write_text(yaml.safe_dump(expected))
    assert '1e-07' in json_path.read_text()
    assert isinstance(yaml.safe_load(json_path.read_text())["tolerance"], str)
    assert module.load_selected_manifest(json_path) == module.load_selected_manifest(yaml_path) == expected
    assert isinstance(module.load_selected_manifest(json_path)["tolerance"], float)
