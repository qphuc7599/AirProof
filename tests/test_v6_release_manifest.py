"""The final manuscript release must remain hash- and scope-bound."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v6_release_manifest_verifies_current_snapshot():
    path = ROOT / "scripts/build_v6_release_manifest.py"
    spec = importlib.util.spec_from_file_location("build_v6_release_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.verify()
    assert result["status"] == "pass"
    assert result["files"] == 39
    assert result["checks"]["pdf_pages"] <= result["checks"]["page_limit"]
    assert result["checks"]["scientific_goal_not_attained"] is True
    assert result["checks"]["mechanistic_v1_scientific_outcomes"] == 0
