import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_registered_terminal_batch_design_artifact_is_integral_and_adverse_history_retained():
    out = ROOT / "reports" / "v6" / "reservation_safe_terminal_batch_design"
    result = json.loads((out / "results.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert result["all_checks_passed"] and manifest["all_checks_passed"]
    assert [r["batch_size"] for r in result["rows"]] == [1, 2, 4, 8, 16, 32]
    assert all(r["terminal_envelope_frames"] >= r["batched_terminal_frames"]
               for r in result["rows"])
    assert all(r["batched_total_return_frames"] <= r["current_total_return_frames"]
               for r in result["rows"])
    assert result["smoke_context"] == {
        "legacy_selected": 181,
        "first_per_record_accountability_selected": 39,
        "stale_gate_selected": 50,
    }
    # The artifact records the pre-integration queue hash; the later opt-in v3
    # implementation adds atomic reservation extension without rewriting it.
    for relative in ("configs/v6_terminal_batch_design.json",
                     "scripts/analyze_v6_reservation_safe_terminal_batch.py",
                     "airproof/audit.py"):
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == manifest["source_hashes"][relative]
    assert all(len(expected)==64 for expected in manifest["source_hashes"].values())
