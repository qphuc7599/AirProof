"""Release checks for the origin-to-v4-to-v6 contribution ledger."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs/v6/contribution_ledger.json"


def test_contribution_ledger_has_required_schema_and_pillars():
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert data["scientific_targets_lowered"] is False
    assert data["primary_v6_launched"] is False
    assert data["confirmation_v6_read"] is False
    required = {
        "claim_id", "pillar", "original_statement", "v4_mechanism_evidence",
        "v6_current_mechanism", "assumptions", "evidence", "reuse_status",
        "changed_part", "acceptance_criteria", "unverified_scope", "status",
    }
    claims = data["claims"]
    assert len({row["claim_id"] for row in claims}) == len(claims) == 6
    assert required <= set.intersection(*(set(row) for row in claims))
    pillars = " ".join(row["pillar"].lower() for row in claims)
    for token in ("robust", "fairness", "dtn", "privacy", "accountability"):
        assert token in pillars


def test_contribution_ledger_evidence_exists_and_keeps_v6_unconfirmed():
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    for row in data["claims"]:
        assert row["evidence"]
        for relative in row["evidence"]:
            assert (ROOT / relative).exists(), (row["claim_id"], relative)
    joined = json.dumps(data).lower()
    assert "target unchanged" in joined or "targets are unchanged" in joined
    assert "confirmation" in joined
    assert not any(
        "v6" in row["status"].lower() and "confirmed" in row["status"].lower()
        for row in data["claims"]
    )
