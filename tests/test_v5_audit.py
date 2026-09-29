import pytest
from airproof.v5_audit import FAULTS, POLICIES, run_same_stream_audit, summarize_audit


@pytest.fixture(scope="module")
def rows():
    return run_same_stream_audit(batch_sizes=(8,16), trials=2)


def test_same_stream_matrix_and_clean_controls(rows):
    assert len(rows)==2*2*len(FAULTS)*len(POLICIES)
    assert all(r["clean_control_valid"] for r in rows)
    for size in (8,16):
        for trial in (0,1):
            assert len({r["stream_sha256"] for r in rows if r["batch_size"]==size and r["trial"]==trial})==1


def test_censoring_does_not_become_zero_detection_delay(rows):
    for row in rows:
        if row["fault"] in ("auditor_partition","checkpoint_loss") and row["policy"]!="anchored_transparency_log":
            assert row["right_censored"] and not row["detected"]
            assert row["detection_epoch"] is None and row["time_from_fault"] is None
            assert row["restricted_detection_time"]==50
        else:
            assert row["detected"]


def test_anchor_does_not_hide_payload_loss_or_claim_finality(rows):
    loss=[r for r in rows if r["fault"]=="payload_loss"]
    assert {r["reason"] for r in loss}=={"retrieval-timeout-no-payload-recovery"}
    assert {r["detection_epoch"] for r in loss}=={21}
    assert all("not-measured-chain" in r["clock_scope"] for r in rows)


def test_membership_cost_and_signature_compromise_are_not_shortcuts(rows):
    for row in rows:
        assert row["total_protocol_bytes"]>row["payload_bytes"]+row["receipt_bytes"]
        if row["fault"]=="equivocation" and row["policy"]=="signed_log":
            assert row["detection_epoch"]==18
        if row["policy"]=="signed_log": assert row["proof_bytes"]==0
        else: assert row["proof_bytes"]>0
    assert all(g["trials"]==2 for g in summarize_audit(rows))


def test_invalid_matrix_rejected():
    with pytest.raises(ValueError): run_same_stream_audit(trials=0)


def test_flat_verifier_rejects_malformed_and_invalid_signatures():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from airproof.v5_audit import _flat_head, _verify_flat
    key = Ed25519PrivateKey.generate()
    payloads = [b"a", b"b"]
    head = _flat_head(payloads, key, "test")
    assert _verify_flat(head, payloads, key.public_key())
    assert not _verify_flat(head, payloads, Ed25519PrivateKey.generate().public_key())
    assert not _verify_flat({}, payloads, key.public_key())
    assert not _verify_flat({**head, "signature_b64": "bad"}, payloads, key.public_key())


def test_flat_verifier_does_not_swallow_programming_errors():
    from airproof.v5_audit import _verify_flat
    class BrokenKey:
        def verify(self, *args):
            raise RuntimeError("implementation failure")
    with pytest.raises(RuntimeError, match="implementation failure"):
        _verify_flat({"signature_b64": ""}, [], BrokenKey())
