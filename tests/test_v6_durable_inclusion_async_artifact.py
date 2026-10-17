import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'v6'/'durable_inclusion_async_return_v1'

def test_durable_async_artifact_is_registered_integral_and_confirmation_blocked():
    reg=(ROOT/'configs'/'v6_durable_inclusion_async_return_v1.json').read_bytes()
    manifest=json.loads((OUT/'manifest.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert (OUT/'registration.json').read_bytes()==reg
    assert hashlib.sha256(reg).hexdigest()==manifest['registration_sha256']
    assert hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()==manifest['results_sha256']
    for path in ('configs/v6_durable_inclusion_async_return_v1.json',
                 'scripts/run_v6_durable_inclusion_async_return_v1.py',
                 'airproof/v6_audit_return.py','airproof/v6_receipts.py','airproof/audit.py'):
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==manifest['source_hashes'][path]
    assert all(len(digest)==64 for digest in manifest['source_hashes'].values())
    assert result['selected']==result['collector_included']==181
    assert result['origin_verified']==result['externally_checkpointed']==73
    assert result['publicly_checkpointed']==0 and not result['confirmation_eligible']
    assert result['durable_storage_bytes']==266734<result['durable_storage_capacity_bytes']
    assert result['inclusion_proof_bytes']==20608
    assert result['return_raw_bytes']==148480
    assert result['return_control_bytes']==result['return_raw_bytes']//512*16==4640
    assert result['pending_return_objects']==21 and result['return_backlog_objects']==179
    assert result['paired_batch_v3']['selected']==45 and result['paired_batch_v3']['origin_verified']==32
