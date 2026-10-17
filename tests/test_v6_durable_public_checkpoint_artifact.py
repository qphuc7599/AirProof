import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'v6'/'durable_public_checkpoint_v2'

def test_public_checkpoint_artifact_is_registered_integral_and_partial():
    reg=(ROOT/'configs'/'v6_durable_public_checkpoint_v2.json').read_bytes()
    manifest=json.loads((OUT/'manifest.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert (OUT/'registration.json').read_bytes()==reg
    assert hashlib.sha256(reg).hexdigest()==manifest['registration_sha256']
    assert hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()==manifest['results_sha256']
    for path,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert result['selected']==result['collector_included']==181
    assert result['origin_verified']==result['externally_checkpointed']==73
    assert result['publicly_checkpointed']==99<result['collector_included']
    assert result['public_heads_verified']==6 and result['public_uplink_contacts']==18
    assert result['public_raw_bytes']==7680
    assert result['public_control_bytes']==result['public_raw_bytes']//512*16==240
    assert result['public_expired']==2 and result['public_pending']==1
    assert result['public_peak_queue_bytes']<=4096 and result['public_peak_reassembly_bytes']<=4096
    assert result['all_fault_checks_passed'] and not result['confirmation_eligible']
    assert result['comparators']['batch_v3']=={'selected':45,'origin_verified':32}
    assert result['comparators']['durable_v1']['publicly_checkpointed']==0
