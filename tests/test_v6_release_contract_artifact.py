import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).parents[1]
OUT=ROOT/'reports/v6/privacy_release_contract'


def test_release_contract_certificate_and_integrity_artifact():
    certificate=json.loads((OUT/'certificate.json').read_text())
    integrity=json.loads((OUT/'integrity.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert certificate['publications']==28
    assert certificate['k_min']==20
    assert certificate['whole_history_epsilon']<=8
    assert certificate['deadline_epochs']==24
    assert certificate['consumer_fixed_lag_epochs']==48
    assert certificate['deployed_laplace_scale']==7
    assert all(value is True or not isinstance(value,bool) for value in integrity.values())
    assert manifest['all_integrity_checks_passed']
    assert 'no protected-release gain supported' in manifest['scientific_disposition']


def test_release_contract_artifact_hashes_bind_sources_and_reused_negative_evidence():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for collection in ('source_hashes','reused_evidence_hashes'):
        for name,digest in manifest[collection].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
