import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).parents[1]
OUT=ROOT/'reports/v6/accountability_integrated_confirmation'


def test_registered_integrated_accountability_artifact_integrity():
    registered=(ROOT/'configs/v6_accountability_confirmation.json').read_bytes()
    manifest=json.loads((OUT/'manifest.json').read_text())
    frozen=(OUT/'registration.json').read_bytes()
    assert frozen==registered
    assert manifest['registration_sha256']==hashlib.sha256(registered).hexdigest()
    assert manifest['case_count']==6 and manifest['all_fault_checks_passed']
    # This v1 artifact preserves execution-time hashes. Later integrated
    # collector/transport revisions intentionally differ; immutable registration,
    # runner and base cryptographic verifier must still match exactly.
    for name in ('configs/v6_accountability_confirmation.json',
                 'scripts/run_v6_accountability_confirmation.py','airproof/audit.py'):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest['source_hashes'][name]
    assert all(len(digest)==64 for digest in manifest['source_hashes'].values())


def test_integrated_accountability_conservation_and_capacity_bounds():
    rows=json.loads((OUT/'results.json').read_text())
    for row in rows:
        assert row['physical_delivered']==row['admitted']+row['accountability_rejected']
        assert row['raw_bytes']==512*row['physical_delivered']
        assert row['allocated']==row['admitted']==row['collector_included']
        assert row['origin_verified']<=row['collector_included']
        assert row['return_bytes']==512*row['return_frames']
        assert row['return_control_bytes']==16*row['return_frames']
        assert row['post_admission_queue_drops']==0
        assert row['max_contact_direction_bytes']<=1024
        assert row['max_control_direction_bytes']<=128
        assert row['token_violations']==0 and row['raw_remaining_copies']==0


def test_adequate_adverse_and_rejection_cells_remain_distinct():
    rows={row['case']:row for row in json.loads((OUT/'results.json').read_text())}
    assert rows['adequate_single']['origin_verified']==1
    assert rows['adequate_serial4']['origin_verified']==4
    assert rows['frame_budget_zero']['physical_delivered']==1
    assert rows['frame_budget_zero']['admitted']==0
    assert rows['bounded_burst8']['accountability_rejected']>0
    for name in ('absent_reverse','partition_past_deadline'):
        assert rows[name]['collector_included']==1
        assert rows[name]['origin_verified']==0
        assert rows[name]['expired_return_objects']==2
