import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'v6'/'accountability_throughput_decomposition_v1'

def test_registered_throughput_decomposition_is_integral_and_retains_adverse_result():
    reg=(ROOT/'configs'/'v6_accountability_throughput_decomposition_v1.json').read_bytes()
    manifest=json.loads((OUT/'manifest.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert (OUT/'registration.json').read_bytes()==reg
    assert hashlib.sha256(reg).hexdigest()==manifest['registration_sha256']
    assert hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest()==manifest['results_sha256']
    for path in ('configs/v6_accountability_throughput_decomposition_v1.json',
                 'scripts/run_v6_accountability_throughput_decomposition.py'):
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==manifest['source_hashes'][path]
    assert all(len(digest)==64 for digest in manifest['source_hashes'].values())
    actual=result['oracles']['actual_v3'];unlimited=result['oracles']['infinite_return_frames_same_contacts_buffer']
    sufficient=result['oracles']['sufficient_buffer_same_contacts'];zero=result['oracles']['zero_return_cost']
    assert (actual['selected'],actual['origin_verified'])==(45,32)
    assert (unlimited['selected'],unlimited['origin_verified'])==(45,32)
    assert (sufficient['selected'],sufficient['origin_verified'])==(181,63)
    assert (zero['selected_ceiling'],zero['origin_verified_ceiling'])==(181,181)
    assert sum(map(int,actual['timely_per_origin'].values()))==181
    assert actual['zero_padding_bytes']==actual['fixed_frame_wire_bytes_issued']-actual['byte_exact_last_fragment_wire_bytes']
    assert not result['decomposition']['padding_gate_passed']
    assert result['decomposition']['sufficient_buffer_still_contact_limited']
    assert not result['decomposition']['codec_prototype_authorized_by_registered_gate']
