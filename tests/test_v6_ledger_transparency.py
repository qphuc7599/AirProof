import json
from pathlib import Path

from airproof.v6_ledger_transparency import run_matrix

ROOT=Path(__file__).resolve().parents[1]

def test_small_same_stream_matrix_has_explicit_applicability_and_censoring():
    reg=json.loads((ROOT/'configs'/'v6_ledger_transparency_same_stream_v2.json').read_text())
    reg['batch_sizes']=[32];reg['trials']=1
    rows=run_matrix(reg)
    assert len(rows)==20 and len({x['stream_sha256'] for x in rows})==1
    assert all(x['clean_control_valid'] for x in rows)
    partition=next(x for x in rows if x['fault']=='auditor_partition')
    assert not partition['detected'] and partition['right_censored']
    assert partition['restricted_detection_time']==50
    assert all(x['return_control_bytes']==x['return_raw_bytes']//512*16 for x in rows)
    assert all(x['public_anchor_control_bytes']==x['public_anchor_raw_bytes']//512*16 for x in rows)

