import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_accountability_manuscript_and_ledgers_match_corrected_artifacts():
    ledger=json.loads((ROOT/'reports/v6/ledger_transparency_same_stream_v2/analysis.json').read_text())
    public=json.loads((ROOT/'reports/v6/durable_public_checkpoint_v2/results.json').read_text())
    assert ledger['rows']==8000 and public['publicly_checkpointed']==99
    partition=[g for g in ledger['groups'] if g['fault']=='auditor_partition']
    assert sum(g['detected'] for g in partition)==0
    assert sum(g['right_censored'] for g in partition)==400
    signed=next(g for g in ledger['groups'] if g['batch_size']==2048 and
        g['mode']=='signed_log' and g['fault']=='mutation')
    transparent=next(g for g in ledger['groups'] if g['batch_size']==2048 and
        g['mode']=='transparency_signed_comparison' and g['fault']=='mutation')
    assert signed['mean_storage_bytes']==2203069.4
    assert (signed['mean_return_raw_bytes'],signed['mean_return_control_bytes'])==(123392.0,3856.0)
    assert transparent['mean_storage_bytes']==3245706.2 and transparent['mean_proof_bytes']==720896.0
    assert (transparent['mean_return_raw_bytes'],transparent['mean_return_control_bytes'])==(2560.0,80.0)
    detailed=[ROOT/'source_paper/AirProof_Elsevier/generated/v6_accountability_late_admission.tex',
        ROOT/'source_paper/AirProof_Elsevier/v6_results_generated.tex',ROOT/'docs/CLAIM_LEDGER.md',
        ROOT/'docs/V6_TOP_TIER_REVIEWER_REPORT_20260908.md',ROOT/'docs/V6_IMPLEMENTATION_MEMORY.md']
    for path in detailed:
        text=path.read_text(encoding='utf-8')
        assert '0/400' in text and '2,203,069.4' in text and '720,896' in text
        assert 'historical' in text.lower() and 'invalid' in text.lower()
    paper=(ROOT/'source_paper/AirProof_Elsevier/paper_v6.tex').read_text(encoding='utf-8')
    checkpoint=(ROOT/'docs/V6_REVIEWER_CHECKPOINT.md').read_text(encoding='utf-8')
    assert '0/400' in paper and '0/400' in checkpoint
    assert '99' in paper and '99' in checkpoint
