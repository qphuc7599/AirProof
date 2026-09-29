import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_v5_campaign import mock_rows,write_outcomes
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('integrated_export',ROOT/'reports/v5/tools/export_integrated_results.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def fixture(tmp_path):
    campaign=tmp_path/'campaign';seeds=list(range(913000,913012))
    rows=mock_rows(seeds,'validation')
    for row in rows:
        row.update(execution_identity='unit',source_hash='source',config_hash='config')
        row['metrics'].update(live_rmse=2.,live_worst_group_rmse=3.)
        row['strata']={f'stratum{i}':dict(cells=0 if i==0 else 8,supported=i!=0,
            live_rmse=None if i==0 else 2.,rmse=None if i==0 else 1.) for i in range(39)}
    outcome=write_outcomes(campaign,rows,'validation',seeds)
    analysis=tmp_path/'analysis.json'
    content=dict(stage=['validation'],source_outcomes_sha256=hashlib.sha256(outcome.read_bytes()).hexdigest(),
        hypotheses={f'H{i}':dict(n=12,valid=False,mean_contrast=0.,holm_p=1.,passed=False) for i in range(1,8)})
    analysis.write_text(json.dumps(content))
    return campaign,analysis,tmp_path/'export',tmp_path/'section.tex'


def test_complete_export_retains_worlds_all_factorials_and_unsupported_strata(tmp_path):
    args=fixture(tmp_path);manifest=tool.export(*args)
    assert manifest['evaluations']==384 and not manifest['primary_claim']
    expected={'per_world_metrics':384,'main_method_means':20,'factorial_per_world':192,
              'factorial_means':16,'all_strata_per_world':384*39,'paired_ap_nf_strata':12*2*39,'hypotheses':7}
    for name,count in expected.items():
        with (args[2]/(name+'.csv')).open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==count
        if name=='paired_ap_nf_strata':
            assert sum(r['ap_supported']=='False' for r in rows)==24
            assert all(r['ap_minus_nf_rmse']=='' for r in rows if r['ap_supported']=='False')
    assert 'descriptive only' in args[3].read_text()


def test_analysis_hash_mismatch_blocks_before_outputs(tmp_path):
    args=fixture(tmp_path);content=json.loads(args[1].read_text());content['source_outcomes_sha256']='wrong'
    args[1].write_text(json.dumps(content))
    with pytest.raises(ValueError,match='binding'):tool.export(*args)
    assert not args[2].exists() and not args[3].exists()


def test_missing_stratum_blocks_before_outputs(tmp_path):
    args=fixture(tmp_path);path=args[0]/'outcomes.json';content=json.loads(path.read_text())
    del content['rows'][0]['strata']['stratum0'];path.write_text(json.dumps(content))
    analysis=json.loads(args[1].read_text());analysis['source_outcomes_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    args[1].write_text(json.dumps(analysis))
    with pytest.raises(ValueError,match='39'):tool.export(*args)
    assert not args[2].exists()
