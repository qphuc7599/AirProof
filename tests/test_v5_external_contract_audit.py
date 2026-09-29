import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/"reports/v5/tools/audit_campaign_integrity.py"
spec=importlib.util.spec_from_file_location("external_contract_audit",path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def transport_fixture():
    cfg=dict(capacity_bytes_per_direction=1024,control_budget_bytes=128,raw_buffer_bytes=18432,release_buffer_bytes=6144,copy_tokens=4)
    metrics=dict(max_contact_direction_bytes=896,max_control_direction_bytes=128,max_raw_buffer_bytes=18432,
        max_release_buffer_bytes=6144,max_live_copies=4,max_live_tokens=4,token_violations=0,
        control_bytes=64,raw_payload_bytes=512,release_payload_bytes=256,total_wire_bytes=832)
    return metrics,cfg


@pytest.mark.parametrize("field,value",[("max_contact_direction_bytes",1025),("max_control_direction_bytes",129),
    ("max_raw_buffer_bytes",18433),("max_release_buffer_bytes",6145),("max_live_copies",5),
    ("max_live_tokens",5),("token_violations",1),("total_wire_bytes",1),("max_live_tokens",None)])
def test_resource_contract_rejects_retained_violation_or_missing_field(field,value):
    metrics,cfg=transport_fixture()
    audit.audit_transport_contract(metrics,cfg)
    if value is None: del metrics[field]
    else: metrics[field]=value
    with pytest.raises(ValueError): audit.audit_transport_contract(metrics,cfg)


def privacy_fixture(tmp_path):
    mask=np.zeros((672,1),bool);mask[::24]=True
    values=np.full((672,1),np.nan);values[mask]=1.
    arrays={name:values.copy() for name in ("raw","residual","raw_query","residual_query")}
    arrays.update(raw_mask=mask.copy(),residual_mask=mask.copy())
    metadata=dict(scheduled_epochs=list(range(0,672,24)),deadline=24,epsilon_history=8.,privacy_budget_violations=0,
        raw=dict(laplace_variance=2450,released_group_queries=26,scheduled_group_queries=26),
        residual=dict(laplace_variance=98,released_group_queries=26,scheduled_group_queries=26))
    return tmp_path/'release.npz',arrays,metadata


@pytest.mark.parametrize("change",["epsilon","violations","deadline","schedule","variance","missing","mask","dense"])
def test_privacy_contract_rejects_wrong_budget_support_or_dense_leak(tmp_path,change):
    path,arrays,metadata=privacy_fixture(tmp_path)
    np.savez(path,**arrays)
    kwargs=dict(steps=672,groups=1,stage="validation",burn=48)
    audit.audit_privacy_contract(metadata,path,**kwargs)
    if change=="epsilon": metadata['epsilon_history']=8.01
    elif change=="violations": metadata['privacy_budget_violations']=1
    elif change=="deadline": metadata['deadline']=25
    elif change=="schedule": metadata['scheduled_epochs'][-1]=649
    elif change=="variance": metadata['residual']['laplace_variance']=97
    elif change=="missing": del metadata['privacy_budget_violations']
    elif change=="mask": arrays['raw_mask'][1]=True
    elif change=="dense": arrays['residual_query_dense']=np.ones((672,1))
    np.savez(path,**arrays)
    with pytest.raises(ValueError): audit.audit_privacy_contract(metadata,path,**kwargs)
