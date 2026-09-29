import numpy as np
from airproof.v6_confirmation import prospective_lock,REQUIRED

def test_confirmation_requires_all_evidence(tmp_path):
    assert prospective_lock({})['sample_size'] is None
    import hashlib
    proof=tmp_path/'evidence.json';proof.write_text('{}')
    digest=hashlib.sha256(proof.read_bytes()).hexdigest()
    packet={'gates':{k:{'passed':True,'evidence_sha256':digest,'artifact':str(proof)} for k in REQUIRED},
      'validation_contrasts':np.random.default_rng(4).normal(-2,.1,(12,7)).tolist(),
      'runtime_seconds_per_world':100,'workers':4}
    assert prospective_lock(packet)['sample_size']==30
    packet['gates']['archive_margin']['passed']=False
    assert prospective_lock(packet)['status']=='blocked'


def test_fabricated_evidence_hash_does_not_unlock():
    packet={'gates':{k:{'passed':True,'evidence_sha256':'made-up','artifact':'absent.json'} for k in REQUIRED}}
    assert prospective_lock(packet)['status']=='blocked'
