"""Fail-closed prospective confirmation lock; never manufactures passed gates."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import norm,nct,t

REQUIRED=('complete_development','candidate_frozen','independent_calibration',
 'validation_H1_H7','archive_margin','public_calibrated_value','event_preservation',
 'transport_targets','privacy_accounting','full_audit_return','architecture_baselines',
 'correctness','source_frozen','data_usage_audited')

def prospective_lock(packet):
    gates=packet.get('gates',{});missing=[key for key in REQUIRED if gates.get(key,{}).get('passed') is not True or not gates[key].get('evidence_sha256')]
    for key in REQUIRED:
        entry=gates.get(key,{})
        if entry.get('passed') is True:
            path=Path(entry.get('artifact',''))
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry.get('evidence_sha256'):
                if key not in missing:missing.append(key)
    result={'status':'blocked','sample_size':None,'missing_or_failed_gates':missing,
            'targets_changed':False,'primary_launched':False}
    if missing:return result
    contrasts=np.asarray(packet.get('validation_contrasts'),float)
    if contrasts.ndim!=2 or contrasts.shape[0]<12 or contrasts.shape[1]!=7 or not np.isfinite(contrasts).all():
        result['reason']='twelve or more complete paired-world H1-H7 validation rows required';return result
    means=contrasts.mean(0);sd=contrasts.std(0,ddof=1)
    if np.any(means>=0) or np.any(sd<=0):
        result['reason']='prospective effects unsupported or variance degenerate';return result
    alpha=.05/7;selected=None
    for n in range(30,101):
        power=nct.cdf(t.ppf(alpha,n-1),n-1,means*np.sqrt(n)/sd)
        if np.all(power>=.9):selected=n;break
    result['prospective_means']=means.tolist();result['prospective_sd']=sd.tolist()
    if selected is None:
        result['reason']='no n in30..100 supports90percent power for every registered contrast';return result
    if packet.get('runtime_seconds_per_world',0)<=0 or packet.get('workers',0)<1:
        result['reason']='measured positive runtime and worker count required';return result
    result.update(status='locked',sample_size=selected,prospective_power=power.tolist(),
                  runtime_seconds_with_1_5_factor=1.5*selected*packet['runtime_seconds_per_world']/packet['workers'],
                  packet_sha256=hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest())
    return result
