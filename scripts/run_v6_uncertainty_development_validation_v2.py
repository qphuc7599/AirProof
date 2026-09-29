"""Run v2 registered uncertainty validation with calibration-frozen event threshold."""
from __future__ import annotations

import hashlib,json
from pathlib import Path
import numpy as np
from airproof.v6_uncertainty import (declared_uncertainty_strata,evaluate_intervals,
                                     fit_frozen_intervals)

ROOT=Path(__file__).resolve().parents[1]
REGISTRATION=ROOT/'configs/v6_uncertainty_development_validation_v2.json'
OUT=ROOT/'reports/v6/uncertainty_development_validation_v2'

def load(seed,cell,stride):
    path=ROOT/f'reports/v6/development_v1/{seed}_{cell}/PUBLIC.npz'
    with np.load(path) as data:
        selected=np.arange(0,data['truth'].shape[1],stride)
        arrays={key:data[key][:,selected] for key in ('live','reconstructed','truth','scales')}
        arrays['cell_groups']=data['cell_groups'][selected]
        arrays['identity']=str(data['identity']);arrays['source_hash']=str(data['source_hash'])
    return path,arrays

def labels(arrays,age,cuts):
    context=np.where(arrays['scales']<=cuts[0],'reference-low',
        np.where(arrays['scales']<=cuts[1],'reference-mid','reference-high'))
    groups=np.broadcast_to(arrays['cell_groups'],arrays['truth'].shape)
    return declared_uncertainty_strata(groups,np.full(groups.shape,age,int),context)

def main():
    registration_bytes=REGISTRATION.read_bytes();registration=json.loads(registration_bytes)
    if registration['registration']!='v6-uncertainty-development-validation-v2':
        raise ValueError('unexpected registration')
    OUT.mkdir(parents=True,exist_ok=True)
    stride=registration['spatial_stride'];loaded={};input_paths=[];input_identities=set()
    for seed in registration['calibration_seeds']:
        path,data=load(seed,registration['calibration_cell'],stride)
        loaded[seed,registration['calibration_cell']]=data;input_paths.append(path)
        input_identities.add(data['identity'])
    calibration_arrays=[loaded[s,registration['calibration_cell']]
                        for s in registration['calibration_seeds']]
    calibration_scales=np.concatenate([a['scales'].ravel() for a in calibration_arrays])
    calibration_truth=np.concatenate([a['truth'].ravel() for a in calibration_arrays])
    cuts=tuple(float(x) for x in np.quantile(calibration_scales,[1/3,2/3]))
    event_threshold=float(np.quantile(calibration_truth,registration['event_quantile']))
    calibrators={}
    for clock,age in registration['clocks'].items():
        prediction=np.concatenate([a[clock] for a in calibration_arrays])
        truth=np.concatenate([a['truth'] for a in calibration_arrays])
        strata=np.concatenate([labels(a,age,cuts) for a in calibration_arrays])
        calibrators[clock]=fit_frozen_intervals(prediction,truth,clock=clock,
            split_id=registration['calibration_split_id'],training_end=registration['calibration_end'],
            target_available_at=registration['calibration_end'],public_strata=strata,
            minimum_stratum_size=registration['minimum_stratum_support'],
            selection_id=registration['selection_id'],selection_end=registration['selection_end'],
            evaluation_split_id=registration['evaluation_split_id'],
            evaluation_start=registration['evaluation_start'])
    results={}
    for cell in registration['cells']:
        worlds=[]
        for seed in registration['evaluation_seeds']:
            path,data=load(seed,cell,stride);input_paths.append(path);worlds.append(data)
            input_identities.add(data['identity'])
        truth=np.concatenate([a['truth'] for a in worlds])
        event=truth>=event_threshold
        results[cell]={}
        for clock,age in registration['clocks'].items():
            prediction=np.concatenate([a[clock] for a in worlds])
            strata=np.concatenate([labels(a,age,cuts) for a in worlds])
            lower,upper=calibrators[clock].predict(prediction,epochs=registration['evaluation_start'],
                evaluation_split_id=registration['evaluation_split_id'],public_strata=strata)
            results[cell][clock]=evaluate_intervals(truth,lower,upper,public_strata=strata,
                event_mask=event,block_length=registration['bootstrap']['block_length'],
                bootstrap_replicates=registration['bootstrap']['replicates'],
                seed=registration['bootstrap']['seed'])
    calibrations={clock:{'clock':model.clock,'count':model.count,
        'pooled_radii':model.pooled_radii,'stratum_counts':model.stratum_counts,
        'fallback_strata':model.fallback_strata,'split_id':model.split_id,
        'selection_id':model.selection_id,'evaluation_split_id':model.evaluation_split_id}
        for clock,model in calibrators.items()}
    disposition=json.loads((ROOT/'reports/v6/development_v1/selection_disposition.json').read_text())
    sources=[REGISTRATION,Path(__file__),ROOT/'airproof/v6_uncertainty.py']
    manifest={'registration_sha256':hashlib.sha256(registration_bytes).hexdigest(),
        'role':registration['role'],'supersedes':registration['supersedes'],
        'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'input_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(input_paths))},
        'input_identities':sorted(input_identities),'reference_context_cuts':cuts,
        'event_threshold':event_threshold,'event_threshold_source':registration['calibration_split_id'],
        'event_threshold_quantile':registration['event_quantile'],
        'citizen_estimator_selected':disposition['selected'],
        'confirmation_authorized':disposition['confirmation_authorized'],
        'physical_cell_equality_reason':'PUBLIC arrays share truth/public backbone per seed; citizen attack and transport mechanisms are absent from this comparator',
        'claim_scope':registration['scientific_limit']}
    (OUT/'registration.json').write_bytes(registration_bytes)
    (OUT/'calibrations.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'calibrations':calibrations,'manifest':manifest},indent=2))

if __name__=='__main__':main()
