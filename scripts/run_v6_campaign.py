"""Single source-bound controller; confirmation refuses unmet prerequisite gates."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,hashlib,json,time,traceback,shutil
import numpy as np
import psutil
from airproof.config import load_config,config_hash
from airproof.simulator import generate_world
from airproof.v5_experiment import cell_configuration,release_evidence
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_experiment import integrated_transport
from airproof.v6_numerical_experiment import prepare_public,evaluate

ROOT=Path(__file__).resolve().parents[1]

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def sources():
    paths=sorted([*ROOT.glob('airproof/*.py'),*ROOT.glob('scripts/*v6*.py'),*ROOT.glob('configs/v6/*.json'),ROOT/'configs/v6/protocol.yaml'])
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def job(spec):
    seed,cell,role,out,source_hash=spec;out=Path(out);start=time.perf_counter()
    cfg=load_config(ROOT/'reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml')
    protocol=load_config(ROOT/'configs/v6/protocol.yaml');cfg['transport']=protocol['transport']|{'drain_epochs':24}
    cfg=cell_configuration(cfg,cell)
    identity=config_hash({'seed':seed,'cell':cell,'role':role,'source':source_hash,'config':cfg})
    target=out/f'{seed}_{cell}';target.mkdir(parents=True,exist_ok=True)
    existing=target/'result.json'
    if existing.exists():
        previous=json.loads(existing.read_text())
        if previous['identity']!=identity:raise ValueError('source/config mismatch on resume')
        return previous
    rows=[]
    try:
        world=generate_world(cfg,seed);trace,_=coupled_public_trace(cfg,seed,world.observations)
        transport,collector=integrated_transport(trace,world.observations,cfg)
        prepared=prepare_public(world,cfg)
        family=json.loads((ROOT/'configs/v6/selection_family.json').read_text())
        specs=[('PUBLIC',1.,False,True)]
        for c in family['candidates']:
            specs.extend([(c['id'],c['regularization_multiplier'],True,False),
                          ('SQ_reg'+str(c['regularization_multiplier']),c['regularization_multiplier'],False,False)])
        if role=='engineering_pilot':specs=[specs[0],specs[1],specs[2]]
        for method,multiplier,robust,public_only in specs:
            row,arrays=evaluate(world,cfg,prepared,collector,method,multiplier,robust=robust,public_only=public_only)
            row.update(seed=seed,cell=cell,role=role,identity=identity);rows.append(row)
            np.savez_compressed(target/(method+'.npz'),**arrays,identity=identity,source_hash=source_hash)
            write(target/(method+'.json'),row)
            print(json.dumps({'seed':seed,'cell':cell,'method':method,'seconds':row['elapsed_seconds']}),flush=True)
        release_summary,release_arrays=release_evidence(world,prepared[0][:len(world.truth)],dict(transport.release_arrivals),cfg)
        np.savez_compressed(target/'releases.npz',**release_arrays)
        tr={k:v for k,v in transport.metrics.items() if k!='local_decisions'}
        write(target/'transport.json',tr);write(target/'allocations.json',collector.allocations)
        record={'status':'complete','identity':identity,'seed':seed,'cell':cell,'role':role,'source_hash':source_hash,
                'config':cfg,'rows':rows,'release':release_summary,'transport':tr,
                'runtime_seconds':time.perf_counter()-start,'rss_bytes':psutil.Process().memory_info().rss,
                'peak_wset_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None)}
    except Exception:
        record={'status':'failed','identity':identity,'seed':seed,'cell':cell,'role':role,
                'source_hash':source_hash,'rows':rows,'error':traceback.format_exc(),'runtime_seconds':time.perf_counter()-start}
    write(existing,record);return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['engineering_pilot','development','confirmation'],required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.role=='confirmation':raise SystemExit('Confirmation not launched: requires completed validation, frozen candidate, scientific prerequisites and prospective power lock.')
    if not 1<=args.workers<=8:raise SystemExit('workers must be1..8')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);inventory=sources();digest=config_hash(inventory)
    registry=json.loads((ROOT/'configs/v6/seed_registry.json').read_text())
    seeds=[6006001] if args.role=='engineering_pilot' else registry['namespaces']['development']
    cells=['severe_clean'] if args.role=='engineering_pilot' else json.loads((ROOT/'configs/v6/selection_family.json').read_text())['cells']
    matrix=[(seed,cell,args.role,str(out),digest) for seed in seeds for cell in cells]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'role':args.role,'source_hash':digest,'source_files':inventory,
              'matrix':[{'seed':s,'cell':c} for s,c,*_ in matrix],'expected_jobs':len(matrix),'workers':args.workers,
              'scope':'changed v6 common execution; development not independent confirmation'}
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['source_hash']!=digest:raise SystemExit('Changed source: use a new labeled campaign directory, preserve old evidence')
    if not mp.exists():
        write(mp,manifest)
        for relative in inventory:
            dest=out/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
    if psutil.virtual_memory().available<3*1024**3:raise SystemExit('Less than3GB free RAM')
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job,spec) for spec in matrix]):
            results.append(future.result());write(out/'progress.json',{'finished':len(results),'expected':len(matrix),'failures':sum(r['status']=='failed' for r in results)})
    write(out/'complete_matrix.json',{'expected':len(matrix),'completed':len(results),'failed':sum(r['status']=='failed' for r in results),'source_hash':digest,
                                    'jobs':[{'seed':r['seed'],'cell':r['cell'],'status':r['status']} for r in results]})
if __name__=='__main__':main()
