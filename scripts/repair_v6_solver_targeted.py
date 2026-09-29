"""Target only two failed numerical arms, preserve objective/seed/tolerance/data."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from dataclasses import replace
import json,hashlib,time
import numpy as np
from airproof.config import load_config
from airproof.simulator import generate_world
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_experiment import integrated_transport
from airproof.v6_numerical_experiment import prepare_public
from airproof.v6_estimator import EstimatorConfig,estimate_public_field
from airproof.meteorology import grid_coordinates
from airproof.field import grid_laplacian
old=Path('reports/v6/development_v1/6201001_severe_hotspot')
historical=json.loads((old/'result.json').read_text());cfg=historical['config'];seed=6201001
out=Path('reports/v6/targeted_solver_repair');out.mkdir(exist_ok=False)
manifest={'role':'numerical repair of two failed development arms, not confirmation or new candidate',
 'original_contribution':'well-defined constrained Huber twin','existing_artifact':str(old),
 'seed':seed,'cell':'severe_hotspot','methods':['HUBER_reg0.1_cap8','SQ_reg0.1'],
 'change':'at most2 same-objective L-BFGS restarts with zero function-tolerance stopping; KKT tolerance unchanged',
 'tolerance':1e-7,'cap':8,'regularization_multiplier':.1,
 'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('airproof/v6_estimator.py'),Path('airproof/v6_solver_refinement.py')]}}
(out/'registration.json').write_text(json.dumps(manifest,indent=2))
start=time.perf_counter();world=generate_world(cfg,seed);trace,_=coupled_public_trace(cfg,seed,world.observations)
transport,collector=integrated_transport(trace,world.observations,cfg)
assert transport.metrics['trace_hash']==historical['transport']['trace_hash']
public,ops,scales,provenance=prepare_public(world,cfg);burn=48
records=[replace(r,epoch=r.epoch-burn) for r in collector.selected if r.epoch>=burn]
arrivals={r.nullifier:collector.selection_times[r.nullifier]-burn for r in records}
rows=[]
for method in manifest['methods']:
 robust=method.startswith('HUBER');config=EstimatorConfig(lambda_zero=.05,lambda_temporal=.05,lambda_spatial=.02,
  loss='huber' if robust else 'quadratic',output_cap=robust,numerical_refinements=2)
 result=estimate_public_field(public[burn:],grid_coordinates(32),records,config,innovation_scales=scales[burn:],
  transitions=ops[burn:],spatial_laplacian=grid_laplacian(32),arrival_map=arrivals)
 with np.load(old/(method+'.npz')) as z:
  delta=float(np.max(abs(result.reconstructed[:624]-z['reconstructed'])))
 rmse=float(np.sqrt(np.mean((result.reconstructed[:624]-world.truth[burn:])**2)))
 np.savez_compressed(out/(method+'.npz'),live=result.live,reconstructed=result.reconstructed)
 rows.append({'method':method,'rmse':rmse,'max_prediction_change':delta,'solver_failure_rate':result.diagnostics['solver_failure_rate'],
 'refined_epochs':[t for t in result.diagnostics['epoch_terms'] if len(t['refinement_history'])>1]})
(out/'result.json').write_text(json.dumps({'rows':rows,'elapsed_seconds':time.perf_counter()-start},indent=2));print(json.dumps(rows))
