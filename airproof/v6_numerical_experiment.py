"""Prepared numerical v6 jobs; no scale fitting on evaluation truth."""
from dataclasses import replace,asdict
import time
import numpy as np
from .experiment import _v4_public_components
from .meteorology import grid_coordinates
from .field import grid_laplacian
from .v6_calibration import frozen_backbone_scale_context
from .v6_estimator import EstimatorConfig,estimate_public_field
from .metrics import prediction_metrics,coverage_metrics


def prepare_public(world,cfg):
    public,operators,diagnostics=_v4_public_components(world,cfg)
    burn=cfg['world']['burn_in_steps'];steps=len(public);lag=cfg['twin']['fixed_lag']
    tail=[];last=public[-1].copy()
    for _ in range(lag):
        last=last.copy() if operators is None else np.maximum(operators[-1]@last,0)
        tail.append(last.copy())
    extended=np.concatenate((public,np.asarray(tail))) if lag else public
    extended_ops=None if operators is None else list(operators)+[operators[-1]]*lag
    scales,scale_provenance=frozen_backbone_scale_context(world.reference_observations,
        side=cfg['world']['grid_side'],calibration_steps=burn,horizon=steps,
        selected=diagnostics['selected'],transitions=operators,
        training_split_id=f'public-prefix-{world.seed}',public_model_id='v4-frozen-backbone',
        initial=cfg['world']['baseline'],smoothing=cfg['twin'].get('reference_smoothing',.001))
    if lag:scales=np.concatenate((scales,np.repeat(scales[-1:],lag,axis=0)))
    scale_provenance['drain_scale']='last scale placeholder; no drain-acquired observations'
    return extended,extended_ops,scales,{'public':diagnostics,'scale':scale_provenance}


def evaluate(world,cfg,prepared,collector,method,multiplier=1.,*,robust=True,public_only=False):
    start=time.perf_counter();public,operators,scales,provenance=prepared
    burn=cfg['world']['burn_in_steps'];steps=len(world.truth);side=cfg['world']['grid_side']
    scored=[r for r in collector.selected if r.epoch>=burn]
    # Calibration prefix is not fed into the solver with subsequently fitted scales.
    records=[replace(r,epoch=r.epoch-burn) for r in scored]
    arrivals={r.nullifier:collector.selection_times[r.nullifier]-burn for r in scored}
    config=EstimatorConfig(lambda_zero=.5*multiplier,lambda_temporal=.5*multiplier,
        lambda_spatial=.2*multiplier,loss='huber' if robust else 'quadratic',output_cap=robust)
    if public_only:
        live=reconstructed=public[burn:steps];diagnostics={'solver_failure_rate':0.,'public_only':True}
    else:
        result=estimate_public_field(public[burn:],grid_coordinates(side),records,config,
            innovation_scales=scales[burn:],transitions=None if operators is None else operators[burn:],
            spatial_laplacian=grid_laplacian(side),arrival_map=arrivals)
        live=result.live[:steps-burn];reconstructed=result.reconstructed[:steps-burn];diagnostics=result.diagnostics
    truth=world.truth[burn:];threshold=float(np.quantile(world.truth[:burn],.95));events=truth>=threshold
    recall=float(np.mean(live[events]>=threshold)) if events.any() else None
    allocations=[a for a in collector.allocations if burn<=a['epoch']<steps]
    metrics={**prediction_metrics(truth,reconstructed,world.cell_groups),
        **{'live_'+k:v for k,v in prediction_metrics(truth,live,world.cell_groups).items()},
        **coverage_metrics(scored,cfg['world']['groups']),
        'event_recall':recall,'event_support':int(events.sum()),'event_threshold':threshold,
        'solver_failure_rate':diagnostics['solver_failure_rate'],
        'selected_records':len(scored),'feasible_epochs':sum(a['feasible'] for a in allocations),
        'mean_avoidable_deficit':float(np.mean([max(a['avoidable'].values()) for a in allocations])),
        'mean_unavoidable_deficit':float(np.mean([max(a['unavoidable'].values()) for a in allocations]))}
    return {'method':method,'estimator':asdict(config) if not public_only else None,
        'metrics':metrics,'diagnostics':{k:v for k,v in diagnostics.items() if k!='epoch_terms'},
        'public_provenance':provenance,'elapsed_seconds':time.perf_counter()-start},\
        {'live':live,'reconstructed':reconstructed,'truth':truth,'public':public[burn:steps],
         'cell_groups':world.cell_groups,'scales':scales[burn:steps]}
