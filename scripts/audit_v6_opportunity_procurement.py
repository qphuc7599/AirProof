"""Static prerequisite audit; never generates a world or opens a seed."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path

from airproof.simulator import SyntheticWorld, generate_world
from airproof.config import load_config


ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    design=ROOT/'configs/v6/fairness_opportunity_procurement_design.json'
    certificate=ROOT/'reports/v6/fairness_ceiling_diagnostic/certificate.json'
    registry=ROOT/'configs/v6/seed_registry.json'
    base_path=ROOT/'reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml'
    base=load_config(base_path)
    cfg=json.loads(design.read_text());cert=json.loads(certificate.read_text())
    registered=json.loads(registry.read_text())['namespaces']
    official={seed for values in registered.values() for seed in values}
    proposed={seed for values in cfg['provisional_collision_free_namespace'].values()
              if isinstance(values,list) for seed in values}
    world_fields={item.name for item in fields(SyntheticWorld)}
    source=inspect.getsource(generate_world)
    missing={
        'eligible_agent_registry': 'eligible_agents' not in world_fields,
        'latent_opt_in_response': 'invitation_response' not in world_fields,
        'counterfactual_measurement_kernel': 'potential_observations' not in world_fields,
        'sensing_energy_ledger': 'sensing_energy' not in world_fields,
        'numerical_invitation_reservation': cfg['invitation_control_reservation_bytes'] is None,
        'numerical_sensing_budget': cfg['sensing_energy_budget_units'] is None,
        'public_spatial_uncertainty_at_decision': 'public_uncertainty' not in world_fields,
    }
    result={
        'role':'static simulator prerequisite audit; no world generated and no new policy outcome read',
        'status':'blocked' if any(missing.values()) else 'ready_for_smoke',
        'population_contract':cfg['population'],
        'configured_participation_propensity':{
            'group_0':base['world']['participation_rate']/base['world']['participation_skew'],
            'groups_1_to_3':base['world']['participation_rate'],
            'interpretation':'This proves a participation-skew mechanism exists, but nonparticipation is not modeled as invitation eligibility or latent consent.'},
        'synthetic_world_fields':sorted(world_fields),
        'source_facts':{
            'latent_paths_created_then_discarded': 'paths = _random_walks' in source,
            'participation_draw_immediately_filters_observation': 'streams["participation"].random()' in source,
            'measurement_reads_truth_only_after_participation': source.index('streams["participation"].random()') < source.index('value = float(truth[epoch, cell]'),
            'returned_world_contains_only_realized_observations': 'tuple(observations)' in source,
        },
        'missing_prerequisites':missing,
        'official_seed_collision':sorted(proposed & official),
        'provisional_seeds_opened':False,
        'confirmation_6204000_opened':False,
        'required_mean_additional_origins':cert['H6']['mean_additional_origin_pairs_required'],
        'eligible_dormant_origins_available':None,
        'can_supply_required_origins':None,
        'reason':'The simulator does not expose an eligible dormant pool, opt-in response, counterfactual sensing kernel, or sensing-energy ledger. Counting or generating invited observations would require hidden truth/RNG reconstruction and would fabricate the policy effect.',
        'required_simulator_extension':[
            'Return a policy-independent per-epoch eligibility envelope for all 1000 agents, containing current public cell/group and invitation reachability.',
            'Draw opt-in response and sensing availability before policy choice, keep them hidden until invitation, and couple every policy to the same latent draws.',
            'Charge each invitation to a fixed reservation within the existing 128-byte control direction budget and each accepted sensing action to a fixed common energy ledger.',
            'Generate a value from the existing measurement kernel only after a selected invitation accepts; never expose value or error before the decision.',
            'Expose a prefix-only contact probability and a public spatial uncertainty map with provenance independent of evaluation truth.'
        ],
        'sha256':{'simulator':sha(ROOT/'airproof/simulator.py'),'design':sha(design),
                  'ceiling_certificate':sha(certificate),'seed_registry':sha(registry),
                  'base_configuration':sha(base_path)}
    }
    out=ROOT/'reports/v6/fairness_opportunity_procurement_design';out.mkdir(parents=True,exist_ok=True)
    audit=out/'prerequisite_audit.json';audit.write_text(json.dumps(result,indent=2)+'\n')
    sources={path:sha(ROOT/path) for path in (
        'airproof/v6_opportunity_procurement.py','scripts/audit_v6_opportunity_procurement.py',
        'configs/v6/fairness_opportunity_procurement_design.json',
        'tests/test_v6_opportunity_procurement.py')}
    document=ROOT/'docs/V6_FAIRNESS_OPPORTUNITY_PROCUREMENT_DESIGN.md'
    manifest={'role':result['role'],'status':result['status'],'audit_sha256':sha(audit),
        'source_sha256':sources,'document_sha256':sha(document),
        'worlds_generated':0,'new_policy_outcomes_read':0,
        'exposed_ceiling_certificate_read':True,'provisional_seeds_opened':False,
        'confirmation_6204000_opened':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
