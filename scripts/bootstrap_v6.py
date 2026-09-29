"""Initialize v6 development while preserving shipped v4/v5 sources and evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import yaml

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'reports/v6/bootstrap'
out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[*sorted((ROOT/'airproof').glob('*.py')),*sorted((ROOT/'scripts').glob('*v5*.py')),
       *sorted((ROOT/'configs/v5').glob('*')),*sorted((ROOT/'docs').glob('V5*.md')),
       ROOT/'output/pdf/AirProof_v5_20260907.pdf',ROOT/'output/pdf/AirProof_v4_final_20260903.pdf',
       ROOT/'docs/V6_MAJOR_REVISION_RESEARCH_PLAN.md']
manifest={}
for p in files:
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest);manifest[str(rel)]=sha(p)
(out/'preserved_manifest.json').write_text(json.dumps(manifest,indent=2))
config=yaml.safe_load((ROOT/'configs/v5/protocol.yaml').read_text())
config.update(protocol_version='airproof-v6-development-1',created_utc=datetime.now(timezone.utc).isoformat(),
              role='development specification; confirmation not authorized until all gates pass')
config.pop('deadline_utc',None)
config['historical_v5']='reports/v5/delivery_manifest.json'
config['selection']={'status':'not frozen; no candidate outcomes permitted until bounded family registered',
                     'cap_default':8.0,'history_envelope_relaxation':False,'loss':'Huber',
                     'extra_confirmation_search':False}
config['seed_namespaces']={'status':'pending collision audit; no campaign seeds allocated'}
config['primary']['status']='not launched; awaits development, calibration, validation and power lock'
dest=ROOT/'configs/v6';dest.mkdir(exist_ok=False)
(dest/'protocol.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
phases=['A0','A1','B1','B2','B3','B4','C','D','E','F']
status={'created_utc':config['created_utc'],'plan':'docs/V6_MAJOR_REVISION_RESEARCH_PLAN.md',
        'implementation':'in progress','scientific_goal':'not attained','primary_launched':False,
        'phases':{p:{'status':'in_progress' if p=='A0' else 'pending','artifacts':[]} for p in phases}}
(ROOT/'reports/v6/status.json').write_text(json.dumps(status,indent=2))
print(json.dumps({'preserved_files':len(manifest),'protocol':'configs/v6/protocol.yaml'}))
