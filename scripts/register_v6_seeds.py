"""Register disjoint v6 namespaces before any scientific world outcomes."""
from pathlib import Path
import hashlib,json,re
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]

def main():
    target=ROOT/'configs/v6/seed_registry.json'
    if target.exists():raise SystemExit('Registry exists; do not overwrite an exposure record')
    pattern=re.compile(r'(?<![\w.])62[0-9]{5}(?![\w.])')
    matches=[];inventory={}
    for folder in ('configs','reports','results','data'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or p.suffix.lower() not in ('.json','.yaml','.yml','.md','.csv','.jsonl') or p.stat().st_size>20_000_000:continue
            if 'v6' in str(p.relative_to(ROOT)):continue
            raw=p.read_bytes();text=raw.decode('utf-8',errors='replace')
            found=sorted(set(pattern.findall(text)))
            if found:matches.append({'path':str(p.relative_to(ROOT)),'tokens':found})
            if 'seed' in p.name or 'manifest' in p.name:inventory[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    reserved=set(range(6201000,6201008))|set(range(6202000,6202008))|set(range(6203000,6203012))|set(range(6204000,6204100))|set(range(6205000,6205004))
    collisions=[m for m in matches if any(int(x) in reserved for x in m['tokens'])]
    if collisions:raise SystemExit(json.dumps({'collision_candidates':collisions}))
    roles={'development':list(range(6201000,6201008)),
           'calibration':list(range(6202000,6202008)),
           'validation':list(range(6203000,6203012)),
           'confirmation_reserved':list(range(6204000,6204100)),
           'descriptive_stress':list(range(6205000,6205004))}
    out={'created_utc':datetime.now(timezone.utc).isoformat(),'namespaces':roles,
         'exposed_engineering_fixtures':[6006001],'consumed':[],
         'audit':{'roots':['configs','reports','results','data'],'token_range':'6200000..6299999',
                  'collision_candidates':collisions,'other_integer_tokens':matches,'manifest_hashes':inventory,
                  'limits':'text artifacts up to20MB; v6 bootstrap excluded; external unrecorded runs cannot be established'},
         'confirmation':'reserved only; must pass source/candidate/validation/power gates before use'}
    target.write_text(json.dumps(out,indent=2));print(target)
if __name__=='__main__':main()
