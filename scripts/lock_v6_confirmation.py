"""Current confirmation disposition, with retained adverse evidence references."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from airproof.v6_confirmation import REQUIRED,prospective_lock
root=Path('reports/v6');gate={k:{'passed':False,'reason':'not yet independently established','evidence_sha256':None} for k in REQUIRED}
for key,relative,reason in [
 ('archive_margin','archive_cap_diagnostic/result.json','Frozen-center cap floor exceeds historical control margin; new archive estimator confirmation absent.'),
 ('data_usage_audited','archive_usage/epa_freshness.json','Known EPA archive has no remaining336h window with4 unexposed supported stations.'),
 ('public_calibrated_value','release_historical_reanalysis/summary.json','Every P5-P0 descriptive interval contains zero on already-exposed histories.'),
 ('full_audit_return','numerical_fullscale_fixture/6006001_severe_clean/transport.json','Only8929 of194675 acceptance receipts returned; terminal common execution incomplete.')]:
 p=root/relative;gate[key]={'passed':False,'reason':reason,'artifact':str(p),'evidence_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
packet={'created_utc':datetime.now(timezone.utc).isoformat(),'gates':gate,
 'role':'current evidence disposition; no confirmatory samples consumed','validation_contrasts':None}
result=prospective_lock(packet)
target=root/'confirmation';target.mkdir(exist_ok=True)
(target/'gate_packet.json').write_text(json.dumps(packet,indent=2));(target/'lock.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
