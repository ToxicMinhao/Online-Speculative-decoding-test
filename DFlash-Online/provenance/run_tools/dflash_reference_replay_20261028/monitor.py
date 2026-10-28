"""Small read-only progress view; no generation, retries or artifact transfers."""
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]); out=root/'campaign_001'
result={}
for name in ('PID','STARTED.utc','FINISHED.utc','EXIT_CODE'):
    p=root/name
    if p.exists(): result[name]=p.read_text().strip()
for name in ('STATUS.json','FIRST_A_UPDATE.json','FIRST_WINDOW.json'):
    p=out/name
    if p.exists():
        value=json.loads(p.read_text());result[name]={k:v for k,v in value.items() if k not in {'evaluation','traceback','parameter_comparisons','responses','rows'}}
for p in out.rglob('REPLAY_PROGRESS.json') if out.exists() else []:
    v=json.loads(p.read_text());result[str(p.relative_to(out))]={k:v[k] for k in ('status','global_step','presentations','active_update','counts') if k in v}
for p in out.glob('evaluation/*/STATUS.json') if out.exists() else []:
    result[str(p.relative_to(out))]=json.loads(p.read_text())
for p in out.rglob('MEASUREMENTS.jsonl') if out.exists() else []:
    result[str(p.relative_to(out))]={'executions':sum(1 for _ in p.open())}
print(json.dumps(result,indent=2))
