"""Read-only progress snapshots; never launches or retries remote work."""
from pathlib import Path
import datetime,json,os,subprocess,time
ROOT=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import json,datetime
r=Path('/data/sources/minhao/runs/dflash_krc_event_20261027')
x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name,path in [('campaign','campaign_001/STATUS.json'),('evaluation','campaign_001/evaluation/STATUS.json'),('first_window','campaign_001/FIRST_WINDOW_REPRODUCTION.json')]:
 p=r/path
 if p.exists():x[name]=json.loads(p.read_text())
ws=sorted((r/'campaign_001/correction').glob('window_*.json'))
x['windows']=len(ws)
if ws:
 s=json.loads(ws[-1].read_text())['summary']
 x['last_window']={k:s[k] for k in ('step','status','correction','integrity_s','window_wall_s')}
if (r/'EXIT_CODE').exists():
 x['exit_code']=int((r/'EXIT_CODE').read_text())
 x['tail']='\n'.join((r/'launch.log').read_text().splitlines()[-12:])
print(json.dumps(x))
'''
env=dict(os.environ,SSH_AUTH_SOCK='/tmp/ssh-XXXXXXymIMtl/agent.60580')
command=['ssh','-o','ConnectTimeout=45','A800','source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh || exit 1; /data/conda_envs/minhao/dflash_specforge_20261022/bin/python -']
for _ in range(370):
 p=subprocess.run(command,input=REMOTE,text=True,capture_output=True,env=env,timeout=65)
 try:
  x=json.loads(p.stdout)
 except ValueError:
  x={'monitor_error':p.stderr[-600:],'returncode':p.returncode,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 with (ROOT/'MONITOR.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 compact={k:x[k] for k in ('utc','windows','exit_code','monitor_error') if k in x}
 compact['phase']=x.get('campaign',{}).get('phase')
 e=x.get('evaluation',{})
 compact.update({k:e[k] for k in ('returned_generations','valid_generations','invalid_generations','qualification') if k in e})
 print(json.dumps(compact),flush=True)
 if 'exit_code' in x:break
 time.sleep(30)
