#!/usr/bin/env python3
"""Independent, metadata-only completion audit for BASE/TAIL58 (no Torch/GPU).

Read-only input inspection. Production reducers/validators are never imported.
Tensor retention uses a separately supplied final live identity inventory; this
script does not deserialize/recompute weights, Adam state or the objective.
Shared implementation authorship/context is disclosed in the output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile

ARMS=('BASE','TAIL')
EVENTS=58
FROZEN_FORMULAS={
 'online_feedback/objective.py':{'bytes':36960,'sha256':'031cdb176e6d367549891a0966abcb6002dd9a6c8976746caee081b4b56dc87c'},
 'online_feedback/kplus.py':{'bytes':6292,'sha256':'218faa074d42676905da26dc4c49e835e4d6e19cd324409f7e9ea07357730872'},
 'online_feedback/prefix_allocation.py':{'bytes':9738,'sha256':'0d967994c6805f04941b67468464d73df310b49966435202e329b4908a32ba4e'},
 'online_feedback/repair_conditional_kl.py':{'bytes':6836,'sha256':'c44f116ce2fe0a6ca808e8f2eca5537624fe98c3fc1804e73ed1df766c8f204e'},
 'online_feedback/continuation_repair.py':{'bytes':10796,'sha256':'da11ae9a13da557dbbe983ee678a8c21c9064ca3645e97faf6571cf2938ef6da'},
}
LOCAL_TARGETS=dict(online_generations=7424,correctness_generations=4,
 terminal_preparation_forwards=7424,response_forwards=14848,response_backwards=14848,optimizer_steps=7424)
MAIN_TARGETS=dict(feedback_generation=14848,teacher_preparation=14848,response_forward=29696,
 response_backward=29696,adam=14848,correctness_generation=8,canonical_generation=1024,
 scored_generation=12288,warmup_generation=96)
LIMITS=dict(elapsed_s=259200,occupied_card_s=432000,new_allocated_bytes=90_000_000_000,free_reserve_bytes=400_000_000_000)


def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def identity(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
 return dict(bytes=Path(path).stat().st_size,sha256=h.hexdigest())


class MissingEvidence(Exception):pass


class Audit:
 def __init__(self):self.checked=0;self.failures=[];self.missing=[];self.private=[];self.totals={}
 def require(self,value,label):
  self.checked+=1
  if not value:raise AssertionError(label)
 def section(self,label,action):
  try:return action()
  except MissingEvidence as error:
   self.missing.append(label);self.private.append(dict(section=label,missing=str(error)))
  except (AssertionError,KeyError,ValueError,TypeError,OSError,IndexError) as error:
   self.failures.append(label);self.private.append(dict(section=label,error=repr(error)))
 def result(self):
  return dict(schema='independent_fullrun_completion_audit_v1',
   status='FAILED' if self.failures else 'INCOMPLETE' if self.missing else 'PASSED',
   checks=self.checked,failed_sections=self.failures,missing_sections=self.missing,totals=self.totals,
   independence='Separate CPU receipt reducer; shared authorship of training/qualification/witness and shared task context disclosed',
   limitations=['Metadata does not independently recompute model/optimizer tensors or full-vocabulary gradients.',
    'Final live tensor inventory is a separately authenticated input, not a new tensor read by this audit.',
    'Observed resource receipts bound recorded checkpoints; they are not a continuously sampled hardware peak.',
    'Serving token/statistical re-reduction belongs to its separate independent audit.'])


class Evidence:
 def __init__(self,root,*,external=None,zstd='zstd'):
  self.root=Path(root).resolve();self.external=external or {};self.zstd=zstd;self.inputs={};self.archives={}
 def path(self,name):
  path=Path(name)
  if not path.is_absolute():return self.root/path
  if str(path) in self.external:return Path(self.external[str(path)])
  if hasattr(self,'original_root'):
   try:return self.root/path.relative_to(self.original_root)
   except ValueError:pass
  if path.is_relative_to(self.root):return path
  raise MissingEvidence('External file needs explicit mapping: '+str(path))
 def raw(self,name):
  path=self.path(name)
  if not path.is_file():raise MissingEvidence(str(path))
  self.inputs[str(path)]=identity(path)
  return path.read_bytes()
 def json(self,name):return json.loads(self.raw(name))
 def jsonl(self,name):return [json.loads(line) for line in self.raw(name).splitlines() if line.strip()]
 def archive(self,arm,event):
  key=(arm,event)
  if key in self.archives:return self.archives[key]
  name=f'compact/{arm}/event_{event:03d}.tar.zst';path=self.path(name)
  index=self.json(name+'.index.json')
  if index['binding_digest']!=digest({k:v for k,v in index.items() if k!='binding_digest'}):
   raise ValueError('Archive index binding changed')
  if not path.is_file():raise MissingEvidence(str(path))
  actual=identity(path);self.inputs[str(path)]=actual
  if actual!={'bytes':index['archive_bytes'],'sha256':index['archive_sha256']}:
   raise ValueError('Archive bytes differ')
  keep={};seen=set();expected=index['members']
  process=subprocess.Popen([self.zstd,'-q','-d','-c',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  try:
   with tarfile.open(fileobj=process.stdout,mode='r|',ignore_zeros=True) as stream:
    for member in stream:
     if not member.isfile() or member.name not in expected or member.name in seen:
      raise ValueError('Unexpected/duplicate archive member')
     seen.add(member.name);h=hashlib.sha256();parts=[]
     wanted=member.name.endswith(('/RESULT.json','/OLDER_BOUNDARY_RETIREMENT.json','/RETIREMENT.json','/RESOURCE_COSTS.json','/ADMISSION.json'))
     with stream.extractfile(member) as payload:
      for chunk in iter(lambda:payload.read(1024*1024),b''):
       h.update(chunk)
       if wanted:parts.append(chunk)
     if member.size!=expected[member.name]['bytes'] or h.hexdigest()!=expected[member.name]['sha256']:
      raise ValueError('Archive member identity changed')
     if wanted:keep[member.name]=json.loads(b''.join(parts))
   process.stdout.close();error=process.stderr.read().decode()
   if process.wait()!=0:raise ValueError('Archive decompression failed: '+error)
  finally:
   process.stdout.close();process.stderr.close()
   if process.poll() is None:process.kill();process.wait()
  if seen!=set(expected):raise ValueError('Archive member missing')
  retirement=self.json(name+'.retirement.json')
  if retirement['status']!='completed' or set(retirement['removed_paths'])!=set(expected):
   raise ValueError('Archive retirement is not complete')
  self.archives[key]=keep
  return keep
 def event(self,arm,event,suffix='trainer'):
  name=f'campaign_001/arms/{arm}/trainer/events/event_{event:03d}/RESULT.json' if suffix=='trainer' else f'campaign_001/arms/{arm}/events/event_{event:03d}/{suffix}'
  if self.path(name).is_file():return self.json(name)
  archive=self.archive(arm,event)
  if name not in archive:raise MissingEvidence(name)
  return archive[name]


def audit_local_work(rows,audit):
 starts={key:set() for key in LOCAL_TARGETS};done={key:set() for key in LOCAL_TARGETS}
 for row in rows:
  kind,ordinal,phase=row['kind'],row['ordinal'],row['phase']
  audit.require(kind in starts and type(ordinal)is int and 1<=ordinal<=LOCAL_TARGETS[kind],'local work kind/count')
  if phase=='started':
   audit.require(ordinal==len(starts[kind])+1 and ordinal not in starts[kind],'local start order');starts[kind].add(ordinal)
  else:
   audit.require(phase=='completed' and ordinal in starts[kind] and ordinal not in done[kind],'local completion order');done[kind].add(ordinal)
 return dict(attempts={k:len(v) for k,v in starts.items()},completed={k:len(v) for k,v in done.items()},
  unknown={k:len(starts[k]-done[k]) for k in starts})


def audit_event(row,arm,event,binding,audit):
 for key,value in dict(status='completed',method=arm,dataset='OpenCodeInstruct',event_id=f'event_{event:03d}',
  objective='osd_first_error',passes=2,response_presentations=256,optimizer_steps=128,start_step=128*(event-1),end_step=128*event,reduction='response_mean').items():
  audit.require(row.get(key)==value,'event '+key)
 contract=row['continuation_repair_fullrun_v1']
 audit.require(contract['arm']==arm and contract['binding']==binding and contract['initial_counters']==[0,0,0]
  and contract['final_counters']==[7424,14848,58],'own event scientific contract')
 ids=[x['record_id'] for x in row['features']];records=row['records']
 audit.require(len(ids)==128 and len(set(ids))==128 and len(records)==256,'event feature/presentation count')
 for offset in (0,128):audit.require(sorted(x['record_id'] for x in records[offset:offset+128])==sorted(ids),'native shuffled pass coverage')
 for ordinal,record in enumerate(records):
  stepped=bool(ordinal%2);obs=record['response_observation']
  audit.require(record['pass']==ordinal//128 and record['optimizer_stepped'] is stepped and record['optimizer_step_after']==128*(event-1)+(ordinal+1)//2,'event accumulation/order')
  audit.require(record['learning_rate']==(3e-4 if stepped else None),'fixed learning rate')
  audit.require(obs['record_id']==record['record_id'] and obs['sampled_anchors']==record['sampled_anchors'] and obs['pass_index']==record['pass'],'original response identity')
  audit.require(('continuation_repair' in obs)==(arm=='TAIL'),'actual TAIL flag')
  groups=record['sampled_anchors'];audit.require(len(groups)==1,'one natural anchor draw')
  draw=groups[0];audit.require(len(draw['positions'])==len(draw['keep_mask'])==1 and 1<=len(draw['positions'][0])<=512 and len(draw['positions'][0])==len(draw['keep_mask'][0]),'native anchor geometry')
  for key in ('mass_violations','bounds_violations','support_violations','nonfinite_values'):
   audit.require(obs['prefix_allocation'][key]==0,'unchanged C invariant')
  f,d=obs['ratio_metrics']['forward_kl'];p,pd=obs['ratio_metrics']['repair_pair_auxiliary'];actual,ad=obs['ratio_metrics']['objective_loss']
  sums=obs['sum_metrics'];ef=f+sums.get('continuation_fk_correction',0.);ep=p+sums.get('continuation_pair_correction',0.)
  audit.require(d==pd==ad and d>0,'one response denominator')
  tolerance=2e-7+5e-6*max(abs(actual),abs(ef)+abs(ep));log=obs['effective_components']
  audit.require(all(math.isfinite(x) for x in (f,p,actual,ef,ep,d)) and abs(ef+ep-actual)<=tolerance,'effective objective reconciliation')
  audit.require(log['effective_FKL_numerator']==ef and log['effective_PAIR_numerator']==ep and log['effective_total_numerator']==actual and log['denominator']==d,'component logging')
 work=row['objective_work']['work']
 for key in ('response_forwards','backwards_attempted','backwards_completed'):
  audit.require(work[key]==256,'native work '+key)
 audit.require(work['draft_forwards']>=256 and work['vocabulary_head_calls']>0,'physical work evidence')
 return set(ids)


def audit_global_work(value,audit,*,witness_forwards,activation_count,initializations):
 audit.require(not value['pending'],'no unresolved global work')
 for key in set(value['attempted'])|set(value['completed']):
  attempted,completed=value['attempted'].get(key,0),value['completed'].get(key,0)
  audit.require(type(attempted)is int and type(completed)is int and attempted>=completed>=0,'valid global work counts')
  audit.require(attempted==completed,'global work needs explicit reconciliation')
 for key,expected in dict(MAIN_TARGETS,witness_response_forward=witness_forwards,activation=activation_count,training_initialization=initializations).items():
  audit.require(value['completed'].get('production.'+key,0)==expected,'complete production '+key)
 for key,cap in (('adam',8),('generation',16)):
  count=value['attempted'].get('qualification.'+key,0)
  audit.require(0<count<=cap,'qualification cap '+key)
 audit.require(value['completed'].get('production.canonical_warmup_generation',0) in (0,2),'optional warmup references')
 return dict(attempted=value['attempted'],completed=value['completed'],pending=len(value['pending']),preserved_failures=len(value['failures']))


def audit_resources(clock,status,processes,storage_receipts,audit):
 audit.require(not clock['active'],'no active card lease')
 occupied=sum(row['occupied_s'] for row in clock['finished'])
 audit.require(abs(occupied-clock['completed_card_s'])<1e-6,'card interval sum')
 audit.require(occupied<=LIMITS['occupied_card_s'],'occupied-card ceiling')
 audit.require(status['finished_epoch']>=clock['first_model_load_epoch'] and status['finished_epoch']-clock['first_model_load_epoch']<=LIMITS['elapsed_s'],'original elapsed ceiling')
 tasks={row['task']:row for row in clock['finished']}
 audit.require(len(tasks)==len(clock['finished']),'unique completed card leases')
 for name,process in processes.items():
  audit.require(process['termination_confirmed'] and process['finished_epoch'] is not None and (not process.get('process_created',True) or process['exit_code'] is not None),'confirmed process termination')
  slots=('A','B') if process['phase']=='train' else (('B',) if process['phase']=='qualification_training' else ('A',))
  for slot in slots:
   audit.require(name+'_'+slot in tasks,'process card lease coverage')
   lease=tasks[name+'_'+slot]
   audit.require(lease['started_epoch']<=process.get('started_epoch',lease['started_epoch'])<=process['finished_epoch']<=lease['finished_epoch']+.01,'process within lease')
 for receipt in storage_receipts:
  audit.require(receipt['charged_bytes']<=LIMITS['new_allocated_bytes'] and receipt['free_bytes']>=LIMITS['free_reserve_bytes'],'recorded storage cap/reserve')
  audit.require(receipt['projected_peak_bytes']<=LIMITS['new_allocated_bytes'],'declared peak cap')
  if 'remaining_peak_bytes' in receipt:
   audit.require(receipt['charged_bytes']+receipt['remaining_peak_bytes']<=LIMITS['new_allocated_bytes'],'remaining-write charge')
   audit.require(receipt['free_bytes']>=receipt['required_free_bytes'],'remaining-write reserve')
 audit.require(bool(storage_receipts),'resource snapshots required')
 return dict(occupied_card_s=occupied,elapsed_s=status['finished_epoch']-clock['first_model_load_epoch'],
  recorded_max_charged_bytes=max(x['charged_bytes'] for x in storage_receipts),
  recorded_min_free_bytes=min(x['free_bytes'] for x in storage_receipts),snapshot_count=len(storage_receipts))


def audit_retention(arm,boundaries,inventory,audit):
 audit.require(inventory['status']=='passed','final live inventory passed')
 files={row['path']:row for row in inventory['files']};absent=set(inventory['absent_paths'])
 audit.require(len(files)==len(inventory['files']) and not set(files)&absent,'unambiguous final inventory')
 for event,row in boundaries.items():
  audit.require(row['status']=='passed' and row['arm']==arm and row['event']==event,'boundary identity')
  check=row['scientific_check']
  audit.require(check['status']=='passed' and check['global_step']==128*event and check['presentations']==256*event and check['live']['exact_resume'] and check['export_exact'],'boundary scientific validation')
  for key,required in (('state',event>=57),('weights',event==58)):
   path=row[key]
   if required:
    audit.require(path in files and files[path].get('exists') is True,'required retained tensor')
    audit.require({k:files[path][k] for k in ('bytes','sha256')}==row[key+'_identity'],'retained tensor identity')
   else:audit.require(path in absent,'retired intermediate explicitly absent')
 return dict(retained_complete_events=[57,58],retained_export_event=58,retired_complete_events=list(range(1,57)))




def audit_qualification_attempts(processes,audit):
 attempts=[row for row in processes.values() if row['phase'].startswith('qualification')]
 failures=[row for row in attempts if row.get('exit_code')!=0]
 audit.require(len(failures)<=1 and len(attempts)<=3,'single shared qualification retry')
 audit.require({row['phase'] for row in attempts if row.get('exit_code')==0}=={'qualification_training','qualification_serving'},'both successful qualification phases')
 return dict(process_attempts=len(attempts),failed_attempts=len(failures),successful_parity_cases=4)

def audit_auxiliary_work(evidence,global_work,audit):
 """Reconcile every attempted case/startup, including failed retained cases."""
 core_forward=core_backward=adam=0;startup_forward=startup_backward=0;case_count=0;startup_count=0
 for path in sorted(evidence.root.glob('qualification/training_*/**/RESULT.json')):
  if path.parent.name=='startup':continue
  if not (path.parent.name.startswith('reference_') or path.parent.name.startswith('integration_')):continue
  case=evidence.json(path);case_count+=1;work=case.get('actual_work',{})
  core_forward+=work.get('response_forwards',0);core_backward+=work.get('backwards_completed',0)
  if 'counters' in case:
   audit.require(case['counters'][0] in (128,129) and case['counters'][1] in (256,257,258),'bounded disposable case counters')
   adam+=case['counters'][0]-128
  elif case.get('endpoint'):
   raise MissingEvidence('Failed case lacks independently reconcilable optimizer counter')
  if case['status']=='completed':
   audit.require(work.get('response_forwards')==work.get('backwards_completed')==2 and case['counters']==[129,258],'completed case work')
 startup_paths=list(evidence.root.glob('qualification/training_*/**/startup/RESULT.json'))
 startup_paths+=list(evidence.root.glob('campaign_001/arms/*/startup_*/RESULT.json'))
 for path in sorted(startup_paths):
  item=evidence.json(path);startup_count+=1;work=item['work']
  audit.require(item['optimizer_steps']==0 and work['completed']['optimizer_steps']==0,'startup never updates Adam')
  audit.require(not any(work['unknown'].values()),'startup unknown work needs reconciliation')
  startup_forward+=work['completed']['response_forwards'];startup_backward+=work['completed']['response_backwards']
  if item['status']=='completed':audit.require(work['completed']['response_forwards']==work['completed']['response_backwards']==2,'qualified startup shape')
 completed=global_work['completed']
 for key,value in (('qualification.response_forward',core_forward),('qualification.response_backward',core_backward),
                   ('qualification.adam',adam),('startup.response_forward',startup_forward),('startup.response_backward',startup_backward)):
  audit.require(completed.get(key,0)==value,'physical case/startup reconciliation '+key)
 audit.require(case_count>=4 and startup_count>=6,'all disposable and production startups retained')
 return dict(disposable_case_attempts=case_count,disposable_core_forwards=core_forward,disposable_core_backwards=core_backward,
  disposable_Adam=adam,startup_attempts=startup_count,startup_forwards=startup_forward,startup_backwards=startup_backward,startup_Adam=0)


def run(root,*,source_manifest=None,tensor_inventory=None,postflight=None,external=None,zstd='zstd'):
 explicit=dict(external or {})
 for path in (source_manifest,tensor_inventory,postflight):
  if path is not None and Path(path).is_absolute():explicit[str(path)]=str(path)
 evidence=Evidence(root,external=explicit,zstd=zstd);audit=Audit()
 plan=audit.section('sealed_plan',lambda:evidence.json('sealed/PLAN.json'))
 if plan is None:return audit.result(),dict(details=audit.private,inputs=evidence.inputs)
 evidence.original_root=Path(plan['root'])
 audit.section('sealed_plan_binding',lambda:audit.require(plan['binding']==digest({k:v for k,v in plan.items() if k!='binding'}),'unchanged sealed plan'))
 def provenance():
  source=evidence.json(source_manifest or plan['source_manifest'])
  audit.require(evidence.inputs[str(evidence.path(source_manifest or plan['source_manifest']))]==plan['source_identity'],'source manifest identity')
  audit.require(source['commit']==plan['execution_commit'] and source['clean'],'clean execution commit')
  audit.require(plan['formula_code_identities']==FROZEN_FORMULAS,'frozen PR65 objective identities')
  for name,pin in FROZEN_FORMULAS.items():audit.require(source['files'][name]==pin,'staged formula '+name)
  for name,value in LIMITS.items():audit.require(plan['limits'][name]==value,'fixed resource limit '+name)
  return dict(execution_commit=plan['execution_commit'],frozen_formula_origin='ab7d8231a70387c7e02f0d9d1feeedd1e80dd258')
 audit.totals['provenance']=audit.section('source_and_formula_identity',provenance)
 status=audit.section('campaign_status',lambda:evidence.json('STATUS.json'))
 if status:audit.section('campaign_complete',lambda:audit.require(status['status']=='completed','campaign complete'))
 inventory=audit.section('final_tensor_inventory',lambda:evidence.json(tensor_inventory or 'FINAL_TENSOR_INVENTORY.json'))
 arm_totals={}
 for arm in ARMS:
  base=f'campaign_001/arms/{arm}';state=audit.section(arm+'_status',lambda:evidence.json(base+'/STATUS.json'))
  if state:
   audit.section(arm+'_endpoint',lambda:audit.require(state['status']=='completed' and [state['events_completed'],state['global_step'],state['presentations']]==[58,7424,14848] and not state['cleanup_errors'],'complete own endpoint'))
  work=audit.section(arm+'_local_work',lambda:audit_local_work(evidence.jsonl(base+'/WORK.jsonl'),audit))
  if work:
   audit.section(arm+'_local_totals',lambda:audit.require(work['attempts']==work['completed']==LOCAL_TARGETS and not any(work['unknown'].values()),'exact local totals'))
   if state:audit.section(arm+'_status_work',lambda:audit.require(state['actual_work']==work,'status matches reduced work'))
  initial=audit.section(arm+'_initial',lambda:evidence.json(base+'/trainer/INITIAL_CHECK.json'))
  if initial:audit.section(arm+'_fresh_initial',lambda:audit.require(initial['status']=='passed' and initial['global_step']==0 and initial['presentations']==0 and initial['adam_state_entries']==0 and initial['algorithm_installed_before_witness'] and initial['export_exact'],'fresh released initializer'))
  if initial and arm=='TAIL':audit.section('paired_initialization',lambda:audit.require(initial['paired_initialization']['exact_weights_empty_Adam_RNG'],'exact initial BASE/TAIL parity'))
  seen=set();boundaries={};completed=0;forward=backward=0
  for number in range(1,59):
   row=audit.section(f'{arm}_event_{number:03d}_receipt',lambda:evidence.event(arm,number))
   if row:
    ids=audit.section(f'{arm}_event_{number:03d}_checks',lambda:audit_event(row,arm,number,plan['binding'],audit))
    if ids is not None:
     audit.section(f'{arm}_event_{number:03d}_novel',lambda:audit.require(not seen&ids,'no repeated feedback identity'));seen.update(ids);completed+=1
     forward+=row['objective_work']['work']['response_forwards'];backward+=row['objective_work']['work']['backwards_completed']
   marker=audit.section(f'{arm}_boundary_{number:03d}',lambda:evidence.json(base+f'/boundaries/event_{number:03d}.json'))
   if marker:boundaries[number]=marker
  def retirements():
   for number in range(3,59):
    removed=evidence.event(arm,number,'OLDER_BOUNDARY_RETIREMENT.json')
    audit.require(removed['status']=='completed' and [x['path'] for x in removed['removed']]==[boundaries[number-2]['state']],'old state retired only after two successors')
    audit.require(removed['retained']['current']==boundaries[number] and removed['retained']['previous']==boundaries[number-1],'retirement retains both successors')
   for number in range(1,58):
    removed=evidence.json(base+f'/boundaries/event_{number:03d}_EXPORT_RETIREMENT.json')
    proof=evidence.json(base+f'/boundaries/event_{number:03d}_TENSOR_FINGERPRINTS.json')
    audit.require(removed['status']=='completed' and [x['path'] for x in removed['removed']]==[boundaries[number]['weights']],'duplicate export retirement')
    audit.require(proof['status']=='passed' and proof['exact_export_tensors'] and proof['identity']==boundaries[number]['state_identity'] and proof['export_identity']==boundaries[number]['weights_identity'],'retained predecessor export equality proof')
  if len(boundaries)==58:audit.section(arm+'_retirement_order',retirements)
  if inventory and len(boundaries)==58:
   audit.section(arm+'_rolling_retention',lambda:audit_retention(arm,boundaries,inventory,audit))
   if state:audit.section(arm+'_final_export',lambda:audit.require(state['checkpoint_identity']==boundaries[58]['weights_identity'],'final endpoint binding'))
  arm_totals[arm]=dict(verified_events=completed,distinct_feedback_responses=len(seen),ordinary_forwards=forward,ordinary_backwards=backward,local_work=work)
 audit.totals['arms']=arm_totals
 def witnesses():
  bindings=evidence.json('witness/BINDINGS.json');rows=bindings['features']
  audit.require(len(rows)==8 and len({r['record_id'] for r in rows})==8 and bindings['selection_seed']==20261126,'sealed shared witness')
  for row in rows:audit.require(row['anchors']==sorted(set(row['anchors'])) and len(row['anchors'])==min(8,row['available_anchors']),'bounded witness anchors')
  forwards=0;reads=0
  for endpoint in ('released',*[f'{arm}_{event:03d}' for arm in ARMS for event in (1,16,32,58)]):
   item=evidence.json('witness/'+endpoint+'/STATUS.json')
   audit.require(item['status']=='completed' and item['training_state_unchanged'] and item['state']['status']=='passed','witness state preservation')
   audit.require(len(item['responses'])==8 and item['work']['response_reads_completed']==8,'eight witness readouts')
   for row,readout in zip(rows,item['responses']):
    audit.require(readout['status']=='passed' and readout['anchors']==row['anchors'] and readout['record_id']==row['record_id'] and readout['feature_identity']==row['feature_identity'],'witness identical conditioning')
   for pair in item['paired_blocks']:audit.require(pair['Y']==pair['useful_repair']+pair['reachable_origin_followers']-pair['lost_prefix'],'signed witness identity')
   forwards+=item['work']['draft_forwards_completed'];reads+=item['work']['response_reads_completed']
  audit.require(reads==72 and forwards==9*sum(bool(r['anchors']) for r in rows),'witness logical/physical counts')
  return dict(response_reads=reads,response_forwards=forwards,origin='fixed released',exposure='shared BASE event1 TRAIN')
 witness=audit.section('shared_witness',witnesses);audit.totals['witness']=witness
 processes={}
 for path in sorted(evidence.root.glob('processes/*.json')):
  item=audit.section('process_'+path.stem,lambda:evidence.json(path))
  if item:processes[path.stem]=item
 def qualification():
  result=evidence.json('qualification/QUALIFICATION.json')
  audit.require(result['status']=='passed' and result['training']['status']=='passed' and result['serving']['status']=='passed','all qualification gates')
  audit.require(result['training']['fresh_initializations_exact'] and result['training']['BASE_TAIL_PR65_integration_exact'],'native parity gates')
  cases=result['training']['cases'];audit.require(len(cases)==4,'four successful parity cases')
  for row in cases:
   case=evidence.json(row['path']);audit.require(case['status']=='completed' and case['counters']==[129,258] and case['restoration_exact'],'disposable own state')
   if row['implementation']=='fullrun':audit.require(case['serialization']['restore']['exact_resume'] and case['serialization']['retired_after_exact_readback'],'honest disposable serialization')
  audit.require(result['serving']['generation_attempts']==result['serving']['completed']==8,'successful bounded serving gate')
  return audit_qualification_attempts(processes,audit)
 audit.totals['qualification']=audit.section('qualification',qualification)
 activations=audit.section('activation_records',lambda:len(evidence.jsonl('evaluation/ACTIVATIONS.jsonl'))+len(evidence.jsonl('canonicals/ACTIVATIONS.jsonl')))
 work=audit.section('global_work_file',lambda:evidence.json('WORK.json'))
 if work and witness and activations is not None:
  audit.totals['global_work']=audit.section('global_work_reduction',lambda:audit_global_work(work,audit,witness_forwards=witness['response_forwards'],activation_count=activations,initializations=sum(r['phase']=='train' for r in processes.values())))
 if work:audit.totals['auxiliary_work']=audit.section('qualification_and_startup_work',lambda:audit_auxiliary_work(evidence,work,audit))
 def resources():
  receipts=[]
  for pattern in ('STORAGE.*.json','forecasts/*.json','phases/*/STATUS.json'):
   for path in evidence.root.glob(pattern):
    value=evidence.json(path)
    if 'charged_bytes' in value:receipts.append(value)
    elif 'storage' in value:receipts.append(value['storage'])
    elif 'admission' in value:receipts.append(value['admission']['storage'])
  for members in evidence.archives.values():
   for name,value in members.items():
    if name.endswith('/ADMISSION.json'):receipts.append(value['storage'])
  final=evidence.json(postflight or 'POSTFLIGHT.json');audit.require(final['status']=='passed','final postflight')
  receipts.append(final['storage'])
  return audit_resources(evidence.json('CLOCK.json'),status,processes,receipts,audit)
 audit.totals['resources']=audit.section('resource_accounting',resources)
 return audit.result(),dict(details=audit.private,inputs=evidence.inputs)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
 parser.add_argument('--source-manifest');parser.add_argument('--tensor-inventory');parser.add_argument('--postflight')
 parser.add_argument('--external-map');parser.add_argument('--zstd',default='zstd')
 args=parser.parse_args();external=json.loads(Path(args.external_map).read_text()) if args.external_map else None
 report,private=run(args.root,source_manifest=args.source_manifest,tensor_inventory=args.tensor_inventory,
                    postflight=args.postflight,external=external,zstd=args.zstd)
 output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
 (output/'AUDIT_WORK.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 (output/'PRIVATE_INPUTS_AND_ERRORS.json').write_text(json.dumps(private,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=report['status'],checks=report['checks'],failed=len(report['failed_sections']),missing=len(report['missing_sections']))))
 return 0 if report['status']=='PASSED' else 2


if __name__=='__main__':raise SystemExit(main())
