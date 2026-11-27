"""Synthetic CPU receipts; no project imports, model loading or remote access."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import audit_completed_work as A


def event(arm='TAIL',number=1):
 features=[dict(record_id=f'e{number}_r{i}') for i in range(128)]
 rows=[]
 for i in range(256):
  rid=features[i%128]['record_id'];anchors=[dict(positions=[[1,3]],keep_mask=[[True,True]])]
  obs=dict(record_id=rid,pass_index=i//128,sampled_anchors=anchors,
   prefix_allocation={key:0 for key in ('mass_violations','bounds_violations','support_violations','nonfinite_values')},
   ratio_metrics=dict(forward_kl=[2.,1.],repair_pair_auxiliary=[1.,1.],objective_loss=[3.,1.]),
   sum_metrics={},effective_components=dict(effective_FKL_numerator=2.,effective_PAIR_numerator=1.,effective_total_numerator=3.,denominator=1.))
  if arm=='TAIL':obs['continuation_repair']={'continuation_changed_high_positions':0}
  rows.append(dict(record_id=rid,pass_=i//128,optimizer_stepped=bool(i%2),optimizer_step_after=128*(number-1)+(i+1)//2,
    learning_rate=3e-4 if i%2 else None,sampled_anchors=anchors,response_observation=obs))
  rows[-1]['pass']=rows[-1].pop('pass_')
 return dict(status='completed',method=arm,dataset='OpenCodeInstruct',event_id=f'event_{number:03d}',objective='osd_first_error',passes=2,
  response_presentations=256,optimizer_steps=128,start_step=128*(number-1),end_step=128*number,reduction='response_mean',
  continuation_repair_fullrun_v1=dict(arm=arm,binding='b',initial_counters=[0,0,0],final_counters=[7424,14848,58]),
  features=features,records=rows,objective_work={'work':dict(response_forwards=256,backwards_attempted=256,backwards_completed=256,draft_forwards=256,vocabulary_head_calls=512)})


def global_work():
 values={'production.'+key:value for key,value in dict(A.MAIN_TARGETS,witness_response_forward=72,activation=49,training_initialization=2).items()}
 values.update({'qualification.adam':4,'qualification.generation':8,'startup.response_forward':12,'startup.response_backward':12})
 return dict(attempted=deepcopy(values),completed=deepcopy(values),pending={},failures=[])


def boundaries():
 rows={};files=[];absent=[]
 for i in range(1,59):
  row=dict(status='passed',arm='BASE',event=i,state=f'/run/state{i}.pt',weights=f'/run/export{i}.safetensors',
   state_identity={'bytes':63,'sha256':str(i)},weights_identity={'bytes':21,'sha256':str(i)},
   scientific_check=dict(status='passed',global_step=128*i,presentations=256*i,live={'exact_resume':True},export_exact=True))
  for key,retain in (('state',i>=57),('weights',i==58)):
   if retain:files.append(dict(path=row[key],exists=True,**row[key+'_identity']))
   else:absent.append(row[key])
  rows[i]=row
 return rows,dict(status='passed',files=files,absent_paths=absent)


class CompletionAuditTests(unittest.TestCase):
 def test_seal_digest_preserves_literal_unicode_like_native_schema(self):
  import hashlib
  value={'description':'測試'}
  literal=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
  self.assertEqual(A.digest(value),hashlib.sha256(literal).hexdigest())

 def test_empty_or_missing_study_is_incomplete_never_passed(self):
  with tempfile.TemporaryDirectory() as directory:
   result,_=A.run(directory)
   self.assertEqual(result['status'],'INCOMPLETE')
   self.assertEqual(result['missing_sections'],['sealed_plan'])

 def test_independent_event_counter_and_flag_checks(self):
  row=event();audit=A.Audit()
  self.assertEqual(len(A.audit_event(row,'TAIL',1,'b',audit)),128)
  for mutate in (lambda x:x.update(end_step=127),lambda x:x['records'][1].update(optimizer_stepped=False),
                 lambda x:x['records'][0]['response_observation'].pop('continuation_repair'),
                 lambda x:x['records'][0]['response_observation']['effective_components'].update(effective_FKL_numerator=3.)):
   bad=deepcopy(row);mutate(bad)
   with self.assertRaises(AssertionError):A.audit_event(bad,'TAIL',1,'b',A.Audit())

 def test_event_neutral_tail_is_allowed_without_efficacy_gate(self):
  self.assertEqual(len(A.audit_event(event(),'TAIL',1,'b',A.Audit())),128)

 def test_local_ledger_reduction_preserves_unknown_and_rejects_duplicate(self):
  rows=[dict(kind='optimizer_steps',ordinal=1,phase='started')]
  result=A.audit_local_work(rows,A.Audit());self.assertEqual(result['unknown']['optimizer_steps'],1)
  with self.assertRaises(AssertionError):A.audit_local_work(rows+rows,A.Audit())
  rows.append(dict(kind='optimizer_steps',ordinal=1,phase='completed'))
  self.assertEqual(A.audit_local_work(rows,A.Audit())['completed']['optimizer_steps'],1)

 def test_global_counts_distinguish_canonical_activation_from48_scoring(self):
  work=global_work()
  result=A.audit_global_work(work,A.Audit(),witness_forwards=72,activation_count=49,initializations=2)
  self.assertEqual(result['completed']['production.activation'],49)
  work['completed']['production.activation']=48
  with self.assertRaises(AssertionError):A.audit_global_work(work,A.Audit(),witness_forwards=72,activation_count=49,initializations=2)

 def test_global_unknown_and_exhausted_qualification_cannot_pass(self):
  for kind in ('pending','cap','missing_main'):
   work=global_work()
   if kind=='pending':work['pending']={'4':{'counts':{'qualification.adam':1}}}
   elif kind=='cap':work['attempted']['qualification.adam']=work['completed']['qualification.adam']=9
   else:work['attempted']['production.adam']=work['completed']['production.adam']=14847
   with self.assertRaises(AssertionError):A.audit_global_work(work,A.Audit(),witness_forwards=72,activation_count=49,initializations=2)

 def test_rolling_retention_needs57_and58_not_only_endpoint_hash(self):
  rows,inventory=boundaries();A.audit_retention('BASE',rows,inventory,A.Audit())
  bad=deepcopy(inventory);bad['files']=[r for r in bad['files'] if r['path']!='/run/state57.pt']
  with self.assertRaises(AssertionError):A.audit_retention('BASE',rows,bad,A.Audit())
  bad=deepcopy(inventory);bad['absent_paths'].remove('/run/state1.pt')
  with self.assertRaises(AssertionError):A.audit_retention('BASE',rows,bad,A.Audit())

 def test_resources_sum_both_cards_original_clock_and_reserve(self):
  clock=dict(first_model_load_epoch=10.,active={},completed_card_s=12.,finished=[
   dict(task='train_BASE_001_A',started_epoch=11.,finished_epoch=17.,occupied_s=6.),
   dict(task='train_BASE_001_B',started_epoch=11.,finished_epoch=17.,occupied_s=6.)])
  processes={'train_BASE_001':dict(phase='train',process_created=True,termination_confirmed=True,exit_code=0,started_epoch=12.,finished_epoch=16.)}
  storage=[dict(charged_bytes=1_000_000,free_bytes=420_000_000_000,projected_peak_bytes=20_000_000_000,
   remaining_peak_bytes=10_000_000_000,required_free_bytes=410_000_000_000)]
  result=A.audit_resources(clock,{'finished_epoch':18.},processes,storage,A.Audit());self.assertEqual(result['occupied_card_s'],12.)
  for target in ('clock','card','reserve'):
   badclock=deepcopy(clock);badstorage=deepcopy(storage);status={'finished_epoch':18.}
   if target=='clock':status['finished_epoch']=A.LIMITS['elapsed_s']+11
   elif target=='card':badclock['completed_card_s']=6.
   else:badstorage[0]['free_bytes']=399_000_000_000
   with self.assertRaises(AssertionError):A.audit_resources(badclock,status,processes,badstorage,A.Audit())

 def test_single_qualification_retry_is_shared_between_phases(self):
  processes={'t1':{'phase':'qualification_training','exit_code':0},'s1':{'phase':'qualification_serving','exit_code':0}}
  self.assertEqual(A.audit_qualification_attempts(processes,A.Audit())['failed_attempts'],0)
  processes['t0']={'phase':'qualification_training','exit_code':1}
  self.assertEqual(A.audit_qualification_attempts(processes,A.Audit())['failed_attempts'],1)
  processes['s0']={'phase':'qualification_serving','exit_code':1}
  with self.assertRaises(AssertionError):A.audit_qualification_attempts(processes,A.Audit())

 def test_startup_and_case_work_reconcile_without_adam_startup(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   def put(path,value):
    path=root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
   startup=dict(status='completed',optimizer_steps=0,work=dict(completed=dict(response_forwards=2,response_backwards=2,optimizer_steps=0),unknown=dict(response_forwards=0,response_backwards=0,optimizer_steps=0)))
   for variant in ('reference_BASE','reference_TAIL','integration_BASE','integration_TAIL'):
    case=f'qualification/training_001/{variant}'
    put(case+'/RESULT.json',dict(status='completed',actual_work={'response_forwards':2,'backwards_completed':2},counters=[129,258]))
    put(case+'/startup/RESULT.json',startup)
   for arm in A.ARMS:put(f'campaign_001/arms/{arm}/startup_000/RESULT.json',startup)
   work=global_work();work['completed'].update({'qualification.response_forward':8,'qualification.response_backward':8})
   result=A.audit_auxiliary_work(A.Evidence(root),work,A.Audit())
   self.assertEqual(result['disposable_Adam'],4)
   self.assertEqual(result['startup_forwards'],12)
   self.assertEqual(result['startup_Adam'],0)
   startup['optimizer_steps']=1;put('campaign_001/arms/TAIL/startup_000/RESULT.json',startup)
   with self.assertRaises(AssertionError):A.audit_auxiliary_work(A.Evidence(root),work,A.Audit())

 def test_repaired_source_can_only_change_four_infrastructure_files(self):
  original=dict(files=dict(A.FROZEN_FORMULAS,**{A.AMENDMENT_FILES[0]:{'sha256':'old'}}))
  repaired=deepcopy(original);repaired['files'].update({A.AMENDMENT_FILES[0]:{'sha256':'new'},A.AMENDMENT_FILES[-1]:{'sha256':'new'}})
  result=A.compare_source_manifests(original,repaired,A.Audit())
  self.assertEqual(set(result['changed_production_paths']),{A.AMENDMENT_FILES[0],A.AMENDMENT_FILES[-1]})
  repaired['files']['online_feedback/objective.py']={'sha256':'different'}
  with self.assertRaises(AssertionError):A.compare_source_manifests(original,repaired,A.Audit())

 def test_archive_recovery_retains_failed_attempt_cost_and_bytes(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);name='compact/BASE/event_001.tar.zst'
   def put(path,row):
    path=root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row));return path
   failed=dict(status='failed',verification={'status':'passed'},error='Loose member changed after archive verification: x',
    removed_paths=['first'],removed_allocated_bytes=5,wall_s=7)
   failed_path=put(name+'.retirement.failed_001.json',failed)
   recovery=dict(status='completed',new_model_work=0,failed_receipt=dict(path=str(failed_path),**A.identity(failed_path)),
    verification=failed['verification'],new_removed_paths=['next'],previous_removed_count=1,remaining_count=1,
    new_removed_allocated_bytes=3,wall_s=2)
   put(name+'.retirement.recovery_001.json',recovery)
   final=dict(recovery=recovery,removed_paths=['first','next'],removed_allocated_bytes=8,wall_s=9)
   A.verify_archive_recovery(A.Evidence(root),name,final)
   final['wall_s']=2
   with self.assertRaisesRegex(ValueError,'accounting'):A.verify_archive_recovery(A.Evidence(root),name,final)

 def test_external_locators_need_explicit_mapping(self):
  with tempfile.TemporaryDirectory() as directory:
   evidence=A.Evidence(directory)
   with self.assertRaises(A.MissingEvidence):evidence.json('/unmapped/private/path.json')


if __name__=='__main__':unittest.main()
