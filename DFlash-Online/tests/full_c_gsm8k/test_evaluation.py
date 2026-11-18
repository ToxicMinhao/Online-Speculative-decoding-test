"""CPU fixtures for exact full-run serving schedules and bounded qualification."""
from collections import Counter
from copy import deepcopy
import hashlib,json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch

from experiments.full_c_gsm8k import evaluation as E, analysis as A
from test_full_k_gsm8k_evaluation import FullBackend, requests, harmful
from test_persistent_prefix_evaluation import references, quiet


def panel(dataset):
 rows=requests(1024)
 for i,row in enumerate(rows):row.update(dataset=dataset,source_id=f'{dataset}:test:{i}')
 return rows

def bound_refs(rows,producer):
 values={k:dict(v,reference_checkpoint=producer) for k,v in references(rows).items()}
 for value in values.values():
  payload={k:v for k,v in value.items()if k!='binding_digest'}
  value['binding_digest']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return values

def baseline(rows):return {r['request_id']:{m:{'verification_calls':2}for m in ('O58','KPLUS58')}for r in rows}

class EvaluationTests(unittest.TestCase):
 def setUp(self):
  directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);self.root=Path(directory.name)
  self.rows=panel('gsm8k');self.warm=deepcopy(self.rows[:2])
  for i,row in enumerate(self.warm):row.update(request_id=f'warm{i}',source_id=f'warm:{i}',prompt_ids=[1,9000+i])
  self.refs=bound_refs(self.rows,'/F');self.wrefs=bound_refs(self.warm,'/K6')
  self.checkpoints={m:'/'+m for m in E.METHODS}
 def run_final(self,backend,name,*,resume=False,boundary_check=quiet):
  return E.evaluate_final(backend,self.rows,self.checkpoints,self.root/name,dataset='gsm8k',canonicals=self.refs,
   warmup_requests=self.warm,warmup_canonicals=self.wrefs,schedule=E.evaluation_schedule(self.rows),
   reference_checkpoints={'scored':'/F','warmup':'/K6'},historical_counts=baseline(self.rows),
   physical_gpu_uuid='GPU-fixed',boundary_check=boundary_check,execution_commit='a'*40,
   checkpoint_identities={m:dict(bytes=123,sha256=str(i)*64)for i,m in enumerate(E.METHODS)},resume=resume)
 def make_plan(self):
  plan={'devices':{'A':'GPU-fixed'},'execution_commit':'a'*40,'datasets':{}}
  for dataset in E.DATASETS:
   rows=panel(dataset);base=self.root/dataset;base.mkdir()
   values={'requests':{'requests':rows},'canonicals':bound_refs(rows,'/F'),
    'schedule':E.evaluation_schedule(rows),'historical_counts':baseline(rows)}
   spec={'reference_checkpoints':{'scored':'/F','warmup':'/K6'}}
   for key,value in values.items():
    path=base/(key+'.json');path.write_text(json.dumps(value));spec[key]=str(path)
   plan['datasets'][dataset]=spec
  return plan
 def test_schedule_full_budget_exact_six_orders_balance(self):
  for dataset in E.DATASETS:
   rows=panel(dataset);schedule=E.evaluation_schedule(rows)
   orders=[tuple(x['method']for x in schedule[i:i+3])for i in range(0,144,3)]
   self.assertEqual(Counter(orders),{p:8 for p in E.ORDER_PERMUTATIONS})
   self.assertEqual(Counter((x['method'],x['position'])for x in schedule),{(m,p):16 for m in E.METHODS for p in range(3)})
   self.assertEqual(orders,[E.ORDER_PERMUTATIONS[i%6]for i in range(48)])
   slots=list(E.execution_slots(schedule,rows,self.warm,dataset=dataset))
   self.assertEqual(Counter(s['phase']for s,_ in slots),{'scored':9216,'warmup':288})
   self.assertEqual(len({E.label_for(s)for s,_ in slots}),9504)
 def test_complete_real_reader_roundtrip_and_public_counts(self):
  first=FullBackend(self.rows);checks=0
  def stop_at_next_boundary():
   nonlocal checks
   checks+=1
   return quiet() if checks==1 else dict(quiet(),host_quiet=False)
  with self.assertRaisesRegex(RuntimeError,'boundary'):self.run_final(first,'full',boundary_check=stop_at_next_boundary)
  self.assertEqual(len(first.executions),66)
  backend=FullBackend(self.rows);result=self.run_final(backend,'full',resume=True);root=self.root/'full'
  self.assertEqual(len(first.executions)+len(backend.executions),9504)
  self.assertEqual(len(first.switches)+len(backend.switches),144)
  self.assertEqual(result['resume_attempts'],1)
  prior=json.loads((root/'ATTEMPTS/resume_001.json').read_text())
  self.assertEqual(prior['completed_generations'],66);self.assertEqual(prior['prior_status']['status'],'failed')
  self.assertEqual(result['completed_valid_generations'],9504)
  self.assertEqual((result['publications'],result['scored_generations'],result['warmup_generations']),(144,9216,288))
  self.assertEqual(set(backend.weight_checks),set(self.checkpoints.values()))
  # This fixture tests the actual full consumer; independent arithmetic tests use all10,000 draws.
  with patch.object(A,'BOOTSTRAP_DRAWS',32):summary=A.write_report(root)
  self.assertEqual(summary['status'],'completed');self.assertEqual(summary['request_count'],1024)
  self.assertEqual(set(summary['comparisons']),{'KPLUS_C58_vs_KPLUS58','KPLUS_C58_vs_O58','KPLUS58_vs_O58'})
  self.assertEqual(len((root/'ALL_REPEATS.csv').read_text().splitlines()),9217)
  self.assertEqual(len((root/'PER_REQUEST.csv').read_text().splitlines()),3073)
  self.assertEqual(len((root/'WARMUPS.csv').read_text().splitlines()),289)
  self.assertTrue((root/'WEIGHTS_KPLUS58_resume_001.json').exists())
  record=root/'0000_r0_KPLUS58.json';raw=json.loads(record.read_text());raw['response_ids'][0]+=1;record.write_text(json.dumps(raw))
  self.assertEqual(A.write_report(root)['status'],'INVALID_OR_INCOMPLETE')
 def test_invalid_and_unknown_preserve_raw_stop_without_retries(self):
  def corrupt(record,options):record['response_ids'][0]+=1
  with self.assertRaisesRegex(RuntimeError,'correctness'):self.run_final(FullBackend(self.rows,corrupt),'bad')
  status=json.loads((self.root/'bad/STATUS.json').read_text())
  self.assertEqual((status['invalid_generations'],status['unknown_generation_attempts'],status['unattempted_generations']),(1,0,9503))
  with self.assertRaisesRegex(ValueError,'invalid or unknown'):self.run_final(FullBackend(self.rows),'bad',resume=True)
  backend=FullBackend(self.rows)
  with patch.object(backend,'generate',side_effect=RuntimeError('lost')):
   with self.assertRaisesRegex(RuntimeError,'lost'):self.run_final(backend,'unknown')
  status=json.loads((self.root/'unknown/STATUS.json').read_text());self.assertEqual(status['unknown_generation_attempts'],1)
 def test_partial_activation_resume_rejected_even_without_unknown_or_invalid(self):
  original=E.write_json;interrupted=False
  def write_once(path,value):
   nonlocal interrupted
   original(path,value)
   if Path(path).name=='STATUS.json' and value.get('completed_valid_generations')==3 and not interrupted:
    interrupted=True
    raise RuntimeError('interrupt after durable third measurement')
  with patch.object(E,'write_json',side_effect=write_once):
   with self.assertRaisesRegex(RuntimeError,'interrupt'):self.run_final(FullBackend(self.rows),'partial')
  status=json.loads((self.root/'partial/STATUS.json').read_text())
  self.assertEqual((status['completed_valid_generations'],status['unknown_generation_attempts'],status['invalid_generations']),(3,0,0))
  backend=FullBackend(self.rows)
  with self.assertRaisesRegex(ValueError,'partial activation'):self.run_final(backend,'partial',resume=True)
  self.assertFalse(backend.executions)
 def test_changed_checkpoint_identity_rejected_before_resume(self):
  with self.assertRaisesRegex(RuntimeError,'boundary'):
   self.run_final(FullBackend(self.rows),'identity',boundary_check=lambda:dict(quiet(),host_quiet=False))
  protocol=self.root/'identity/PROTOCOL.json';value=json.loads(protocol.read_text());value['checkpoint_identities']['O58']['sha256']='f'*64;protocol.write_text(json.dumps(value))
  backend=FullBackend(self.rows)
  with self.assertRaisesRegex(ValueError,'identities'):self.run_final(backend,'identity',resume=True)
  self.assertFalse(backend.executions)
 def test_historical_counts_gate_controls_only(self):
  counts=baseline(self.rows)
  row={'phase':'scored','method':'KPLUS58','request_id':'eval0','verification_calls':3}
  self.assertEqual(E.historical_errors(row,counts),['historical_verification_count_mismatch'])
  row['method']='KPLUS_C58';self.assertEqual(E.historical_errors(row,counts),[])
  row.update(method='O58',phase='warmup');self.assertEqual(E.historical_errors(row,counts),[])
 def test_references_producers_and_digest_fail_before_publication(self):
  backend=FullBackend(self.rows);self.refs['eval0']['reference_checkpoint']='/bad'
  with self.assertRaisesRegex(ValueError,'producer'):self.run_final(backend,'producer')
  self.assertFalse(backend.switches)
  self.refs=bound_refs(self.rows,'/F');self.refs['eval0']['response_ids'][0]+=1
  with self.assertRaisesRegex(ValueError,'digest'):self.run_final(backend,'digest')
  self.assertFalse(backend.switches)
 def test_serving_qualification_exact24_no_warmups(self):
  plan=self.make_plan();backend=FullBackend(self.rows)
  result=E.evaluate_qualification(backend,plan,self.checkpoints,self.root/'qual',boundary_check=quiet)
  self.assertEqual((result['status'],result['attempted_generations'],result['warmup_generations']),('passed',24,0))
  self.assertEqual(len(backend.switches),12)
  self.assertEqual({x[2]['request_index']for x in backend.executions},{0,1})
  rows=[json.loads(x)for x in(self.root/'qual/MEASUREMENTS.jsonl').read_text().splitlines()]
  self.assertEqual(Counter((x['dataset'],x['method'])for x in rows),{(d,m):4 for d in E.DATASETS for m in E.METHODS})
 def test_qualification_control_mismatch_stops_and_preserves(self):
  plan=self.make_plan()
  def mutation(record,options):harmful(record)
  with self.assertRaisesRegex(RuntimeError,'historical'):
   E.evaluate_qualification(FullBackend(self.rows,mutation),plan,self.checkpoints,self.root/'qual',boundary_check=quiet)
  result=json.loads((self.root/'qual/STATUS.json').read_text())
  self.assertEqual((result['attempted_generations'],result['invalid_generations'],result['unattempted_generations']),(1,1,23))
 def test_qualification_candidate_different_rounds_allowed(self):
  plan=self.make_plan()
  def mutation(record,options):
   if record['checkpoint_id']=='/KPLUS_C58':harmful(record)
  result=E.evaluate_qualification(FullBackend(self.rows,mutation),plan,self.checkpoints,self.root/'qual',boundary_check=quiet)
  self.assertEqual(result['status'],'passed')

if __name__=='__main__':unittest.main()
