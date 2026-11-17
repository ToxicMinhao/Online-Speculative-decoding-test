"""Execution guards independent of model or serving fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.unchanged_c_mathqa_transfer import runtime,launch,preflight
from online_feedback.storage import write_json

class RuntimeTests(unittest.TestCase):
 def test_caps_and_complete_work(self):
  self.assertEqual(runtime.LIMITS['adam'],272)
  self.assertEqual(runtime.LIMITS['response_forward'],576)
  self.assertEqual(sum(runtime.LIMITS[k] for k in ('feedback_generation','canonical_generation','qualification_generation','warmup_generation','scored_generation')),2770)
  self.assertEqual(runtime.ALLOCATION_BYTES,90_000_000_000)
 def test_failed_work_stays_charged_and_clock_cannot_reset(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);runtime.initialize_work(root);ledger=runtime.WorkLedger(root,'qualification')
   def fail():raise ValueError('fixture failure')
   with self.assertRaisesRegex(ValueError,'fixture failure'):ledger.call({'adam':1},'step',fail)
   state=json.loads((root/'WORK.json').read_text())
   self.assertEqual(state['attempted']['adam'],1);self.assertEqual(state['completed']['adam'],0)
   self.assertEqual(len(state['pending']),1)
   with self.assertRaises(FileExistsError):runtime.initialize_work(root)
 def test_serial_own_feedback_chain(self):
  plan={'devices':{'A':'physical_A','B':'physical_B'}}
  groups=launch.production_groups(plan)
  self.assertEqual([[r[0] for r in g] for g in groups],[['C5'],['C_collect6'],['C6'],['final']])
  self.assertTrue(all(r[1]=='A' for g in groups for r in g))
 def test_no_historical_feature_proof_is_not_a_pass(self):
  from experiments.unchanged_c_mathqa_transfer.prepare import identity
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'receipt.json';write_json(p,{'status':'passed','historical_feature_identity_verified':False,'files':{}})
   with self.assertRaisesRegex(ValueError,'Historical teacher-feature'):
    preflight.read_authentication({'asset_authentication':{'path':str(p),'identity':identity(p)}})
 def test_unchanged_receipt_required(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'receipt.json';write_json(p,{'status':'passed','historical_feature_identity_verified':True,'files':{}})
   with self.assertRaisesRegex(ValueError,'receipt changed'):
    preflight.read_authentication({'asset_authentication':{'path':str(p),'identity':{'bytes':1,'sha256':'0'*64}}})
 def test_identity_failure_stops_before_serving_qualification(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);plan_path=root/'PLAN.json'
   plan={'root':td,'execution_commit':'fixture','compiler_root':td,'source_root':td,
         'training_python':'fixture-python','estimated_seconds':{'training_A':1},
         'devices':{'A':'A','B':'B'}}
   write_json(plan_path,plan);jobs=[]
   def read(path):
    if Path(path)==plan_path:return plan
    if Path(path).name=='CPU_PREFLIGHT.json':return {'status':'passed','parallel_qualification_admitted':True}
    return {'result':{}}
   bad={'status':'failed','same_card_checks_passed':True,'errors':['foreign origin']}
   from contextlib import ExitStack
   from experiments.unchanged_c_mathqa_transfer import qualification
   with ExitStack() as stack:
    for owner,name,value in [(launch,'read',read),(launch,'check_compilers',lambda:None),
      (launch,'child_environment',lambda p:{}),(launch,'initialize_work',lambda r:None),
      (launch,'require_time_for',lambda *a:{}),(launch,'run_children',lambda p,q,g:jobs.extend(g)),
      (qualification,'compare_training_cards',lambda a,b:bad)]:
     stack.enter_context(patch.object(owner,name,side_effect=value))
    stack.enter_context(patch.object(launch.subprocess,'check_output',return_value='0 fixture'))
    stack.enter_context(patch.object(launch.subprocess,'run'))
    with self.assertRaisesRegex(ValueError,'provenance'):
     launch.run(plan_path)
   self.assertFalse(any(job[0].startswith('serving_') for job in jobs))

 def test_original_clocks_include_new_work(self):
  with patch.object(runtime,'clock_status',return_value={'occupied_s':14000,'elapsed_s':1000}):
   with self.assertRaises(RuntimeError):runtime.require_time_for({'root':'fixture'},100,100)

if __name__=='__main__':unittest.main()
