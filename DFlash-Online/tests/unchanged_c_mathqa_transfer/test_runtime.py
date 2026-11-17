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
  self.assertEqual(runtime.LIMITS['adam'],528)
  self.assertEqual(runtime.LIMITS['response_forward'],1088)
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
 def test_original_clock_and_work_required_for_continuation(self):
  with tempfile.TemporaryDirectory() as td:
   plan=dict(root=td,binding='same',execution_commit='abc',prerequisite={'feature_manifest':{'identity':'pool'}})
   runtime.start_campaign(plan)
   clock_before=runtime.read(Path(td)/'CLOCK.json')['first_model_load_epoch']
   ledger=runtime.WorkLedger(td,'replay')
   ledger.call(dict(adam=256,response_forward=516,response_backward=516,training_initialization=2),'two_replays',lambda:None)
   self.assertEqual(runtime.validate_continuation(plan)['status'],'passed')
   self.assertEqual(runtime.read(Path(td)/'CLOCK.json')['first_model_load_epoch'],clock_before)
   with self.assertRaises(ValueError):runtime.validate_continuation(dict(plan,binding='changed'))
   with self.assertRaises(FileExistsError):runtime.start_campaign(plan)
 def test_pending_replay_or_missing_work_cannot_start_main(self):
  with tempfile.TemporaryDirectory() as td:
   plan=dict(root=td,binding='same',execution_commit='abc',prerequisite={'feature_manifest':{'identity':'pool'}})
   runtime.start_campaign(plan)
   with self.assertRaises(ValueError):runtime.validate_continuation(plan)
   work=runtime.read(Path(td)/'WORK.json')
   work['attempted'].update(adam=256,response_forward=516,response_backward=516,training_initialization=2)
   work['completed']=dict(work['attempted']);work['pending']={'1':{'failed':True}}
   write_json(Path(td)/'WORK.json',work)
   with self.assertRaises(ValueError):runtime.validate_continuation(plan)
 def test_identity_failure_has_no_cross_card_fallback(self):
  with self.assertRaisesRegex(ValueError,'provenance'):
   launch.require_cross_card_identity({'status':'failed','same_card_checks_passed':True,'errors':['foreign origin']})

 def test_original_clocks_include_new_work(self):
  with patch.object(runtime,'clock_status',return_value={'occupied_s':14000,'elapsed_s':1000}):
   with self.assertRaises(RuntimeError):runtime.require_time_for({'root':'fixture'},100,100)

if __name__=='__main__':unittest.main()
