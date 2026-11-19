"""Explicit initial-only retry retains evidence, science and cumulative work."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.full_c_gsm8k import retry_initial as R
from experiments.full_c_gsm8k.contract import plan_binding, QUALIFICATION
from experiments.full_c_gsm8k.runtime import LIMITS, initialize_work


class InitialRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'run';self.root.mkdir()
        initialize_work(self.root)
        counts={k:0 for k in LIMITS};counts['training_initialization']=2
        work=R.read(self.root/'WORK.json');work.update(attempted=counts,completed=counts,next_operation=2)
        write_json(self.root/'WORK.json',work)
        self.old=dict(root=str(self.root),source_root='/old/DFlash-Online',source_manifest='/old/SOURCE.json',
            execution_commit='a'*40,execution_source=dict(commit='a'*40,dirty=False),seed=17,
            training_python='/python',events=58,storage=dict(reserve_bytes=400_000_000_000,
                external_roots=['/old'],source_root='/old/DFlash-Online',source_manifest='/old/SOURCE.json',execution_commit='a'*40))
        self.old['binding']=plan_binding(self.old)
        self.new=deepcopy(self.old)
        self.new.update(source_root='/new/DFlash-Online',source_manifest='/new/SOURCE.json',
            execution_commit='b'*40,execution_source=dict(commit='b'*40,dirty=False))
        self.new['storage'].update(source_root=self.new['source_root'],source_manifest=self.new['source_manifest'],
            execution_commit=self.new['execution_commit'],external_roots=['/old','/new'])
        self.new['binding']=plan_binding(self.new)
        write_json(self.root/'sealed/PLAN.json',self.old)
        self.plan=self.root/'sealed/REPAIR_PLAN.json';write_json(self.plan,self.new)
        write_json(self.root/'LAUNCH.json',dict(status='stopped',completed=[],
            execution_commit=self.old['execution_commit'],binding=self.old['binding']))
        write_json(self.root/'phases/qualification_training/STATUS.json',dict(status='failed'))
        write_json(self.root/'qualification/training_B/DETAILS.json',dict(status='failed',phase='loading',cases=[],
            error='ValueError: Independent arm initialization differs'))
        (self.root/'logs').mkdir();(self.root/'logs/qualification_training.log').write_text('original failure\n')
        write_json(self.root/'processes/qualification_training.json',dict(exit_code=1))

    def test_accepts_only_source_metadata_and_preserves_failed_constructor_counts(self):
        receipt=R.validate_retry(self.old,self.new)
        self.assertTrue(receipt['scientific_plan_unchanged'])
        self.assertEqual(receipt['prior_work']['completed']['training_initialization'],2)
        self.assertNotEqual(receipt['binding'],receipt['original_binding'])

    def test_refuses_scientific_or_limit_changes_even_with_recomputed_binding(self):
        for key,value in [('seed',18),('events',57)]:
            changed=deepcopy(self.new);changed[key]=value;changed['binding']=plan_binding(changed)
            with self.assertRaisesRegex(ValueError,'scientific plan'):R.validate_retry(self.old,changed)
        changed=deepcopy(self.new);changed['storage']['reserve_bytes']=0;changed['binding']=plan_binding(changed)
        with self.assertRaisesRegex(ValueError,'resource limits'):R.validate_retry(self.old,changed)

    def test_refuses_unknown_or_executed_update_work(self):
        for kind in ('adam','response_forward','qualification_generation'):
            work=R.read(self.root/'WORK.json');work['completed'][kind]=work['attempted'][kind]=1
            write_json(self.root/'WORK.json',work)
            with self.assertRaisesRegex(ValueError,'constructor-only'):R.validate_retry(self.old,self.new)
            work['completed'][kind]=work['attempted'][kind]=0;write_json(self.root/'WORK.json',work)
        work['pending']={'unknown':{}};write_json(self.root/'WORK.json',work)
        with self.assertRaisesRegex(ValueError,'constructor-only'):R.validate_retry(self.old,self.new)

    def test_refuses_completed_qualification_or_active_clock(self):
        clock=R.read(self.root/'CLOCK.json');clock['active']={'worker':{}};write_json(self.root/'CLOCK.json',clock)
        with self.assertRaisesRegex(ValueError,'idle clock'):R.validate_retry(self.old,self.new)
        clock['active']={};write_json(self.root/'CLOCK.json',clock)
        write_json(self.root/'QUALIFICATION.json',{'status':'passed'})
        with self.assertRaisesRegex(ValueError,'must not be repeated'):R.validate_retry(self.old,self.new)

    def test_preserves_failed_bytes_without_modifying_ledgers_or_original_plan(self):
        protected={name:(self.root/name).read_bytes() for name in ('WORK.json','CLOCK.json','LAUNCH.json','sealed/PLAN.json')}
        rows=R.preserve_failed_paths(self.root,self.root/'retry/prior_failure')
        self.assertEqual(len(rows),4)
        for row in rows:self.assertEqual(R.identity(row['retained_path']),row['identity'])
        for name,data in protected.items():self.assertEqual((self.root/name).read_bytes(),data)
        self.assertFalse((self.root/'phases/qualification_training').exists())

    def test_completed_qualification_survives_forecast_block_without_production(self):
        old_clock=(self.root/'CLOCK.json').read_bytes()
        calls=[]
        def preflight(command,**kwargs):
            write_json(command[command.index('--output')+1],{'status':'passed'})
        def child(path,plan,phase):
            calls.append(phase)
            work=R.read(self.root/'WORK.json')
            for key,amount in QUALIFICATION.items():
                if (phase=='qualification_serving') == (key=='qualification_generation'):
                    work['attempted'][key]+=amount;work['completed'][key]+=amount
            write_json(self.root/'WORK.json',work)
            return {'status':'completed','result':{'status':'passed'}}
        with patch.object(R,'check_compilers'),patch.object(R,'enforce_clock'),patch.object(R,'child_environment',return_value={}),\
             patch.object(R.subprocess,'run',side_effect=preflight),patch.object(R,'child',side_effect=child),\
             patch('experiments.full_c_gsm8k.storage.retire_qualification',return_value={'status':'passed'}),\
             patch.object(R,'forecast',side_effect=RuntimeError('complete fixed campaign no longer fits')):
            with self.assertRaisesRegex(RuntimeError,'no longer fits'):R.run(self.plan)
        self.assertEqual(calls,['qualification_training','qualification_serving'])
        self.assertEqual(R.read(self.root/'QUALIFICATION.json')['status'],'passed')
        work=R.read(self.root/'WORK.json')
        self.assertEqual(work['completed']['training_initialization'],3)
        self.assertEqual(work['completed']['adam'],6)
        self.assertEqual(work['completed']['scored_generation'],0)
        self.assertEqual((self.root/'CLOCK.json').read_bytes(),old_clock)
        receipt=R.read(self.root/'initial_retry_001/LAUNCH.json')
        self.assertEqual(receipt['status'],'stopped')
        self.assertTrue(receipt['ledgers_unchanged_before_retry'])
        with self.assertRaises((FileNotFoundError,ValueError,FileExistsError)):R.run(self.plan)


if __name__=='__main__':unittest.main()
