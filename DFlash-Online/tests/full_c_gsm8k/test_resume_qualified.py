"""Reuse passed qualification after the authorized elapsed-only amendment."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.full_c_gsm8k import resume_qualified as R
from experiments.full_c_gsm8k.contract import plan_binding, PRODUCTION, QUALIFICATION
from experiments.full_c_gsm8k.runtime import LIMITS, initialize_work, ELAPSED_AMENDMENT


class QualifiedResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'run';self.root.mkdir();initialize_work(self.root)
        self.original=dict(root=str(self.root),source_root='/qualified/DFlash-Online',
            source_manifest=str(self.root/'QUALIFIED_SOURCE.json'),execution_commit='a'*40,
            execution_source=dict(commit='a'*40,dirty=False),seed=17,events=58,training_python='/python',
            storage=dict(reserve_bytes=400_000_000_000,external_roots=['/qualified'],
                source_root='/qualified/DFlash-Online',source_manifest=str(self.root/'QUALIFIED_SOURCE.json'),
                execution_commit='a'*40))
        self.original['binding']=plan_binding(self.original)
        original_path=self.root/'sealed/QUALIFIED_PLAN.json';write_json(original_path,self.original)
        self.plan=deepcopy(self.original)
        self.plan.update(source_root='/production/DFlash-Online',source_manifest=str(self.root/'PRODUCTION_SOURCE.json'),
            execution_commit='b'*40,execution_source=dict(commit='b'*40,dirty=False),
            elapsed_budget_amendment=dict(ELAPSED_AMENDMENT))
        self.plan['storage'].update(external_roots=['/qualified','/production'],source_root=self.plan['source_root'],
            source_manifest=self.plan['source_manifest'],execution_commit=self.plan['execution_commit'])
        self.plan['binding']=plan_binding(self.plan)
        self.path=self.root/'sealed/PRODUCTION_PLAN.json';write_json(self.path,self.plan)
        files={'online_feedback/objective.py':{'bytes':1,'sha256':'1'},
               'experiments/full_c_gsm8k/runtime.py':{'bytes':2,'sha256':'2'}}
        write_json(self.original['source_manifest'],dict(commit='a'*40,dirty=False,files=files))
        revised=deepcopy(files);revised['experiments/full_c_gsm8k/runtime.py']['sha256']='3'
        revised['experiments/full_c_gsm8k/resume_qualified.py']={'bytes':4,'sha256':'4'}
        write_json(self.plan['source_manifest'],dict(commit='b'*40,dirty=False,files=revised))
        prior={k:0 for k in LIMITS};prior['training_initialization']=2
        counts={k:prior[k]+QUALIFICATION[k] for k in LIMITS}
        work=R.read(self.root/'WORK.json');work.update(attempted=counts,completed=counts,next_operation=36)
        write_json(self.root/'WORK.json',work)
        qualified=dict(status='passed',execution_commit=self.original['execution_commit'],binding=self.original['binding'],
            active_plan=str(original_path),work=deepcopy(work),prior_work={'completed':prior},
            training={'status':'completed','result':{'status':'passed'}},serving={'status':'completed'})
        write_json(self.root/'QUALIFICATION.json',qualified)
        write_json(self.root/'initial_retry_001/LAUNCH.json',dict(status='stopped',
            completed=['qualification_training','qualification_serving'],
            error='RuntimeError: Complete fixed campaign no longer fits: recorded original forecast',
            clock=R.read(self.root/'CLOCK.json')))
        write_json(self.root/'initial_retry_001/CPU_PREFLIGHT.json',dict(status='passed',
            source={'manifest_identity':R.identity(self.original['source_manifest'])}))

    def test_passed_qualification_and_elapsed_only_plan_are_accepted(self):
        report=R.validate_continuation(self.plan)
        self.assertTrue(report['qualification_reused'])
        self.assertEqual(report['work_before']['completed']['adam'],6)
        self.assertEqual(report['work_before']['completed']['training_initialization'],3)
        self.assertEqual(report['elapsed_budget_amendment']['elapsed_cap_s'],108000)

    def test_scientific_or_unapproved_budget_change_is_rejected(self):
        changed=deepcopy(self.plan);changed['seed']=29;changed['binding']=plan_binding(changed)
        with self.assertRaisesRegex(ValueError,'qualified science'):R.validate_continuation(changed)
        changed=deepcopy(self.plan);changed['elapsed_budget_amendment']['elapsed_cap_s']=200000
        changed['binding']=plan_binding(changed)
        with self.assertRaisesRegex(ValueError,'elapsed-only'):R.validate_continuation(changed)
        changed=deepcopy(self.plan);changed['storage']['reserve_bytes']=0;changed['binding']=plan_binding(changed)
        with self.assertRaisesRegex(ValueError,'Storage'):R.validate_continuation(changed)

    def test_changed_scientific_source_is_rejected(self):
        manifest=R.read(self.plan['source_manifest'])
        manifest['files']['online_feedback/objective.py']['sha256']='unexpected'
        write_json(self.plan['source_manifest'],manifest)
        with self.assertRaisesRegex(ValueError,'scientific source changed'):R.validate_continuation(self.plan)

    def test_other_stop_started_production_and_unknown_work_are_rejected(self):
        path=self.root/'initial_retry_001/LAUNCH.json';status=R.read(path)
        status['error']='Unexpected numerical failure';write_json(path,status)
        with self.assertRaisesRegex(ValueError,'forecast-only'):R.validate_continuation(self.plan)
        status['error']='Complete fixed campaign no longer fits: old';write_json(path,status)
        (self.root/'campaign_001').mkdir()
        with self.assertRaisesRegex(ValueError,'no started production'):R.validate_continuation(self.plan)
        (self.root/'campaign_001').rmdir()
        work=R.read(self.root/'WORK.json');work['pending']={'unknown':{}};write_json(self.root/'WORK.json',work)
        with self.assertRaisesRegex(ValueError,'reconciled qualification'):R.validate_continuation(self.plan)

    def test_only_three_production_phases_run_and_qualification_is_not_repeated(self):
        protected={name:(self.root/name).read_bytes() for name in ('CLOCK.json','QUALIFICATION.json')}
        calls=[]
        def preflight(command,**kwargs):write_json(command[command.index('--output')+1],{'status':'passed'})
        def child(path,plan,phase):
            calls.append(phase)
            if phase=='final_mathqa':
                work=R.read(self.root/'WORK.json')
                for key,amount in PRODUCTION.items():
                    work['attempted'][key]+=amount;work['completed'][key]+=amount
                write_json(self.root/'WORK.json',work)
            return {'status':'completed'}
        with patch.object(R,'check_compilers'),patch.object(R,'enforce_clock'),patch.object(R,'child_environment',return_value={}),\
             patch.object(R.subprocess,'run',side_effect=preflight),patch.object(R,'child',side_effect=child),\
             patch.object(R,'forecast',return_value={'admission':{'elapsed_cap_s':108000}}),\
             patch.object(R,'boundary',return_value={'status':'passed'}),patch.object(R,'final_audit',return_value={'status':'passed'}):
            report=R.run(self.path)
        self.assertEqual(calls,['train','final_gsm8k','final_mathqa'])
        self.assertEqual(report['qualification_optimizer_steps_repeated'],0)
        self.assertEqual(report['work']['completed']['adam'],7430)
        self.assertEqual(report['work']['completed']['training_initialization'],4)
        for name,value in protected.items():self.assertEqual((self.root/name).read_bytes(),value)


if __name__=='__main__':unittest.main()
