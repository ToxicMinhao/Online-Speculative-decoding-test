"""Metadata-only campaign orchestration and fail-closed handoffs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.survival_head import campaign as C
from experiments.survival_head.contract import ARMS, budgets, expected_work
from online_feedback.storage import write_json


def schedule():
    ids=[f'feedback-{i}' for i in range(128)]
    return dict(features=[{'record_id':key,'path':key+'.pt'} for key in ids],
        records=[dict(record_id=key,**{'pass':p},sampled_anchors=[{'positions':[[1]],'keep_mask':[[True]]}])
                 for p in (0,1) for key in ids])


def result(arm, baseline):
    return dict(status='completed',method=arm,dataset='gsm8k',event_id='event_005',
        start_step=512,end_step=640,optimizer_steps=128,response_presentations=256,passes=2,
        records=deepcopy(baseline['records']),features=deepcopy(baseline['features']),
        survival_work=expected_work(arm),head_optimizer_steps=0 if arm=='KPLUS' else 128,
        boundary=dict(status='passed',export_exact=True))


class CampaignTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        self.plan=dict(root=str(self.root),binding='a'*64,execution_source={'commit':'b'*40},
                       budgets=budgets(),automatic_retry=False)
        write_json(self.root/'sealed/PLAN.json',self.plan)

    def test_shared_schedule_mismatch_missing_head_and_extra_work_fail(self):
        baseline=schedule()
        for arm in ARMS: self.assertEqual(C.validate_update(result(arm,baseline),arm,baseline)['status'],'passed')
        for mutate in (lambda row:row['records'][0].update(record_id='different'),
                       lambda row:row['records'][0]['sampled_anchors'][0]['positions'][0].append(5),
                       lambda row:row['survival_work'].update(response_forwards=257),
                       lambda row:row['survival_work'].update(unknown=1),
                       lambda row:row.update(head_optimizer_steps=127)):
            broken=result('SURVIVAL',baseline);mutate(broken)
            with self.assertRaises(ValueError):C.validate_update(broken,'SURVIVAL',baseline)

    def test_admission_revalidates_source_priority_and_raw_qualification_evidence(self):
        admission=dict(status='passed',binding='a'*64,execution_commit='b'*40,checks={'all':True})
        write_json(self.root/'qualification/ADMISSION.json',admission)
        with patch('experiments.survival_head.preflight.verify_seal',return_value=(self.plan,10)), \
                patch('experiments.survival_head.preflight.priority_finished',return_value={'status':'passed'}) as priority, \
                patch('experiments.survival_head.run_qualification.summarize',return_value=admission) as summarize:
            self.assertEqual(C.require_admission(self.plan),admission)
            priority.assert_called_once_with(self.plan);summarize.assert_called_once_with(str(self.root))
            summarize.return_value=dict(admission,checks={'all':False})
            with self.assertRaisesRegex(ValueError,'raw evidence'):C.require_admission(self.plan)

    def run_campaign(self, train=None, evaluate=None):
        baseline=schedule()
        train=train or Mock(side_effect=lambda plan,arm:result(arm,baseline))
        evaluate=evaluate or Mock(return_value={'status':'passed','errors':[],'rows':[{}]*768})
        with patch.object(C,'require_admission',return_value={'status':'passed'}), \
                patch.object(C,'require_capacity',return_value={'status':'passed'}), \
                patch.object(C,'shared_inputs',return_value=({'records':[]},baseline)):
            outcome=C.run(self.root,train_arm=train,evaluate=evaluate)
        return outcome,train,evaluate

    def test_three_arms_then_exact768_evaluation(self):
        outcome,train,evaluate=self.run_campaign()
        self.assertEqual(outcome['status'],'completed')
        self.assertEqual([call.args[1] for call in train.call_args_list],list(ARMS))
        self.assertEqual(len(outcome['completed_tasks']),4)
        self.assertEqual(set(evaluate.call_args.args[1]),set(ARMS))
        with self.assertRaises(FileExistsError):self.run_campaign()

    def test_failed_training_result_cannot_advance_or_evaluate(self):
        baseline=schedule(); train=Mock(side_effect=lambda plan,arm:dict(result(arm,baseline),boundary={'status':'failed'}))
        evaluate=Mock()
        with self.assertRaisesRegex(ValueError,'boundary'):self.run_campaign(train,evaluate)
        self.assertEqual(train.call_count,1);evaluate.assert_not_called()
        saved=C.read(self.root/'campaign_001/STATUS.json')
        self.assertEqual(saved['status'],'failed')
        self.assertEqual(saved['completed_tasks'],[])

    def test_failed_or_incomplete_evaluation_cannot_mark_campaign_completed(self):
        evaluate=Mock(return_value={'status':'passed','rows':[{}]*767,'errors':[]})
        with self.assertRaisesRegex(ValueError,'768'):self.run_campaign(evaluate=evaluate)
        saved=C.read(self.root/'campaign_001/STATUS.json')
        self.assertEqual(saved['status'],'failed')
        self.assertEqual(len(saved['completed_tasks']),3)

    def test_failed_admission_or_storage_calls_no_worker(self):
        train,evaluate=Mock(),Mock()
        with patch.object(C,'require_admission',side_effect=ValueError('priority still active')):
            with self.assertRaisesRegex(ValueError,'priority'):C.run(self.root,train_arm=train,evaluate=evaluate)
        train.assert_not_called();evaluate.assert_not_called()


if __name__=='__main__':unittest.main()
