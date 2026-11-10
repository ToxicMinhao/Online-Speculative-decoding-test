from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.ok_coding.campaign import WorkLedger, read_work
from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from experiments.survival_head import qualification as Q, run_qualification as R
from experiments.survival_head.contract import QUALIFICATION_CASES, QUALIFICATION_BUDGET


def complete_work(expected):
    return dict(attempts=dict(expected),completed=dict(expected),unknown={key:0 for key in expected})


def plan(root):
    value=dict(root=str(root),binding='a'*64,execution_source=dict(commit='b'*40,source_path=str(root)),
        training_python='/python',worker_python='/worker',training_gpu=0,serving_gpu=0)
    write_json(root/'sealed/PLAN.json',value)
    write_json(root/'PREFLIGHT_QUALIFICATION.json',dict(status='passed',binding=value['binding']))
    return value


def case_receipt(case):
    return dict(status='completed',work=complete_work(Q.case_limits(case)),
        startup=dict(status='completed',work=complete_work(dict(response_forwards=2,response_backwards=2,optimizer_steps=0))))


class QualificationTests(unittest.TestCase):
    def test_exact_case_work_and_no_disabled_auxiliary(self):
        self.assertEqual(Q.case_limits('DISABLED')['auxiliary_backwards'],0)
        self.assertEqual(Q.case_limits('LOCAL')['head_optimizer_steps'],1)
        self.assertEqual(sum(Q.case_limits(c)['optimizer_steps'] for c in QUALIFICATION_CASES),4)
        self.assertEqual(sum(Q.case_limits(c)['auxiliary_backwards'] for c in QUALIFICATION_CASES),4)

    def test_same_graph_auxiliary_backward_preserves_existing_gradient(self):
        with tempfile.TemporaryDirectory() as temp:
            p=torch.nn.Parameter(torch.tensor([1.,2.]))
            p.grad=torch.tensor([3.,4.])
            ledger=WorkLedger(Path(temp)/'WORK.jsonl',dict(auxiliary_backwards=1))
            auxiliary=(p.square()).mean()
            check=Q.auxiliary_gradient_check(auxiliary,[p],ledger)
            self.assertEqual(check,dict(finite=True,norm_sq=5.,nonzero=True,native_gradient_buffer_unchanged=True))
            self.assertTrue(torch.equal(p.grad,torch.tensor([3.,4.])))
            auxiliary.backward()  # Original native graph remains usable.
            self.assertTrue(torch.equal(p.grad,torch.tensor([4.,6.])))
            self.assertEqual(read_work(ledger.path,ledger.limits)['completed']['auxiliary_backwards'],1)

    def test_auxiliary_nonfinite_is_recorded_as_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            p=torch.nn.Parameter(torch.tensor([1.]))
            ledger=WorkLedger(Path(temp)/'WORK.jsonl',dict(auxiliary_backwards=1))
            check=Q.auxiliary_gradient_check(p.sum()*float('nan'),[p],ledger)
            self.assertFalse(check['finite'])
            self.assertTrue(check['native_gradient_buffer_unchanged'])
            self.assertIsNone(p.grad)

    def test_first_two_feedback_identities_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);records=[]
            for i in range(128):
                feature=root/f'{i}.pt';feature.write_bytes(b'fixture')
                canonical=root/f'{i}.json';write_json(canonical,dict(record_id=f'r{i}'))
                records.append(dict(record_id=f'r{i}',feature=dict(path=str(feature),**identity(feature)),
                    canonical=dict(path=str(canonical),**identity(canonical))))
            write_json(root/'feedback/POOL.json',dict(records=records))
            selected,_=Q.verified_inputs(root)
            self.assertEqual([r['record_id'] for r in selected],['r0','r1'])
            Path(records[1]['feature']['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):Q.verified_inputs(root)


class CoordinatorTests(unittest.TestCase):
    def test_summary_counts_diagnostic_backward_once_and_no_extra_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan(root)
            for case in QUALIFICATION_CASES:
                write_json(root/'qualification'/case/'RESULT.json',case_receipt(case))
            with patch.object(R,'validate_case',side_effect=lambda r,c,p:case_receipt(c)),\
                 patch.object(R,'validate_serving',return_value=dict(work=complete_work(dict(serving_generations=2)))):
                summary=R.summarize(root)
            self.assertEqual(summary['counts'],dict(optimizer_steps=4,head_optimizer_steps=2,
                response_forwards=16,response_backwards=20,auxiliary_backwards=4,serving_generations=6))
            self.assertEqual(len(summary['cases']),4)

    def test_incomplete_or_wrong_work_prevents_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan(root)
            value=case_receipt('KPLUS');value['work']['completed']['response_backwards']=1
            write_json(root/'qualification/KPLUS/RESULT.json',value)
            with self.assertRaisesRegex(ValueError,'Incomplete or unknown'):
                R.validate_case(root,'KPLUS',R.read(root/'sealed/PLAN.json'))

    def test_missing_bulk_file_is_not_called_archived(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path=folder/'ENDPOINT.pt';path.write_bytes(b'endpoint')
            expected=identity(path);path.unlink()
            with self.assertRaises(ValueError):R.identity_or_retirement(path,expected,folder)
            write_json(folder/'RETIREMENT.json',dict(status='completed',removed=[dict(path=str(path),identity=expected)]))
            self.assertEqual(R.identity_or_retirement(path,expected,folder),'retired_not_archived')

    def test_worker_failure_no_retry_no_retirement_and_unattempted_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan(root)
            with patch.object(R.subprocess,'run',return_value=type('Exited',(),{'returncode':1})()) as launch,\
                 patch.object(R,'retire_case') as retire:
                with self.assertRaises(RuntimeError):R.run(root)
            self.assertEqual(launch.call_count,1);retire.assert_not_called()
            status=R.read(root/'qualification/STATUS.json')
            self.assertEqual(status['status'],'failed')
            self.assertEqual(status['completed'],[])
            self.assertEqual(status['unattempted_cases'],['DISABLED','LOCAL','SURVIVAL'])
            self.assertFalse((root/'qualification/ADMISSION.json').exists())

    def test_fresh_preflight_required_before_any_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan(root)
            write_json(root/'PREFLIGHT_QUALIFICATION.json',dict(status='failed',binding='a'*64))
            with patch.object(R.subprocess,'run') as launch,self.assertRaises(ValueError):R.run(root)
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
