"""Pure CPU fixtures for native-audit gates and admission bookkeeping."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.kplus_abt.qualification import NativeInputAudit
from experiments.kplus_abt.run_qualification import summarize
from experiments.kplus_abt.contract import ARMS, DATASETS, QUALIFICATION_CASES
from online_feedback.storage import write_json


def work(**counts):
    return dict(attempts=dict(counts), completed=dict(counts), unknown={key:0 for key in counts})


class AdversarialBlockQualificationTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def audit_fixture(self):
        draft = torch.nn.Linear(3, 3).bfloat16()
        runtime = SimpleNamespace(view='clean', embeddings=torch.ones(1, 1, 4, 3, dtype=torch.bfloat16),
                                  eligible=torch.tensor([[[False, True, True, False]]]))
        model = SimpleNamespace(draft_model=draft, adversarial_block=runtime)
        state = dict(step=torch.tensor(512.), exp_avg=torch.zeros_like(draft.weight),
                     exp_avg_sq=torch.zeros_like(draft.weight))
        optimizer = SimpleNamespace(state={draft.weight:state})
        trainer = SimpleNamespace(bundle=SimpleNamespace(model=model),
                                  backend=SimpleNamespace(optimizer=SimpleNamespace(optimizer=optimizer)))
        audit = NativeInputAudit(trainer)
        self.addCleanup(audit.close)
        kwargs = dict(target_hidden=torch.zeros(1, 6, 9, dtype=torch.bfloat16),
                      position_ids=torch.arange(10).reshape(1, 10),
                      attention_mask=torch.zeros(1, 4, 10),
                      noise_embedding=runtime.embeddings.reshape(1, 4, 3).clone())
        return audit, draft, runtime, state, kwargs

    def test_native_input_audit_accepts_only_eligible_change_with_unchanged_conditioning(self):
        audit, draft, runtime, state, kwargs = self.audit_fixture()
        audit.observe(draft, (), kwargs)
        runtime.view = 'second'
        kwargs['noise_embedding'][0, 1] += .015625
        audit.observe(draft, (), kwargs)
        self.assertEqual(len(audit.pairs), 1)
        self.assertEqual(audit.pairs[0]['changed_slots'], 1)
        self.assertIsNone(audit.pending)

    def test_native_input_audit_rejects_moment_mutation_without_counter_change(self):
        audit, draft, runtime, state, kwargs = self.audit_fixture()
        audit.observe(draft, (), kwargs)
        runtime.view = 'second'
        state['exp_avg'].add_(1)
        with self.assertRaisesRegex(ValueError, 'Adam changed'):
            audit.observe(draft, (), kwargs)

    def test_native_input_audit_rejects_seed_change_and_unknown_attention(self):
        audit, draft, runtime, state, kwargs = self.audit_fixture()
        audit.observe(draft, (), kwargs)
        runtime.view = 'second'
        kwargs['noise_embedding'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'Seed or ineligible'):
            audit.observe(draft, (), kwargs)
        with self.assertRaisesRegex(TypeError, 'Unsupported'):
            NativeInputAudit.mask_parts(object())

    def receipt_fixture(self, root):
        write_json(root/'sealed/PLAN.json', dict(execution_source=dict(commit='fixture'), binding='fixture'))
        rows = {}
        for dataset in DATASETS:
            for case in QUALIFICATION_CASES:
                views = 4 if case in ('CLEAN2', 'RANDOM', 'ADV') else 2
                row = dict(status='completed', complete_state=dict(status='passed'),
                    work=work(response_forwards=views, response_backwards=views, optimizer_steps=1),
                    startup=dict(work=work(response_forwards=2,response_backwards=2,optimizer_steps=0)),
                    neutral_exact=dict(exact_equal=True), perturbation=dict(status='passed'),
                    sampling_state_exact=dict(exact_equal=True), native_input_pairs=[{}, {}],
                    resource=dict(status='passed',work=work(response_forwards=4,response_backwards=4,optimizer_steps=0)))
                path = root/'qualification'/dataset/case/'RESULT.json'
                write_json(path, row)
                rows[dataset, case] = (path, row)
                if case in ARMS:
                    write_json(path.with_name('SERVING_CHECK.json'),
                        dict(status='passed', work=work(serving_generations=1)))
        return rows

    def test_admission_requires_all_exact_case_startup_resource_and_serving_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.receipt_fixture(root)
            receipt = summarize(root)
            self.assertEqual(receipt['status'], 'passed')
            self.assertEqual(receipt['counts'], dict(optimizer_steps=12,response_forwards=68,
                                                    response_backwards=68,serving_generations=8))

    def test_unknown_started_work_cannot_hide_behind_complete_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.receipt_fixture(root)
            path, row = rows['gsm8k', 'ADV']
            row['resource']['work']['attempts']['response_forwards'] += 1
            row['resource']['work']['unknown']['response_forwards'] = 1
            write_json(path, row)
            with self.assertRaisesRegex(ValueError, 'Incomplete or unknown'):
                summarize(root)

    def test_compensating_case_counts_cannot_hide_behind_correct_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.receipt_fixture(root)
            for case, adjustment in (('KPLUS', 1), ('DISABLED', -1)):
                path, row = rows['gsm8k', case]
                for key in ('attempts', 'completed'):
                    row['work'][key]['response_forwards'] += adjustment
                write_json(path, row)
            with self.assertRaisesRegex(ValueError, 'Incomplete or unknown'):
                summarize(root)


if __name__ == '__main__':
    unittest.main()
