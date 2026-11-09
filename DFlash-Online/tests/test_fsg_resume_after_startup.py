"""Accounting and scheduling checks; fake callbacks never execute a model."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.ok_coding.storage import identity
from experiments.functional_step_geometry import resume_after_startup as resume
from experiments.functional_step_geometry.work import LIMITS, QualificationWork


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ResumeAfterStartupTests(unittest.TestCase):
    def fixture(self, root):
        prior = root / 'previous'
        work = QualificationWork(prior / 'QUALIFICATION_WORK.jsonl')
        for kind in LIMITS:
            for _ in range(resume.PRIOR_COUNTS.get(kind + '_attempted', 0)):
                with work.operation(kind):
                    pass
        write(prior / 'qualification/QUALIFICATION.json', dict(
            status='failed', work=resume.PRIOR_COUNTS, main_pilot_started=False,
            error='Exact qualification gate failed: /qualification/gsm8k/SCALAR/NATIVE_GRADIENTS.json'))
        formula = dict(status='passed', numerical_protocol='independent_same_native_layout',
            tolerances=dict(atol=1e-5, rtol=.016, maximum_relative_l2=.016),
            historical_cross_layout_gate_passed=False, native_vjp=dict(status='passed'),
            chunks=[8, 1], fixture=dict(fixture_anchors=9))
        oracle = prior / 'qualification/gsm8k/revised_oracles'
        write(oracle / 'RESULT.json', dict(status='passed', formula=formula,
                                         resource=dict(status='passed', wall_s=24.)))
        write(oracle / 'EXACT_STATE_RESTORATION.json', dict(exact_equal=True))
        plan = dict(binding='fixed', geometry={'precision': 'BF16'},
            serving={'block_size': 16}, numerical_recipe={'rtol': .016},
            execution_source={'commit': 'test-only'},
            revised_qualification=dict(policy=resume.POLICY, previous_root='older',
                                       separate_diagnostics=[]),
            startup_repair=dict(previous_root=str(prior), prior_counts=resume.PRIOR_COUNTS,
                cumulative_limits=resume.CUMULATIVE_LIMITS,
                remaining_work=resume.remaining_work_plan(),
                reference_startup=resume.REFERENCE_STARTUP,
                previous_ledger_identity=identity(prior / 'QUALIFICATION_WORK.jsonl'),
                retained_oracles=str(oracle)))
        write(prior / 'sealed/PLAN.json', plan)
        destination = root / 'new'
        destination.mkdir()
        return prior, destination, plan

    def test_plan_accounts_all_remaining_work_without_repeating_completed_checks(self):
        remaining = resume.remaining_work_plan()
        for kind in LIMITS:
            self.assertEqual(resume.PRIOR_COUNTS.get(kind + '_attempted', 0) + remaining[kind],
                             resume.CUMULATIVE_LIMITS[kind])
        self.assertEqual(sum(row['forward'] for row in remaining['additional_parts'].values()), 33)
        self.assertEqual(sum(row['backward'] for row in remaining['additional_parts'].values()), 16)
        self.assertEqual(remaining['repeated_completed_adam_steps'], 0)
        self.assertEqual(remaining['repeated_gsm_oracle_work'], 0)
        self.assertEqual(remaining['repeated_resource_fixture_work'], 0)
        self.assertEqual(remaining['main_pilot_work'], 0)

    def test_default_work_limits_are_unchanged_and_explicit_limits_are_copied(self):
        self.assertEqual((LIMITS['ordinary_forward'], LIMITS['additional_forward'],
                          LIMITS['additional_backward']), (16, 96, 32))
        with tempfile.TemporaryDirectory() as directory:
            limits = dict(resume.CUMULATIVE_LIMITS)
            work = QualificationWork(Path(directory) / 'ledger', limits=limits)
            limits['adam'] = 100
            self.assertEqual(work.limits['adam'], 8)
            for _ in range(18):
                with work.operation('ordinary_forward'):
                    pass
            with self.assertRaisesRegex(RuntimeError, 'ceiling'):
                work.record('ordinary_forward', 'attempted')
            self.assertEqual(work.summary()['ordinary_forward_attempted'], 18)

    def test_admission_retains_byte_exact_latest_ledger_and_does_not_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            prior, output, plan = self.fixture(Path(directory))
            _, work = resume.admit_resume(plan, output)
            old_bytes = (prior / 'QUALIFICATION_WORK.jsonl').read_bytes()
            self.assertEqual((output / 'QUALIFICATION_WORK.jsonl').read_bytes(), old_bytes)
            self.assertEqual(work.summary(), resume.PRIOR_COUNTS)
            with work.operation('adam'):
                pass
            self.assertTrue((output / 'QUALIFICATION_WORK.jsonl').read_bytes().startswith(old_bytes))
            self.assertEqual(work.summary()['adam_completed'], 4)
            with self.assertRaises(FileExistsError):
                resume.admit_resume(plan, output)

    def test_stale_or_tampered_contract_is_rejected_before_copy(self):
        mutations = (
            lambda p: p['startup_repair'].update(cumulative_limits=LIMITS),
            lambda p: p['startup_repair'].update(reference_startup={}),
            lambda p: p['startup_repair'].update(prior_counts={'adam_completed': 3}),
            lambda p: p['startup_repair']['previous_ledger_identity'].update(sha256='wrong'),
            lambda p: p['geometry'].update(precision='FP32'),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                _, output, plan = self.fixture(Path(directory))
                plan = deepcopy(plan)
                mutate(plan)
                with self.assertRaises(ValueError):
                    resume.admit_resume(plan, output)
                self.assertFalse((output / 'QUALIFICATION_WORK.jsonl').exists())

    def test_oracle_reuse_has_zero_new_work_and_labels_historical_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output, plan = self.fixture(Path(directory))
            receipt = resume.reuse_gsm_oracles(plan, output)
            self.assertEqual(receipt['new_work'], dict(additional_forward=0, additional_backward=0, adam=0))
            self.assertIn('historical', receipt['cost_and_work_fields'])
            self.assertFalse(receipt['synthetic_resource_is_representative_correction_evidence'])
            self.assertEqual(receipt['resource']['wall_s'], 24.)

    def test_failed_or_wrong_layout_oracle_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output, plan = self.fixture(Path(directory))
            path = Path(plan['startup_repair']['retained_oracles']) / 'RESULT.json'
            receipt = json.loads(path.read_text())
            receipt['formula']['chunks'] = [9]
            write(path, receipt)
            with self.assertRaisesRegex(ValueError, 'not qualified'):
                resume.reuse_gsm_oracles(plan, output)

    def test_fake_complete_schedule_runs_only_missing_cases_and_stops_before_pilot(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            _, output, plan = self.fixture(Path(directory))
            config = Path(directory) / 'config.yaml'
            config.write_text('fake config')
            plan['original_config'] = str(config)
            plan['datasets'] = {name: dict(configs={'KPLUS': 'fake'},
                origin_state='own_' + name, origin_export='own_export_' + name)
                for name in ('gsm8k', 'mathqa')}
            prepared = dict(status='completed', datasets={})
            for name in plan['datasets']:
                prepared['datasets'][name] = []
                for index in range(2):
                    path = Path(directory) / (name + str(index))
                    path.write_bytes(b'fixed-feature')
                    prepared['datasets'][name].append(dict(feature_path=str(path), feature_identity=identity(path)))
            write(output / 'sealed/PLAN.json', plan)
            write(output / 'qualification_feedback/PREPARATION.json', prepared)
            calls = []

            def charge(work, kind, count):
                for _ in range(count):
                    with work.operation(kind):
                        pass

            def trainer_factory(*args, **kwargs):
                return SimpleNamespace(dataset=kwargs['dataset'], bundle=SimpleNamespace(model=object()),
                                       close=lambda: None)

            def startup(trainer, features, work, baseline, root):
                calls.append((trainer.dataset, 'startup'))
                charge(work, 'additional_forward', 2)
                charge(work, 'additional_backward', 2)

            def recover(trainer, features, retained, work, baseline, root, **kwargs):
                calls.append((trainer.dataset, 'warmed_reference'))
                charge(work, 'additional_forward', 4)
                charge(work, 'additional_backward', 4)
                return {}, dict(case='KPLUS', export='retained')

            step = dict(decision=dict(accepted=False, reason='rounded_norm_exceeds_native'),
                        committed=dict(identical_native=True), raw_fsg=dict(norm=1.01, native_norm=1.))

            def run_case(trainer, features, label, strength, work, baseline, root, reference, **kwargs):
                calls.append((trainer.dataset, label))
                charge(work, 'ordinary_forward', 2)
                charge(work, 'ordinary_backward', 2)
                charge(work, 'adam', 1)
                if label != 'KPLUS':
                    charge(work, 'additional_forward', 4 if label == 'FSG_ZERO' else 6)
                    charge(work, 'additional_backward', 2)
                if label == 'FSG':
                    self.assertIs(kwargs['oracle_runner'], resume.formula_only)
                    charge(work, 'additional_forward', 5)
                    charge(work, 'additional_backward', 2)
                return {}, dict(case=label, export='new', hook=dict(steps=[step]))

            def serving(plan, dataset, cases, prepared, work, root):
                calls.append((dataset, 'serving'))
                self.assertEqual([row['case'] for row in cases], ['KPLUS', 'FSG_ZERO', 'FSG', 'SCALAR'])
                charge(work, 'serving_generation', 4)
                return dict(status='passed', generations=4)

            replacements = dict(FunctionalGeometryTrainer=trainer_factory, runtime_state=lambda model: {},
                recover_warmed_ordinary_reference=recover, warmup=startup, run_case=run_case,
                check_serving=serving, restore_origin=lambda *args: None,
                pin_retained_graph_runtime=lambda *args: None, retained_reference=lambda plan: {},
                reuse_gsm_cases=lambda *args: (dict(case='FSG_ZERO', export='old'),
                                              dict(case='FSG', export='old', reused_step=step)))
            for name, value in replacements.items():
                patches.enter_context(patch.object(resume, name, value))
            patches.enter_context(patch.object(resume.torch.cuda, 'reset_peak_memory_stats'))
            patches.enter_context(patch.object(resume.torch.cuda, 'empty_cache'))
            result = resume.run(output)
            self.assertEqual(calls, [('gsm8k', 'warmed_reference'), ('gsm8k', 'SCALAR'),
                ('gsm8k', 'serving'), ('mathqa', 'startup'), ('mathqa', 'KPLUS'),
                ('mathqa', 'FSG_ZERO'), ('mathqa', 'FSG'), ('mathqa', 'SCALAR'), ('mathqa', 'serving')])
            self.assertEqual(result['status'], 'passed')
            for kind, cap in resume.CUMULATIVE_LIMITS.items():
                self.assertEqual(result['work'][kind + '_completed'], cap)
            self.assertFalse(result['main_pilot_started'])
            self.assertFalse(result['pilot_mechanism_active'])
            self.assertEqual(result['next_scope'], 'stop_before_pilot_all_representative_corrections_fallback')


if __name__ == '__main__':
    unittest.main()
