"""CPU fixtures for qualification trust, immutable source and blocked admission."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.repair_conditional_kl_pilot import prepare_production as P
from experiments.repair_conditional_kl_pilot import production_storage as S
from experiments.repair_conditional_kl_pilot.test_resources_contract import (
    inputs_fixture,
)


def qualification_fixture(root):
    old = root / 'qualified'
    source = root / 'qualified_source.json'
    write_json(source, dict(commit=P.QUALIFIED_COMMIT, files={}))
    plan = dict(
        root=str(old),
        base_commit=P.BASE_COMMIT,
        execution_commit=P.QUALIFIED_COMMIT,
        execution_mode='qualification_only',
        production_admitted=False,
        source_manifest=str(source),
        sealed_files={str(source): identity(source)},
    )
    plan['binding'] = P.digest(plan)
    write_json(old / 'sealed/PLAN.json', plan)
    write_json(
        old / 'QUALIFICATION.json',
        dict(
            status='passed',
            execution_commit=P.QUALIFIED_COMMIT,
            arms=['BASE', 'PAIR'],
            optimizer_steps=2,
            production_started=False,
        ),
    )
    write_json(
        old / 'LAUNCH.json',
        dict(
            status='completed',
            execution_commit=P.QUALIFIED_COMMIT,
            automatic_retry=False,
            clock=dict(elapsed_s=288.5, occupied_s=554.4),
        ),
    )
    write_json(old / 'CPU_PREFLIGHT.json', dict(status='passed', model_work=0))
    write_json(
        old / 'FINAL_CPU_AUDIT.json',
        dict(
            status='passed',
            execution_commit=P.QUALIFIED_COMMIT,
            cuda_initialized=False,
            exact_initialization_comparison=dict(exact_weights_empty_Adam_RNG=True),
            raw_native_records=[dict(valid=True)] * 8,
        ),
    )
    work = dict(
        feedback_generation=2,
        teacher_preparation=2,
        training_initialization=1,
        response_forward=2,
        response_backward=2,
        adam=1,
        evaluation_generation=2,
    )
    for arm in P.ARMS:
        write_json(
            old / f'qualification/{arm}/TRAINING_QUALIFICATION.json',
            dict(
                status='passed',
                arm=arm,
                execution_commit=P.QUALIFIED_COMMIT,
                binding=plan['binding'],
                global_step=1,
                presentations=2,
                exact_export_and_state_reload=True,
            ),
        )
        write_json(
            old / f'qualification_inputs/{arm}/STATUS.json',
            dict(
                status='passed',
                phase='completed',
                execution_commit=P.QUALIFIED_COMMIT,
                canonical_and_repeat_checks=True,
            ),
        )
        write_json(
            old / f'qualification_inputs/{arm}/WORK.json',
            dict(pending={}, attempted=work, completed=work),
        )
        write_json(
            old / f'qualification/{arm}/startup/EXACT_RESTORATION.json',
            dict(exact_equal=True),
        )
        for name in (
            'EXPORT_RELOAD_0.json',
            'FROZEN_RELOAD.json',
            'EXPORT_RELOAD_1.json',
        ):
            write_json(
                old / f'qualification_inputs/{arm}/{name}',
                dict(errors=[], weights=dict(all_equal=True)),
            )
    publication = root / 'published.json'
    write_json(
        publication,
        dict(
            execution_commit=P.QUALIFIED_COMMIT,
            compact_retrieval={
                str(path.relative_to(old)): identity(path)
                for path in old.rglob('*.json')
            },
        ),
    )
    return old, source, publication


class ReuseAuthenticationTests(unittest.TestCase):
    def test_preparation_cannot_authenticate_a_different_child_interpreter(self):
        with patch.object(P, 'verify_native_runtime') as native:
            with self.assertRaisesRegex(ValueError, 'sealed training interpreter'):
                P.authenticate_runtime(
                    '/worker/python',
                    {},
                    {},
                    training_python='/different/python',
                    source_root=Path(P.__file__).parents[2],
                )
            native.assert_not_called()

    def test_actual_receipts_reuse_zero_work_and_reject_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            old, source, publication = qualification_fixture(Path(directory))
            with patch.multiple(
                P,
                PUBLISHED_ARTIFACTS=publication,
                PUBLISHED_ARTIFACTS_IDENTITY=identity(publication),
                QUALIFIED_SOURCE_IDENTITY=identity(source),
            ):
                plan, manifest, receipt = P.authenticate_qualification(old)
                self.assertEqual(receipt['status'], 'reused_passed')
                self.assertEqual(receipt['optimizer_steps_executed'], 0)
                self.assertEqual(receipt['model_forwards_executed'], 0)
                self.assertEqual(receipt['qualified_plan_binding'], plan['binding'])
                self.assertTrue(receipt['qualification_states_are_not_origins'])
                status = old / 'qualification_inputs/PAIR/STATUS.json'
                write_json(status, {**P.read(status), 'status': 'failed'})
                with self.assertRaisesRegex(
                    ValueError, 'Authenticated artifact changed'
                ):
                    P.authenticate_qualification(old)

    def test_self_consistent_failed_reload_cannot_be_called_reused_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            old, source, publication = qualification_fixture(Path(directory))
            reload = old / 'qualification_inputs/BASE/EXPORT_RELOAD_1.json'
            write_json(reload, dict(errors=['mismatch'], weights=dict(all_equal=False)))
            published = P.read(publication)
            published['compact_retrieval'][str(reload.relative_to(old))] = identity(
                reload
            )
            write_json(publication, published)
            with patch.multiple(
                P,
                PUBLISHED_ARTIFACTS=publication,
                PUBLISHED_ARTIFACTS_IDENTITY=identity(publication),
                QUALIFIED_SOURCE_IDENTITY=identity(source),
            ):
                with self.assertRaisesRegex(ValueError, 'reload did not pass'):
                    P.authenticate_qualification(old)

    def test_source_allows_only_reviewed_plumbing_and_rejects_science_or_hidden_code(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scientific = root / 'online_feedback/objective.py'
            scientific.parent.mkdir()
            scientific.write_text('qualified = True\n')
            plumbing = root / (P.PILOT + 'witness.py')
            plumbing.parent.mkdir(parents=True)
            plumbing.write_text('old = True\n')
            old = dict(
                commit=P.QUALIFIED_COMMIT,
                files={
                    str(p.relative_to(root)): identity(p)
                    for p in (scientific, plumbing)
                },
            )
            plumbing.write_text('new_observation = True\n')
            new = dict(
                commit='new-commit',
                files={
                    str(p.relative_to(root)): identity(p)
                    for p in (scientific, plumbing)
                },
            )
            checked = P.verify_source_reuse(old, new, root, 'new-commit')
            self.assertEqual(
                checked['reviewed_plumbing_changes'], [P.PILOT + 'witness.py']
            )
            hidden = root / 'hidden.py'
            hidden.write_text('unbound = True\n')
            with self.assertRaisesRegex(ValueError, 'Unmanifested executable'):
                P.verify_source_reuse(old, new, root, 'new-commit')
            hidden.unlink()
            scientific.write_text('changed_loss = True\n')
            new['files']['online_feedback/objective.py'] = identity(scientific)
            with self.assertRaisesRegex(ValueError, 'Qualification-invalidating'):
                P.verify_source_reuse(old, new, root, 'new-commit')


class PreparationTests(unittest.TestCase):
    def prepare_fixture(
        self, root, *, admitted, boundary_error=None, compiler_elsewhere=False
    ):
        source = root / 'source/DFlash-Online'
        source.mkdir(parents=True)
        source_file = source / (P.PILOT + 'production.py')
        source_file.parent.mkdir(parents=True)
        source_file.write_text('production = True\n')
        manifest = dict(
            commit='new-commit',
            files={P.PILOT + 'production.py': identity(source_file)},
        )
        write_json(source.parent / 'SOURCE.json', manifest)
        originals = root / 'originals'
        originals.mkdir()
        original = P.PUBLISHED_ARTIFACTS.parent / 'BASE.yaml'
        (originals / 'ORIGINAL_CONFIG.yaml').write_bytes(original.read_bytes())
        for name in ('MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
            write_json(originals / name, {})
        paths = {p.name: p for p in originals.iterdir()}
        prior = dict(
            target='/released/target',
            draft='/released/draft',
            worker_python='/worker/python',
            training_python='/training/python',
            devices={'A': 'GPU-A', 'B': 'GPU-B'},
            canonical_provenance={},
        )
        evidence_path = root / 'original_pass.json'
        write_json(evidence_path, dict(status='passed'))
        reuse = dict(
            status='reused_passed',
            qualified_execution_commit=P.QUALIFIED_COMMIT,
            optimizer_steps_executed=0,
            evidence={str(evidence_path): identity(evidence_path)},
        )
        authenticated = dict(
            prior=prior,
            paths=paths,
            selected=inputs_fixture(),
            canonicals={},
            warmups={},
            model_identities=[{'role': 'fixture'}],
            source_identities={},
        )
        storage = dict(
            admitted=admitted,
            strategy='local_single',
            projected_peak_bytes=52_615_699_048,
            groups=[dict(name='main', filesystem=dict(capacity_key='main', device=1))],
            blockers=(
                []
                if admitted
                else ['main: projected peak misses reserve by 27469621864 bytes']
            ),
        )
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.object(
                P,
                'authenticate_qualification',
                return_value=({}, dict(commit=P.QUALIFIED_COMMIT, files={}), reuse),
            )
        )
        stack.enter_context(
            patch.object(P, 'authenticate_inputs', return_value=authenticated)
        )
        stack.enter_context(
            patch.object(
                P, 'authenticate_runtime', return_value=dict(cuda_initialized=False)
            )
        )
        stack.enter_context(
            patch.object(
                P,
                'compiler_snapshot',
                return_value=dict(
                    directories=(
                        {'cache': '/compiler/cache'} if compiler_elsewhere else {}
                    ),
                    allocated_bytes=0,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                S,
                'filesystem',
                return_value=dict(
                    capacity_key='other', device=2, readonly=False, durable=True
                ),
            )
        )
        stack.enter_context(patch.object(S, 'prepare_storage', return_value=storage))
        boundary = stack.enter_context(
            patch.object(
                P,
                'boundary',
                return_value={'checked': True},
                side_effect=boundary_error,
            )
        )
        run = root / 'new-run'
        result = P.prepare(
            '/old-source', '/old-qualification', run, 'new-commit', source
        )
        return run, result, boundary

    def test_blocked_preflight_seals_fresh_configs_reuse_and_exact_counts_without_clock(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            run, result, boundary = self.prepare_fixture(
                Path(directory), admitted=False
            )
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['authentication_status'], 'passed')
            boundary.assert_not_called()
            plan = P.read(run / 'sealed/PLAN.json')
            self.assertFalse(plan['production_admitted'])
            self.assertEqual(plan['execution_mode'], 'one_event_production')
            self.assertEqual(plan['expected_production_work']['adam'], 256)
            self.assertEqual(plan['expected_production_work']['scored_generation'], 768)
            self.assertEqual(plan['expected_production_work']['warmup_generation'], 24)
            self.assertEqual(
                P.read(plan['qualification_reuse'])['optimizer_steps_executed'], 0
            )
            self.assertIn(plan['qualification_reuse'], plan['sealed_files'])
            self.assertFalse((run / 'CLOCK.json').exists())
            self.assertFalse((run / 'QUALIFICATION.json').exists())
            self.assertNotEqual(
                Path(plan['configs']['BASE']).read_text(),
                Path(plan['configs']['PAIR']).read_text(),
            )
            self.assertIn('max_steps: 7424', Path(plan['configs']['PAIR']).read_text())
            with self.assertRaisesRegex(FileExistsError, 'unused run root'):
                P.prepare(
                    '/old-source',
                    '/old-qualification',
                    run,
                    'new-commit',
                    run.parent / 'source/DFlash-Online',
                )

    def test_admitted_preflight_requires_live_boundary_and_can_retain_rejection(self):
        for error in (None, RuntimeError('filesystem changed before admission')):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                run, result, boundary = self.prepare_fixture(
                    Path(directory), admitted=True, boundary_error=error
                )
                boundary.assert_called_once()
                self.assertEqual(
                    result['status'], 'passed' if error is None else 'blocked'
                )
                self.assertEqual(
                    P.read(run / 'sealed/PLAN.json')['production_admitted'],
                    error is None,
                )
                self.assertFalse((run / 'CLOCK.json').exists())

    def test_compiler_cache_on_another_filesystem_blocks_before_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            run, result, boundary = self.prepare_fixture(
                Path(directory), admitted=True, compiler_elsewhere=True
            )
            self.assertEqual(result['status'], 'blocked')
            self.assertTrue(
                any('Compiler cache' in item for item in result['blockers'])
            )
            boundary.assert_not_called()
            self.assertFalse((run / 'CLOCK.json').exists())


if __name__ == '__main__':
    unittest.main()
