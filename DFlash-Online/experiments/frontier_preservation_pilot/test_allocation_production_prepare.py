"""CPU plan-sealing fixtures; parent model authentication is independently reused."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml

from experiments.ok_coding.storage import identity
from experiments.repair_conditional_kl_pilot.resources import verify_inputs
from experiments.repair_conditional_kl_pilot.test_evaluation import inputs
from online_feedback.storage import write_json
from . import allocation_production_prepare as prepare
from .allocation_contract import numerical_contract


def fixture(root):
    root = Path(root)
    parent_root, numerical, integration = (
        root / name for name in ('parent', 'numerical', 'integration')
    )
    source = root / 'source/DFlash-Online'
    source.mkdir(parents=True)
    (source / 'integrated.py').write_text('unchanged qualified code\n')
    (source / 'preparation.py').write_text('new CPU preparation only\n')
    integrated_source = root / 'integrated/SOURCE.json'
    write_json(
        integrated_source,
        dict(
            commit='integrated',
            files={'integrated.py': identity(source / 'integrated.py')},
        ),
    )
    source_manifest = root / 'source/SOURCE.json'
    write_json(
        source_manifest,
        dict(
            commit='production',
            files={
                name: identity(source / name)
                for name in ('integrated.py', 'preparation.py')
            },
        ),
    )
    panel, warmups, canonicals, warmup_canonicals = inputs()
    training = [
        dict(
            request_id='train' + str(index),
            request_index=index,
            prompt_ids=[index + 5000],
        )
        for index in range(256)
    ]
    full_manifest = root / 'full/INPUTS.json'
    write_json(
        full_manifest,
        dict(
            train_requests=training,
            evaluation_requests=panel + [{'unused_tail': True}] * 896,
            warmup_requests=warmups,
        ),
    )
    parent_seal = parent_root / 'sealed'
    write_json(
        parent_seal / 'INPUTS.json',
        dict(
            train_requests=training[:128],
            evaluation_requests=panel,
            warmup_requests=warmups,
        ),
    )
    write_json(parent_seal / 'CANONICALS.json', canonicals)
    write_json(parent_seal / 'WARMUP_CANONICALS.json', warmup_canonicals)
    native_config = dict(
        model={'dtype': 'bfloat16'},
        training={'max_steps': 7424, 'learning_rate': 0.0003},
        online_feedback={'max_events': 58},
        run_id='original-pair',
        output_dir='/original/output',
    )
    for name in (*prepare.PIN_NAMES, 'PAIR.yaml'):
        destination = parent_seal / name
        if name.endswith('.yaml'):
            destination.write_text(yaml.safe_dump(native_config))
        else:
            write_json(destination, {'pin': name})
    parent = dict(
        root=str(parent_root),
        manifest=str(parent_seal / 'INPUTS.json'),
        configs={'PAIR': str(parent_seal / 'PAIR.yaml')},
        target='/fixed/target',
        draft='/fixed/released-draft',
        worker_python='/fixed/worker-python',
        training_python='/fixed/train-python',
        devices=dict(A='GPU-a', B='GPU-b'),
        canonical_provenance={'source': 'PR56'},
        model_identities={'fixed': {'bytes': 1}},
        serving={'block_size': 16},
        sealed_files={str(path): identity(path) for path in parent_seal.iterdir()},
    )
    parent_plan = parent_seal / 'PLAN.json'
    write_json(parent_plan, parent)
    export = parent_root / 'export/model.safetensors'
    export.parent.mkdir()
    export.write_bytes(b'parent')
    clock = root / 'ALLOCATION_PHASE_CLOCK.json'
    clock_data = dict(
        first_gpu_action_epoch=time.time() - 200,
        elapsed_limit_s=14400,
        occupied_card_limit_s=28800,
        prior_occupied_card_s=267.424,
    )
    write_json(clock, clock_data)
    write_json(numerical / 'CLOCK.json', clock_data)
    contract = numerical / 'NUMERICAL_CONTRACT.json'
    write_json(contract, numerical_contract())
    numerical_plan = dict(
        parent_plan=str(parent_plan),
        numerical_contract=str(contract),
        numerical_contract_identity=identity(contract),
        full_input_manifest=str(full_manifest),
        parent_export=str(export),
        prior_record_identities={},
        compiler_baseline={'directories': {}},
        prior_charged_roots=[],
        source_manifest=str(integrated_source),
    )
    write_json(numerical / 'PLAN.json', numerical_plan)
    numerical_result = numerical / 'native/RESULT.json'
    write_json(
        numerical_result,
        dict(
            status='numerically_qualified',
            passed=True,
            native_adam_steps=0,
            restoration=dict(exact_resume=True),
            numerical_contract_identity=identity(contract),
        ),
    )
    write_json(
        numerical / 'QUALIFICATION_PROCESS.json',
        dict(
            exit_code=0,
            stopped_reason=None,
            final_storage_error=None,
            occupied_card_s=120.0,
            cumulative_occupied_card_s=387.424,
        ),
    )
    integration_plan = dict(
        source_root=str(source),
        parent_plan=str(parent_plan),
        numerical_result=str(numerical_result),
        numerical_result_identity=identity(numerical_result),
        prior_occupied_card_s=387.424,
        phase_clock=str(clock),
        source_manifest=str(integrated_source),
        source_identity=identity(integrated_source),
    )
    write_json(integration / 'PLAN.json', integration_plan)
    write_json(
        integration / 'native/RESULT.json',
        dict(
            status='passed',
            passed=True,
            production_qualified=True,
            candidate_active=True,
            native_adam_steps=4,
            restoration=dict(exact_resume=True),
            numerical_audit_identity=identity(numerical_result),
            complete_candidate_window_s=10.0,
        ),
    )
    write_json(
        integration / 'QUALIFICATION_PROCESS.json',
        dict(
            exit_code=0,
            stopped_reason=None,
            final_storage_error=None,
            occupied_card_s=100.0,
            prior_occupied_card_s=387.424,
            cumulative_occupied_card_s=487.424,
        ),
    )
    write_json(
        root / 'ALLOCATION_REFERENCE_LEDGER.json',
        dict(attempts={'1': {'records': [{}] * 58}}),
    )
    write_json(
        root / 'ALLOCATION_STEP_LEDGER.json', dict(steps=[{}] * 4, fixed_windows=[{}])
    )
    return dict(
        parent=parent,
        parent_plan=parent_plan,
        numerical=numerical,
        integration=integration,
        source=source,
        source_manifest=source_manifest,
        manifest=full_manifest,
        export=export,
        output=root / 'production_001',
        clock=clock,
    )


def fixture_patches(values):
    context = ExitStack()
    context.enter_context(patch.object(prepare, 'SOURCE_ROOT', values['source']))
    context.enter_context(
        patch.object(prepare, 'INPUT_IDENTITY', identity(values['manifest']))
    )
    context.enter_context(
        patch.object(prepare, 'PARENT_EXPORT_IDENTITY', identity(values['export']))
    )
    context.enter_context(
        patch.object(
            prepare,
            'select_parent',
            return_value=(values['parent'], '/parent/recovery.pt', []),
        )
    )
    context.enter_context(patch.object(prepare, 'validate_manifest'))
    return context


def execute(values):
    return prepare.prepare(
        values['parent_plan'],
        values['numerical'],
        values['integration'],
        values['output'],
        values['source_manifest'],
        'production',
        500_000_000,
    )


class ProductionPrepareTests(unittest.TestCase):
    def test_metadata_only_config_clone_preserves_native_recipe(self):
        original = {
            'training': {'max_steps': 7424, 'lr': 0.0003},
            'online_feedback': {'max_events': 58},
            'run_id': 'parent',
            'output_dir': '/parent',
        }
        for arm in prepare.ARMS:
            current = yaml.safe_load(
                prepare.clone_parent_config(
                    yaml.safe_dump(original), arm, '/new/' + arm
                )
            )
            self.assertEqual(current['training'], original['training'])
            self.assertEqual(current['online_feedback'], original['online_feedback'])
            self.assertEqual(current['output_dir'], '/new/' + arm)
            self.assertNotEqual(current['run_id'], original['run_id'])

    def test_forecast_accounts_for_full_outputs_one_overlap_and_qualified_window(self):
        requests = [dict(prompt_ids=[1] * (index + 1)) for index in range(128)]
        result = prepare.forecasts(requests, 10.0)
        rows = sum(index + 1 + 2048 for index in range(128)) + 128 + 2049
        self.assertEqual(
            result['collection_write_bound_bytes'],
            (49164 * rows * 105 + 99) // 100 + 2_000_000_000,
        )
        self.assertEqual(result['continuation_write_bound_bytes'], 20_000_000_000)
        self.assertEqual(
            result['continuation_wall_bound_s'], 1.5 * (1280 + 554 + 914 + 600)
        )
        self.assertEqual(result['collection_wall_bound_s'], 900.0)
        with self.assertRaises(ValueError):
            prepare.forecasts(requests, float('nan'))

    def test_production_cannot_change_any_integrated_source_file(self):
        old = dict(commit='old', files={'native.py': {'sha256': 'a'}})
        new = dict(
            commit='new',
            files={'native.py': {'sha256': 'a'}, 'prepare.py': {'sha256': 'b'}},
        )
        self.assertEqual(prepare.source_continuity(new, old)['status'], 'passed')
        new['files']['native.py']['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'integrated source'):
            prepare.source_continuity(new, old)

    def test_complete_seal_has_correct_panels_parent_export_directory_and_same_clock(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            values = fixture(directory)
            original_clock = values['clock'].read_bytes()
            with fixture_patches(values):
                result = execute(values)
            self.assertEqual(result['status'], 'prepared')
            self.assertEqual(result['model_work'], 0)
            self.assertFalse(result['live_resource_admission'])
            plan = json.loads(Path(result['plan']).read_text())
            self.assertEqual(plan['prior_occupied_card_s'], 487.424)
            self.assertEqual(plan['parent_export'], str(values['export'].parent))
            self.assertEqual(plan['manifest'], str(values['manifest']))
            self.assertEqual(plan['bootstrap_seed'], 20261125)
            self.assertEqual(
                set(json.loads(Path(plan['canonicals']).read_text())),
                {'r' + str(i) for i in range(128)},
            )
            selected = json.loads(
                (values['output'] / 'sealed/INPUT_SELECTION.json').read_text()
            )['training']
            self.assertEqual(
                [row['source_index'] for row in selected], list(range(128, 256))
            )
            self.assertEqual(selected[0]['local_ordinal'], 0)
            self.assertEqual(values['clock'].read_bytes(), original_clock)
            self.assertFalse((values['output'] / 'OCCUPIED_CARDS.json').exists())
            self.assertEqual(verify_inputs(plan)['status'], 'passed')
            with fixture_patches(values), self.assertRaisesRegex(
                FileExistsError, 'new root'
            ):
                execute(values)

    def test_incomplete_integration_and_spent_oracle_allowance_reject_before_output(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            values = fixture(directory)
            path = values['integration'] / 'native/RESULT.json'
            original = json.loads(path.read_text())
            write_json(path, dict(original, candidate_active=False))
            with fixture_patches(values), self.assertRaisesRegex(
                ValueError, 'activity'
            ):
                execute(values)
            self.assertFalse(values['output'].exists())
            write_json(path, original)
            write_json(
                Path(directory) / 'ALLOCATION_REFERENCE_LEDGER.json',
                dict(attempts={'1': {'records': [{}] * 61}}),
            )
            with fixture_patches(values), self.assertRaisesRegex(
                ValueError, 'allowance'
            ):
                execute(values)
            self.assertFalse(values['output'].exists())

    def test_insufficient_existing_clock_seals_blocked_plan_without_new_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            values = fixture(directory)
            old = json.loads(values['clock'].read_text())
            old['first_gpu_action_epoch'] = time.time() - 14000
            write_json(values['clock'], old)
            write_json(values['numerical'] / 'CLOCK.json', old)
            with fixture_patches(values):
                result = execute(values)
            self.assertEqual(result['status'], 'time_forecast_blocked')
            self.assertFalse(result['production_admitted'])
            self.assertFalse(result['clock_started'])
            self.assertFalse(
                json.loads(Path(result['plan']).read_text())['production_admitted']
            )

    def test_only_exact_reviewed_three_reporting_lines_can_differ(self):
        relative = 'experiments/frontier_preservation_pilot/allocation_training.py'
        before = (
            "                fast = scorer.fast_scores(preview['w'])\n"
            "                    pass_index=pass_index)\n"
            "                diagnostics.update(total_window_s=time.perf_counter()-started,\n"
        )
        after = (
            "                fast = scorer.fast_scores(preview['w'])\n"
            "                torch.cuda.synchronize()\n"
            "                    pass_index=pass_index)\n"
            "                work.value['native_adam_steps'] = 1\n"
            "                work.save()\n"
            "                diagnostics.update(total_window_s=time.perf_counter()-started,\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path, new_path = root / 'old' / relative, root / 'new' / relative
            old_path.parent.mkdir(parents=True)
            new_path.parent.mkdir(parents=True)
            old_path.write_text(before)
            new_path.write_text(after)
            old = dict(commit='integrated', files={relative: identity(old_path)})
            new = dict(commit='production', files={relative: identity(new_path)})
            receipt = prepare.source_continuity(
                new, old, source_root=root / 'new', integrated_root=root / 'old'
            )
            self.assertEqual(receipt['unchanged_integrated_files'], 0)
            self.assertEqual(len(receipt['reviewed_reporting_changes']), 1)
            self.assertIn(
                '+                torch.cuda.synchronize()',
                receipt['reviewed_reporting_changes'][0]['diff'],
            )
            new_path.write_text(after + 'unreviewed_scientific_change = True\n')
            new['files'][relative] = identity(new_path)
            with self.assertRaisesRegex(ValueError, 'beyond the reviewed'):
                prepare.source_continuity(
                    new, old, source_root=root / 'new', integrated_root=root / 'old'
                )


if __name__ == '__main__':
    unittest.main()
