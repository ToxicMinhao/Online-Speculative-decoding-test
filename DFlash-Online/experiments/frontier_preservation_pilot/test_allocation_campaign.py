"""CPU-only continuation ordering, source-index and inherited evaluation checks."""

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.ok_coding.storage import identity
from experiments.repair_conditional_kl_pilot import evaluation, analysis
from experiments.repair_conditional_kl_pilot.test_evaluation import Backend, inputs
from online_feedback.storage import write_json

from . import allocation_campaign as campaign


def fixture(root):
    root = Path(root)
    numerical, integration, clock = (
        root / name
        for name in ('NUMERICAL.json', 'INTEGRATION.json', 'PHASE_CLOCK.json')
    )
    write_json(
        numerical,
        dict(
            status='numerically_qualified',
            passed=True,
            restoration=dict(exact_resume=True),
        ),
    )
    write_json(
        integration,
        dict(
            status='passed',
            passed=True,
            production_qualified=True,
            candidate_active=True,
            restoration=dict(exact_resume=True),
        ),
    )
    write_json(
        clock,
        dict(
            first_gpu_action_epoch=time.time() - 60,
            elapsed_limit_s=14400,
            occupied_card_limit_s=28800,
            prior_occupied_card_s=267.424,
        ),
    )
    plan = dict(
        schema='frontier_allocation_production_v1',
        production_admitted=True,
        arms=list(campaign.ARMS),
        bootstrap_seed=20261125,
        event_id='event_002',
        configs={arm: '/native/' + arm + '.yaml' for arm in campaign.ARMS},
        devices=dict(A='GPU-a', B='GPU-b'),
        root=str(root),
        numerical_result=str(numerical),
        numerical_result_identity=identity(numerical),
        integration_result=str(integration),
        integration_result_identity=identity(integration),
        phase_clock=str(clock),
        phase_clock_identity=identity(clock),
        prior_occupied_card_s=400.0,
        collection_write_bound_bytes=4_000_000_000,
        continuation_write_bound_bytes=18_000_000_000,
        collection_wall_bound_s=500.0,
        continuation_wall_bound_s=6000.0,
        training_python='/native/python',
        source_root='/native/source',
    )
    path = root / 'PLAN.json'
    write_json(path, plan)
    return plan, path


class CampaignTests(unittest.TestCase):
    def test_exact_phase_order_and_physical_card_isolation_preserve_old_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, path = fixture(root)
            original = Path(plan['phase_clock']).read_bytes()
            calls = []

            def child(plan_path, current, phase, arm):
                command, environment = campaign.child_command(
                    plan_path, current, phase, arm
                )
                calls.append((phase, arm, environment['CUDA_VISIBLE_DEVICES']))
                self.assertEqual(
                    environment['PYTHONPATH'],
                    '/native/source:/native/source/training:/native/source/inference',
                )
                self.assertEqual(environment['PYTHONDONTWRITEBYTECODE'], '1')
                self.assertIn(campaign.MODULE, command)
                return dict(status='completed')

            with patch.object(campaign, 'verify_inputs'), patch.object(
                campaign, 'admit_stage'
            ) as admit, patch.object(
                campaign, 'boundary', return_value={}
            ), patch.object(
                campaign, 'timing_barrier'
            ) as quiet, patch.object(
                campaign, 'run_child', side_effect=child
            ), patch.object(
                campaign, 'shared_features'
            ), patch.object(
                campaign, 'summarize', return_value={}
            ):
                result = campaign.launch(path)
            self.assertEqual(
                calls,
                [
                    ('collect', None, 'GPU-a'),
                    ('train', 'CONTROL', 'GPU-b'),
                    ('train', 'FRONTIER_P', 'GPU-b'),
                    ('evaluate', None, 'GPU-a'),
                ],
            )
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['unattempted'], [])
            self.assertEqual(quiet.call_count, 8)
            self.assertEqual(
                [call.args[1] for call in admit.call_args_list],
                ['collection', 'continuation'],
            )
            self.assertEqual(Path(plan['phase_clock']).read_bytes(), original)

    def test_failed_phase_stops_and_cannot_reset_production_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, path = fixture(root)
            with patch.object(campaign, 'verify_inputs'), patch.object(
                campaign, 'admit_stage'
            ), patch.object(campaign, 'boundary', return_value={}), patch.object(
                campaign, 'timing_barrier'
            ), patch.object(
                campaign,
                'run_child',
                side_effect=RuntimeError('native collection failure'),
            ) as child, patch.object(
                campaign, 'summarize'
            ) as summary:
                with self.assertRaisesRegex(RuntimeError, 'native collection failure'):
                    campaign.launch(path)
                with self.assertRaisesRegex(FileExistsError, 'cost reset'):
                    campaign.launch(path)
            self.assertEqual(child.call_count, 1)
            summary.assert_not_called()
            receipt = campaign.read(root / 'LAUNCH.json')
            self.assertEqual(receipt['pending'], 'collect')
            self.assertEqual(
                receipt['unattempted'],
                ['train_CONTROL', 'train_FRONTIER_P', 'evaluate'],
            )

    def test_failed_numerical_or_inactive_candidate_cannot_admit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _ = fixture(directory)
            campaign.validate_production(plan)
            write_json(
                plan['integration_result'],
                dict(status='passed', candidate_active=False),
            )
            plan['integration_result_identity'] = identity(plan['integration_result'])
            with self.assertRaisesRegex(ValueError, 'activity'):
                campaign.validate_production(plan)
            write_json(
                plan['integration_result'],
                dict(
                    status='passed',
                    passed=True,
                    production_qualified=True,
                    candidate_active=True,
                    restoration=dict(exact_resume=True),
                ),
            )
            plan['integration_result_identity'] = identity(plan['integration_result'])
            write_json(
                plan['numerical_result'], dict(status='numerical_qualification_failed')
            )
            plan['numerical_result_identity'] = identity(plan['numerical_result'])
            with self.assertRaisesRegex(ValueError, 'Numerical'):
                campaign.validate_production(plan)

    def test_selection_keeps_original_ids_and_source_indices(self):
        rows = [
            dict(request_id='original_' + str(index), request_index=index)
            for index in range(256)
        ]
        with patch.object(
            campaign, 'identity', return_value=campaign.INPUT_IDENTITY
        ), patch.object(
            campaign, 'read', return_value=dict(train_requests=rows)
        ), patch.object(
            campaign, 'validate_manifest'
        ):
            _, selected = campaign.selected_inputs({'manifest': 'authenticated-full'})
        self.assertEqual(selected, rows[128:256])
        self.assertEqual(selected[0]['request_index'], 128)
        self.assertEqual(selected[-1]['request_index'], 255)
        self.assertEqual(selected[0]['request_id'], 'original_128')

    def test_watchdog_accounts_failed_owned_child_and_keeps_prior_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, path = fixture(root)
            write_json(
                root / 'OCCUPIED_CARDS.json',
                dict(completed_card_s=12.0, active=None, finished=[]),
            )
            process = SimpleNamespace(pid=123, returncode=None)
            process.poll = lambda: process.returncode

            def stop(current):
                self.assertIs(current, process)
                current.returncode = -15

            with patch.object(
                campaign.subprocess, 'Popen', return_value=process
            ), patch.object(
                campaign, 'boundary', side_effect=TimeoutError('global deadline')
            ), patch.object(
                campaign, 'stop_owned', side_effect=stop
            ), patch.object(
                campaign, 'timing_barrier'
            ):
                with self.assertRaisesRegex(TimeoutError, 'global deadline'):
                    campaign.run_child(path, plan, 'train', 'CONTROL')
            clock = campaign.clock_status(plan)
            self.assertGreaterEqual(clock['production_occupied_card_s'], 12.0)
            self.assertGreaterEqual(clock['cumulative_occupied_card_s'], 412.0)
            self.assertIsNone(clock['active'])
            receipt = campaign.read(root / 'phases/train_CONTROL.launch.json')
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['exit_code'], -15)

    def test_unresolved_runtime_closure_remains_actively_charged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, path = fixture(root)
            write_json(
                root / 'OCCUPIED_CARDS.json',
                dict(completed_card_s=0.0, active=None, finished=[]),
            )
            process = SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)
            with patch.object(
                campaign.subprocess, 'Popen', return_value=process
            ), patch.object(campaign, 'stop_owned'), patch.object(
                campaign,
                'timing_barrier',
                side_effect=RuntimeError('orphan still active'),
            ):
                with self.assertRaisesRegex(RuntimeError, 'orphan still active'):
                    campaign.run_child(path, plan, 'collect')
            charge = campaign.read(root / 'OCCUPIED_CARDS.json')
            self.assertEqual(charge['active']['phase'], 'collect')
            self.assertFalse(charge['finished'])
            receipt = campaign.read(root / 'phases/collect.launch.json')
            self.assertFalse(receipt['clean_process_closure'])
            self.assertEqual(
                receipt['occupied_card_status'], 'unresolved_still_charged'
            )

    def test_stage_admission_uses_actual_charged_bytes_and_old_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _ = fixture(root)
            write_json(
                root / 'OCCUPIED_CARDS.json',
                dict(completed_card_s=0.0, active=None, finished=[]),
            )
            with patch.object(
                campaign,
                'total_storage',
                return_value=dict(
                    all_locations_charged_bytes=5_000_000_000,
                    free_bytes=425_000_000_000,
                ),
            ):
                receipt = campaign.admit_stage(plan, 'continuation')
            self.assertEqual(receipt['projected_peak_allocated_bytes'], 23_000_000_000)
            with patch.object(
                campaign,
                'total_storage',
                return_value=dict(
                    all_locations_charged_bytes=24_000_000_000,
                    free_bytes=425_000_000_000,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, 'storage'):
                    campaign.admit_stage(plan, 'continuation')

    def test_evaluation_uses_explicit_new_labels_all792_and_new_bootstrap_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _ = fixture(root)
            panel, warmups, canonicals, warmup_canonicals = inputs()
            plan.update(
                canonicals=str(root / 'CANONICALS.json'),
                warmup_canonicals=str(root / 'WARMUPS.json'),
                canonical_provenance={'source': 'PR56'},
                execution_commit='fixture',
            )
            write_json(plan['canonicals'], canonicals)
            write_json(plan['warmup_canonicals'], warmup_canonicals)
            for arm in campaign.ARMS:
                export = root / arm
                export.mkdir()
                (export / 'model.safetensors').write_bytes(arm.encode())
                write_json(
                    root / 'arms' / arm / 'STATUS.json',
                    dict(
                        status='completed',
                        global_step=256,
                        checkpoint=str(export),
                        checkpoint_identity=identity(export / 'model.safetensors'),
                    ),
                )
            backend = Backend()
            backend.failed, backend.loading_s = False, 0.0
            backend.publish = lambda checkpoint: setattr(
                backend, 'checkpoint', checkpoint
            )
            backend._call = lambda name: dict(status='passed')
            backend.close = Mock()
            with patch.object(
                campaign,
                'selected_inputs',
                return_value=(
                    dict(
                        evaluation_requests=panel
                        + [{'unauthorized_panel_tail': True}] * 896,
                        warmup_requests=warmups,
                    ),
                    [],
                ),
            ), patch.object(
                campaign, 'create_backend', return_value=backend
            ), patch.object(
                campaign, 'boundary', return_value={}
            ), patch.object(
                campaign, 'quiet_evaluation', return_value={}
            ):
                result = campaign.evaluate_endpoints(plan)
            backend.close.assert_called_once()
            self.assertEqual(result['completed_valid'], 792)
            rows = [
                json.loads(line)
                for line in (root / 'evaluation/MEASUREMENTS.jsonl')
                .read_text()
                .splitlines()
            ]
            self.assertEqual({row['method'] for row in rows}, set(campaign.ARMS))
            summary = analysis.summarize(
                rows, methods=campaign.ARMS, bootstrap_seed=20261125
            )
            self.assertEqual(summary['bootstrap']['seed'], 20261125)
            self.assertEqual(summary['requests'], 128)
            self.assertEqual(set(summary['contrasts']), {'FRONTIER_P_vs_CONTROL'})
            self.assertEqual(
                summary['contrasts']['FRONTIER_P_vs_CONTROL']['baseline'], 'CONTROL'
            )
            old_schedule = evaluation.evaluation_schedule(panel)
            self.assertEqual({row['method'] for row in old_schedule}, {'BASE', 'PAIR'})


if __name__ == '__main__':
    unittest.main()
