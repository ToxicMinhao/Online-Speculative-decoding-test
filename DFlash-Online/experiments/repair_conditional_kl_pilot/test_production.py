"""CPU sequencing fixtures for the new one-event production coordinator."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from . import campaign, production
from .test_evaluation import Backend, inputs


def plan_fixture(root):
    root = Path(root)
    reuse = root / 'QUALIFICATION_REUSE.json'
    write_json(
        reuse,
        dict(
            status='reused_passed',
            qualified_execution_commit='79b8c076610e9b4d310169b3891cb76ac6d0f9bb',
            optimizer_steps_executed=0,
            historical_qualification_costs=dict(elapsed_s=288.524),
        ),
    )
    write_json(root / 'CPU_PREFLIGHT.json', dict(status='passed'))
    plan = dict(
        root=str(root),
        execution_commit='fixture',
        execution_mode='one_event_production',
        production_admitted=True,
        maximum_events=1,
        arms=['BASE', 'PAIR'],
        feature_strategy='local_split',
        qualification_reuse=str(reuse),
        feature_roots={arm: str(root / 'features' / arm) for arm in ('BASE', 'PAIR')},
        devices=dict(A='GPU-a', B='GPU-b'),
        training_python='/native/python',
        source_root='/native/source',
    )
    path = root / 'PLAN.json'
    write_json(path, plan)
    return plan, path


class ProductionTests(unittest.TestCase):
    def test_serial_arms_share_training_b_then_quiet_a_evaluation_without_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, path = plan_fixture(root)
            calls = []

            def child(plan_path, current, phase, arm):
                command, environment = production.child_command(plan_path, current, phase, arm)
                active = production.clock_status(root)['active']
                calls.append(
                    (
                        phase,
                        arm,
                        environment['CUDA_VISIBLE_DEVICES'],
                        sorted(row['physical_gpu_uuid'] for row in active.values()),
                    )
                )
                self.assertIn(production.MODULE, command)
                self.assertEqual(
                    environment['PYTHONPATH'],
                    '/native/source:/native/source/training:/native/source/inference',
                )
                self.assertEqual(environment['PYTHONDONTWRITEBYTECODE'], '1')
                return dict(status='completed', phase=phase, arm=arm, exit_code=0)

            with patch.object(production, 'verify_inputs'), patch.object(
                production, 'boundary', return_value={}
            ), patch.object(production, 'timing_barrier', return_value={}), patch.object(
                production, 'run_child', side_effect=child
            ), patch.object(
                production, 'summarize_production', return_value={'status': 'completed'}
            ), patch(
                'experiments.repair_conditional_kl_pilot.qualification.run_training_qualification'
            ) as qualify:
                result = production.launch(path)
            self.assertEqual(
                calls,
                [
                    ('train', 'BASE', 'GPU-b', ['GPU-a', 'GPU-b']),
                    ('train', 'PAIR', 'GPU-b', ['GPU-a', 'GPU-b']),
                    ('evaluate', None, 'GPU-a', ['GPU-a']),
                ],
            )
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['unattempted'], [])
            self.assertFalse(result['qualification_executed'])
            self.assertEqual(result['clock']['active'], {})
            self.assertEqual(len(production.read(root / 'CLOCK.json')['finished']), 5)
            qualify.assert_not_called()

    def test_failure_stops_later_phases_keeps_unknown_work_and_cannot_restart_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, path = plan_fixture(root)
            pending = dict(
                attempted={'adam': 1}, completed={}, pending={'0': {'counts': {'adam': 1}}}
            )

            def failed(*args):
                write_json(root / 'arms' / 'BASE' / 'WORK.json', pending)
                raise RuntimeError('unexplained native failure')

            with patch.object(production, 'verify_inputs'), patch.object(
                production, 'boundary', return_value={}
            ), patch.object(production, 'timing_barrier', return_value={}), patch.object(
                production, 'run_child', side_effect=failed
            ) as child, patch.object(
                production, 'summarize_production'
            ) as summary:
                with self.assertRaisesRegex(RuntimeError, 'unexplained'):
                    production.launch(path)
                first_clock = production.read(root / 'CLOCK.json')
                with self.assertRaisesRegex(FileExistsError, 'already exists'):
                    production.launch(path)
            self.assertEqual(child.call_count, 1)
            summary.assert_not_called()
            result = production.read(root / 'LAUNCH.json')
            self.assertEqual(result['pending'], 'train_BASE')
            self.assertEqual(result['unattempted'], ['train_PAIR', 'evaluate'])
            self.assertEqual(result['clock']['active'], {})
            self.assertEqual(production.read(root / 'arms' / 'BASE' / 'WORK.json'), pending)
            self.assertEqual(production.read(root / 'CLOCK.json'), first_clock)

    def test_blocked_or_unimplemented_storage_never_initializes_a_gpu_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            plan, path = plan_fixture(folder)
            for updates in (
                dict(production_admitted=False),
                dict(feature_strategy='verified_offload'),
                dict(maximum_events=2),
                dict(execution_mode='qualification_only'),
            ):
                current = dict(plan, **updates)
                write_json(path, current)
                with patch.object(production, 'verify_inputs'), patch.object(
                    production, 'initialize'
                ) as initialize, patch.object(production, 'timing_barrier') as gpu_check:
                    with self.assertRaises(ValueError):
                        production.launch(path)
                    initialize.assert_not_called()
                    gpu_check.assert_not_called()

    def test_watchdog_terminates_only_the_owned_child_and_preserves_failed_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, path = plan_fixture(root)
            process = SimpleNamespace(pid=123, returncode=None)
            process.poll = lambda: process.returncode

            def terminate(current):
                self.assertIs(current, process)
                current.returncode = -15

            with patch.object(
                production.subprocess, 'Popen', return_value=process
            ) as spawn, patch.object(
                production, 'boundary', side_effect=TimeoutError('cumulative ceiling')
            ), patch.object(
                production, 'stop_owned', side_effect=terminate
            ) as stop:
                with self.assertRaisesRegex(TimeoutError, 'cumulative ceiling'):
                    production.run_child(path, plan, 'train', 'BASE')
            self.assertEqual(spawn.call_count, 1)
            stop.assert_called_once_with(process)
            result = production.read(root / 'phases' / 'train_BASE.launch.json')
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['terminated_by_watchdog'])
            self.assertEqual(result['exit_code'], -15)
            self.assertFalse(result['automatic_retry'])

    def test_direct_child_rejects_wrong_training_card_before_runtime_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            plan, path = plan_fixture(folder)
            with patch.object(production, 'verify_inputs'), patch.dict(
                production.os.environ, CUDA_VISIBLE_DEVICES='GPU-a'
            ), patch.object(production, 'run_arm') as train:
                with self.assertRaisesRegex(ValueError, 'declared physical card'):
                    production.run_phase(path, 'train', 'PAIR')
            train.assert_not_called()

    def test_reuse_is_explicit_and_feature_paths_do_not_require_a_fake_offload_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _ = plan_fixture(root)
            for arm in ('BASE', 'PAIR'):
                allocation, event, native = campaign.production_paths(plan, arm)
                self.assertEqual(allocation, Path(plan['feature_roots'][arm]))
                self.assertEqual(event, allocation / 'event_001')
                self.assertEqual(native, event / 'native')
            self.assertFalse((root / 'BASE_FEATURE_OFFLOAD.json').exists())
            receipt = production.read(plan['qualification_reuse'])
            receipt['optimizer_steps_executed'] = 2
            write_json(plan['qualification_reuse'], receipt)
            with self.assertRaisesRegex(ValueError, 'reuse'):
                production.validate_production(plan)

    def test_evaluation_rejects_any_compute_on_other_physical_card(self):
        plan = dict(devices=dict(A='GPU-a', B='GPU-b'))
        backend = SimpleNamespace(worker=SimpleNamespace(pid=40))
        with patch.object(
            production, 'timing_barrier', return_value=dict(status='passed')
        ) as quiet:
            with patch.object(production.subprocess, 'check_output', return_value='40, GPU-a\n'):
                self.assertEqual(
                    production.quiet_evaluation(plan, backend)['physical_gpu_uuid'], 'GPU-a'
                )
            quiet.assert_called_with(40)
            with patch.object(
                production.subprocess, 'check_output', return_value='40, GPU-a\n41, GPU-b\n'
            ):
                with self.assertRaisesRegex(ValueError, 'second card idle'):
                    production.quiet_evaluation(plan, backend)

    def test_main_filesystem_witness_retention_keeps_exact_first_eight_packets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'external'
            source.mkdir()
            paths = []
            for index in range(9):
                path = source / f'{index}.pt'
                path.write_bytes(bytes([index]) * 25)
                write_json(path.with_suffix('.json'), dict(source_ordinal=index))
                paths.append(path)
            retained = campaign.retain_witness_features(paths, root / 'witness')
            self.assertEqual(len(retained), 8)
            for source, destination in zip(paths, retained):
                self.assertEqual(identity(source), identity(destination))
                self.assertEqual(
                    production.read(destination.with_suffix('.json')),
                    production.read(source.with_suffix('.json')),
                )
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(
                production.read(root / 'witness' / 'FEATURE_RETENTION.json')['status'], 'verified'
            )

    def test_production_invokes_all_existing_scored_warmup_validators_and_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _ = plan_fixture(root)
            panel, warmups, canonicals, warmup_canonicals = inputs()
            plan.update(manifest=str(root / 'INPUTS.json'), canonical_provenance={'source': 'PR56'})
            write_json(plan['manifest'], dict(evaluation_requests=panel, warmup_requests=warmups))
            write_json(root / 'sealed' / 'CANONICALS.json', canonicals)
            write_json(root / 'sealed' / 'WARMUP_CANONICALS.json', warmup_canonicals)
            for arm in ('BASE', 'PAIR'):
                export = root / arm
                export.mkdir()
                (export / 'model.safetensors').write_bytes(arm.encode())
                write_json(
                    root / 'arms' / arm / 'STATUS.json',
                    dict(
                        status='completed',
                        phase='completed',
                        checkpoint=str(export),
                        checkpoint_identity=identity(export / 'model.safetensors'),
                    ),
                )
                write_json(
                    root / 'arms' / arm / 'WORK.json', dict(pending={}, attempted={}, completed={})
                )
            backend = Backend()
            backend.failed, backend.loading_s = False, 0.0
            backend.worker = SimpleNamespace(pid=40)
            backend.publish = lambda checkpoint: setattr(backend, 'checkpoint', checkpoint)
            backend._call = lambda name: dict(status='passed')
            backend.close = unittest.mock.Mock()
            with patch.object(production, 'boundary', return_value={}), patch.object(
                production, 'quiet_evaluation', return_value=dict(quiet=True)
            ), patch.object(production, 'create_backend', return_value=backend) as create:
                result = production.evaluate_endpoints(plan)
            self.assertEqual(result['completed_valid'], 792)
            self.assertEqual(
                production.read(root / 'evaluation_work.json')['completed'],
                dict(scored_generation=768, warmup_generation=24),
            )
            create.assert_called_once_with(plan, root / 'evaluation_native', card='A')
            backend.close.assert_called_once()
            production.initialize(plan)
            for name in ('origin', 'BASE1', 'PAIR1'):
                write_json(root / 'witness' / (name + '.json'), dict(endpoint=name))
            with patch(
                'experiments.repair_conditional_kl_pilot.witness.compare_matched_witnesses',
                return_value=dict(summary={'blocks': 64}, per_block=[]),
            ) as compare:
                summary = production.summarize_production(plan)
            compare.assert_called_once()
            self.assertEqual(production.read(summary['serving'])['requests'], 128)
            self.assertFalse(
                production.read(summary['costs'])['historical_qualification_reexecuted']
            )


if __name__ == '__main__':
    unittest.main()
