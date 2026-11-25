"""CPU supervisor fixtures: same clock, sealed imports, failed work retained."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from . import integration_launch as launch

USAGE = dict(all_locations_charged_bytes=1_000_000_000, free_bytes=430_000_000_000)


def fixture(root):
    root = Path(root)
    clock, audit = root / 'PHASE_CLOCK.json', root / 'audit/RESULT.json'
    write_json(
        clock,
        dict(
            first_gpu_action_epoch=time.time() - 200,
            elapsed_limit_s=14400,
            occupied_card_limit_s=28800,
        ),
    )
    write_json(
        audit,
        dict(
            status='numerically_qualified',
            passed=True,
            restoration=dict(status='passed', exact_resume=True),
        ),
    )
    plan = dict(
        schema='frontier_native_integration_admission_v2',
        root=str(root),
        production_admitted=False,
        phase_clock=str(clock),
        numerical_result=str(audit),
        numerical_result_identity=identity(audit),
        qualification_wall_limit_s=600,
        prior_occupied_card_s=390.0,
        additional_phase_peak_bytes=2_000_000_000,
        reserve_bytes=400_000_000_000,
        limits=dict(additional_allocated_bytes=40_000_000_000),
        training_python='/native/python',
        source_root='/native/source',
        parent_plan='/parent/PLAN.json',
        original_probe='/original/native',
        execution_commit='sealed',
        devices=dict(B='GPU-b'),
    )
    path = root / 'PLAN.json'
    write_json(path, plan)
    return plan, path


class IntegrationLaunchTests(unittest.TestCase):
    def test_child_uses_only_b_and_sealed_sources_without_resetting_phase_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, path = fixture(root)
            before = Path(plan['phase_clock']).read_bytes()
            child = SimpleNamespace(pid=321, returncode=0, poll=lambda: 0)
            with patch.object(launch, 'verify_source'), patch.object(
                launch.subprocess, 'check_output', return_value=''
            ), patch.object(launch, 'total_storage', return_value=USAGE), patch.object(
                launch.subprocess, 'Popen', return_value=child
            ) as spawn, patch.object(
                launch, 'stop_child'
            ), redirect_stdout(
                io.StringIO()
            ):
                result = launch.run(path)
            command = spawn.call_args.args[0]
            environment = spawn.call_args.kwargs['env']
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'GPU-b')
            self.assertEqual(
                environment['PYTHONPATH'],
                '/native/source:/native/source/training:/native/source/inference',
            )
            self.assertEqual(environment['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertIn(
                'experiments.frontier_preservation_pilot.allocation_integration',
                command,
            )
            self.assertIn(str(root.parent / 'ALLOCATION_STEP_LEDGER.json'), command)
            self.assertEqual(Path(plan['phase_clock']).read_bytes(), before)
            self.assertGreaterEqual(result['cumulative_occupied_card_s'], 390.0)
            self.assertFalse(result['production_started'])

    def test_stale_pass_boolean_with_failed_restoration_cannot_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, path = fixture(directory)
            write_json(
                plan['numerical_result'],
                dict(
                    status='failed', passed=True, restoration=dict(exact_resume=False)
                ),
            )
            plan['numerical_result_identity'] = identity(plan['numerical_result'])
            write_json(path, plan)
            with patch.object(launch.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'Numerical'):
                    launch.run(path)
            spawn.assert_not_called()

    def test_final_storage_failure_retains_executed_child_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path = fixture(root)
            child = SimpleNamespace(pid=321, returncode=0, poll=lambda: 0)
            with patch.object(launch, 'verify_source'), patch.object(
                launch.subprocess, 'check_output', return_value=''
            ), patch.object(
                launch,
                'total_storage',
                side_effect=[USAGE, RuntimeError('final scan unavailable')],
            ), patch.object(
                launch.subprocess, 'Popen', return_value=child
            ), patch.object(
                launch, 'stop_child'
            ), redirect_stdout(
                io.StringIO()
            ):
                result = launch.run(path)
            saved = json.loads((root / 'QUALIFICATION_PROCESS.json').read_text())
            self.assertEqual(saved['exit_code'], 0)
            self.assertGreaterEqual(saved['cumulative_occupied_card_s'], 390.0)
            self.assertIn('final scan unavailable', saved['final_storage_error'])
            self.assertEqual(result, saved)

    def test_shared_deadline_stops_owned_child_and_charges_failed_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path = fixture(root)
            child = SimpleNamespace(pid=321, returncode=None)
            child.poll = lambda: child.returncode

            def stop(process):
                self.assertIs(process, child)
                process.returncode = -15

            with patch.object(launch, 'verify_source'), patch.object(
                launch.subprocess, 'check_output', return_value=''
            ), patch.object(launch, 'total_storage', return_value=USAGE), patch.object(
                launch.subprocess, 'Popen', return_value=child
            ), patch.object(
                launch, 'remaining_budget', side_effect=[1000.0, 0.0]
            ), patch.object(
                launch.shutil,
                'disk_usage',
                return_value=SimpleNamespace(free=430_000_000_000),
            ), patch.object(
                launch, 'stop_child', side_effect=stop
            ), redirect_stdout(
                io.StringIO()
            ):
                result = launch.run(path)
            self.assertEqual(result['exit_code'], -15)
            self.assertEqual(result['stopped_reason'], 'shared phase time/card limit')
            self.assertGreaterEqual(result['cumulative_occupied_card_s'], 390.0)


if __name__ == '__main__':
    unittest.main()
