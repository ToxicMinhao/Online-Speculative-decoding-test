"""The revised elapsed phase never erases old cost or admits production itself."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.ok_coding.storage import identity
from .allocation_contract import numerical_contract
from .allocation_launch import (
    audit_command,
    initialize_clock,
    remaining_budget,
    run,
    validate_plan,
)


class AllocationPhaseTests(unittest.TestCase):
    def test_new_elapsed_clock_retains_old_interval_and_card_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'audit'
            root.mkdir()
            plan = dict(
                root=str(root),
                prior_gpu_start_epoch=100,
                prior_elapsed_s=60,
                prior_occupied_card_s=10,
                limits=dict(elapsed_s=14400, occupied_card_s=28800),
            )
            clock = initialize_clock(plan, 200)
            self.assertEqual(clock['first_gpu_action_epoch'], 200)
            self.assertEqual(clock['prior_completion_to_new_phase_s'], 40)
            self.assertEqual(clock['prior_occupied_card_s'], 10)
            self.assertEqual(remaining_budget(clock, 10, 0, 300), 14300)
            self.assertEqual(remaining_budget(clock, 28799, 0, 200), 1)
            self.assertEqual(json.loads((root / 'CLOCK.json').read_text()), clock)
            with self.assertRaises(FileExistsError):
                initialize_clock(plan, 250)

    def test_audit_command_is_zero_update_and_same_retained_probe(self):
        plan = dict(
            root='/study/audit',
            training_python='/env/bin/python',
            parent_plan='/parent/PLAN.json',
            original_probe='/original/probe',
            execution_commit='reviewed',
            numerical_contract='/study/audit/NUMERICAL_CONTRACT.json',
        )
        command = audit_command(plan)
        self.assertIn(
            'experiments.frontier_preservation_pilot.allocation_qualify', command
        )
        self.assertEqual(
            command[command.index('--original-probe') + 1], '/original/probe'
        )
        self.assertEqual(command[command.index('--attempt') + 1], '1')
        self.assertEqual(
            command[command.index('--reference-ledger') + 1],
            '/study/ALLOCATION_REFERENCE_LEDGER.json',
        )
        self.assertNotIn('--train', command)

    def test_empty_historical_bindings_rejected_before_receipt_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = Path(directory) / 'NUMERICAL_CONTRACT.json'
            contract.write_text(json.dumps(numerical_contract()))
            plan = dict(
                schema='frontier_allocation_audit_v2',
                allowed_phase='allocation_numerical_audit',
                production_admitted=False,
                native_adam_steps_this_phase=0,
                old_hidden_gate_status='failed_retained_diagnostic',
                numerical_contract=str(contract),
                numerical_contract_identity=identity(contract),
                original_probe='/original/derivative_support_001',
                previous_root='/transpose_003',
                prior_record_identities={},
            )
            with self.assertRaisesRegex(
                ValueError, 'Missing historical failure and cost bindings'
            ):
                validate_plan(plan)

    def test_exhausted_historical_card_allowance_blocks_before_clock_and_child(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / 'PLAN.json'
            plan = dict(
                root=directory,
                status='admitted',
                reserve_bytes=400_000_000_000,
                additional_phase_peak_bytes=2_000_000_000,
                prior_occupied_card_s=28_800,
                limits=dict(
                    occupied_card_s=28_800, additional_allocated_bytes=40_000_000_000
                ),
            )
            plan_path.write_text(json.dumps(plan))
            module = 'experiments.frontier_preservation_pilot.allocation_launch'
            with mock.patch(module + '.validate_plan'), mock.patch(
                module + '.verify_source'
            ), mock.patch(
                module + '.subprocess.check_output', return_value=''
            ), mock.patch(
                module + '.total_storage',
                return_value=dict(
                    free_bytes=430_000_000_000,
                    all_locations_charged_bytes=2_000_000_000,
                ),
            ), mock.patch(
                module + '.initialize_clock'
            ) as clock, mock.patch(
                module + '.subprocess.Popen'
            ) as child:
                with self.assertRaisesRegex(
                    RuntimeError, 'exhausts the occupied-card allowance'
                ):
                    run(plan_path)
            clock.assert_not_called()
            child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
