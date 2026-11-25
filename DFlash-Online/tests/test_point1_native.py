"""CPU integration of the Point1 coordinator using the native tiny-model fixture."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from experiments.frontier_preservation_pilot import point1_native as N
from experiments.frontier_preservation_pilot.allocation_scoring import NativeAllocationScorer
from experiments.frontier_preservation_pilot.native import WorkReceipt, rng_digest
from experiments.frontier_preservation_pilot.point1_launch import Point1StepLedger
from experiments.frontier_preservation_pilot.test_allocation_integration import fixture, verify_parent
from experiments.functional_step_geometry.qualify import runtime_state
from online_feedback.rng import capture_rng


class Point1NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def setup_window(self, root):
        trainer, state, features, responses, _, _, _ = fixture(root)
        trainer.events = [dict(event=1)]
        state['events'] = deepcopy(trainer.events)
        runtime = runtime_state(trainer.bundle.model)
        ancillary = N.ancillary_state(trainer)
        work = WorkReceipt(root / 'WORK_POINT1.json')
        scorer = NativeAllocationScorer(trainer, responses, work, collect_diagnostics=True)
        return trainer, state, features, responses, runtime, ancillary, work, scorer

    def test_actual_C_capture_matches_frozen_C_and_preserves_state(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, _, _, _, _, _, _, scorer = self.setup_window(root)
            scorer.freeze_current()
            before_rng = rng_digest(capture_rng())
            before_parameters = [p.detach().clone() for p in scorer.parameters]
            before_costs = deepcopy(trainer.bundle.model.costs)
            extension = trainer.bundle.model.prefix_allocation
            result = N.native_c_coefficients(scorer)
            self.assertEqual(len(result), 2)
            for coefficients, snapshot in zip(result, scorer.snapshots):
                torch.testing.assert_close(coefficients, snapshot['old_C_coefficients'], rtol=0, atol=0)
            self.assertIs(trainer.bundle.model.prefix_allocation, extension)
            self.assertEqual(before_rng, rng_digest(capture_rng()))
            self.assertEqual(before_costs, trainer.bundle.model.costs)
            for before, after in zip(before_parameters, scorer.parameters):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertIsNone(after.grad)

    def test_four_independent_steps_and_C_repeat_restore_complete_parent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, state, features, responses, runtime, ancillary, work, scorer = self.setup_window(root)
            output = root / 'point1'
            output.mkdir()
            allocations, retained, receipt = N.seal_scoring(scorer, output)
            self.assertTrue(receipt['frozen_before_native_updates'])
            self.assertEqual(set(allocations), {'BETA', 'C', 'F'})
            ledger = Point1StepLedger(output / 'STEP_LEDGER.json', {'sha256': 'fixture'})
            ledger.start_attempt(1)
            with mock.patch.object(N, 'verify_live', side_effect=verify_parent), \
                    mock.patch.object(torch.cuda, 'synchronize'):
                cases = N.run_cases(trainer, scorer, state, runtime, ancillary, features,
                                    responses, allocations, retained, ledger, work, output)
            self.assertEqual(list(cases), ['BETA', 'C', 'F', 'C_REPEAT'])
            self.assertTrue(cases['BETA']['beta_g0_exact'])
            self.assertTrue(cases['C_REPEAT']['exact_C_repeat'])
            self.assertEqual(work.value['native_adam_steps'], 4)
            self.assertFalse(work.value['pending'])
            self.assertEqual(len(ledger.value['calls']), 4)
            self.assertTrue(all(row['status'] == 'completed' for row in ledger.value['calls']))
            for case in cases.values():
                self.assertEqual(case['status'], 'completed')
                self.assertEqual(case['endpoint']['global_step'], 129)
                self.assertEqual(case['endpoint']['presentations'], 258)
                self.assertTrue(case['restoration']['passed'])
            self.assertEqual(set(retained['native']), {'BETA', 'C', 'F'})
            self.assertEqual(set(retained['shadows']), {'BETA', 'C', 'F'})
            verify_parent(trainer, state, None)
            self.assertEqual(trainer.events, state['events'])
            self.assertEqual(trainer._active_event, ancillary['active_event'])
            self.assertEqual(json.loads((output/'C_REPEAT.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
