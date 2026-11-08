"""CPU checks of bounded-oracle fixtures and restoration; no native GPU claim."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from experiments.functional_step_geometry.native_oracles import (
    ORACLE_BUDGET, gradient_comparison, response_fixture, run_oracles,
)
from experiments.functional_step_geometry import native_oracles
from online_feedback.functional_geometry import install_functional_geometry
from test_functional_geometry_runtime import tiny_trainer, run_window


class OracleTests(unittest.TestCase):
    def test_fixture_repeats_original_bindings_without_sampling(self):
        original = dict(anchors=torch.tensor([[3, 7]]), keep=torch.tensor([[True, False]]),
            labels=torch.tensor([[[1, 2], [3, 4]]]), validity_weights=torch.tensor([[[0., 1.], [0., 0.]]]),
            input_ids=torch.tensor([[5, 6, 7]]), hidden_states=torch.ones(1, 3, 2),
            loss_mask=torch.ones(1, 3))
        state = torch.get_rng_state().clone()
        repeated = response_fixture(original, 5)
        self.assertEqual(repeated['anchors'].tolist(), [[3, 7, 3, 7, 3]])
        self.assertEqual(repeated['keep'].tolist(), [[True, False, True, False, True]])
        self.assertIs(repeated['input_ids'], original['input_ids'])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_gradient_failure_retains_coordinate_and_global_evidence(self):
        expected = (torch.tensor([1., 2., 3.], dtype=torch.bfloat16),)
        changed = (torch.tensor([1., 2., 4.], dtype=torch.bfloat16),)
        result = gradient_comparison(changed, expected, ['weight'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['violations'], 1)
        self.assertGreater(result['relative_l2'], .016)
        self.assertEqual(result['parameters'][0]['failing_sample'][0]['index'], 2)

    def test_absolute_floor_cannot_admit_wholly_inaccurate_tiny_vector(self):
        expected = (torch.tensor([1e-7], dtype=torch.bfloat16),)
        result = gradient_comparison((expected[0]*2,), expected, ['tiny'])
        self.assertEqual(result['violations'], 0)
        self.assertEqual(result['status'], 'failed')

    def test_bounded_full_helper_restores_committed_state_and_uses_no_adam(self):
        torch.manual_seed(17)
        trainer = tiny_trainer()
        receipts = []
        with tempfile.TemporaryDirectory() as directory:
            def observer(phase, hook):
                if phase == 'committed':
                    receipts.append(run_oracles(hook, Path(directory)/'oracles'))
            hook = install_functional_geometry(trainer, observer=observer)
            run_window(trainer)
            receipt = receipts[0]
            self.assertEqual(receipt['status'], 'passed')
            self.assertTrue(receipt['committed_weights_exact'])
            self.assertTrue(receipt['optimizer_tensor_identity_and_versions_exact'])
            self.assertTrue(receipt['rng_exact'])
            self.assertEqual(receipt['actual_work']['additional_response_forwards_attempted'], 12)
            self.assertEqual(receipt['actual_work']['additional_backwards_attempted'], 6)
            self.assertEqual(receipt['actual_work']['optimizer_steps_attempted'], 0)
            self.assertEqual(ORACLE_BUDGET['optimizer_steps'], 0)
            self.assertEqual(trainer.global_step, 1)
            hook.close()

    def test_primary_oracle_error_survives_restore_and_report_failures(self):
        torch.manual_seed(17)
        trainer = tiny_trainer()
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            def observer(phase, hook):
                if phase != 'committed':
                    return
                committed = tuple(p.detach().clone() for p in hook.active_parameters)
                restore = hook.restore_endpoint
                writer = native_oracles.write_json
                failed = [False]
                primary = ValueError('primary expanded oracle failure')
                def fail_expanded(*args, **kwargs):
                    failed[0] = True
                    raise primary
                def fail_restore(values):
                    if failed[0]:
                        raise RuntimeError('secondary restoration failure')
                    return restore(values)
                def fail_report(path, payload):
                    if Path(path).name == 'RESULT.json':
                        raise OSError('secondary report failure')
                    return writer(path, payload)
                hook.restore_endpoint = fail_restore
                try:
                    with mock.patch.object(native_oracles, '_expanded_reference', side_effect=fail_expanded), \
                         mock.patch.object(native_oracles, 'write_json', side_effect=fail_report):
                        with self.assertRaisesRegex(ValueError, 'primary expanded oracle failure') as caught:
                            run_oracles(hook, Path(directory)/'oracles')
                    self.assertIs(caught.exception, primary)
                    notes = '\n'.join(primary.__notes__)
                    self.assertIn('secondary restoration failure', notes)
                    self.assertIn('secondary report failure', notes)
                    observed.append(True)
                finally:
                    hook.restore_endpoint = restore
                    restore(committed)
            hook = install_functional_geometry(trainer, observer=observer)
            run_window(trainer)
            self.assertEqual(observed, [True])
            hook.close()


if __name__ == '__main__':
    unittest.main()
