"""CPU wiring for revised qualification; these tests do not certify native GPU."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from experiments.functional_step_geometry.qualification_checks import (
    maximum_resource_fixture, run_resource_checks, run_same_layout_check,
)
from experiments.functional_step_geometry import same_layout_oracle
from online_feedback.functional_geometry import install_functional_geometry
from test_functional_geometry_runtime import tiny_trainer, run_window


class ResourceFixtureTests(unittest.TestCase):
    def fixture(self):
        return dict(anchors=torch.tensor([[1, 3]]), keep=torch.tensor([[True, False]]),
            labels=torch.tensor([[[1, 2], [3, 4]]]),
            validity_weights=torch.tensor([[[0., 1.], [0., 0.]]]),
            input_ids=torch.tensor([[5, 6, 7, 8, 9]]),
            hidden_states=torch.arange(10).reshape(1, 5, 2).bfloat16(),
            loss_mask=torch.tensor([[0., 0., 1., 1., 1.]]))

    def test_maximum_fixture_shifts_bindings_and_preserves_original_suffix(self):
        source = self.fixture()
        rng = torch.get_rng_state().clone()
        padded = maximum_resource_fixture(source, 9, anchors=5)
        self.assertEqual(padded['anchors'].tolist(), [[5, 7, 5, 7, 5]])
        self.assertEqual(padded['keep'].tolist(), [[True, False, True, False, True]])
        self.assertEqual(padded['input_ids'].tolist(), [[5, 5, 5, 5, 5, 6, 7, 8, 9]])
        for name in ('input_ids', 'hidden_states', 'loss_mask'):
            self.assertTrue(torch.equal(padded[name][:, 4:], source[name]))
        self.assertEqual(float(padded['hidden_states'][:, :4].abs().sum()), 0.)
        self.assertEqual(float(padded['loss_mask'][:, :4].sum()), 0.)
        self.assertEqual(padded['labels'][:, 0].tolist(), source['labels'][:, 0].tolist())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(padded['qualification_fixture']['maximum_training_sequence_length_tested'])
        self.assertFalse(padded['qualification_fixture']['scientific_feedback'])

    def test_resource_fixture_never_truncates(self):
        with self.assertRaisesRegex(ValueError, 'never truncated'):
            maximum_resource_fixture(self.fixture(), 4)


class QualificationHelperTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def trainer(self):
        trainer = tiny_trainer()
        trainer.max_length = 12
        trainer.bundle.model.objective_chunk_blocks = 8
        return trainer

    def test_same_layout_and_resource_checks_restore_and_respect_fixed_work(self):
        trainer = self.trainer()
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            def observer(phase, hook):
                if phase != 'committed':
                    return
                receipt, reference = run_same_layout_check(hook, Path(directory) / 'same')
                self.assertEqual(receipt['status'], 'passed')
                self.assertEqual(receipt['chunks'], [8, 1])
                self.assertTrue(receipt['prefix_mask_exact'])
                self.assertTrue(receipt['weights_exact'])
                self.assertTrue(receipt['optimizer_tensor_identity_and_versions_exact'])
                self.assertTrue(receipt['gradients_unchanged'])
                self.assertEqual(receipt['actual_work']['additional_response_forwards_attempted'], 5)
                self.assertEqual(receipt['actual_work']['additional_backwards_attempted'], 2)
                self.assertEqual(receipt['actual_work']['optimizer_steps_attempted'], 0)
                resource = run_resource_checks(hook, Path(directory) / 'resource', reference,
                                               sequence_length=trainer.max_length)
                self.assertEqual(resource['status'], 'passed')
                self.assertEqual(resource['actual_work']['additional_response_forwards_attempted'], 9)
                self.assertEqual(resource['actual_work']['additional_backwards_attempted'], 4)
                self.assertEqual(resource['actual_work']['optimizer_steps_attempted'], 0)
                self.assertEqual(resource['expanded']['response_anchor_counts'], [512, 512])
                self.assertEqual(resource['expanded']['input_lengths'], [12, 12])
                self.assertEqual(resource['expanded']['simultaneous_native_vjp_vectors'], 4)
                self.assertTrue(resource['restored_hidden_exact'])
                self.assertTrue(resource['optimizer_tensor_identity_and_versions_exact'])
                observed.append(True)
            hook = install_functional_geometry(trainer, observer=observer)
            run_window(trainer)
            self.assertEqual(observed, [True])
            self.assertEqual(trainer.global_step, 1)
            hook.close()

    def test_same_layout_mismatch_still_fails_and_preserves_evidence_and_state(self):
        trainer = self.trainer()
        ordinary = same_layout_oracle.same_layout_reference
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            def altered(*args, **kwargs):
                reference = ordinary(*args, **kwargs)
                reference.gradients = tuple(None if value is None else value + 1
                                            for value in reference.gradients)
                return reference

            def observer(phase, hook):
                if phase != 'committed':
                    return
                committed = [parameter.detach().clone() for parameter in hook.active_parameters]
                output = Path(directory) / 'failed'
                with mock.patch.object(same_layout_oracle, 'same_layout_reference', side_effect=altered):
                    with self.assertRaisesRegex(ValueError, 'oracle failed'):
                        run_same_layout_check(hook, output)
                import json
                result = json.loads((output / 'RESULT.json').read_text())
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['native_vjp']['status'], 'failed')
                self.assertTrue(result['committed_weights_exact'])
                self.assertTrue(result['rng_exact'])
                self.assertTrue(result['optimizer_tensor_identity_and_versions_exact'])
                self.assertEqual(result['actual_work']['additional_backwards_attempted'], 2)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(hook.active_parameters, committed)))
                observed.append(True)
            hook = install_functional_geometry(trainer, observer=observer)
            run_window(trainer)
            self.assertEqual(observed, [True])
            hook.close()

    def test_resource_limit_must_be_the_qualified_configuration(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            hook = install_functional_geometry(trainer)
            with self.assertRaisesRegex(ValueError, 'qualified training limit'):
                run_resource_checks(hook, Path(directory), None, sequence_length=11)
            self.assertEqual(hook.work['additional_response_forwards_attempted'], 0)
            hook.close()


if __name__ == '__main__':
    unittest.main()
