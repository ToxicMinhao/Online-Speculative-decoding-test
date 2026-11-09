"""Complete tiny native graph diagnostic, including restoration and work bounds."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.functional_step_geometry.pilot_diagnostics import run_window_diagnostics, validate_bindings
from online_feedback.functional_geometry import FEASIBLE_REALIZATION, install_functional_geometry
from test_functional_geometry_runtime import tiny_trainer, run_window


def teacher_alignment(hidden, indices):
    selected = (indices - 1).clamp_min(0)
    gathered = hidden.gather(1, selected.reshape(1, -1, 1).expand(-1, -1, hidden.shape[-1]))
    return gathered.reshape(*indices.shape, hidden.shape[-1])


def fixture_batches(hook):
    fitting = []
    for index, response in enumerate(hook.responses):
        fitting.append(dict(record_id='A' + str(index),
            **{key: response[key].clone() for key in ('input_ids', 'hidden_states', 'loss_mask')},
            target_last_hidden_states=response['hidden_states'].clone()))
    off_fit = []
    for index, response in enumerate(hook.responses):
        features = response['hidden_states'].clone() * .9
        off_fit.append(dict(record_id='D' + str(index), input_ids=response['input_ids'].clone(),
            hidden_states=features, loss_mask=response['loss_mask'].clone(),
            target_last_hidden_states=features.clone()))
    binding = dict(dataset='gsm8k', event=5, window=1, pass_index=0,
                   fitting_ids=['A0', 'A1'], off_fit_ids=['D0', 'D1'],
                   diagnostic_rng_key=[20261108, 0, 5, 1])
    return fitting, off_fit, binding


class PilotDiagnosticTests(unittest.TestCase):
    def test_complete_window_preserves_training_state_and_separates_work(self):
        torch.manual_seed(17)
        trainer = tiny_trainer()
        trainer.bundle.model.loss_dtype = torch.bfloat16
        trainer.bundle.model._aligned_target_hidden = teacher_alignment
        trainer._original_sampler = trainer.bundle.model._sample_anchor_positions
        receipts = []
        with tempfile.TemporaryDirectory() as temporary:
            def observe(phase, hook):
                if phase != 'committed':
                    return
                fitting, off_fit, binding = fixture_batches(hook)
                before_work = dict(hook.work)
                receipts.append(run_window_diagnostics(hook, fitting, off_fit,
                                                       output=Path(temporary), binding=binding))
                self.assertEqual(hook.work, before_work)
            hook = install_functional_geometry(trainer, realization_revision=FEASIBLE_REALIZATION,
                                               observer=observe)
            try:
                run_window(trainer)
            finally:
                hook.close()
            receipt, = receipts
            self.assertEqual(receipt['status'], 'completed')
            self.assertTrue(receipt['restoration_exact'])
            self.assertLessEqual(receipt['response_forwards'], 40)
            self.assertEqual(receipt['backwards'], 4)
            self.assertEqual(trainer.global_step, 1)
            self.assertTrue((Path(temporary) / 'prestate.pt').is_file())
            self.assertTrue((Path(temporary) / 'native.pt').is_file())
            prestate = torch.load(Path(temporary) / 'prestate.pt', weights_only=True)
            native = torch.load(Path(temporary) / 'native.pt', weights_only=True)
            self.assertIn('inactive', prestate)
            self.assertTrue(torch.equal(prestate['inactive'], trainer.bundle.draft_model.inactive))
            self.assertTrue(torch.equal(native['inactive'], trainer.bundle.draft_model.inactive))
            self.assertEqual(set(receipt['measurements']), {'prestate', 'native', 'raw_fsg',
                'feasible_fsg', 'committed', 'scalar', 'uniform_raw', 'uniform', 'expanded_raw', 'expanded'})
            for measurements in receipt['measurements'].values():
                self.assertEqual([row['role'] for row in measurements], ['A', 'A', 'D', 'D'])
                self.assertTrue(all('first_error' in row['teacher_kl'] for row in measurements))
            self.assertEqual(receipt['candidates']['raw_fsg']['role'],
                             'unadjusted a=1; expanded second direction')

    def test_changed_fitting_input_is_rejected_before_diagnostic_work(self):
        trainer = tiny_trainer()
        received = []
        def observe(phase, hook):
            if phase != 'committed':
                return
            fitting, off_fit, binding = fixture_batches(hook)
            fitting[0]['input_ids'][0, 0] = 3
            with self.assertRaisesRegex(ValueError, 'Fitting diagnostic tensors differ'):
                validate_bindings(hook, fitting, off_fit, binding)
            received.append(True)
        hook = install_functional_geometry(trainer, realization_revision=FEASIBLE_REALIZATION,
                                           observer=observe)
        try:
            run_window(trainer)
        finally:
            hook.close()
        self.assertEqual(received, [True])

    def test_failed_replay_restores_committed_state_and_keeps_failed_work(self):
        trainer = tiny_trainer()
        trainer.bundle.model.loss_dtype = torch.bfloat16
        trainer.bundle.model._aligned_target_hidden = teacher_alignment
        trainer._original_sampler = trainer.bundle.model._sample_anchor_positions
        with tempfile.TemporaryDirectory() as temporary:
            def observe(phase, hook):
                if phase != 'committed':
                    return
                fitting, off_fit, binding = fixture_batches(hook)
                committed = [parameter.detach().clone() for parameter in hook.active_parameters]
                before_work = dict(hook.work)
                original = hook.original_forward
                def failed(*args, **kwargs):
                    if hook.phase == 'diagnostic_raw_fsg':
                        raise ValueError('diagnostic failed replay')
                    return original(*args, **kwargs)
                with patch.object(hook, 'original_forward', side_effect=failed):
                    with self.assertRaisesRegex(ValueError, 'diagnostic failed replay'):
                        run_window_diagnostics(hook, fitting, off_fit,
                                               output=Path(temporary), binding=binding)
                self.assertTrue(all(torch.equal(parameter, expected)
                    for parameter, expected in zip(hook.active_parameters, committed)))
                self.assertEqual(hook.work, before_work)
                receipt = json.loads((Path(temporary) / 'RESULT.json').read_text())
                self.assertEqual(receipt['status'], 'failed')
                self.assertEqual(receipt['model_work']['additional_forward_failed'], 1)
                self.assertTrue(receipt['committed_weights_exact'])
            hook = install_functional_geometry(trainer, realization_revision=FEASIBLE_REALIZATION,
                                               observer=observe)
            try:
                run_window(trainer)
            finally:
                hook.close()

    def test_zero_native_window_still_completes_its_scheduled_diagnostic(self):
        trainer = tiny_trainer()
        trainer.backend.optimizer.optimizer.param_groups[0]['lr'] = 0.
        trainer.bundle.model.loss_dtype = torch.bfloat16
        trainer.bundle.model._aligned_target_hidden = teacher_alignment
        trainer._original_sampler = trainer.bundle.model._sample_anchor_positions
        receipts = []
        with tempfile.TemporaryDirectory() as temporary:
            def observe(phase, hook):
                if phase == 'committed':
                    fitting, off_fit, binding = fixture_batches(hook)
                    receipts.append(run_window_diagnostics(hook, fitting, off_fit,
                                                           output=Path(temporary), binding=binding))
            hook = install_functional_geometry(trainer, realization_revision=FEASIBLE_REALIZATION,
                                               observer=observe)
            try:
                run_window(trainer)
            finally:
                hook.close()
        self.assertEqual(receipts[0]['status'], 'completed')
        self.assertTrue(receipts[0]['restoration_exact'])
        self.assertLessEqual(receipts[0]['response_forwards'], 40)
        self.assertLessEqual(receipts[0]['backwards'], 4)
        self.assertEqual(receipts[0]['candidates']['expanded']['bypass'], 'zero_production_sigma')


if __name__ == '__main__':
    unittest.main()
