"""CPU integration checks for supplementary qualification, not GPU evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from experiments.functional_step_geometry import qualify, qualify_realization as revised
from experiments.functional_step_geometry.work import QualificationWork
from specforge.training.strategies.base import StepContext
from test_fsg_ordinary_startup import restore_fixture, scalar_fixture
from test_fsg_qualification import replay_fixture


def ordinary_control(trainer, paths):
    """A small real native step supplies retained weights and boundary records."""
    qualify.set_observation_directory(trainer, 'cpu_ordinary_control')
    rows, gradients = [], {}
    original = trainer.backend.optimizer.step

    def record_step():
        gradients.update({name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in trainer.bundle.draft_model.named_parameters()})
        return original()

    with mock.patch.object(trainer.backend.optimizer, 'step', side_effect=record_step):
        for path in paths:
            raw = torch.load(path, weights_only=True)
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(raw, StepContext(global_step=512, total_steps=7424))
            rows.append(dict(record_id=raw['record_id'],
                sampled_anchors=deepcopy(trainer._anchor_observations),
                metrics=result.materialize_metrics(), optimizer_stepped=result.optimizer_stepped))
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
    return gradients, rows, qualify.scientific_state(trainer)


class ReferenceCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def fixture(self, root, *, native_hook=False):
        trainer, paths = (scalar_fixture if native_hook else replay_fixture)(root)
        baseline = qualify.runtime_state(trainer.bundle.model)
        origin = qualify.scientific_state(trainer)
        expected, rows, native = ordinary_control(trainer, paths)
        restore_fixture(trainer, origin, baseline)
        return trainer, paths, baseline, origin, expected, rows, native

    def test_warm_reference_captures_preclip_gradients_without_adam_and_restores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths, baseline, origin, expected, rows, _ = self.fixture(root)
            work = QualificationWork(root/'work.jsonl', limits=revised.LIMITS)
            (root/'startup').mkdir()
            restore = lambda *_: restore_fixture(trainer, origin, baseline)
            with mock.patch.object(qualify, 'restore_origin', side_effect=restore), \
                 mock.patch.object(revised, 'restore_origin', side_effect=restore), \
                 mock.patch.object(trainer.backend.optimizer.optimizer, 'step',
                                   side_effect=AssertionError('No Adam during recapture')) as adam:
                qualify.warmup(trainer, paths, work, baseline, root/'startup')
                reference = revised.capture_reference(trainer, paths, rows, work, baseline, root/'reference')
            adam.assert_not_called()
            self.assertFalse(qualify.differences(reference['gradients'], expected))
            self.assertGreater(sum(float(g.float().square().sum()) for g in expected.values())**.5, 1.)
            self.assertFalse(qualify.differences(qualify.scientific_state(trainer), origin))
            self.assertEqual(work.summary()['additional_forward_completed'], 4)
            self.assertEqual(work.summary()['additional_backward_completed'], 4)
            self.assertNotIn('adam_attempted', work.summary())
            self.assertEqual(reference['rows'], rows)
            self.assertNotIn('native', reference)
            self.assertEqual(json.loads((root/'reference/RESULT.json').read_text())['adam'], 0)

    def test_retained_anchor_mismatch_is_preserved_without_an_adam_application(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths, baseline, origin, _, rows, _ = self.fixture(root)
            rows = deepcopy(rows); rows[0]['record_id'] = 'wrong retained response'
            work = QualificationWork(root/'work.jsonl', limits=revised.LIMITS)
            with mock.patch.object(revised, 'restore_origin',
                    side_effect=lambda *_: restore_fixture(trainer, origin, baseline)), \
                 mock.patch.object(trainer.backend.optimizer.optimizer, 'step',
                                   side_effect=AssertionError('No Adam during recapture')) as adam, \
                 self.assertRaisesRegex(AssertionError, 'Exact qualification gate failed'):
                revised.capture_reference(trainer, paths, rows, work, baseline, root/'reference')
            adam.assert_not_called()
            self.assertEqual(json.loads((root/'reference/RESULT.json').read_text())['status'], 'failed')
            self.assertEqual(work.summary()['additional_backward_completed'], 2)

    def test_live_fsg_scalar_share_actual_native_state_not_retired_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths, baseline, origin, expected, rows, native = self.fixture(root, native_hook=True)
            export = root/'ordinary_export'; export.mkdir()
            save_file(native['weights'], str(export/'model.safetensors'))
            reference = dict(gradients=expected, rows=rows)
            work = QualificationWork(root/'work.jsonl', limits=revised.LIMITS)

            def disposable_export(_trainer, _path, destination, payload):
                destination.mkdir()
                save_file(payload['weights'], str(destination/'model.safetensors'))
                return dict(status='passed')

            restore = lambda *_: restore_fixture(trainer, origin, baseline)
            with mock.patch.object(qualify, 'restore_origin', side_effect=restore), \
                 mock.patch.object(qualify, 'export_disposable_state', side_effect=disposable_export):
                for label in ('FSG', 'SCALAR'):
                    _, receipt = qualify.run_case(trainer, paths, label, 1., work, baseline,
                        root/label, reference, oracle_runner=revised.reused_oracle,
                        realization_revision=revised.REVISION, reference_export=export)
                    self.assertEqual(receipt['status'], 'passed')
                    self.assertTrue(json.loads((root/label/'AUTHENTICATED_NATIVE_EXPORT.json').read_text())['exact_weights'])
                    self.assertTrue(json.loads((root/label/'FINAL_OPTIMIZER.json').read_text())['exact_equal'])
                    self.assertTrue(json.loads((root/label/'FINAL_MICRO.json').read_text())['exact_equal'])
                    self.assertTrue(json.loads((root/label/'FINAL_PRESENTATIONS.json').read_text())['exact_equal'])
            self.assertEqual(reference['native']['global_step'], 512)
            self.assertEqual(reference['native']['presentations'], 1025)
            self.assertEqual(reference['native']['micro'], 1026)
            self.assertEqual(reference['final']['global_step'], 513)
            self.assertEqual(reference['final']['presentations'], 1026)
            self.assertEqual(reference['final']['micro'], 1026)
            self.assertEqual(reference['final']['accumulation'], 0)
            self.assertFalse(qualify.differences(reference['native']['optimizer'], native['optimizer']))
            self.assertEqual(work.summary()['adam_completed'], 2)
            self.assertEqual(work.summary()['ordinary_forward_completed'], 4)
            self.assertEqual(work.summary()['ordinary_backward_completed'], 4)

    def test_wrong_retained_native_export_stops_after_first_included_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths, baseline, origin, expected, rows, native = self.fixture(root, native_hook=True)
            export = root/'wrong_export'; export.mkdir()
            save_file({name: value+1 for name, value in native['weights'].items()}, str(export/'model.safetensors'))
            work = QualificationWork(root/'work.jsonl', limits=revised.LIMITS)
            reference = dict(gradients=expected, rows=rows)
            with mock.patch.object(qualify, 'restore_origin',
                    side_effect=lambda *_: restore_fixture(trainer, origin, baseline)), \
                 self.assertRaisesRegex(ValueError, 'differs from exact export'):
                qualify.run_case(trainer, paths, 'FSG', 1., work, baseline,
                    root/'FSG', reference, oracle_runner=revised.reused_oracle,
                    realization_revision=revised.REVISION, reference_export=export)
            self.assertEqual(work.summary()['adam_attempted'], 1)
            self.assertEqual(work.summary()['adam_completed'], 1)
            self.assertNotIn('additional_forward_attempted', work.summary())
            self.assertEqual(json.loads((root/'FSG/RESULT.json').read_text())['status'], 'failed')
            self.assertNotIn('native', reference)


class ActivityTests(unittest.TestCase):
    @staticmethod
    def cases(*, fsg_native=False, fsg_prestate=False, scalar_matching=True):
        fsg = dict(decision=dict(accepted=True), committed=dict(identical_native=fsg_native,
                   identical_prestate=fsg_prestate), final_candidate_check='actual_bf16_replay')
        scalar = dict(decision=dict(accepted=False), committed=dict(identical_native=True,
                      identical_prestate=False), scalar_matching_exercised=scalar_matching,
                      final_candidate_check='exact_native_risk_reuse')
        return {'gsm8k':dict(cases=[dict(case=name, hook=dict(steps=[step]))
                    for name,step in [('FSG',fsg),('SCALAR',scalar)]])}

    def test_native_and_prestate_endpoints_cannot_supply_activity(self):
        self.assertFalse(revised.activity(self.cases(fsg_native=True))['mechanism_gate'])
        self.assertFalse(revised.activity(self.cases(fsg_prestate=True))['mechanism_gate'])

    def test_scalar_matching_is_required_but_scalar_acceptance_is_not(self):
        self.assertFalse(revised.activity(self.cases(scalar_matching=False))['mechanism_gate'])
        self.assertTrue(revised.activity(self.cases())['mechanism_gate'])


if __name__ == '__main__':
    unittest.main()
