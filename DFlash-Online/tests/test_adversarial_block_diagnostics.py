"""CPU tensor/stub checks for bounded diagnostics and response observations."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from experiments.kplus_abt.diagnostics import BlockCapture, FixedObjective, capture_blocks
from experiments.kplus_abt.training import AdversarialBlockTrainer
from online_feedback.first_error import first_error_mask
from online_feedback.kplus import group_terms, redistribute_response
from specforge.training.strategies.base import StepOutput


def fixture():
    labels = torch.arange(20).reshape(1, 5, 4)
    clean = labels.clone()
    clean[0, 0, 1:] = -1
    clean[0, 1, 1] = -1
    clean[0, 3, 2:] = -1
    weights = torch.tensor([0., 1., .7, .5]).repeat(1, 5, 1)
    selection = first_error_mask(clean, labels, weights)
    denominator = sum((weights[:, start:start+2] * selection[:, start:start+2]).sum()
                      for start in (0, 2, 4)) + 1e-6
    return dict(labels=labels, weights=weights, clean_proposals=clean,
                selection=selection, denominator=denominator)


class AdversarialBlockDiagnosticTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_capture_storage_offsets_reassemble_partial_chunks_once(self):
        snapshot = fixture()
        capture = BlockCapture()
        for start in (0, 2, 4, 4, 2, 0):
            capture(*(snapshot[name][:, start:start+2] for name in
                      ("clean_proposals", "labels", "weights", "selection")))
        self.assertEqual(sorted(capture.chunks), [0, 8, 16])
        for actual, name in zip(capture.joined(), ("clean_proposals", "labels", "weights", "selection")):
            torch.testing.assert_close(actual, snapshot[name], rtol=0, atol=0)

    def test_fixed_objective_keeps_old_support_priorities_denominator(self):
        snapshot = fixture()
        fixed = FixedObjective(snapshot, "cpu", ())
        new_predictions = torch.full_like(snapshot["labels"], -1)
        rows = torch.arange(20).reshape(1, 5, 4).float()
        ordinary, fixed_terms = [], []
        for start in (0, 2, 4):
            sl = (slice(None), slice(start, start+2))
            labels, weights = snapshot["labels"][sl], snapshot["weights"][sl]
            new_selection = first_error_mask(new_predictions[sl], labels, weights)
            selected = fixed.freeze_selection(new_predictions[sl], labels, weights, new_selection)
            self.assertTrue(torch.equal(selected, snapshot["selection"][sl]))
            ordinary.append(group_terms(rows[sl], snapshot["clean_proposals"][sl],
                                         labels, weights, selected))
            fixed_terms.append(fixed.group_terms(rows[sl], labels))
        for original_chunk, fixed_chunk in zip(ordinary, fixed_terms):
            for original, actual in zip(original_chunk, fixed_chunk):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
        self.assertIs(fixed.freeze_response(None, snapshot["denominator"]), fixed.snapshot["denominator"])
        with self.assertRaisesRegex(ValueError, "denominator"):
            fixed.freeze_response(None, snapshot["denominator"] + 1)

    def test_fixed_objective_rejects_changed_label_or_position_weight(self):
        snapshot = fixture()
        fixed = FixedObjective(snapshot, "cpu", ())
        labels = snapshot["labels"][:, :2].clone()
        labels[0, 0, 1] += 1
        with self.assertRaisesRegex(ValueError, "identity"):
            fixed.freeze_selection(labels, labels, snapshot["weights"][:, :2],
                                   snapshot["selection"][:, :2])

    def test_observer_context_restores_existing_observer_even_on_failure(self):
        prior = object()
        model = SimpleNamespace(block_observer=prior)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with capture_blocks(model, BlockCapture()):
                self.assertIsNot(model.block_observer, prior)
                raise RuntimeError("fixture")
        self.assertIs(model.block_observer, prior)

    def test_trainer_emits_one_mean_loss_observation_with_cpu_probe_snapshot(self):
        trainer = object.__new__(AdversarialBlockTrainer)
        trainer.response_records, trainer.view_records, trainer.work = [], [], {}
        trainer._active_event = "event_005"
        trainer._anchor_observations = [dict(positions=[[1, 3]], keep_mask=[[True, True]])]
        emitted, remembered = [], []
        trainer._append_observation = lambda path, row: emitted.append((path, deepcopy(row)))
        trainer.response_observer = lambda *values: remembered.append(values)
        output = StepOutput(loss=torch.tensor(3.), metrics={}, loss_terms=(torch.tensor(3.), torch.tensor(1.)),
            ratio_metrics={"objective_loss": (torch.tensor(6.), torch.tensor(2.))},
            sum_metrics={"kplus_response_count": torch.tensor(1.), "response_presentations": torch.tensor(1.)})
        snapshot = fixture()
        snapshot["intended_norm"] = torch.tensor([[.01, .01]])
        diagnostic = dict(arm="ADV", key=("gsm8k", 5, 0, 0), clean_loss=2., second_loss=4.,
                          snapshots=snapshot)
        batch = SimpleNamespace(sample_ids=["fixture"])
        trainer._observe_views(batch, output, diagnostic)
        self.assertEqual(len(trainer.response_records), 1)
        self.assertEqual(len(trainer.view_records), 1)
        self.assertEqual(trainer.response_records[0]["loss"], 3.)
        self.assertEqual(trainer.response_records[0]["sum_metrics"]["kplus_response_count"], 1)
        self.assertEqual(trainer.work["response_forwards"], 2)
        self.assertEqual(len(emitted), 2)
        self.assertNotIn("snapshots", trainer.view_records[0])
        self.assertIs(remembered[0][2], snapshot)
        self.assertEqual(remembered[0][3]["clean_loss"], 2.)


if __name__ == "__main__":
    unittest.main()
