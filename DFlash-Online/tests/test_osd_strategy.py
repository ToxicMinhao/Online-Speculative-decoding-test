"""Teacher forwarding must be independent of optional diagnostic metrics."""
import unittest

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext

from online_feedback.objective import OSDTrainStrategy, TargetObjectiveModel
from test_objective_integration import tiny_author, parameter_gradients
from test_osd_objective import osd_batch


class OSDStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_strategy_and_batch(self):
        model = TargetObjectiveModel(tiny_author(dtype=torch.float32), 'osd')
        batch = TrainBatch(sample_ids=['response'], strategy='dflash', tensors=osd_batch(model))
        return OSDTrainStrategy(model), batch

    def test_metrics_toggle_does_not_change_loss_gradients_or_sampling(self):
        strategy, batch = self.make_strategy_and_batch()
        model = strategy.dflash_model
        self.assertFalse(model.teacher_metrics)
        results = []
        for enabled in (True, False):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(17)
            output = strategy.forward_loss(batch, StepContext(collect_detailed_metrics=enabled))
            output.loss.backward()
            results.append((output, parameter_gradients(model), torch.get_rng_state()))
        torch.testing.assert_close(results[0][0].loss, results[1][0].loss, atol=0, rtol=0)
        self.assertTrue(torch.equal(results[0][2], results[1][2]))
        for name, gradient in results[0][1].items():
            torch.testing.assert_close(gradient, results[1][1][name], atol=0, rtol=0)
        numerator, denominator = results[0][0].loss_terms
        torch.testing.assert_close(results[0][0].loss, numerator / denominator)

    def test_missing_teacher_fails_before_forward(self):
        strategy, batch = self.make_strategy_and_batch()
        del batch.tensors['target_last_hidden_states']
        with self.assertRaisesRegex(ValueError, 'missing required features.*target_last_hidden_states'):
            strategy.forward_loss(batch, StepContext(collect_detailed_metrics=False))

    def test_checkpoint_filter_excludes_teacher_head_and_embedding(self):
        strategy, _ = self.make_strategy_and_batch()
        model = strategy.dflash_model
        exported = strategy.checkpoint_state_filter(model.state_dict())
        self.assertEqual(set(exported), set(model.draft_model.state_dict()))
        self.assertFalse(any(name.startswith(('lm_head.', 'embed_tokens.')) for name in exported))


if __name__ == '__main__':
    unittest.main()
