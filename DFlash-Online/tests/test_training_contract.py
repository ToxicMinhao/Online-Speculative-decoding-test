"""CPU checks of response reduction and real upstream optimizer persistence."""
from pathlib import Path
from types import SimpleNamespace
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from specforge.optimizer import BF16Optimizer
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepOutput

from online_feedback import training as T
from online_feedback.rng import capture_rng


def feature(record_id="r1", response=(4, 5)):
    tokens = [1, 2, *response]
    length = len(tokens)
    return {
        "record_id": record_id,
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "loss_mask": torch.tensor([[0, 0, *([1] * len(response))]]),
        "hidden_states": torch.ones(1, length, 3),
        "target_last_hidden_states": torch.ones(1, length, 3),
        "prompt_length": 2,
        "label_binding": {"labels": "canonical_response_tokens"},
    }


def trees_equal(left, right):
    """Compare small fixture state directly, including RNG arrays and tensors."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(trees_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            trees_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class FeatureTests(unittest.TestCase):
    def validate(self, raw, cap=4):
        return T.validate_feature(raw, max_length=cap, hidden_width=3, vocab_size=32)

    def test_complete_response_has_expected_adjacent_positions(self):
        metadata = self.validate(feature())
        self.assertEqual(metadata["response_tokens"], 2)
        self.assertEqual(metadata["adjacent_response_positions"], 1)

    def test_overlength_rejected_before_truncation(self):
        with self.assertRaisesRegex(ValueError, "truncation"):
            self.validate(feature(), cap=3)

    def test_one_token_response_rejected(self):
        with self.assertRaisesRegex(ValueError, "two adjacent"):
            self.validate(feature(response=(4,)))

    def test_prompt_or_incomplete_response_mask_rejected(self):
        for position in (0, 2):
            with self.subTest(position=position):
                raw = feature()
                raw["loss_mask"][0, position] = 1 - raw["loss_mask"][0, position]
                with self.assertRaisesRegex(ValueError, "Mask"):
                    self.validate(raw)

    def test_invalid_feature_dimensions_and_values_rejected(self):
        invalid_features = [
            ("input_ids", torch.ones(4, dtype=torch.long)),
            ("input_ids", torch.tensor([[1, 2, 3, 32]])),
            ("loss_mask", torch.ones(1, 3)),
            ("hidden_states", torch.ones(1, 3, 3)),
            ("hidden_states", torch.ones(1, 4, 4)),
            ("hidden_states", torch.full((1, 4, 3), float("nan"))),
            ("target_last_hidden_states", torch.ones(1, 3, 3)),
        ]
        for name, value in invalid_features:
            with self.subTest(name=name, shape=value.shape):
                raw = feature()
                raw[name] = value
                with self.assertRaises(ValueError):
                    self.validate(raw)

    def test_upstream_batch_keeps_exact_tokens_and_response_mask(self):
        trainer = object.__new__(T.SpecForgeEventTrainer)
        trainer.max_length = 4
        raw = feature()
        batch = trainer._batch(raw)
        for name in ("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"):
            self.assertTrue(torch.equal(batch.tensors[name], raw[name]), name)
        self.assertEqual(batch.sample_ids, ["r1"])
        self.assertEqual(batch.metadata["label_binding"], raw["label_binding"])


class ScalarStrategy:
    """Small differentiable objective with unequal author response weights."""
    name = "dflash"

    def __init__(self, model):
        self.model = model
        self.calls = 0

    def forward_loss(self, batch, ctx=None):
        self.calls += 1
        label = batch.tensors["input_ids"][0, -1].float()
        loss = (self.model.draft_model.weight.float().reshape(()) - label).square()
        return StepOutput(loss, {}, loss_terms=(loss * label.detach(), label.detach()))

    def checkpoint_state_filter(self, state):
        return {
            name.removeprefix("draft_model."): value
            for name, value in state.items()
            if name.startswith("draft_model.")
        }


class StochasticScalarStrategy(ScalarStrategy):
    """Make all CPU RNG streams affect the resume comparison's next update."""
    def forward_loss(self, batch, ctx=None):
        output = super().forward_loss(batch, ctx)
        factor = torch.rand(()) + random.random() + np.random.rand()
        loss = output.loss * factor
        numerator, denominator = output.loss_terms
        return StepOutput(loss, {}, loss_terms=(numerator * factor, denominator))


class DirectResponseMeanStrategy(StochasticScalarStrategy):
    """Let TrainerCore average plain per-response losses across accumulation."""
    def forward_loss(self, batch, ctx=None):
        output = super().forward_loss(batch, ctx)
        return StepOutput(output.loss, {})


class GradientBackend:
    checks_loss_denominator = False

    def __init__(self, parameter):
        self.parameter = parameter
        self.last_gradient = None

    def backward(self, loss, **kwargs):
        loss.backward()

    def scale_gradients(self, factor):
        self.parameter.grad.mul_(factor)

    def step(self):
        self.last_gradient = self.parameter.grad.item()
        return None


class ReductionTests(unittest.TestCase):
    def gradient(self, reduction):
        model = torch.nn.Module()
        model.draft_model = torch.nn.Linear(1, 1, bias=False)
        model.draft_model.weight.data.zero_()
        strategy = ScalarStrategy(model)
        backend = GradientBackend(model.draft_model.weight)
        core = TrainerCore(T.ReductionAdapter(strategy, reduction), backend, accumulation_steps=2)
        for token in (1, 3):
            batch = SimpleNamespace(tensors={"input_ids": torch.tensor([[token]])})
            result = core.train_step(batch)
        self.assertEqual(strategy.calls, 2)
        self.assertEqual(core.accumulation_remainder, 0)
        self.assertEqual(result.metrics["response_presentations"], 2)
        return backend.last_gradient, result.metrics

    def test_response_mean_uses_equal_response_weights(self):
        gradient, metrics = self.gradient("response_mean")
        self.assertAlmostEqual(gradient, -4)
        self.assertAlmostEqual(metrics["loss"], 5)
        self.assertAlmostEqual(metrics["author_weighted_objective"], 7)

    def test_upstream_weight_mean_keeps_author_ratio(self):
        gradient, metrics = self.gradient("upstream_weight_mean")
        self.assertAlmostEqual(gradient, -5)
        self.assertAlmostEqual(metrics["loss"], 7)

    def test_unknown_reduction_rejected(self):
        with self.assertRaisesRegex(ValueError, "reduction"):
            T.ReductionAdapter(None, "silent_default")


class TinyTrainer(T.SpecForgeEventTrainer):
    """Use real TrainerCore and BF16Optimizer with a one-parameter CPU model."""
    def _build_runtime(self):
        torch.manual_seed(17)
        random.seed(17)
        np.random.seed(17)
        self.horizon = 4
        self.accumulation = 2
        self.max_length = 16
        self.hidden_width = 3
        self.vocab_size = 32
        self.contract = {"tiny_cpu_contract": True}
        self.cfg = SimpleNamespace(
            run_id="cpu",
            model=SimpleNamespace(
                draft_model_config=str(self.config_path),
                target_model_path="unused",
                embedding_key="unused",
            ),
        )
        model = torch.nn.Module()
        model.target_table = torch.nn.Parameter(torch.tensor([0.3]), requires_grad=False)
        model.draft_model = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
        self.bundle = SimpleNamespace(model=model, draft_model=model.draft_model)

        def optimizer_factory(draft):
            return BF16Optimizer(
                draft,
                lr=0.01,
                total_steps=4,
                warmup_ratio=0,
                lr_scheduler="cosine",
                offload_master=True,
                max_grad_norm=10,
            )

        self.backend = FSDPTrainingBackend(
            ParallelConfig(sharding_strategy="NO_SHARD"),
            optimizer_factory=optimizer_factory,
        )
        self.backend.prepare_model(model, wrap=False, optimizer_target=model.draft_model)
        self.author_strategy = StochasticScalarStrategy(model)
        self.strategy = T.ReductionAdapter(self.author_strategy, self.reduction)
        self.core = TrainerCore(self.strategy, self.backend, accumulation_steps=2)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text("{}")
        self.features = []
        for index in range(2):
            path = self.root / f"feature{index}.pt"
            torch.save(feature(f"r{index}", response=(4, 5 + index)), path)
            self.features.append(path)

        def export(state, config, output, **kwargs):
            Path(output).mkdir()
            (Path(output) / "model.safetensors").write_bytes(b"fixture export")
            torch.rand(3)
            random.random()
            np.random.rand()

        self.export = mock.patch("specforge.export.to_hf.export_to_hf", side_effect=export)
        self.export.start()
        self.addCleanup(self.export.stop)
        logger = mock.patch("specforge.optimizer.print_on_rank0")
        logger.start()
        self.addCleanup(logger.stop)

    def trainer(self, name="run", resume=None, reduction="response_mean"):
        trainer = TinyTrainer(self.config, self.root / name, resume_state=resume, reduction=reduction)
        self.addCleanup(trainer.close)
        return trainer

    def state(self, trainer):
        return torch.load(trainer.latest, map_location="cpu", weights_only=False)

    def test_two_events_keep_optimizer_master_scheduler_and_pass_order(self):
        trainer = self.trainer()
        optimizer = trainer.backend.optimizer
        master = optimizer.fp32_params[0]
        original_draft = trainer.bundle.draft_model.weight.detach().clone()
        frozen_target = trainer.bundle.model.target_table.detach().clone()
        first = trainer.train_event("e1", self.features)
        learning_rate_after_first = optimizer.optimizer.param_groups[0]["lr"]
        second = trainer.train_event("e2", self.features)
        self.assertIs(trainer.backend.optimizer, optimizer)
        self.assertIs(optimizer.fp32_params[0], master)
        self.assertEqual(trainer.global_step, 4)
        self.assertEqual(trainer.presentations, 8)
        self.assertEqual((first["optimizer_steps"], second["optimizer_steps"]), (2, 2))
        self.assertEqual([row["record_id"] for row in first["records"]], ["r0", "r1", "r0", "r1"])
        self.assertEqual([row["optimizer_stepped"] for row in first["records"]], [False, True, False, True])
        self.assertEqual(float(optimizer.optimizer.state[master]["step"]), 4)
        self.assertLess(optimizer.optimizer.param_groups[0]["lr"], learning_rate_after_first)
        self.assertFalse(torch.equal(original_draft, trainer.bundle.draft_model.weight))
        self.assertTrue(torch.equal(frozen_target, trainer.bundle.model.target_table))
        self.assertTrue(trainer.latest.is_file())
        self.assertFalse(trainer.pending_state.exists())

    def test_resume_restores_weights_optimizer_rng_and_next_event_exactly(self):
        uninterrupted = self.trainer("uninterrupted")
        uninterrupted.train_event("e1", self.features)
        saved = self.root / "saved.pt"
        saved.write_bytes(uninterrupted.latest.read_bytes())
        initial_state = self.state(uninterrupted)
        uninterrupted.train_event("e2", self.features)
        expected = self.state(uninterrupted)

        resumed = self.trainer("resumed", resume=saved)
        self.assertTrue(trees_equal(resumed.bundle.draft_model.state_dict(), initial_state["draft_state_dict"]))
        self.assertTrue(trees_equal(resumed.backend.optimizer.state_dict(), initial_state["backend"]["optimizer"]))
        self.assertTrue(trees_equal(capture_rng(), initial_state["rng"]))
        self.assertEqual(resumed.core.accumulation_remainder, 0)
        resumed.train_event("e2", self.features)
        actual = self.state(resumed)
        for key in ("draft_state_dict", "backend", "rng", "global_step", "presentations"):
            self.assertTrue(trees_equal(expected[key], actual[key]), key)

    def test_two_events_match_direct_upstream_training_for_both_reductions(self):
        for reduction in ("response_mean", "upstream_weight_mean"):
            with self.subTest(reduction=reduction):
                event_trainer = self.trainer(f"events_{reduction}", reduction=reduction)
                event_trainer.train_event("e1", self.features)
                event_trainer.train_event("e2", self.features)
                expected = self.state(event_trainer)

                # Recreate identical weights, optimizer and RNG, then train with
                # upstream TrainerCore directly: no event or ReductionAdapter.
                reference = self.trainer(f"direct_{reduction}", reduction=reduction)
                if reduction == "response_mean":
                    strategy = DirectResponseMeanStrategy(reference.bundle.model)
                else:
                    strategy = reference.author_strategy
                core = TrainerCore(strategy, reference.backend, accumulation_steps=2)
                optimizer_steps = 0
                presentations = 0
                for event in range(2):
                    for pass_index in range(2):
                        for path in self.features:
                            raw = torch.load(path, weights_only=True)
                            result = core.train_step(reference._batch(raw))
                            optimizer_steps += int(result.optimizer_stepped)
                            presentations += 1

                self.assertTrue(trees_equal(
                    reference.bundle.draft_model.state_dict(), expected["draft_state_dict"]
                ))
                self.assertTrue(trees_equal(
                    reference.backend.optimizer.state_dict(), expected["backend"]["optimizer"]
                ))
                self.assertTrue(trees_equal(capture_rng(), expected["rng"]))
                self.assertEqual(optimizer_steps, expected["global_step"])
                self.assertEqual(presentations, expected["presentations"])
                self.assertEqual(core.accumulation_remainder, 0)

    def test_export_restores_every_cpu_rng_stream(self):
        trainer = self.trainer()
        trainer.train_event("e1", self.features)
        self.assertTrue(trees_equal(capture_rng(), self.state(trainer)["rng"]))

    def test_close_releases_runtime_and_keeps_status(self):
        trainer = self.trainer()
        trainer.train_event("e1", self.features)
        trainer.close()
        trainer.close()
        self.assertTrue(trainer.status()["closed"])
        self.assertEqual(trainer.status()["global_step"], 2)
        self.assertFalse(hasattr(trainer, "bundle"))
        self.assertFalse(hasattr(trainer, "core"))

    def test_invalid_event_budget_does_not_train(self):
        trainer = self.trainer()
        with self.assertRaisesRegex(ValueError, "remainder"):
            trainer.train_event("partial", self.features[:1], passes=1)
        with self.assertRaisesRegex(ValueError, "horizon"):
            trainer.train_event("beyond_horizon", self.features, passes=5)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.author_strategy.calls, 0)
        self.assertFalse(list(trainer.events_root.iterdir()))

    def test_feature_validation_happens_before_any_update(self):
        trainer = self.trainer()
        bad_feature = feature("bad")
        bad_feature["loss_mask"][0, 0] = 1
        torch.save(bad_feature, self.features[1])
        with self.assertRaisesRegex(ValueError, "Mask"):
            trainer.train_event("bad", self.features)
        self.assertEqual(trainer.author_strategy.calls, 0)
        self.assertFalse(trainer.failed)

    def test_duplicate_event_and_response_refused(self):
        trainer = self.trainer()
        with self.assertRaisesRegex(ValueError, "Duplicate response"):
            trainer.train_event("bad", [self.features[0], self.features[0]])
        trainer.train_event("e1", self.features)
        original = trainer.bundle.draft_model.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "Duplicate event"):
            trainer.train_event("e1", self.features)
        self.assertTrue(torch.equal(original, trainer.bundle.draft_model.weight))

    def test_resume_rejects_changed_reduction_and_old_schema(self):
        trainer = self.trainer()
        trainer.train_event("e1", self.features)
        with self.assertRaisesRegex(ValueError, "reduction"):
            self.trainer("different_reduction", resume=trainer.latest, reduction="upstream_weight_mean")
        state = self.state(trainer)
        state["event_schema"] = "specforge_event_trainer_v1"
        old_state = self.root / "old.pt"
        torch.save(state, old_state)
        with self.assertRaisesRegex(ValueError, "archived code"):
            self.trainer("old_schema", resume=old_state)

    def test_export_failure_keeps_commit_and_blocks_more_updates(self):
        trainer = self.trainer()
        trainer.train_event("e1", self.features)
        committed_bytes = trainer.latest.read_bytes()
        with mock.patch("specforge.export.to_hf.export_to_hf", side_effect=OSError("copy failure")):
            with self.assertRaisesRegex(OSError, "copy failure"):
                trainer.train_event("e2", self.features)
        self.assertEqual(trainer.latest.read_bytes(), committed_bytes)
        self.assertTrue(trainer.failed)
        self.assertTrue(trainer.pending_state.exists())
        self.assertTrue((trainer.events_root / "e2" / "FAILED.json").is_file())
        with self.assertRaisesRegex(ValueError, "failed"):
            trainer.train_event("retry", self.features)
        with self.assertRaisesRegex(ValueError, "Uncommitted"):
            self.trainer(resume=trainer.latest)

    def test_partial_training_failure_blocks_live_and_directory_reuse(self):
        trainer = self.trainer()
        original_step = trainer.core.train_step

        def interrupt_second_presentation(*args, **kwargs):
            if trainer.presentations == 1:
                raise RuntimeError("training failure")
            return original_step(*args, **kwargs)

        with mock.patch.object(trainer.core, "train_step", side_effect=interrupt_second_presentation):
            with self.assertRaisesRegex(RuntimeError, "training failure"):
                trainer.train_event("failed", self.features)
        self.assertTrue(trainer.failed)
        self.assertEqual(trainer.core.accumulation_remainder, 1)
        self.assertFalse(trainer.latest.exists())
        with self.assertRaisesRegex(ValueError, "failed"):
            trainer.train_event("retry", self.features)
        with self.assertRaisesRegex(ValueError, "Uncommitted event"):
            self.trainer()

    def test_failure_record_error_does_not_hide_training_error(self):
        trainer = self.trainer()
        training_error = RuntimeError("original training failure")
        write_json = T.write_json

        def persist(path, value):
            if path.name == "FAILED.json":
                raise OSError("disk full")
            write_json(path, value)

        with mock.patch.object(trainer.core, "train_step", side_effect=training_error):
            with mock.patch.object(T, "write_json", side_effect=persist):
                with self.assertRaises(RuntimeError) as caught:
                    trainer.train_event("failed", self.features)
        self.assertIs(caught.exception, training_error)
        self.assertTrue(trainer.failed)
        self.assertIn("disk full", training_error.__notes__[0])


if __name__ == "__main__":
    unittest.main()
