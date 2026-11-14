"""CPU native-shaped state fixtures; no model downloads or GPU execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from experiments.full_k_gsm8k.training import FullKTrainer
from experiments.ok_continuation.resume import differences
from experiments.threatened_repair_replay.state import (
    capture_state, copy_tree, restore_state, tensor_change, verify_state,
)
from experiments.threatened_repair_replay.training import ThreatReplayTrainer
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.training import ReductionAdapter
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepOutput


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
        self.draft_model.layers = torch.nn.ModuleList()
        self.lm_head = torch.nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
        self.lm_head.requires_grad_(False)
        self.register_buffer("marker", torch.tensor([3.]), persistent=False)
        self.costs = {"teacher": 0}
        self._cached_seq_len = 16
        self._cached_block_mask = object()
        self._sample_anchor_positions = lambda: dict(
            positions=[torch.randperm(7)[:3].tolist()], keep_mask=[[True] * 3])

    def begin_event(self):
        self.costs.clear()
        self.costs["teacher"] = 0

    def end_event(self):
        return dict(self.costs)

    def _chunk_loss(self, *args):
        return None


class TinyBackend:
    def __init__(self, model):
        self.model = model
        self.optimizer = DirectBF16AdamW(model)
        # A continuation fixture with nonempty BF16 moments and step counters.
        for parameter in self.optimizer.model_params:
            self.optimizer.optimizer.state[parameter] = dict(
                step=torch.tensor(768.), exp_avg=torch.full_like(parameter, .01),
                exp_avg_sq=torch.full_like(parameter, .02))

    def backward(self, loss, *, is_boundary):
        loss.backward()

    def scale_gradients(self, scale):
        torch._foreach_mul_([p.grad for p in self.optimizer.model_params], scale)

    def step(self):
        return self.optimizer.step()

    def state_dict(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])


class TinyStrategy:
    def __init__(self, trainer):
        self.trainer = trainer

    def checkpoint_state_filter(self, state):
        return state

    def forward_loss(self, batch, ctx=None):
        trainer, raw = self.trainer, batch.tensors
        observation = trainer.bundle.model._sample_anchor_positions()
        trainer._anchor_observations.append(observation)
        trainer._presentation_schedule.check_anchors(observation)
        predictions = trainer.bundle.draft_model(raw["x"])
        numerator = (predictions.float() - raw["y"]).square().sum()
        denominator = raw["denominator"]
        loss = numerator / denominator
        return StepOutput(loss=loss, metrics={}, loss_terms=(numerator, denominator),
            ratio_metrics=dict(fit=(numerator.detach(), denominator)),
            sum_metrics=dict(positions=denominator))


def tiny_trainer(root, *, count=4):
    """Use the real TrainerCore, response adapter, observation wrapper and AdamW."""
    root = Path(root)
    torch.manual_seed(17)
    model = TinyModel()
    trainer = object.__new__(ThreatReplayTrainer)
    trainer.bundle = SimpleNamespace(model=model, draft_model=model.draft_model)
    trainer.backend = TinyBackend(model.draft_model)
    trainer.author_strategy = TinyStrategy(trainer)
    trainer.strategy = ReductionAdapter(trainer.author_strategy, "response_mean")
    trainer.core = TrainerCore(trainer.strategy, trainer.backend, accumulation_steps=2)
    trainer.core._micro = trainer.presentations = 1536
    trainer.global_step, trainer.horizon, trainer.accumulation = 768, 7424, 2
    trainer.reduction, trainer.config_text = "response_mean", "native fixture"
    trainer.cfg = SimpleNamespace(run_id="fixture", data=SimpleNamespace(max_length=2272))
    trainer.contract, trainer.events = {"native": True}, []
    trainer.closed = trainer.failed = False
    trainer.output_root, trainer.events_root = root, root / "events"
    trainer.events_root.mkdir(parents=True)
    trainer._active_event = "event_007"
    (trainer.events_root / trainer._active_event).mkdir()
    trainer._anchor_observations, trainer.response_records, trainer.window_records = [], [], []
    trainer.work, trainer._work_handles = {}, []
    trainer.presentation_ordinal = None
    trainer._pending_capture = trainer._pending_gradient = trainer.firing_capture = None
    trainer._replaying = trainer.capture_first_firing = False
    trainer.diagnostic_capture_s = 0.
    trainer.on_window = trainer.ledger = None
    trainer._original_sampler = model._sample_anchor_positions
    trainer._batch = lambda raw: SimpleNamespace(tensors=raw, sample_ids=[raw["record_id"]])
    FullKTrainer._install_observations(trainer)
    features = []
    for index in range(count):
        path = root / f"feature_{index}.pt"
        torch.save(dict(record_id=f"r{index:03d}", x=torch.tensor([[1., 2., 3.]], dtype=torch.bfloat16)
                        / (index % 4 + 1), y=torch.tensor([[.5, -.25]]),
                        denominator=torch.tensor(float(index % 3 + 1))), path)
        features.append(dict(path=str(path), record_id=f"r{index:03d}", prompt_length=4, response_tokens=20))
    rows = []
    with preserve_rng():
        for pass_index in range(2):
            for index in torch.randperm(count).tolist():
                rows.append(dict(record_id=features[index]["record_id"], **{"pass": pass_index},
                                 sampled_anchors=[model._sample_anchor_positions()]))
    trainer.schedule = dict(passes=2, records=[dict(record_id=row["record_id"],
        pass_index=row["pass"], presentation_id=index, sampled_anchors=row["sampled_anchors"])
        for index, row in enumerate(rows)])
    trainer._presentation_schedule = PresentationSchedule(dict(status="completed", passes=2,
        response_presentations=count * 2, records=rows), [f["record_id"] for f in features], 2)

    def prepare(_event, _paths, _passes):
        trainer.response_records.clear()
        trainer.window_records.clear()
        trainer.work.clear()
        return features
    trainer._prepare_event = prepare
    return trainer, features


def first_pair(trainer, features):
    order = torch.randperm(len(features)).tolist()
    return [features[index] for index in order[:2]]


class StateTests(unittest.TestCase):
    def test_active_schedule_and_every_restorable_part_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(directory)
            pair = first_pair(trainer, features)
            before = capture_state(trainer)
            trainer.ledger = {"physical_steps": 0}
            trainer.ledger["physical_steps"] += 1
            trainer.ledger = None  # no timing fixture needed for the real pair
            trainer.run_pair(pair, 0, 0)
            trainer.bundle.model.eval()
            trainer.bundle.model.marker.add_(3)
            trainer.bundle.model._cached_seq_len = 99
            trainer.bundle.model._cached_block_mask = object()
            trainer.bundle.model.costs["extra"] = 1
            trainer.bundle.model._sample_anchor_positions = lambda: None
            trainer._anchor_observations.append({"dirty": True})
            trainer.presentation_ordinal = 99
            trainer.core._ratio_totals["dirty"] = (torch.tensor(1.), torch.tensor(1.))
            for parameter in trainer.bundle.draft_model.parameters():
                parameter.grad = torch.ones_like(parameter)
            torch.rand(9)
            restore_state(trainer, before)
            receipt = verify_state(trainer, before)
            self.assertEqual(receipt["status"], "passed")
            self.assertIsNotNone(trainer._presentation_schedule)
            self.assertEqual(trainer._presentation_schedule.position, 0)
            self.assertEqual(trainer.global_step, 768)
            self.assertEqual(trainer.work, {})

    def test_snapshot_optimizer_is_not_aliased_to_restored_live_state(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(directory)
            pair = first_pair(trainer, features)
            before = capture_state(trainer)
            original = copy_tree(before["scientific"]["backend"])
            restore_state(trainer, before)
            trainer.run_pair(pair, 0, 0)
            self.assertEqual(differences(before["scientific"]["backend"], original), [])
            restore_state(trainer, before)
            self.assertEqual(verify_state(trainer, before)["status"], "passed")

    def test_monotonic_work_ledger_is_not_part_of_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = tiny_trainer(directory)
            trainer.ledger = {"physical_steps": 5}
            before = capture_state(trainer)
            trainer.ledger["physical_steps"] = 9
            restore_state(trainer, before)
            verify_state(trainer, before)
            self.assertEqual(trainer.ledger["physical_steps"], 9)

    def test_frozen_parameter_mutation_saves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = tiny_trainer(directory)
            before = capture_state(trainer)
            with torch.no_grad():
                trainer.bundle.model.lm_head.weight.add_(1)
            receipt = Path(directory) / "failed.json"
            with self.assertRaisesRegex(ValueError, "frozen_parameter_version"):
                verify_state(trainer, before, receipt)
            self.assertEqual(json.loads(receipt.read_text())["status"], "failed")

    def test_partial_window_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = tiny_trainer(directory)
            trainer.core._micro += 1
            with self.assertRaisesRegex(ValueError, "complete native optimizer window"):
                capture_state(trainer)

    def test_tensor_accounting_matches_hand_calculation_and_rejects_nonfinite(self):
        first = {"w": torch.tensor([1., 2.], dtype=torch.bfloat16)}
        second = {"w": torch.tensor([2., 2.], dtype=torch.bfloat16)}
        result = tensor_change(first, second)
        self.assertEqual(result["difference_norm"], 1.)
        self.assertEqual(result["left_dot_difference"], 1.)
        self.assertEqual(result["changed_entries"], 1)
        second["w"][0] = torch.inf
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            tensor_change(first, second)


if __name__ == "__main__":
    unittest.main()
