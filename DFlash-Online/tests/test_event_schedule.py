"""Configured event sizes, continuation and stopping without model execution."""
from pathlib import Path
import tempfile
import unittest

import yaml

from online_feedback.controller import OnlineController
from online_feedback.recipe import OnlineRecipe, load_event_config
from test_controller import FakeBackend, FakeTrainer


class CountingTrainer(FakeTrainer):
    def train_event(self, event_id, paths, passes=2):
        self.calls.append((event_id, list(paths), passes))
        steps = len(paths) * passes // 2
        self.steps += steps
        return {"checkpoint_path": event_id, "optimizer_steps": steps}


class EventScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="event_schedule_")
        self.root = Path(self.temp.name)
        self.config = Path(__file__).parents[1] / "configs/osd.yaml"
        self.cfg, self.recipe = load_event_config(self.config)
        self.trainer = CountingTrainer()
        self.trainer.recipe = self.recipe
        self.backend = FakeBackend()
        self.controller = OnlineController(self.backend, self.trainer, self.root / "loop")

    def tearDown(self):
        self.temp.cleanup()

    def serve(self, start, count):
        for index in range(start, start + count):
            self.controller.serve(
                [101, 7], request_id=f"r{index}", session_id="stream",
                request_index=index, max_new_tokens=4,
            )

    def test_configured_buffers_are_disjoint_and_trainer_continues(self):
        self.assertEqual(self.controller.buffer.capacity, 128)
        self.serve(1, 128)
        first = self.controller.update("e1")
        self.assertEqual(first["training"]["optimizer_steps"], 128)
        self.assertEqual(self.controller.buffer.pending(), [])
        self.serve(129, 128)
        second = self.controller.update("e2")
        self.assertEqual(second["training"]["optimizer_steps"], 128)
        self.assertIs(self.controller.trainer, self.trainer)
        self.assertEqual(self.trainer.steps, 256)
        for number, (_, paths, passes) in enumerate(self.trainer.calls):
            self.assertEqual(passes, 2)
            self.assertEqual([p.stem for p in paths],
                             [f"r{i}" for i in range(number * 128 + 1, (number + 1) * 128 + 1)])
        self.assertEqual(self.controller.buffer.pending(), [])

    def test_event_59_stops_before_sealing_or_preparing(self):
        self.trainer.calls = [(f"e{i}", [], 2) for i in range(1, 59)]
        self.trainer.steps = 7424
        self.serve(1, 1)
        pending = self.controller.buffer.pending()
        with self.assertRaisesRegex(ValueError, "event limit"):
            self.controller.update("e59")
        self.assertEqual(self.controller.buffer.pending(), pending)
        self.assertEqual(self.backend.prepared, [])
        self.assertEqual(self.backend.published, [])
        self.assertEqual(self.trainer.steps, 7424)

    def test_explicit_qualification_size_and_legacy_default_remain_available(self):
        self.controller = OnlineController(
            self.backend, self.trainer, self.root / "qualification", buffer_capacity=2,
        )
        self.serve(1, 2)
        self.controller.update("q1", count=2, passes=2)
        self.assertEqual(self.trainer.steps, 2)
        self.trainer.recipe = OnlineRecipe()
        historical = OnlineController(self.backend, self.trainer, self.root / "historical")
        self.assertEqual(historical.buffer.capacity, 8)
        self.assertIsNone(historical.max_events)

    def test_schedule_must_match_optimizer_horizon(self):
        for field, value in (("max_steps", 8), ("accumulation_steps", 3)):
            raw = yaml.safe_load(self.config.read_text())
            raw["training"][field] = value
            path = self.root / f"{field}.yaml"
            path.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError, "schedule and optimizer horizon"):
                load_event_config(path)


if __name__ == "__main__":
    unittest.main()
