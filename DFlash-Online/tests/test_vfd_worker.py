"""CPU-only worker orchestration, work ceilings, identity and wall-time checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.ok_events_5_8.campaign import identity
from experiments.vfd_second_pass import contract as C
from experiments.vfd_second_pass import worker as W
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json
from test_vfd_scoring import snapshot


class FakeTrainer:
    """Count orchestration operations only; native numerical tests are separate."""

    instances = []

    def __init__(self, config, output, **kwargs):
        self.output = Path(output)
        self.output.mkdir(parents=True)
        self.global_step = self.presentations = 0
        self.events = []
        self.failed = False
        self.bundle = SimpleNamespace(model=object())
        self.core = SimpleNamespace(
            strategy=SimpleNamespace(forward_loss=lambda batch: None)
        )
        self.backend = SimpleNamespace(
            backward=lambda loss: None,
            optimizer=SimpleNamespace(optimizer=SimpleNamespace(step=lambda: None)),
        )
        self.instances.append(self)

    def restore_original(self, path):
        self.global_step = self.presentations = 0
        self.events = []
        return dict(status="passed", global_step=0)

    def restore_boundary(self, path, *, arm, event, shared_fork=False):
        saved = json.loads(Path(path).read_text())
        self.global_step, self.presentations = saved["steps"], saved["presentations"]
        self.events = saved["events"]
        return dict(status="passed", global_step=self.global_step)

    def features(self, event, paths):
        return [
            dict(
                path=path, record_id=Path(path).stem, response_tokens=8, prompt_length=4
            )
            for path in paths
        ]

    def _batch(self, raw):
        return raw

    def run_pass(self, features, *, event, pass_index, full_order=None, selected=None):
        ids = [row["record_id"] for row in features]
        order = list(reversed(ids)) if full_order is None else list(full_order)
        executed = (
            order if selected is None else [item for item in order if item in selected]
        )
        for item in executed:
            self.core.strategy.forward_loss(item)
            self.backend.backward(None)
            self.presentations += 1
            if self.presentations % 2 == 0:
                self.backend.optimizer.optimizer.step()
                self.global_step += 1
        return dict(
            full_order=order,
            own_permutation=ids,
            executed_order=executed,
            optimizer_steps=len(executed) // 2,
            response_presentations=len(executed),
            training_s=len(executed) * 0.001,
            pass_wall_s=len(executed) * 0.002,
            records=[dict(record_id=item) for item in executed],
        )

    def save_boundary(self, path, *, arm, event, shared_fork=False, event_record=None):
        if event_record is not None:
            self.events.append(deepcopy(event_record))
        expected = 64 if shared_fork else C.logical_steps(arm, event)
        if self.global_step != expected or self.presentations != 2 * expected:
            raise AssertionError("Fake worker crossed the intended native boundary")
        write_json(
            path,
            dict(
                steps=self.global_step,
                presentations=self.presentations,
                events=self.events,
                arm=arm,
                event=event,
            ),
        )
        return dict(
            status="passed",
            state_path=str(path),
            identity=identity(path),
            global_step=self.global_step,
            presentations=self.presentations,
            persistence_s=0.0,
        )

    def export(self, path):
        Path(path).mkdir(parents=True)
        return dict(export_s=0.0, checkpoint_path=str(path))

    def close(self):
        return None


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "original.yaml"
        self.original.write_text("fixture")
        self.plan = dict(
            schema=C.SCHEMA,
            arms=list(C.ARMS),
            events=4,
            seed=17,
            budget=C.BUDGET,
            serving=C.SERVING,
            original_execution=C.ORIGINAL_EXECUTION,
            automatic_retry=False,
            inherited_optimizer_horizon=7424,
            training_requests=list(range(512)),
            evaluation_requests=list(range(200)),
            execution_source=dict(dirty=False, commit="a" * 40),
            config="fixture",
            original_config=str(self.original),
            original_initial_state="fixture-initial.pt",
        )
        write_json(self.root / "sealed/PLAN.json", self.plan)
        (self.root / "campaign_001/states").mkdir(parents=True)

    def make_pool(self, event):
        paths = [str(self.root / f"e{event}_r{index:03d}.pt") for index in range(128)]
        pool = dict(
            features=paths, identities={path: dict(sha256="b" * 64) for path in paths}
        )
        write_json(self.root / f"campaign_001/events/event_{event:03d}/POOL.json", pool)
        return pool

    def test_invalid_worker_identity_stops_before_pool_or_model(self):
        with mock.patch.object(W, "VFDTrainer") as trainer:
            for event, arm in (
                (0, "FULL"),
                (5, "FULL"),
                (1, "UNKNOWN"),
                (2, "SHARED"),
                (2, "QUALIFICATION"),
            ):
                with self.subTest(event=event, arm=arm), self.assertRaisesRegex(
                    ValueError, "outside"
                ):
                    W.run(self.root, event, arm)
            trainer.assert_not_called()

    def test_changed_plan_stops_before_pool_or_model(self):
        self.plan["budget"] = dict(self.plan["budget"], scoring_forwards=1500)
        write_json(self.root / "sealed/PLAN.json", self.plan)
        with mock.patch.object(W, "VFDTrainer") as trainer, self.assertRaisesRegex(
            ValueError, "fixed handoff"
        ):
            W.run(self.root, 1, "FULL")
        trainer.assert_not_called()

    def test_boundary_source_checks_hash_and_expected_path(self):
        path = self.root / "state.pt"
        path.write_bytes(b"committed-state")
        receipt = dict(state_path=str(path), identity=identity(path))
        self.assertEqual(W.verify_boundary_source(path, receipt)["status"], "passed")
        path.write_bytes(b"replaced-state")
        with self.assertRaisesRegex(ValueError, "committed identity"):
            W.verify_boundary_source(path, receipt)

    def test_scoring_worker_timer_includes_loading_outside_snapshot(self):
        fake = SimpleNamespace(
            bundle=SimpleNamespace(model=None), _batch=lambda raw: raw
        )
        ledger = SimpleNamespace(
            call=lambda kind, action, *args, **kwargs: action(*args, **kwargs)
        )
        pool = dict(identities={"one.pt": dict(sha256="c" * 64)})
        with mock.patch.object(
            W.time, "perf_counter", side_effect=[10.0, 15.0]
        ), mock.patch("torch.load", return_value={}), mock.patch.object(
            W, "score_response_snapshot", return_value=dict(costs=dict(total_s=1.5))
        ):
            rows = W.scoring(
                fake, [dict(path="one.pt", record_id="one")], pool, 1, ledger
            )
        self.assertEqual(W.scoring_seconds(rows), 5.0)
        self.assertEqual(W.scoring_compute_seconds(rows), 1.5)

    def test_all_workers_enforce_exact_physical_budgets_and_own_continuity(self):
        for event in range(1, 5):
            self.make_pool(event)
        FakeTrainer.instances = []

        def score(model, batch, *, event_id, response_id, conditioning_id):
            row = snapshot([], response_id=response_id, event_id=event_id)
            row["costs"] = dict(total_s=0.0)
            return row

        def live(trainer):
            return dict(
                draft_state_dict=dict(weight=torch.tensor(float(trainer.global_step))),
                backend=dict(steps=trainer.global_step),
                rng=capture_rng(),
                global_step=trainer.global_step,
                presentations=trainer.presentations,
                events=deepcopy(trainer.events),
                accumulation_steps=2,
                horizon=7424,
            )

        patches = [
            mock.patch.object(W, "VFDTrainer", FakeTrainer),
            mock.patch.object(W, "score_response_snapshot", side_effect=score),
            mock.patch.object(W, "live_state", side_effect=live),
            mock.patch(
                "torch.load",
                side_effect=lambda path, **kwargs: dict(record_id=Path(path).stem),
            ),
            mock.patch("experiments.ok_events_5_8.campaign.verify_pool"),
            mock.patch("torch.cuda.max_memory_allocated", return_value=0),
            mock.patch("torch.cuda.max_memory_reserved", return_value=0),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        qualification = W.run(self.root, 1, "QUALIFICATION")
        expected_qualification = dict(
            training_presentations=4, backwards=4, optimizer_steps=2, scoring_forwards=8
        )
        self.assertEqual(qualification["work"]["completed"], expected_qualification)
        self.assertEqual(qualification["work"]["attempts"], expected_qualification)
        self.assertEqual(FakeTrainer.instances[0].global_step, 0)
        self.assertEqual(FakeTrainer.instances[0].presentations, 0)
        self.assertTrue(qualification["fork_equality"])
        completed = [W.run(self.root, 1, "SHARED")]
        previous = {}
        for event in range(1, 5):
            current = {}
            for arm in C.ARMS:
                row = W.run(self.root, event, arm)
                current[arm] = row
                completed.append(row)
                self.assertEqual(
                    row["state"]["global_step"], C.logical_steps(arm, event)
                )
                if event > 1:
                    self.assertEqual(
                        row["predecessor_state"]["identity"],
                        previous[arm]["state"]["identity"],
                    )
                    self.assertEqual(
                        row["state"]["replaced_predecessor_identity"],
                        previous[arm]["state"]["identity"],
                    )
                    self.assertEqual(
                        row["state"]["retired_direct_resume_boundary"][
                            "completed_event"
                        ],
                        event - 1,
                    )
                if arm == "ONCE":
                    self.assertIsNone(row["pass2"])
                elif arm != "FULL":
                    chosen = set(row["selection"]["selected_ids"])
                    expected_order = [
                        item
                        for item in current["FULL"]["pass2"]["full_order"]
                        if item in chosen
                    ]
                    self.assertEqual(row["pass2"]["executed_order"], expected_order)
                if event > 1 and arm != "FULL":
                    self.assertEqual(
                        row["pass1"]["executed_order"],
                        current["FULL"]["pass1"]["full_order"],
                    )
                self.assertGreaterEqual(
                    row["logical_updater_s"], row["costs"]["training_s"]
                )
                self.assertGreater(
                    row["costs"]["training_s"], row["costs"]["training_compute_s"]
                )
                if event == 1 or arm not in ("STATIC64", "PROGRESS64"):
                    self.assertEqual(row["physical_scoring_wall_s"], 0.0)
            previous = current
        physical = {
            key: sum(row["work"]["completed"][key] for row in completed)
            for key in (
                "optimizer_steps",
                "training_presentations",
                "backwards",
                "scoring_forwards",
            )
        }
        self.assertEqual(
            physical,
            dict(
                optimizer_steps=1664,
                training_presentations=3328,
                backwards=3328,
                scoring_forwards=1408,
            ),
        )
        self.assertTrue(
            all(not any(row["work"]["unknown"].values()) for row in completed)
        )
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
