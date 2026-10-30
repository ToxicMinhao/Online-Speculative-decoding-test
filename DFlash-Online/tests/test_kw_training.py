"""KW wiring with tiny real DFlash forwards, TrainerCore and direct BF16 AdamW.

The CPU source boundary has synthetic K8 counters/history. Only the production
58-tensor/input-history admission gate is replaced; restore equality, forward,
checkpoint recomputation, backward, accumulation, optimizer and KW are real.
These fixtures are separate from the bounded full-model qualification budget.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import specforge.training.assembly as assembly

from online_feedback import training as T
from online_feedback.kw_training import (
    CONTRACT_KEY, KWContinuationTrainer, validate_kw_snapshot_metadata,
    verify_kw_live_resume,
)
from online_feedback.response_weighting import KW_CONTRACT, response_features
from online_feedback.rng import capture_rng
from test_training_contract import trees_equal


class KWTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_osd_recipe
        cls.legacy = test_osd_recipe.OSDRecipeTests
        cls.legacy.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.legacy.tearDownClass.__func__(cls)

    def setUp(self):
        self.legacy.setUp(self)
        patch = mock.patch.object(T, "ranking_eos_token_ids", return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)
        tiny_loader = assembly.build_model_bundle.side_effect

        def block16_tiny(*args, **kwargs):
            bundle = tiny_loader(*args, **kwargs)
            bundle.model.block_size = 16
            bundle.draft_model.config.block_size = 16
            bundle.model.loss_decay_gamma = 7.0
            bundle.model.objective_chunk_blocks = 8
            return bundle

        assembly.build_model_bundle.side_effect = block16_tiny
        self.source = self.root / "synthetic_K8.pt"
        initial = self.ordinary("initial")
        for index, parameter in enumerate(initial.bundle.draft_model.parameters()):
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, (index + 1) / 100)
        initial.backend.optimizer.step()
        for state in initial.backend.optimizer.optimizer.state.values():
            state["step"].fill_(1024)
        initial.global_step, initial.presentations = 1024, 2048
        initial.core._micro = 2048
        initial.events = [{"event_id": f"event_{i:03d}"} for i in range(1, 8)]
        state = initial._checkpoint_state({"event_id": "event_008"})
        torch.save(state, self.source)
        initial.close()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def ordinary(self, name, *, resume_state=None):
        trainer = T.SpecForgeEventTrainer(
            self.config, self.root / name, resume_state=resume_state,
            objective="osd_first_error", reduction="response_mean", ranking_lambda=0,
        )
        self.addCleanup(trainer.close)
        return trainer

    def kw(self, name, *, resume_state=None, completed_event=8):
        # Only architecture/complete historical receipt admission is too large
        # for this fixture. All live scientific-state comparisons remain active.
        with mock.patch("experiments.kw_v1.resume.validate_snapshot") as admit:
            trainer = KWContinuationTrainer(
                self.config, self.root / name,
                resume_state=self.source if resume_state is None else resume_state,
                completed_event=completed_event,
            )
        self.addCleanup(trainer.close)
        self.assertEqual(admit.call_args.kwargs["algorithm_arm"],
                         "K" if completed_event == 8 else "KW")
        return trainer

    @staticmethod
    def scientific_state(trainer):
        return deepcopy({
            "draft": dict(trainer.bundle.draft_model.state_dict()),
            "optimizer": trainer.backend.optimizer.state_dict(),
            "rng": capture_rng(), "global_step": trainer.global_step,
            "presentations": trainer.presentations, "micro": trainer.core._micro,
            "ratio_totals": trainer.core._ratio_totals,
            "sum_totals": trainer.core._sum_totals,
        })

    def run_small(self, trainer, paths):
        metadata = T.SpecForgeEventTrainer._prepare_event(trainer, "fixture", paths, 1)
        if isinstance(trainer, KWContinuationTrainer):
            trainer._event_start_presentations = trainer.presentations
        trainer.bundle.model.begin_event()
        try:
            return trainer._train_passes(metadata, 1)
        finally:
            trainer.bundle.model.end_event()

    def test_two_native_windows_match_K_and_observe_each_response_once(self):
        native = self.ordinary("native", resume_state=self.source)
        expected_boundaries = []
        native_step = native.backend.optimizer.step

        def native_observer():
            norm = native_step()
            expected_boundaries.append(self.scientific_state(native))
            return norm

        with mock.patch.object(native.backend.optimizer, "step", new=native_observer):
            expected_rows = self.run_small(native, self.features[:4])
        expected = self.scientific_state(native)

        weighted = self.kw("weighted")
        observed_boundaries, response_outputs, applications = [], [], []
        weighted.optimizer_application_observer = applications.append

        def boundary_observer(stage, trainer, payload):
            if stage == "post_step":
                observed_boundaries.append(self.scientific_state(trainer))

        weighted.qualification_observer = boundary_observer
        forward = weighted.author_strategy.forward_loss

        def response_observer(batch, ctx=None):
            output = forward(batch, ctx)
            length = int(batch.tensors["loss_mask"].sum())
            response_outputs.append((float(output.loss.detach().float()),
                                     response_features(output.loss, output.sum_metrics, length)))
            return output

        model = weighted.bundle.model
        with mock.patch.object(weighted.author_strategy, "forward_loss", side_effect=response_observer), \
                mock.patch.object(model, "_chunk_loss", wraps=model._chunk_loss) as chunks:
            actual_rows = self.run_small(weighted, self.features[:4])
        self.assertTrue(trees_equal(observed_boundaries, expected_boundaries))
        self.assertTrue(trees_equal(self.scientific_state(weighted), expected))
        self.assertEqual(len(response_outputs), 4)
        self.assertGreater(chunks.call_count, len(response_outputs))
        self.assertEqual([row["a"] for row in weighted.window_rows], [0.0, 0.0])
        self.assertEqual(len(applications), 6)  # Four previews and two real steps.
        self.assertEqual(weighted.controller.n, 1)
        for index, (actual, reference) in enumerate(zip(actual_rows, expected_rows)):
            for key in ("record_id", "pass", "optimizer_step_after", "optimizer_stepped",
                        "learning_rate", "metrics", "sampled_anchors"):
                self.assertTrue(trees_equal(actual[key], reference[key]), (index, key))
        for index, row in enumerate(weighted.window_rows):
            first, second = response_outputs[index * 2:index * 2 + 2]
            self.assertEqual(row["ordinary_response_losses"], [first[0], second[0]])
            self.assertEqual(row["features"], [first[1].tolist(), second[1].tolist()])
            self.assertEqual(row["presentation_ordinals"], [index * 2, index * 2 + 1])
            self.assertEqual(row["window_in_pass"], index + 1)
            self.assertEqual(row["global_step"], 1025 + index)
            self.assertEqual(row["pass_index"], 0)
            self.assertEqual(row["response_ids"],
                             [entry["record_id"] for entry in actual_rows[index * 2:index * 2 + 2]])
        self.assertIsNone(weighted.window_rows[0]["credit"])
        self.assertEqual(weighted.window_rows[1]["credit"]["producing_identity"]["global_step"], 1025)

    def test_complete_event_clears_each_pass_and_controller_restores_exactly(self):
        paths = []
        for index in range(128):
            raw = torch.load(self.features[index % len(self.features)], weights_only=True)
            raw["record_id"] = f"event9_response_{index:03d}"
            raw["label_binding"]["record_id"] = raw["record_id"]
            path = self.root / f"event9_response_{index:03d}.pt"
            torch.save(raw, path)
            paths.append(path)
        trainer = self.kw("event")
        receipt = trainer.train_event("event_009", paths)
        self.assertEqual((trainer.global_step, trainer.presentations), (1152, 2304))
        self.assertEqual((trainer.controller.windows, trainer.controller.previews, trainer.controller.n),
                         (128, 256, 126))
        self.assertEqual(len(trainer.window_rows), 128)
        self.assertIsNone(trainer.controller.pending)
        self.assertEqual([i for i, row in enumerate(trainer.window_rows) if row["credit"] is None],
                         [0, 64])
        self.assertEqual([i for i, row in enumerate(trainer.window_rows)
                          if row["pending_discarded_at_pass_end"]], [63, 127])
        self.assertEqual([row["window_in_pass"] for row in trainer.window_rows],
                         list(range(1, 65)) * 2)
        self.assertEqual([row["pass_index"] for row in trainer.window_rows], [0] * 64 + [1] * 64)
        self.assertEqual(receipt["kw"]["credit_opportunities"], 126)
        self.assertEqual(receipt["kw"]["previews"], 256)
        saved = torch.load(trainer.latest, weights_only=False)
        self.assertEqual(validate_kw_snapshot_metadata(saved, completed_event=9)["status"], "passed")
        expected_state = self.scientific_state(trainer)
        expected_controller = trainer.kw_checkpoint_state()
        restored = self.kw("restored", resume_state=trainer.latest, completed_event=9)
        self.assertTrue(trees_equal(self.scientific_state(restored), expected_state))
        self.assertTrue(trees_equal(restored.kw_checkpoint_state(), expected_controller))
        self.assertEqual(verify_kw_live_resume(restored, saved)["controller_exact"], True)
        self.assertEqual(restored.kw_migration, saved["kw_migration"])
        features = restored._prepare_event("event_010", paths, 2)
        restored._window_log = None
        restored.bundle.model.begin_event()
        try:
            restored._train_passes(features[:2], 1)
        finally:
            restored.bundle.model.end_event()
        self.assertIsNone(restored.window_rows[0]["credit"])
        self.assertEqual(restored.controller.n, 126)
        self.assertEqual(restored.window_rows[0]["phi_entry"], expected_controller["phi"].tolist())

    def test_migration_adds_only_zero_controller_and_preserves_original_contract(self):
        source = torch.load(self.source, weights_only=False)
        trainer = self.kw("migration")
        self.assertTrue(verify_kw_live_resume(trainer, source)["migration_from_K8"])
        contract = dict(trainer.contract)
        self.assertEqual(contract.pop(CONTRACT_KEY), KW_CONTRACT)
        self.assertEqual(contract, source["algorithm_contract"])
        self.assertEqual((trainer.controller.n, trainer.controller.windows, trainer.controller.previews), (0, 0, 0))
        self.assertTrue(torch.equal(trainer.controller.phi, torch.zeros(5, dtype=torch.float64)))
        self.assertEqual(trainer.kw_migration["global_step"], 1024)
        trainer.controller.phi[0] = 0.1
        with self.assertRaisesRegex(ValueError, "zero controller"):
            verify_kw_live_resume(trainer, source)

    def test_event_guard_rejects_partial_or_wrong_boundary_before_model_calls(self):
        trainer = self.kw("guards")
        original = self.scientific_state(trainer)
        for event_id, paths, passes in (("event_010", self.features * 16, 2),
                                         ("event_009", self.features, 2),
                                         ("event_009", self.features * 16, 1)):
            with self.subTest(event=event_id, count=len(paths), passes=passes):
                with self.assertRaisesRegex(ValueError, "exactly its next"):
                    trainer._prepare_event(event_id, paths, passes)
        self.assertTrue(trees_equal(self.scientific_state(trainer), original))
        with self.assertRaisesRegex(ValueError, "incomplete event"):
            trainer._checkpoint_state({"event_id": "event_009"})

    def test_actual_qualification_flow_spends_eight_pairs_and_fourteen_applications(self):
        from experiments.kw_v1 import qualify as Q
        rows = []
        for path in self.features[:4]:
            raw = torch.load(path, weights_only=True)
            rows.append({"path": str(path), "record_id": raw["record_id"],
                         "length": raw["input_ids"].shape[-1]})
        selection = self.root / "qualification_selection.json"
        selection.write_text(json.dumps({
            "source_state": str(self.source),
            "source_state_identity": {"sha256": hashlib.sha256(self.source.read_bytes()).hexdigest()},
            "rows": rows,
        }))
        root = self.root / "qualification_run"
        root.mkdir()
        plan = {"qualification": str(selection), "config": str(self.config),
                "execution_source": {"commit": "tiny_CPU_fixture"}}

        def ordinary_factory(config, output, *, resume_state, completed_event, arm):
            self.assertEqual((completed_event, arm), (8, "K"))
            return self.ordinary(str(Path(output).relative_to(self.root)), resume_state=resume_state)

        def weighted_factory(config, output, *, resume_state, completed_event):
            return self.kw(str(Path(output).relative_to(self.root)), resume_state=resume_state,
                           completed_event=completed_event)

        # The production harness measures CUDA peak counters. Only those device
        # telemetry calls are substituted; CPU model/optimizer arithmetic is real.
        with mock.patch.object(Q, "OKContinuationTrainer", side_effect=ordinary_factory), \
                mock.patch.object(Q, "KWContinuationTrainer", side_effect=weighted_factory), \
                mock.patch("torch.cuda.reset_peak_memory_stats"), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=0), \
                mock.patch("torch.cuda.max_memory_reserved", return_value=0), \
                mock.patch("torch.cuda.synchronize"):
            result = Q.qualify(plan, root)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["work"], {
            "response_forwards": 8, "response_backwards": 8,
            "disposable_optimizer_applications": 14, "serving_generations": 0,
            "persistent_optimizer_updates": 0,
        })
        self.assertTrue(all(row["status"] == "passed" for row in result["checks"]))
        self.assertEqual(result["controller_after_two_windows"]["n"], 1)
        ledger = json.loads((root / "implementation_qualification/WORK.json").read_text())
        self.assertEqual(len(ledger["attempts"]), 30)
        self.assertTrue(all(row["status"] == "completed" for row in ledger["attempts"]))
        with self.assertRaises(FileExistsError):
            Q.qualify(plan, root)


class QualificationLedgerTests(unittest.TestCase):
    @staticmethod
    def plan_fixture(root):
        source = root / "source.pt"
        source.write_bytes(b"fixture source")
        selection = root / "selection.json"
        selection.write_text(json.dumps({
            "source_state": str(source),
            "source_state_identity": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            "rows": [],
        }))
        return {"qualification": str(selection), "config": "fixture",
                "execution_source": {"commit": "fixture"}}

    def test_budget_is_durable_before_dispatch_and_never_expands(self):
        from experiments.kw_v1.qualify import LIMITS, WorkLedger
        with TemporaryDirectory() as directory:
            path = Path(directory) / "WORK.json"
            ledger = WorkLedger(path)
            record = ledger.start("response_forwards", arm="K", response="one")
            saved = json.loads(path.read_text())
            self.assertEqual(saved["attempts"][0]["status"], "started")
            self.assertEqual(saved["work"]["response_forwards"], 1)
            ledger.complete(record)
            self.assertEqual(json.loads(path.read_text())["attempts"][0]["status"], "completed")
            for _ in range(1, LIMITS["response_forwards"]):
                ledger.start("response_forwards", arm="K")
            before = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                ledger.start("response_forwards")
            self.assertEqual(path.read_bytes(), before)
            for prohibited in ("serving_generations", "persistent_optimizer_updates"):
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    ledger.start(prohibited)
            self.assertEqual(path.read_bytes(), before)

    def test_constructor_failure_keeps_original_error_and_zero_work_receipt(self):
        from experiments.kw_v1 import qualify as Q
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan_fixture(root)
            failure = RuntimeError("fixture runtime construction failed")
            with mock.patch.object(Q, "OKContinuationTrainer", side_effect=failure):
                with self.assertRaises(RuntimeError) as caught:
                    Q.qualify(plan, root)
            self.assertIs(caught.exception, failure)
            result = json.loads((root / "IMPLEMENTATION_QUALIFICATION.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("fixture runtime construction failed", result["error"])
            self.assertFalse(result["automatic_retry"])
            self.assertTrue(all(count == 0 for count in result["work"].values()))

    def test_missing_selection_leaves_zero_work_failure_receipt(self):
        from experiments.kw_v1 import qualify as Q
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan_fixture(root)
            Path(plan["qualification"]).unlink()
            with self.assertRaises(FileNotFoundError):
                Q.qualify(plan, root)
            result = json.loads((root / "IMPLEMENTATION_QUALIFICATION.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertTrue(all(count == 0 for count in result["work"].values()))

    def test_cleanup_failure_does_not_mask_original_training_failure(self):
        from experiments.kw_v1 import qualify as Q
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan_fixture(root)
            trainer = SimpleNamespace(
                backend=SimpleNamespace(optimizer=SimpleNamespace(step=lambda: None)),
                close=mock.Mock(side_effect=OSError("fixture cleanup failed")), failed=False,
            )
            original = RuntimeError("fixture backward failed")
            with mock.patch.object(Q, "OKContinuationTrainer", return_value=trainer), \
                    mock.patch.object(Q, "run_two_windows", side_effect=original), \
                    mock.patch("torch.cuda.reset_peak_memory_stats"):
                with self.assertRaises(RuntimeError) as caught:
                    Q.qualify(plan, root)
            self.assertIs(caught.exception, original)
            result = json.loads((root / "IMPLEMENTATION_QUALIFICATION.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("fixture backward failed", result["error"])
            self.assertIn("fixture cleanup failed", str(result["cleanup_errors"]))

    def test_receipt_failure_does_not_mask_original_constructor_failure(self):
        from experiments.kw_v1 import qualify as Q
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan_fixture(root)
            write = Q.write_json

            def fail_final_receipt(path, value):
                if Path(path).name == "IMPLEMENTATION_QUALIFICATION.json" and value["status"] == "failed":
                    raise OSError("fixture final receipt unavailable")
                return write(path, value)

            original = RuntimeError("fixture constructor failed")
            with mock.patch.object(Q, "OKContinuationTrainer", side_effect=original), \
                    mock.patch.object(Q, "write_json", side_effect=fail_final_receipt):
                with self.assertRaises(RuntimeError) as caught:
                    Q.qualify(plan, root)
            self.assertIs(caught.exception, original)
            self.assertIn("fixture final receipt unavailable", str(getattr(original, "__notes__", [])))
            self.assertTrue((root / "implementation_qualification/WORK.json").is_file())


if __name__ == "__main__":
    unittest.main()
