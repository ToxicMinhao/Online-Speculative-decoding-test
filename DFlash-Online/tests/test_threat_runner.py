"""Controller wiring with real CPU Adam branches and declared fake observations."""
import json
from pathlib import Path
import tempfile
import signal
import unittest
from unittest.mock import patch

import torch

from experiments.threatened_repair_replay.core import selection_hash
from experiments.threatened_repair_replay.runner import (
    HOST_RESERVE, WindowController, export_weights, host_admission, qualification_fixture,
    scalar_update_rows,
    termination_cleanup,
)
from experiments.threatened_repair_replay.state import capture_state, copy_tree, verify_state
from experiments.threatened_repair_replay.work import Ledger, TrainingInstrumentation, WorkLimit
from online_feedback.rng import capture_rng
from experiments.ok_continuation.resume import differences
from test_threat_state import tiny_trainer


def setup_window(root, *, opportunity=True, identical=False, corrupt_observer=False):
    trainer, features = tiny_trainer(root / "trainer", count=128)
    order = torch.randperm(128).tolist()
    for ordinal in range(0, 14, 2):
        trainer.run_pair([features[index] for index in order[ordinal:ordinal+2]], 0, ordinal)
    pair = [features[index] for index in order[14:16]]
    current = {row["record_id"] for row in pair}
    records = {row["record_id"]: row for row in trainer.schedule["records"][:128]}
    choices = [row for row in features if row["record_id"] not in current][:3]
    tags, layouts = [], []
    for index, feature in enumerate(choices):
        record = records[feature["record_id"]]
        layout_id = f"{feature['record_id']}:{record['presentation_id']}"
        layouts.append(dict(layout_id=layout_id, request_id=feature["record_id"],
            reference_presentation_id=record["presentation_id"], binding=record, feature_path=feature["path"]))
        tags.append(dict(layout_id=layout_id, request_id=feature["record_id"], canonical_start=index,
            proposal_slot=1, reference_presentation_id=record["presentation_id"],
            anchor_ordinal=0, canonical_token=7, weight=1.))
    ordered = sorted(tags, key=lambda row:selection_hash("U", 8, row))
    threatened = ordered[0] if identical else ordered[-1]

    def observer(model, original_layouts, *, max_length, ledger, context):
        state = context["state"]
        rows = []
        for layout in original_layouts:
            for tag in layout["candidate_tags"]:
                failed = opportunity and state == "N" and tag == threatened
                rows.append(dict(tag, accepted_prefix=0 if failed else 3,
                    valid_proposals=3, first_error=1 if failed else None, valid=True,
                    native_correct=not failed, direct_correct=not failed,
                    margin=-.25 if failed else .25, wrong_tie=False))
        trainer.work["synthetic_observation"] = trainer.work.get("synthetic_observation", 0) + 1
        if corrupt_observer:
            torch.rand(1)
        return rows, [dict(cpu_fixture=True, actual_model_forwards=0)]

    def replayer(current, _layouts, _selected, **kwargs):
        for parameter in current.bundle.draft_model.parameters():
            parameter.grad.add_(torch.full_like(parameter, .125))
        return dict(cpu_synthetic_gradient=True)

    def exporter(weights, template, destination):
        destination.mkdir(parents=True)
        (destination / "cpu_fixture.json").write_text(json.dumps(dict(tensors=len(weights))))
        return dict(checkpoint=str(destination), export_s=0., bytes=0)

    origin = root / "origin.pt"
    origin.write_bytes(b"CPU admission placeholder")
    ledger = Ledger(root / "ledger.jsonl", limits={"optimizer_step":168})
    instrumentation = TrainingInstrumentation(trainer, ledger)
    trainer.ledger = ledger
    controller = WindowController(dict(origin_state=str(origin), origin_export=str(root)), root / "controller", ledger,
        tags, layouts, observer=observer, replayer=replayer, exporter=exporter,
        admit=lambda _extra=0:dict(status="CPU fixture"))
    controller.output.mkdir()
    return trainer, pair, controller, ledger, instrumentation, tags, layouts


class ControllerTests(unittest.TestCase):
    def test_active_branches_restore_native_and_do_not_accumulate_disposable_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, pair, controller, ledger, instruments, _, _ = setup_window(Path(directory))
            try:
                result = controller(trainer, 8, pair, 0, 14)
                self.assertEqual(trainer.global_step, 776)
                self.assertEqual(trainer.presentations, 1552)
                self.assertEqual(len(trainer.response_records), 16)
                self.assertEqual(len(trainer.window_records), 8)
                self.assertEqual(result["preclip_gradients"], {})
                self.assertEqual(ledger.completed["optimizer_step"], 4)
                self.assertEqual(len(controller.probe_exports), 4)
                self.assertEqual({row["state"] for row in controller.probe_exports}, {"P","N","U","T"})
                self.assertTrue(all(row["reused_from"] is None for row in controller.probe_exports))
                self.assertEqual(len(controller.memory_rows), 8)
                self.assertEqual(controller.windows[0]["status"], "completed")
                self.assertNotIn("synthetic_observation", trainer.work)
                self.assertEqual(trainer._presentation_schedule.position, 16)
                self.assertTrue((controller.output / "window_008" / "SEALED_SELECTION.json").exists())
                for state in trainer.backend.optimizer.optimizer.state.values():
                    self.assertEqual(int(state["step"]), 776)
            finally:
                instruments.close(); ledger.close()

    def test_no_threat_reuses_native_without_disposable_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, pair, controller, ledger, instruments, _, _ = setup_window(Path(directory), opportunity=False)
            try:
                controller(trainer, 8, pair, 0, 14)
                self.assertEqual(ledger.completed["optimizer_step"], 2)  # real N + required duplicate
                self.assertEqual(controller.memory_rows, [])
                self.assertEqual([row["reused_from"] for row in controller.probe_exports], [None,None,"N","N"])
                self.assertFalse(controller.windows[0]["active"])
            finally:
                instruments.close(); ledger.close()

    def test_identical_selection_reuses_one_actual_replay_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, pair, controller, ledger, instruments, _, _ = setup_window(Path(directory), identical=True)
            try:
                controller(trainer, 8, pair, 0, 14)
                self.assertEqual(ledger.completed["optimizer_step"], 3)
                self.assertEqual(controller.probe_exports[-1]["reused_from"], "U")
                self.assertTrue(controller.windows[0]["identical_sets"])
                self.assertEqual(len(controller.memory_rows), 4)
                alias = next(row for row in controller.memory_rows if row["state"] == "T")
                self.assertEqual(alias["reused_from"], "U")
            finally:
                instruments.close(); ledger.close()

    def test_observation_rng_corruption_fails_before_any_new_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, pair, controller, ledger, instruments, _, _ = setup_window(Path(directory), corrupt_observer=True)
            original = capture_state(trainer)
            try:
                with self.assertRaisesRegex(ValueError, "Memory observation changed"):
                    controller(trainer, 8, pair, 0, 14)
                self.assertEqual(ledger.completed["optimizer_step"], 0)
                self.assertEqual(controller.windows[0]["status"], "failed")
                self.assertTrue((controller.output / "window_008" / "FAILED.json").exists())
                self.assertEqual(len(controller.witnesses), 1)
                verify_state(trainer, original)
            finally:
                instruments.close(); ledger.close()

    def test_qualification_fixture_uses_identity_order_and_preserves_teachers(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, pair, controller, ledger, instruments, tags, layouts = setup_window(Path(directory))
            try:
                before = copy_tree(tags)
                one = qualification_fixture(tags, layouts, [])
                two = qualification_fixture(list(reversed(tags)), list(reversed(layouts)), [])
                self.assertEqual(one, two)
                self.assertEqual(one["selected_U"][0]["proposal_slot"], one["selected_T"][0]["proposal_slot"])
                self.assertNotEqual(one["selected_U"][0]["layout_id"], one["selected_T"][0]["layout_id"])
                self.assertEqual(tags, before)
                self.assertFalse(one["efficacy_evidence"])
                with self.assertRaisesRegex(ValueError, "No bounded"):
                    qualification_fixture(tags, layouts, [row["request_id"] for row in tags])
            finally:
                instruments.close(); ledger.close()

    def test_weights_only_export_preserves_bits_config_and_rng(self):
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template"; template.mkdir()
            initial = dict(weight=torch.tensor([[1.,2.]], dtype=torch.bfloat16))
            save_file(initial, str(template / "model.safetensors"))
            (template / "config.json").write_text('{"unchanged":true}')
            updated = dict(weight=torch.tensor([[1.5,2.]], dtype=torch.bfloat16))
            rng = capture_rng()
            result = export_weights(updated, template, root / "export")
            self.assertEqual(differences(updated, load_file(str(root / "export/model.safetensors"))), [])
            self.assertEqual((root / "export/config.json").read_text(), (template / "config.json").read_text())
            self.assertEqual(differences(rng, capture_rng()), [])
            self.assertTrue(result["weights_only"])
            self.assertEqual(result["weights_identity"]["bytes"], (root / "export/model.safetensors").stat().st_size)
            self.assertEqual(len(result["weights_identity"]["sha256"]), 64)

    def test_host_admission_keeps_32_gib_reserved(self):
        with patch("pathlib.Path.read_text", return_value=f"MemAvailable: {(HOST_RESERVE+8192)//1024} kB\n"):
            self.assertEqual(host_admission(4096)["status"], "passed")
            with self.assertRaisesRegex(MemoryError, "Host memory admission"):
                host_admission(16384)

    def test_scalar_csv_has_stable_fields_without_private_paths_or_layouts(self):
        records = scalar_update_rows([dict(window=8, state="U", feature_path="private.pt",
            replay=dict(coefficient=.1, loss=2., layout_receipts=[dict(private="tokens")]),
            displacement=dict(difference_norm=.3))])
        self.assertEqual(records, [dict(window=8, state="U", **{
            "replay.coefficient":.1, "replay.loss":2., "displacement.difference_norm":.3})])

    def test_termination_stops_work_ignores_second_signal_then_restores_handler(self):
        bindings = []
        original = object()
        with patch("signal.getsignal", return_value=original), patch("signal.signal", side_effect=lambda *args:bindings.append(args)):
            with self.assertRaisesRegex(WorkLimit, "safe stop"):
                with termination_cleanup():
                    handler = bindings[-1][1]
                    handler(signal.SIGTERM, None)
            self.assertIs(bindings[-1][1], original)
            self.assertEqual(bindings[-2], (signal.SIGTERM, signal.SIG_IGN))


if __name__ == "__main__":
    unittest.main()
