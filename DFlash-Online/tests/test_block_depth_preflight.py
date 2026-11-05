"""CPU admission tests: frozen export, strict scope, and durable failure receipts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.block_depth import preflight as P
from experiments.block_depth.data import build_manifests


def plan():
    root = str(P.PROJECT_ROOT / "runs/dflash_block_depth_20261105/qualification_001")
    files = {name: dict(bytes=1, sha256="a" * 64) for name in (
        "experiments/block_depth/native_routes.py", "experiments/block_depth/preflight.py",
        "online_feedback/sglang_hooks.py", "online_feedback/native.py",
    )}
    inputs = dict(model_pin=root + "/MODEL_PIN.json", runtime_pin=root + "/RUNTIME_PIN.json",
                  manifest_file=root + "/MANIFESTS.json")
    return dict(
        schema="block_depth_qualification_v1", stage="native_qualification_and_pricing",
        checkpoint_path=str(P.EXPORT), target_path=str(P.TARGET), worker_python=P.WORKER_PYTHON,
        output_root=root, source_root=str(P.SOURCE_ROOT), source_files=files,
        code_commit="a" * 40, budget=deepcopy(P.BUDGET), serving=deepcopy(P.SERVING),
        storage=P.capacity_plan(), input_identities={p: dict(bytes=1, sha256="b" * 64) for p in inputs.values()},
        **inputs,
    )


def boundary(export):
    return dict(
        status="passed", arm="KPLUS", event=58,
        weights=str(export / "model.safetensors"),
        weights_identity=deepcopy(P.EXPORT_FILES["model.safetensors"]),
        scientific_check=dict(status="passed", completed_event=58, global_step=7424,
                              presentations=14848, export_exact=True),
    )


def manifests():
    def rows(split, count, offset):
        return [dict(source_row=i, source_id=f"gsm8k:{split}:{i}",
                     question=f"{split} question {i}", rendered_prompt=f"user:{split} question {i}",
                     prompt_ids=[offset + i, 151644]) for i in range(count)]
    return build_manifests(rows("train", 512, 100), rows("test", 384, 10000))


class PlanTests(unittest.TestCase):
    def test_fixed_first_milestone_has_zero_fitting_and_final_work(self):
        result = P.validate_plan(plan())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["budget"]["readout_generations"], 16)
        self.assertEqual(result["budget"]["timed_route_measurements_max"], 2048)
        for key in ("readout_optimizer_steps", "backbone_optimizer_steps", "gate_fit_rows", "final_generations"):
            changed = plan()
            changed["budget"][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "budget"):
                P.validate_plan(changed)

    def test_no_later_checkpoint_or_protocol_substitution(self):
        for key, value in (("checkpoint_path", str(P.EXPORT).replace("058", "057")),
                           ("worker_python", "/usr/bin/python"), ("stage", "final_evaluation")):
            changed = plan()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                P.validate_plan(changed)
        changed = plan()
        changed["serving"]["block_size"] = 10
        with self.assertRaisesRegex(ValueError, "serving"):
            P.validate_plan(changed)

    def test_scope_cannot_expand_generation_or_pricing_budget(self):
        for key in ("readout_generations", "qualification_generations_max", "timed_route_measurements_max"):
            changed = plan()
            changed["budget"][key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "budget"):
                P.validate_plan(changed)

    def test_resource_budget_cannot_be_nan_or_lower_reserve(self):
        for key, value in (("peak_additional_bytes", float("nan")), ("reserve_bytes", 0)):
            changed = plan()
            changed["storage"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "storage"):
                P.validate_plan(changed)

    def test_absolute_authorized_paths_and_source_traversal(self):
        changed = plan()
        changed["output_root"] = "/tmp/depth"
        with self.assertRaisesRegex(ValueError, "escapes"):
            P.validate_plan(changed)
        changed = plan()
        changed["source_files"]["../wrong.py"] = dict(bytes=1, sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "escapes"):
            P.validate_plan(changed)

    def test_missing_manifest_identity_and_source_dependency_are_failures(self):
        changed = plan()
        del changed["input_identities"][changed["manifest_file"]]
        with self.assertRaisesRegex(ValueError, "sealed identity"):
            P.validate_plan(changed)
        changed = plan()
        del changed["source_files"]["experiments/block_depth/native_routes.py"]
        with self.assertRaisesRegex(ValueError, "dependencies"):
            P.validate_plan(changed)


class ResourceTests(unittest.TestCase):
    def test_peak_and_reserve_are_both_accounted(self):
        self.assertEqual(P.capacity_plan()["peak_required_bytes"], 14 * P.GIB)
        result = P.check_capacity(free_bytes=14 * P.GIB, host_available_bytes=32 * P.GIB)
        self.assertEqual(result["status"], "passed")
        with self.assertRaises(RuntimeError):
            P.check_capacity(free_bytes=14 * P.GIB - 1, host_available_bytes=32 * P.GIB)
        with self.assertRaises(RuntimeError):
            P.check_capacity(free_bytes=30 * P.GIB, host_available_bytes=32 * P.GIB - 1)

    def test_postflight_keeps_reserve_without_recharging_consumed_writes(self):
        result = P.check_capacity(free_bytes=8 * P.GIB, host_available_bytes=32 * P.GIB, initial=False)
        self.assertEqual(result["required_bytes"], 8 * P.GIB)

    def test_nonfinite_fractional_or_negative_capacity_is_rejected(self):
        for value in (float("nan"), float("inf"), 1.2, -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                P.check_capacity(free_bytes=value, host_available_bytes=32 * P.GIB)


class ExportTests(unittest.TestCase):
    def test_retained_export_receipt_is_bound_to_event58(self):
        with tempfile.TemporaryDirectory() as directory:
            export = Path(directory)
            (export / "config.json").write_text(json.dumps(dict(num_hidden_layers=5, block_size=16)))
            with patch.object(P, "EXPORT", export), patch.object(P, "verify_files") as check:
                receipt = boundary(export)
                result = P.verify_export(export, receipt)
                self.assertFalse(result["optimizer_state_loaded"])
                check.assert_called_once_with(export, P.EXPORT_FILES, exact_files=True)
                for field, value in (("event", 57), ("arm", "K"), ("weights_identity", dict(bytes=1, sha256="bad"))):
                    wrong = deepcopy(receipt)
                    wrong[field] = value
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        P.verify_export(export, wrong)

    def test_changed_file_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("original")
            expected = P.identity(path)
            path.write_text("changed!")
            with self.assertRaisesRegex(ValueError, "changed"):
                P.verify_files(directory, {"config.json": expected})

    def test_three_layer_export_cannot_replace_frozen_five(self):
        with tempfile.TemporaryDirectory() as directory:
            export = Path(directory)
            (export / "config.json").write_text(json.dumps(dict(num_hidden_layers=3, block_size=16)))
            with patch.object(P, "EXPORT", export), patch.object(P, "verify_files"):
                with self.assertRaisesRegex(ValueError, "five-layer"):
                    P.verify_export(export, boundary(export))


class ManifestTests(unittest.TestCase):
    def test_sealed_panels_are_inputs_only_for_this_milestone(self):
        result = P.verify_manifests(manifests())
        self.assertEqual(result["selected_requests"], 896)
        self.assertFalse(result["final_outputs_read"])
        self.assertEqual(result["final_generations"], 0)

    def test_overlap_by_text_or_source_is_rejected(self):
        for field in ("source_id", "question", "prompt_ids"):
            payload = manifests()
            payload["manifests"]["FINAL"]["requests"][0][field] = payload["manifests"]["READOUT"]["requests"][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlap"):
                P.verify_manifests(payload)

    def test_cannot_shrink_panel_or_change_output_cap(self):
        payload = manifests()
        payload["manifests"]["READOUT"]["requests"].pop()
        with self.assertRaisesRegex(ValueError, "count"):
            P.verify_manifests(payload)
        payload = manifests()
        payload["manifests"]["READOUT"]["requests"][0]["max_new_tokens"] = 1024
        with self.assertRaisesRegex(ValueError, "context/termination"):
            P.verify_manifests(payload)


class FailureReceiptTests(unittest.TestCase):
    def test_failed_preflight_persists_error_and_zero_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "PREFLIGHT.json"
            with patch.object(P, "owned_path", side_effect=Path), patch.object(P, "validate_plan", side_effect=ValueError("wrong export")):
                with self.assertRaisesRegex(ValueError, "wrong export"):
                    P.preflight(dict(code_commit="a" * 40), output)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("wrong export", result["error"])
            self.assertEqual(result["model_work"], 0)
            self.assertEqual(result["optimizer_steps"], 0)
            with patch.object(P, "owned_path", side_effect=Path):
                with self.assertRaises(FileExistsError):
                    P.preflight(dict(code_commit="a" * 40), output)


if __name__ == "__main__":
    unittest.main()
