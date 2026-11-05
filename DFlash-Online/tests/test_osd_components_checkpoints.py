"""CPU-only exact-substitution tests using a tiny five-layer BF16 fixture."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file

from experiments.osd_components import checkpoints as C


class StorageBudgetTests(unittest.TestCase):
    def test_decimal_minimum_and_binary_reserve(self):
        budget = C.storage_budget(2_100_000_000, 18_000_000_000)
        self.assertEqual(budget["reserve_bytes"], 8 * 2**30)
        self.assertEqual(budget["required_bytes"], 18_000_000_000)
        self.assertTrue(budget["admitted"])
        self.assertFalse(C.storage_budget(2_100_000_000, 17_999_999_999)["admitted"])

    def test_large_export_peak_budget_is_not_replaced_by_minimum(self):
        required = 3 * 4_000_000_000 + 2_000_000_000 + 8 * 2**30
        self.assertEqual(C.storage_budget(4_000_000_000, required)["required_bytes"], required)
        self.assertTrue(C.storage_budget(4_000_000_000, required)["admitted"])
        self.assertFalse(C.storage_budget(4_000_000_000, required - 1)["admitted"])

    def test_invalid_size_inputs(self):
        for export, free in ((0, 10), (-1, 10), (1, -1), (True, 10), (1.5, 10), (1, False)):
            with self.subTest(export=export, free=free), self.assertRaises(ValueError):
                C.storage_budget(export, free)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.frozen, self.trained, self.output = (
            self.root / "frozen", self.root / "trained", self.root / "hybrids")
        self.config = {
            "model_type": "qwen3", "vocab_size": 16, "hidden_size": 2,
            "intermediate_size": 4, "num_hidden_layers": 5,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 2,
            "rope_theta": 1000000, "rope_scaling": None, "torch_dtype": "bfloat16",
        }
        self.f_weights = {}
        for index, key in enumerate(sorted(C.EXPECTED_KEYS)):
            if key == "fc.weight":
                shape = (2, 6)
            elif "gate_proj" in key or "up_proj" in key:
                shape = (4, 2)
            elif "down_proj" in key:
                shape = (2, 4)
            elif "_proj" in key:
                shape = (2, 2)
            else:
                shape = (2,)
            self.f_weights[key] = torch.full(shape, index / 8, dtype=torch.bfloat16)
        self.o_weights = {key: value.clone() for key, value in self.f_weights.items()}
        # Deliberately leave some tensors unchanged and change one element of others.
        for index, key in enumerate(sorted(C.EXPECTED_KEYS)):
            if index % 3:
                self.o_weights[key].view(-1)[0] += 1
        self.write_checkpoint(self.frozen, self.f_weights)
        self.write_checkpoint(self.trained, self.o_weights)
        disk = patch.object(C.shutil, "disk_usage", return_value=Mock(free=50_000_000_000))
        disk.start()
        self.addCleanup(disk.stop)

    def write_checkpoint(self, path, weights, config=None):
        path.mkdir(exist_ok=True)
        (path / "config.json").write_text(json.dumps(self.config if config is None else config))
        save_file(weights, path / "model.safetensors")

    def build(self):
        return C.build_hybrids(self.frozen, self.trained, self.output)

    def source_snapshot(self):
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for folder in (self.frozen, self.trained) for path in folder.iterdir()}

    def test_all_58_keys_have_exact_origins_and_originals_remain_untouched(self):
        before = self.source_snapshot()
        result = self.build()
        self.assertEqual(before, self.source_snapshot())
        self.assertEqual(result["arms"]["F"], str(self.frozen))
        self.assertEqual(result["arms"]["O"], str(self.trained))
        for arm in ("C-only", "B-only"):
            path = Path(result["arms"][arm])
            actual = load_file(path / "model.safetensors")
            self.assertEqual(set(actual), C.EXPECTED_KEYS)
            self.assertEqual((path / "config.json").read_bytes(),
                             (self.frozen / "config.json").read_bytes())
            checks = result["qualification"]["hybrids"][arm]["per_key"]
            for key in C.EXPECTED_KEYS:
                learned = (key in C.CONDITIONING_KEYS) == (arm == "C-only")
                expected = (self.o_weights if learned else self.f_weights)[key]
                self.assertTrue(torch.equal(actual[key].view(torch.int16), expected.view(torch.int16)))
                self.assertEqual(checks[key]["origin"], "O" if learned else "F")
                self.assertTrue(checks[key]["exact_bytes_equal"])
        qualified = result["qualification"]
        self.assertEqual(qualified["status"], "passed")
        self.assertEqual(qualified["original_sources_unchanged"], {"F": True, "O": True})
        for arm in ("F", "O"):
            self.assertEqual(len(qualified["endpoint_reconstruction_exact"][arm]), 58)
            self.assertTrue(all(qualified["endpoint_reconstruction_exact"][arm].values()))
        inventory = result["inventory"]
        self.assertEqual(inventory["C"]["tensor_count"], 2)
        self.assertEqual(inventory["B"]["tensor_count"], 56)
        self.assertEqual(inventory["all"]["numel"], sum(t.numel() for t in self.f_weights.values()))
        self.assertEqual(inventory["all"]["bytes"], 2 * inventory["all"]["numel"])
        self.assertEqual(inventory["all"]["changed_elements"], 38)
        for name, key in zip(C.RECEIPTS, ("storage_admission", "inventory", "qualification")):
            self.assertEqual(json.loads((self.output / name).read_text()), result[key])

    def test_sharded_source_exports_are_supported(self):
        (self.trained / "model.safetensors").unlink()
        keys = sorted(self.o_weights)
        for index, subset in enumerate((keys[:20], keys[20:])):
            save_file({key: self.o_weights[key] for key in subset},
                      self.trained / f"model-{index}.safetensors")
        self.assertEqual(self.build()["qualification"]["status"], "passed")

    def test_changed_bit_count_distinguishes_signed_zero(self):
        self.f_weights["hidden_norm.weight"][0] = 0.0
        self.o_weights["hidden_norm.weight"][0] = -0.0
        self.write_checkpoint(self.frozen, self.f_weights)
        self.write_checkpoint(self.trained, self.o_weights)
        result = self.build()
        row = next(row for row in result["inventory"]["tensors"] if row["key"] == "hidden_norm.weight")
        self.assertEqual(row["changed_elements"], 1)
        saved = load_file(Path(result["arms"]["C-only"]) / "model.safetensors")
        self.assertTrue(torch.signbit(saved["hidden_norm.weight"][0]))

    def test_missing_and_target_owned_extra_keys_are_refused(self):
        variants = [dict(self.o_weights) for _ in range(4)]
        variants[0].pop("layers.4.mlp.down_proj.weight")
        variants[1]["embed_tokens.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
        variants[2]["lm_head.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
        variants[3]["layers.0.unexpected.weight"] = torch.ones(2, dtype=torch.bfloat16)
        for weights in variants:
            with self.subTest(keys=set(weights) - C.EXPECTED_KEYS):
                self.write_checkpoint(self.trained, weights)
                with self.assertRaisesRegex(ValueError, "fixed 58 draft tensors"):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_duplicate_shard_key_is_refused(self):
        save_file({"fc.weight": self.o_weights["fc.weight"]}, self.trained / "duplicate.safetensors")
        with self.assertRaisesRegex(ValueError, "Duplicate checkpoint tensor"):
            self.build()

    def test_shape_or_dtype_mismatch_is_refused(self):
        for changed in (torch.ones(3, 6, dtype=torch.bfloat16), torch.ones(2, 6)):
            with self.subTest(dtype=changed.dtype, shape=changed.shape):
                self.write_checkpoint(self.trained, {**self.o_weights, "fc.weight": changed})
                with self.assertRaisesRegex(ValueError, "keys, shapes or dtypes differ"):
                    self.build()

    def test_matching_non_bf16_endpoints_are_still_refused(self):
        for path, weights in ((self.frozen, self.f_weights), (self.trained, self.o_weights)):
            self.write_checkpoint(path, {key: value.float() for key, value in weights.items()})
        with self.assertRaisesRegex(ValueError, "must have BF16"):
            self.build()

    def test_nonfinite_in_either_endpoint_is_refused(self):
        for path, weights in ((self.frozen, self.f_weights), (self.trained, self.o_weights)):
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(endpoint=path.name, nonfinite=nonfinite):
                    changed = {**weights, "norm.weight": torch.full((2,), nonfinite, dtype=torch.bfloat16)}
                    self.write_checkpoint(path, changed)
                    with self.assertRaisesRegex(ValueError, "Nonfinite checkpoint tensor"):
                        self.build()
            self.write_checkpoint(path, weights)

    def test_semantic_config_change_is_refused(self):
        self.write_checkpoint(self.trained, self.o_weights, {**self.config, "rope_theta": 2000000})
        with self.assertRaisesRegex(ValueError, "semantic draft configuration"):
            self.build()

    def test_native_semantic_config_accepts_redundant_export_metadata(self):
        exported = {
            **self.config, "_name_or_path": "a-different-local-export-name",
            "architectures": ["DFlashDraftModel"], "transformers_version": "5.12.1",
            "draft_vocab_size": self.config["vocab_size"],
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        }
        self.write_checkpoint(self.trained, self.o_weights, exported)
        self.assertTrue(self.build()["qualification"]["semantic_config_equal"])

    def test_equal_configs_with_wrong_layer_count_are_refused(self):
        for path, weights in ((self.frozen, self.f_weights), (self.trained, self.o_weights)):
            self.write_checkpoint(path, weights, {**self.config, "num_hidden_layers": 4})
        with self.assertRaisesRegex(ValueError, "five draft layers"):
            self.build()

    def test_output_inside_original_is_refused(self):
        before = self.source_snapshot()
        with self.assertRaisesRegex(ValueError, "inside either source"):
            C.build_hybrids(self.frozen, self.trained, self.trained / "hybrids")
        self.assertEqual(before, self.source_snapshot())

    def test_existing_work_is_never_replaced(self):
        self.output.mkdir()
        (self.output / "C-only").mkdir()
        keep = self.output / "C-only" / "keep.txt"
        keep.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(keep.read_text(), "preserve")

    def test_storage_denial_precedes_materialization(self):
        with patch.object(C.shutil, "disk_usage", return_value=Mock(free=17_999_999_999)), \
                patch("safetensors.torch.save_file") as save:
            with self.assertRaisesRegex(ValueError, "Insufficient storage"):
                self.build()
        save.assert_not_called()
        receipt = json.loads((self.output / "STORAGE_ADMISSION.json").read_text())
        self.assertFalse(receipt["admitted"])
        self.assertFalse((self.output / "C-only").exists())

    def test_reload_detects_wrong_origin_and_does_not_publish_corruption(self):
        def corrupt(state, path, metadata=None):
            wrong = dict(state)
            wrong["hidden_norm.weight"] = state["hidden_norm.weight"].clone() + 1
            save_file(wrong, path, metadata=metadata)

        with patch("safetensors.torch.save_file", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "wrong exact origin"):
                self.build()
        self.assertFalse((self.output / "C-only").exists())
        self.assertFalse(list(self.output.glob(".*.pending-*")))


if __name__ == "__main__":
    unittest.main()
