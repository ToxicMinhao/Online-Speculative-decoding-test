"""Configuration and current CLI boundaries; no model weights or GPU execution."""

from contextlib import chdir, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import tempfile
import tomllib
import unittest

from graduate.__main__ import main
from graduate.batches import FeatureLayout, read_feature_layout
from graduate.config import Configuration, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fenrir.toml"


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.values = tomllib.loads(CONFIG.read_text())

    def test_qualified_recipe_and_derived_counts_are_preserved(self):
        config = load_config(CONFIG)
        expected = {
            "output_dir": config.training.output_dir, "seed": 17, "responses_per_event": 128, "passes_per_event": 2,
            "microbatch_size": 1, "accumulation_steps": 2, "context_limit": 4096,
            "max_anchors_per_response": 512, "projection_chunk_blocks": 8,
            "learning_rate": 3e-4, "weight_decay": 0.01, "max_grad_norm": 1.0,
            "loss_decay_gamma": 7.0, "shuffle_responses": True,
            "parameter_dtype": "bfloat16", "probability_dtype": "bfloat16",
            "optimizer": "bf16_adamw", "learning_rate_schedule": "constant",
            "attention_backend": "flex_attention",
        }
        self.assertEqual(config.training.model_dump(), expected)
        self.assertEqual(config.objective.model_dump(), {
            "first_error_redistribution_fraction": 0.5,
            "prefix_probability_smoothing": 0.5,
            "prefix_relative_cap": 0.25, "denominator_epsilon": 1e-6,
        })
        self.assertEqual(config.training.presentations_per_event, 256)

    def test_misspelled_and_missing_options_do_not_get_defaults(self):
        for section in self.values:
            with self.subTest(section=section):
                values = deepcopy(self.values)
                values[section]["misspelled_option"] = 1
                with self.assertRaises(ValueError):
                    Configuration.model_validate(values)
        del self.values["training"]["learning_rate"]
        with self.assertRaises(ValueError):
            Configuration.model_validate(self.values)

    def test_nonfinite_values_and_wrong_types_are_rejected(self):
        cases = [
            ("training", "learning_rate", float("nan")),
            ("training", "learning_rate", float("inf")),
            ("training", "weight_decay", float("-inf")),
            ("training", "seed", "17"),
            ("training", "responses_per_event", True),
            ("training", "learning_rate", "0.0003"),
            ("training", "microbatch_size", True),
            ("training", "microbatch_size", 1.0),
            ("training", "shuffle_responses", 1),
            ("serving", "concurrency", True),
            ("serving", "greedy", 1),
            ("serving", "runtime", "transformers"),
        ]
        for section, name, value in cases:
            with self.subTest(section=section, name=name, value=value):
                values = deepcopy(self.values)
                values[section][name] = value
                with self.assertRaises(ValueError):
                    Configuration.model_validate(values)

    def test_inconsistent_context_boundaries_are_rejected(self):
        cases = [
            ("training", "context_limit", 2048),
            ("serving", "max_new_tokens", 4096),
            ("serving", "block_size", 8192),
        ]
        for section, name, value in cases:
            with self.subTest(section=section, name=name):
                values = deepcopy(self.values)
                values[section][name] = value
                with self.assertRaises(ValueError):
                    Configuration.model_validate(values)

    def test_accumulation_can_cross_passes_but_must_complete_the_event(self):
        self.values["training"]["responses_per_event"] = 3
        config = Configuration.model_validate(self.values)
        self.assertEqual(config.training.presentations_per_event, 6)
        self.values["training"]["passes_per_event"] = 1
        with self.assertRaises(ValueError):
            Configuration.model_validate(self.values)


class EntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="graduate-cli-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.target = self.directory / "target"
        self.draft = self.directory / "draft"
        for directory in (self.target, self.draft):
            directory.mkdir()
        self.target_metadata = {"hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 4}
        self.draft_metadata = dict(self.target_metadata, num_hidden_layers=1, block_size=16,
                                   dflash_config={"target_layer_ids": [0, 2]})
        (self.target / "config.json").write_text(json.dumps(self.target_metadata))
        (self.draft / "config.json").write_text(json.dumps(self.draft_metadata))
        text = CONFIG.read_text()
        original = tomllib.loads(text)
        for section, field, value in (
            ("models", "target", "target"), ("models", "draft", "draft"),
            ("feedback", "manifest", "feedback-data/manifest.json"),
            ("feedback", "buffer_dir", "buffer"), ("training", "output_dir", "training-output"),
        ):
            text = text.replace(json.dumps(original[section][field]), json.dumps(value))
        self.config_path = self.directory / "fixture.toml"
        self.config_path.write_text(text)

    def test_relative_configuration_paths_do_not_depend_on_working_directory(self):
        with chdir(self.directory.parent):
            config = load_config(self.config_path)
        self.assertEqual(config.models.target, self.target)
        self.assertEqual(config.models.draft, self.draft)
        self.assertEqual(config.feedback.manifest, self.directory / "feedback-data/manifest.json")
        self.assertEqual(config.feedback.buffer_dir, self.directory / "buffer")
        self.assertEqual(config.training.output_dir, self.directory / "training-output")
        self.assertEqual(read_feature_layout(config), FeatureLayout(4096, 32, 16, (0, 2)))

    def test_incompatible_model_pair_is_rejected(self):
        cases = [
            ({"hidden_size": 8}, "hidden_size"), ({"vocab_size": 64}, "vocab_size"),
            ({"block_size": 8}, "block_size"),
            ({"dflash_config": {"target_layer_ids": []}}, "target-layer"),
            ({"dflash_config": {"target_layer_ids": [-1, 2]}}, "target-layer"),
            ({"dflash_config": {"target_layer_ids": [0, 4]}}, "target-layer"),
        ]
        config = load_config(self.config_path)
        for changes, message in cases:
            with self.subTest(changes=changes):
                metadata = dict(self.draft_metadata, **changes)
                (self.draft / "config.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, message):
                    read_feature_layout(config)

    def test_target_directory_cannot_be_used_as_the_draft(self):
        text = self.config_path.read_text().replace('draft = "draft"', 'draft = "target"')
        self.config_path.write_text(text)
        with self.assertRaises(KeyError):
            read_feature_layout(load_config(self.config_path))

    def test_missing_model_metadata_preserves_the_original_error(self):
        (self.draft / "config.json").unlink()
        with self.assertRaises(FileNotFoundError) as error:
            read_feature_layout(load_config(self.config_path))
        self.assertEqual(Path(error.exception.filename), self.draft / "config.json")

    def test_missing_feedback_settings_fail_before_any_output_or_buffer_write(self):
        text = re.sub(r"(?ms)^\[feedback\]\n.*?(?=^\[|\Z)", "", self.config_path.read_text())
        self.config_path.write_text(text)
        before = {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "feedback"):
            main(["train", "--config", str(self.config_path)])
        self.assertEqual(
            {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}, before,
        )
        self.assertFalse((self.directory / "buffer").exists())
        self.assertFalse((self.directory / "training-output").exists())

    def test_previous_demonstration_commands_are_not_exposed(self):
        for command in ("inspect", "feedback"):
            with self.subTest(command=command), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([command, "--config", str(self.config_path)])
                self.assertEqual(error.exception.code, 2)

    def test_single_f5_workflow_uses_the_live_collection_cli_and_interpreter(self):
        launch = json.loads((ROOT / ".vscode" / "launch.json").read_text())
        self.assertEqual(len(launch["configurations"]), 1)
        entry = launch["configurations"][0]
        self.assertEqual(entry["name"], "Graduate: collect live feedback (two requests)")
        self.assertEqual(entry["python"], "/home/minhao2/.venvs/dflash-specforge/bin/python")
        self.assertEqual(entry["module"], "graduate")
        self.assertEqual(entry["args"], [
            "collect", "--config", "${workspaceFolder}/configs/fenrir.toml", "--debug-responses", "2",
        ])
        self.assertEqual(entry["cwd"], "${workspaceFolder}")
        self.assertEqual(entry["env"]["PYTHONPATH"], "${workspaceFolder}/src")
        self.assertEqual((entry["type"], entry["request"]), ("debugpy", "launch"))
        self.assertIs(entry["justMyCode"], True)
        self.assertIs(entry["subProcess"], False)
        self.assertFalse(entry.get("stopOnEntry", False))
        self.assertNotIn("program", entry)
        self.assertNotIn("inputs", launch)


if __name__ == "__main__":
    unittest.main()
