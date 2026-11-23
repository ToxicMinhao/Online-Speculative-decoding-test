"""CPU orchestration with real capture packets; no native/GPU qualification."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from graduate.collection import FeedbackCapture
from graduate.config import Configuration
from graduate.feedback import read_response
from graduate.live import collect_feedback


ROOT = Path(__file__).resolve().parents[1]


class CPUGenerator:
    """Emit prescribed tokens and exercise the real canonical-row collector."""

    def __init__(self, responses, hidden_size=2):
        self.responses = iter(responses)
        self.hidden_size = hidden_size
        self.calls = []
        self.stop_token_ids = [2]
        self.peak_cuda_bytes = 0
        self.runtime = {"verification": "CPU orchestration fixture; no native model execution"}

    def features(self, positions):
        auxiliary = torch.tensor([[position] * (2 * self.hidden_size) for position in positions],
                                 dtype=torch.bfloat16)
        final = torch.tensor([[100 + position] * self.hidden_size for position in positions],
                             dtype=torch.bfloat16)
        return auxiliary, final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def generate(self, prompt, cap, stop_ids, block_size, *, capture):
        self.calls.append({"prompt": prompt, "cap": cap, "stop_ids": stop_ids,
                           "block_size": block_size, "capture": capture})
        response = next(self.responses)
        collected = None
        trace = [{"accepted_draft_tokens": 0} for _ in response[1:]]
        if capture:
            collected = FeedbackCapture(prompt, [1, 3], "sglang_live_capture")
            collected.prefill(response[0], *self.features(range(len(prompt))))
            for offset, successor in enumerate(response[1:]):
                position = len(prompt) + offset
                collected.verify([response[offset], 77], [successor, 88], 1,
                                 *self.features([position, 99]))
            collected.final_token(response[-1], *self.features([len(prompt) + len(response) - 1]))
        return SimpleNamespace(response_ids=response, finish_reason="eos", trace=trace,
                               capture=collected)


class LiveCollectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="graduate-live-cpu-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        target, draft = self.root / "target", self.root / "draft"
        target.mkdir()
        draft.mkdir()
        (target / "config.json").write_text(json.dumps({
            "hidden_size": 2, "vocab_size": 128, "num_hidden_layers": 4,
        }))
        (draft / "config.json").write_text(json.dumps({
            "hidden_size": 2, "vocab_size": 128, "block_size": 3,
            "dflash_config": {"target_layer_ids": [1, 3]},
        }))
        with (ROOT / "configs/fenrir.toml").open("rb") as stream:
            values = tomllib.load(stream)
        values["models"] = {"target": target, "draft": draft}
        values["feedback"] = {"manifest": self.root / "manifest.json", "buffer_dir": self.root / "buffer"}
        values["collection"] = {"requests": self.root / "requests.json", "output_dir": self.root / "runs"}
        values["training"].update(context_limit=16, output_dir=self.root / "training")
        values["serving"].update(context_limit=16, max_new_tokens=8, block_size=3)
        self.config = Configuration.model_validate(values)
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.return_value = [11, 12]
        self.tokenizer_loader = self.enterContext(patch(
            "graduate.live.AutoTokenizer.from_pretrained", return_value=self.tokenizer))
        self.generator_loader = self.enterContext(patch("graduate.live.load_generator"))

    def run_requests(self, requests, responses, *, hidden_size=2):
        self.config.collection.requests.write_text(json.dumps(requests))
        generator = CPUGenerator(responses, hidden_size)
        self.generator_loader.return_value = generator
        with redirect_stdout(io.StringIO()):
            result = collect_feedback(self.config)
        return result, generator

    def test_feedback_is_serialized_and_evaluation_never_enters_feedback(self):
        result, generator = self.run_requests([
            {"record_id": "feedback-1", "prompt": "Feedback example", "role": "feedback"},
            {"record_id": "eval-1", "prompt": "Separate evaluation", "role": "evaluation"},
        ], [[21, 22, 2], [31, 2]])
        run_dir = Path(result["output_directory"])
        manifest = json.loads(self.config.feedback.manifest.read_text())
        self.assertEqual([call["capture"] for call in generator.calls], [True, False])
        self.assertEqual(len(manifest), 1)
        response = read_response(Path(manifest[0]["response"]))
        packet = torch.load(manifest[0]["features"], map_location="cpu", weights_only=True)
        self.assertEqual(response.prompt_ids, (11, 12))
        self.assertEqual(response.response_ids, (21, 22, 2))
        self.assertEqual(packet["input_ids"].tolist(), [[11, 12, 21, 22, 2]])
        self.assertEqual(packet["loss_mask"].tolist(), [[0, 0, 1, 1, 1]])
        self.assertEqual(packet["hidden_states"][0, :, 0].tolist(), [0, 1, 2, 3, 4])
        evaluation = run_dir / "0001"
        self.assertEqual({path.name for path in evaluation.iterdir()}, {"generation.json"})
        self.assertEqual(json.loads((evaluation / "generation.json").read_text())["purpose"], "evaluation")
        self.assertEqual(result["requests"][1]["status"], "evaluation_only")
        self.assertFalse(result["requests"][1]["training_eligible"])
        self.assertEqual(result["requests"][1]["terminal_feature_forwards"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(self.config.feedback.buffer_dir.exists())
        self.assertFalse(self.config.training.output_dir.exists())
        self.assertEqual(self.tokenizer.apply_chat_template.call_count, 2)
        for call in self.tokenizer.apply_chat_template.call_args_list:
            self.assertIs(call.kwargs["return_dict"], False)

    def test_first_token_eos_saves_evidence_and_is_excluded_without_replacement(self):
        result, generator = self.run_requests([
            {"record_id": "short", "prompt": "A short feedback example"},
        ], [[2]])
        request_dir = Path(result["output_directory"]) / "0000"
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(result["training_responses"], 0)
        self.assertEqual(result["requests"][0]["status"], "no_training_anchors")
        self.assertEqual(json.loads(self.config.feedback.manifest.read_text()), [])
        response = read_response(request_dir / "response.json")
        self.assertEqual(response.response_ids, (2,))
        packet = torch.load(request_dir / "features.pt", map_location="cpu", weights_only=True)
        self.assertEqual(packet["input_ids"].tolist(), [[11, 12, 2]])
        self.assertEqual(packet["hidden_states"].shape, (1, 3, 4))
        self.assertEqual(packet["label_binding"]["target_argmax_ids"], [2])

    def test_evaluation_only_collection_does_not_modify_training_manifest(self):
        self.config.feedback.manifest.parent.mkdir(parents=True, exist_ok=True)
        original = '[{"response":"retained.json","features":"retained.pt"}]\n'
        self.config.feedback.manifest.write_text(original)
        result, generator = self.run_requests([
            {"record_id": "eval-only", "prompt": "Evaluate separately", "role": "evaluation"},
        ], [[31, 2]])
        self.assertEqual(self.config.feedback.manifest.read_text(), original)
        self.assertEqual(result["training_responses"], 0)
        self.assertFalse(generator.calls[0]["capture"])

    def test_context_failure_precedes_model_loading_and_artifact_creation(self):
        self.config.collection.requests.write_text(json.dumps([
            {"record_id": "oversized", "prompt": "A long input"},
        ]))
        self.tokenizer.apply_chat_template.return_value = list(range(10))
        with self.assertRaisesRegex(ValueError, "Prompt plus output allowance"):
            collect_feedback(self.config)
        self.generator_loader.assert_not_called()
        self.assertFalse(self.config.collection.output_dir.exists())
        self.assertFalse(self.config.feedback.manifest.exists())

    def test_raw_generation_is_retained_before_real_packet_validation_failure(self):
        with self.assertRaisesRegex(ValueError, "must be detached CPU BF16 features"):
            self.run_requests([
                {"record_id": "wrong-width", "prompt": "Retain a failed capture"},
            ], [[21, 2]], hidden_size=3)
        generation_files = list(self.config.collection.output_dir.glob("*/0000/generation.json"))
        self.assertEqual(len(generation_files), 1)
        saved = json.loads(generation_files[0].read_text())
        self.assertEqual(saved["response_ids"], [21, 2])
        self.assertEqual(saved["runtime"]["verification"], "CPU orchestration fixture; no native model execution")
        self.assertTrue(generation_files[0].with_name("response.json").exists())
        self.assertTrue(generation_files[0].with_name("features.pt").exists())
        self.assertEqual(json.loads(self.config.feedback.manifest.read_text()), [])

    def test_new_collection_manifest_contains_only_its_own_panel(self):
        first, _ = self.run_requests([
            {"record_id": "first", "prompt": "First panel"},
        ], [[21, 2]])
        first_manifest = json.loads(self.config.feedback.manifest.read_text())
        second, _ = self.run_requests([
            {"record_id": "second", "prompt": "Second panel"},
        ], [[31, 2]])
        second_manifest = json.loads(self.config.feedback.manifest.read_text())
        self.assertNotEqual(first["output_directory"], second["output_directory"])
        self.assertEqual(len(second_manifest), 1)
        self.assertEqual(read_response(Path(second_manifest[0]["response"])).metadata["request_id"], "second")
        self.assertEqual(json.loads((Path(first["output_directory"]) / "manifest.json").read_text()), first_manifest)
        self.assertTrue(Path(first_manifest[0]["features"]).exists())


if __name__ == "__main__":
    unittest.main()
