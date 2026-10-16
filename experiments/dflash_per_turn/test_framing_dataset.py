"""CPU-only protocol tests; optional actual package/tokenizer qualification.

Set PER_TURN_DATASET and PER_TURN_TOKENIZER to run the two local artifact tests.
No prompts, model weights, GPU, SSH or inference are used by this test module.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.dflash_per_turn import dataset, framing


class CharacterTokenizer:
    """Token-boundary fixture that retains every whitespace character."""
    bos_token = framing.BOS
    specials = {framing.BOS: 128000, framing.EOT: 128009,
                "<|start_header_id|>": 128006, "<|end_header_id|>": 128007}

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            for special, token in self.specials.items():
                if text.startswith(special):
                    result.append(token)
                    text = text[len(special):]
                    break
            else:
                result.append(ord(text[0]) + 1000)
                text = text[1:]
        return result

    def decode(self, ids, **kwargs):
        inverse = {v: k for k, v in self.specials.items()}
        return "".join(inverse.get(x, chr(x - 1000) if 1000 <= x < 128000 else "?") for x in ids)


class FramingTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharacterTokenizer()
        self.validation = patch.object(framing, "validate_tokenizer")
        self.validation.start()
        self.addCleanup(self.validation.stop)

    def test_exact_whitespace_and_generated_ids(self):
        text = "\n  def f():\n\treturn 2\n\n  "
        output = [555, 444, 128009]  # Must be copied, even without reversible text.
        messages = [{"role": "user", "content": text},
                    {"role": "assistant", "output_ids": output, "content": "DO NOT RETOKENIZE"},
                    {"role": "user", "content": " \nnext\t "}]
        ids = framing.encode_history(self.tok, messages)
        prefix = self.tok.encode(framing.BOS + framing.SYSTEM_HEADER + framing.COMMON_SYSTEM_TEXT + framing.EOT)
        self.assertEqual(ids, prefix + framing.user_segment_ids(self.tok, text) + output +
                         framing.user_segment_ids(self.tok, " \nnext\t "))
        self.assertEqual(messages[0]["content"], text)

    def test_append_eot_only_when_missing(self):
        for output in ([700, 128009], [700]):
            ids = framing.encode_history(self.tok, [{"role": "user", "content": "x"},
                {"role": "assistant", "output_ids": list(output)}, {"role": "user", "content": "y"}])
            self.assertEqual(ids.count(128009), 4)  # system, user1, answer, user2

    def test_source_assistant_text_without_generated_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "Original generated"):
            framing.encode_history(self.tok, [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "source answer"}, {"role": "user", "content": "y"}])

    def test_source_system_is_not_admitted(self):
        with self.assertRaises(ValueError):
            framing.encode_history(self.tok, [{"role": "system", "content": "persona"},
                                             {"role": "user", "content": "x"}])

    def test_replay_uses_identical_ids_not_equivalent_text(self):
        framing.require_identical_input([1, 2], [1, 2])
        with self.assertRaisesRegex(ValueError, "differ"):
            framing.require_identical_input([1, 2], [1, 3])

    def test_no_truncation_context_boundary(self):
        framing.check_input([1] * 31735)
        with self.assertRaisesRegex(ValueError, "exceed"):
            framing.check_input([1] * 31736)
        with self.assertRaises(ValueError):
            framing.check_input([1] * 31745)

    def test_output_cap_does_not_become_truncation(self):
        with self.assertRaises(ValueError):
            framing.encode_history(self.tok, [{"role": "user", "content": "x"},
                {"role": "assistant", "output_ids": [1] * 1025}, {"role": "user", "content": "y"}])

    def test_budget_adds_full_unknown_answers_and_proposal_allowance(self):
        rows = framing.conservative_budget(self.tok, ["x", "y"])
        self.assertEqual(rows[1]["input_tokens_upper_bound"] - rows[0]["input_tokens_upper_bound"],
                         len(framing.user_segment_ids(self.tok, "y")) + 1025)
        self.assertEqual(rows[1]["including_current_answer_and_speculation_upper_bound"] -
                         rows[1]["input_tokens_upper_bound"], 1033)


class DatasetGuardTests(unittest.TestCase):
    def test_counts_and_plan(self):
        self.assertEqual(list(dataset.EXPECTED_STREAM_COUNTS.values()), [20, 16, 17, 23, 18, 22])
        self.assertEqual(sum(dataset.EXPECTED_STREAM_COUNTS.values()), dataset.TOTAL_REQUESTS)
        self.assertEqual(dataset.TOTAL_REQUESTS - len(dataset.STREAM_IDS), dataset.UPDATE_EVENTS)
        self.assertEqual(2 * dataset.UPDATE_EVENTS, dataset.OPTIMIZER_STEPS)

    def test_group_consecutive_users_without_stripping(self):
        source = [{"role": "system", "content": "not served"},
            {"role": "user", "content": "  first\n"},
            {"role": "user", "content": "\tsecond "},
            {"role": "assistant", "content": "not served"},
            {"role": "user", "content": "third"}]
        groups = dataset._groups(source)
        self.assertEqual([[i for i, _ in g] for g in groups], [[1, 2], [4]])
        self.assertEqual("\n\n".join(m["content"] for _, m in groups[0]), "  first\n\n\n\tsecond ")

    def test_manifest_edits_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text("{}")
            (path / "PACKAGE_SHA256.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                dataset.load_streams(path)

    def test_artifact_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "escapes"):
                dataset._path(Path(directory), "../anything")


@unittest.skipUnless(os.getenv("PER_TURN_DATASET") and os.getenv("PER_TURN_TOKENIZER"),
                     "Set local approved dataset and pinned tokenizer paths for artifact qualification")
class ActualArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.streams = dataset.load_streams(os.environ["PER_TURN_DATASET"])
        cls.tokenizer = framing.load_pinned_tokenizer(os.environ["PER_TURN_TOKENIZER"])

    def test_all_116_source_requests_and_budgets(self):
        checked = whitespace = 0
        for stream in self.streams:
            bound = json.loads((Path(os.environ["PER_TURN_DATASET"]) / stream["token_budget_file"]).read_text())
            computed = framing.conservative_budget(self.tokenizer, [r["prompt"] for r in stream["requests"]])
            for request, actual, expected in zip(stream["requests"], computed, bound["per_request"]):
                self.assertEqual(actual["input_tokens_upper_bound"], expected["input_tokens_upper_bound"])
                self.assertEqual(actual["including_current_answer_and_speculation_upper_bound"],
                                 expected["including_current_answer_and_speculation_upper_bound"])
                ids = framing.encode_history(self.tokenizer, [{"role": "user", "content": request["prompt"]}])
                native = list(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": request["prompt"]}], tokenize=True,
                    add_generation_prompt=True, return_dict=False))
                changed = ids != native
                self.assertEqual(changed, not expected["native_encode_history_preserves_exact_content"])
                whitespace += changed
                checked += 1
        self.assertEqual((checked, whitespace), (116, 22))

    def test_saved_replay_identity_across_complete_fixture_history(self):
        # Synthetic token-only fixture answers are used solely to check framing;
        # these are not model results or training feedback.
        for stream in self.streams:
            history = []
            for request in stream["requests"]:
                history.append({"role": "user", "content": request["prompt"]})
                canonical = framing.encode_history(self.tokenizer, history)
                saved = json.loads(json.dumps(canonical))
                framing.require_identical_input(canonical, saved)
                self.assertEqual(canonical, framing.encode_history(self.tokenizer, copy.deepcopy(history)))
                history.append({"role": "assistant", "output_ids": [42, 128009]})


if __name__ == "__main__":
    unittest.main()
