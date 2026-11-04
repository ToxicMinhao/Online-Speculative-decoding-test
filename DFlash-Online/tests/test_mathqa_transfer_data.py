"""Input-only tests for the prespecified MathQA transfer panel."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mathqa_transfer.data import (
    assemble, history_candidates, history_matches, inventory_history, normalize, read_source,
    render_inputs, select_panel, token_digest,
)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(text.encode())


def rows(questions):
    return render_inputs([{"Problem": q, "options": "a ) 1 , b ) 2"} for q in questions], Tokenizer())


class MathqaDataTests(unittest.TestCase):
    def test_only_problem_and_options_are_rendered(self):
        source = {"Problem": "  Keep\nindent\t  ", "options": " a ) 1 ",
                  "Rationale": "SECRET", "correct": "SECRET", "category": "SECRET"}
        self.assertEqual(assemble(source), "Keep\nindent\n\nOptions: a ) 1")
        rendered = render_inputs([source], Tokenizer())[0]
        self.assertNotIn("SECRET", rendered["rendered_prompt"])
        self.assertEqual(rendered["prompt_ids_sha256"], token_digest(rendered["prompt_ids"]))

    def test_malformed_rows_never_render(self):
        for source in ({}, {"Problem": "", "options": "ok"}, {"Problem": "ok", "options": []}):
            self.assertIsNone(assemble(source))
            self.assertEqual(render_inputs([source], Tokenizer())[0]["invalid_reason"], "malformed_or_empty_input")

    def test_normalization_nfkc_case_and_whitespace(self):
        self.assertEqual(normalize(" ＡＢＣ\nStraße \t42 "), "abc strasse 42")

    def test_source_order_dedup_and_fixed_hash_order(self):
        inputs = rows(["Question A", "QUESTION\tＡ", "Question B", "Question C"])
        panel = select_panel(inputs, {}, count=3)
        self.assertEqual(panel["eligible_count"], 3)
        self.assertEqual(panel["exclusions"][0]["source_row"], 1)
        self.assertEqual(panel["exclusions"][0]["duplicate_of"], 0)
        expected = sorted((0, 2, 3), key=lambda i: hashlib.sha256(f"mathqa-transfer-v1:{i}".encode()).hexdigest())
        self.assertEqual([r["source_row"] for r in panel["requests"]], expected)

    def test_prompt_cap_inclusive_and_shortfall_visible(self):
        inputs = rows(["Q0", "Q1"])
        inputs[0]["prompt_tokens"] = 2048
        inputs[1]["prompt_tokens"] = 2049
        panel = select_panel(inputs, {}, count=2)
        self.assertEqual(panel["status"], "selection_shortfall")
        self.assertEqual([r["source_row"] for r in panel["requests"]], [0])
        self.assertEqual(panel["exclusions"][0]["reason"], "rendered_prompt_over_2048_tokens")

    def test_historical_overlap_is_excluded(self):
        panel = select_panel(rows(["Q0", "Q1"]), {0: [{"path": "old", "reasons": ["normalized_stem"]}]}, count=1)
        self.assertEqual(panel["requests"][0]["source_row"], 1)
        self.assertEqual(panel["exclusions"][0]["reason"], "available_project_inventory_overlap")

    def test_stem_matching_in_prior_options_wrapper(self):
        found, counts = history_matches({"prompt": "ＳＡＭＥ\nquestion\n\nOptions: other formatting"}, rows(["Same question"]))
        self.assertIn(0, found)
        self.assertEqual(counts["text_input_fields"], 1)

    def test_assistant_output_source_catalogs_do_not_count_as_usage(self):
        value = {"messages": [{"role": "assistant", "content": "Same question"}],
                 "answer": "Same question", "response": {"text": "Same question"},
                 "sources": [{"question": "Same question"}], "candidate_pools": [{"prompt": "Same question"}]}
        found, _ = history_matches(value, rows(["Same question"]))
        self.assertEqual(found, {})

    def test_retained_result_input_still_counts_as_usage(self):
        found, _ = history_matches({"results": [{"prompt": "Q0", "output": "Q1"}]}, rows(["Q0", "Q1"]))
        self.assertEqual(set(found), {0})

    def test_source_identity_does_not_confuse_aqua_row_number(self):
        found, _ = history_matches({"rows": [{"source_id": "mathqa:test:1"}, {"source_id": "aqua:test:0"}]}, rows(["Q0", "Q1"]))
        self.assertEqual(set(found), {1})

    def test_exact_token_only_and_chat_text_matching(self):
        inputs = rows(["Q0", "Q1"])
        value = {"requests": [{"prompt_ids": inputs[0]["prompt_ids"]}, {"prompt": inputs[1]["rendered_prompt"]}]}
        found, counts = history_matches(value, inputs)
        self.assertEqual(set(found), {0, 1})
        self.assertEqual(counts["token_input_records"], 1)

    def test_source_blob_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.zip"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "Git blob"):
                read_source(path)

    def test_inventory_receipts_include_duplicates_and_bad_json(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("a.json", "b.json", "bad.json")]
            paths[0].write_text(json.dumps({"prompt": "Q0"}))
            paths[1].write_bytes(paths[0].read_bytes())
            paths[2].write_text("{")
            overlaps, report = inventory_history(paths, rows(["Q0"]))
            self.assertEqual(set(overlaps), {0})
            self.assertEqual(report["distinct_scanned_files"], 1)
            self.assertEqual(report["duplicate_files"], 1)
            self.assertEqual(len(report["unreadable_files"]), 1)

    def test_allocated_input_directories_are_inventoried(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ["research/old/inputs/heldout.jsonl", "research/old/prompts/DEV.jsonl",
                     "research/old/source/test.json", "research/new/panel.json"]
            with patch("experiments.mathqa_transfer.data.subprocess.check_output", return_value="\n".join(names)):
                found = list(history_candidates(root, root / "research/new"))
            self.assertEqual(found, [root / n for n in names[:2]])

    def test_missing_referenced_inventory_stays_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "prior-selected.json"
            _, report = inventory_history([missing], rows(["Q0"]))
            self.assertEqual(report["unavailable_referenced_inventories"], [str(missing)])
            self.assertEqual(report["distinct_scanned_files"], 0)

    def test_no_outcome_fields_change_selection(self):
        raw = [{"Problem": f"Q{i}", "options": "a ) 1", "Rationale": f"answer {i}"} for i in range(8)]
        first = select_panel(render_inputs(raw, Tokenizer()), {}, count=4)
        for r in raw:
            r.update(Rationale="changed", correct="different", generated_length=999)
        second = select_panel(render_inputs(raw, Tokenizer()), {}, count=4)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
