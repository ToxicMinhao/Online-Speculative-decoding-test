"""Original request identity, split membership, and explicit context budgets."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.osd_persistent import data


class PersistentDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="osd_manifest_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def reference(self, count):
        rows = [{"dataset_index": i, "source_line_1based": i + 1,
                 "source_record_id": f"t{i}", "record_sha256": f"original-record-{i}"}
                for i in range(count)]
        path = self.root / "train_reference.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows, path

    def archive(self, count=3):
        reference, manifest = self.reference(count)
        root = self.root / "archive"
        root.mkdir()
        for index, row in enumerate(reference):
            self.write_json(f"archive/{index:05d}.json", {
                **row, "prompt_token_ids": [151644, index, 151668],
                "response": "DO_NOT_REUSE_HISTORICAL_RESPONSE",
                "native_output_token_ids": [999],
            })
        return root, manifest

    def heldout(self, count=3):
        timed = [{"request_index": i, "dataset_id": f"e{i}", "kind": "timed",
                  "rendered_prompt_token_ids": [151644, 200 + i, 151668],
                  "max_new_tokens": 2048, "reference_answer": "DO_NOT_USE_LABEL"}
                 for i in range(1, count + 1)]
        warmup = {**timed[-1], "request_index": 0, "kind": "warmup", "max_new_tokens": 64}
        return self.write_json("heldout.json", {
            "seed": 42, "num_timed": count, "max_samples": count,
            "qualification_only": False, "requests": [warmup, *timed],
        })

    @staticmethod
    def full_rows():
        train = [{"dataset_index": i, "dataset_id": f"t{i}", "prompt_ids": [151644, i, 151668]}
                 for i in range(data.TRAIN_REQUESTS)]
        evaluation = [{"request_index": i, "dataset_id": f"e{i}", "prompt_ids": [151644, i]}
                      for i in range(1, data.EVAL_REQUESTS + 1)]
        return train, evaluation

    def test_archive_copies_only_exact_prompt_ids_and_identity(self):
        archive, reference = self.archive()
        rows = data.load_training_archive(archive, reference, expected_count=3)
        self.assertEqual([row["prompt_ids"] for row in rows],
                         [[151644, i, 151668] for i in range(3)])
        self.assertEqual([row["dataset_id"] for row in rows], ["t0", "t1", "t2"])
        self.assertNotIn("DO_NOT_REUSE", json.dumps(rows))
        self.assertNotIn("response", json.dumps(rows))

    def test_missing_archive_row_does_not_resample(self):
        archive, reference = self.archive()
        (archive / "00001.json").unlink()
        with self.assertRaises(FileNotFoundError):
            data.load_training_archive(archive, reference, expected_count=3)

    def test_archive_identity_mismatch_is_rejected(self):
        archive, reference = self.archive()
        path = archive / "00001.json"
        raw = json.loads(path.read_text())
        raw["source_record_id"] = "different-request"
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "archive identity differs"):
            data.load_training_archive(archive, reference, expected_count=3)

    def test_source_order_and_token_types_are_checked(self):
        archive, reference = self.archive()
        rows = reference.read_text().splitlines()
        reference.write_text("\n".join([rows[1], rows[0], rows[2]]))
        with self.assertRaisesRegex(ValueError, "chronological"):
            data.load_training_archive(archive, reference, expected_count=3)
        reference.write_text("\n".join(rows))
        raw = json.loads((archive / "00001.json").read_text())
        raw["prompt_token_ids"] = [True, 4]
        (archive / "00001.json").write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            data.load_training_archive(archive, reference, expected_count=3)

    def test_heldout_excludes_warmup_preserves_timed_order_and_omits_answers(self):
        path = self.heldout()
        rows = data.load_evaluation_manifest(path, expected_count=3)
        self.assertEqual([row["dataset_id"] for row in rows], ["e1", "e2", "e3"])
        self.assertEqual([row["request_index"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[-1]["prompt_ids"], [151644, 203, 151668])
        self.assertNotIn("DO_NOT_USE", json.dumps(rows))

    def test_heldout_changed_cap_or_order_is_rejected(self):
        path = self.heldout()
        source = json.loads(path.read_text())
        source["requests"][1]["max_new_tokens"] = 512
        path.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "output cap"):
            data.load_evaluation_manifest(path, expected_count=3)
        source["requests"][1]["max_new_tokens"] = 2048
        source["requests"][1], source["requests"][2] = source["requests"][2], source["requests"][1]
        path.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "order changed"):
            data.load_evaluation_manifest(path, expected_count=3)

    def test_explicit_context_policies_never_truncate_prompts(self):
        self.assertEqual(data.generation_cap(200, cap_policy="remaining", train_context_limit=2048), 1848)
        self.assertEqual(data.generation_cap(200, cap_policy="remaining", train_context_limit=4096), 2048)
        self.assertEqual(data.generation_cap(200, cap_policy="fixed2048", train_context_limit=4096), 2048)
        with self.assertRaisesRegex(ValueError, "Fixed 2048"):
            data.generation_cap(200, cap_policy="fixed2048", train_context_limit=2048)
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            data.generation_cap(2047, cap_policy="remaining", train_context_limit=2048)

    def test_full_manifest_has_58_events_and_49_serve_only_requests(self):
        train, evaluation = self.full_rows()
        manifest = data.build_manifest(train, evaluation, cap_policy="remaining")
        requests = manifest["train_requests"]
        self.assertEqual(requests[127]["event_id"], "event_001")
        self.assertEqual(requests[128]["event_id"], "event_002")
        self.assertEqual(requests[7423]["event_id"], "event_058")
        self.assertTrue(requests[7423]["collect_feedback"])
        self.assertTrue(all(row["event_id"] is None and not row["collect_feedback"]
                            for row in requests[7424:]))
        self.assertEqual(len(requests[7424:]), 49)
        self.assertEqual(requests[0]["prompt_ids"], train[0]["prompt_ids"])
        self.assertEqual(requests[0]["max_new_tokens"], 2045)
        self.assertEqual({row["max_new_tokens"] for row in manifest["eval_requests"]}, {2048})
        self.assertEqual(manifest["policy"]["serving_context_limit"], 2050)
        self.assertEqual(data.validate_manifest(manifest)["counts"]["feedback_requests"], 7424)

    def test_fixed_cap_and_context_are_jointly_validated(self):
        train, evaluation = self.full_rows()
        with self.assertRaisesRegex(ValueError, "Fixed 2048"):
            data.build_manifest(train, evaluation, cap_policy="fixed2048")
        manifest = data.build_manifest(train, evaluation, cap_policy="fixed2048",
                                       train_context_limit=4096, serving_context_limit=4096)
        self.assertEqual({row["max_new_tokens"] for row in manifest["train_requests"]}, {2048})
        with self.assertRaisesRegex(ValueError, "Serving context"):
            data.build_manifest(train, evaluation, cap_policy="remaining", serving_context_limit=2048)

    def test_runtime_rejects_split_leakage_and_changed_feedback_membership(self):
        manifest = data.build_manifest(*self.full_rows(), cap_policy="remaining")
        bad = copy.deepcopy(manifest)
        bad["eval_requests"][0]["dataset_id"] = bad["train_requests"][0]["dataset_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            data.validate_manifest(bad)
        bad = copy.deepcopy(manifest)
        bad["train_requests"][7424]["collect_feedback"] = True
        with self.assertRaisesRegex(ValueError, "feedback membership"):
            data.validate_manifest(bad)

    def test_raw_fallback_uses_only_user_question_and_checks_original_prompt_ids(self):
        _, reference = self.reference(data.TRAIN_REQUESTS)
        raw = [{"id": f"t{i}", "conversations": [
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": "SECRET_GROUND_TRUTH"}]}
            for i in range(data.TRAIN_REQUESTS)]
        raw_path = self.root / "raw.jsonl"
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw))
        provenance = [{"dataset_index": i, "source_record_id": f"t{i}",
                       "prompt_token_ids_sha256": hashlib.sha256(
                           json.dumps([151644, i], separators=(",", ":")).encode()).hexdigest()}
                      for i in range(data.TRAIN_REQUESTS)]
        evidence = self.write_json("ONLINE_ROWS.json", {"arms": {"osd": provenance}})
        case = self

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                case.assertEqual(len(messages), 1)
                case.assertEqual(messages[0]["role"], "user")
                case.assertEqual(kwargs, {"tokenize": False, "add_generation_prompt": True,
                                          "enable_thinking": False})
                return messages[0]["content"]

            def encode(self, text, *, add_special_tokens):
                case.assertFalse(add_special_tokens)
                case.assertNotIn("SECRET", text)
                return [151644, int(text.split()[-1])]

        with patch.object(data, "TRAIN_DATA_SHA256", hashlib.sha256(raw_path.read_bytes()).hexdigest()):
            rows = data.load_training_raw(raw_path, reference, Tokenizer(), evidence)
            self.assertEqual(len(rows), data.TRAIN_REQUESTS)
            self.assertEqual(rows[-1]["prompt_ids"], [151644, 7472])
            self.assertNotIn("SECRET", json.dumps(rows))
            provenance[0]["prompt_token_ids_sha256"] = "wrong"
            evidence.write_text(json.dumps({"arms": {"osd": provenance}}))
            with self.assertRaisesRegex(ValueError, "Reconstructed prompt differs"):
                data.load_training_raw(raw_path, reference, Tokenizer(), evidence)
        with self.assertRaisesRegex(ValueError, "pinned normalized"):
            data.load_training_raw(raw_path, reference, Tokenizer(), evidence)


if __name__ == "__main__":
    unittest.main()
