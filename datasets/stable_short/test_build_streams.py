"""Synthetic contract tests: never load or sample research questions."""

import copy
from pathlib import Path
import tempfile
import unittest

import build_streams as builder


def fixture():
    rows = []
    for replica in range(1, 5):
        for label, prefix in (("code", "C"), ("math", "M")):
            for block in range(1, 5):
                for position in range(1, 9):
                    index = (replica - 1) * 32 + (block - 1) * 8 + position - 1
                    prompt = f"Synthetic {label} question {index}.\nKeep   these spaces."
                    rows.append({"source_dataset": builder.DATASETS[label], "source_revision": "a" * 40,
                                 "source_split": "train", "source_row_index": index,
                                 "source_id": f"source-{label}-{index}", "label": label,
                                 "prompt": prompt, "question_id": f"{label}-{index}",
                                 "prompt_sha256": builder.sha256(prompt.encode("utf-8")),
                                 "replica": replica, "block": f"{prefix}{block}",
                                 "position_in_block": position, "sampling_rank": index + 1,
                                 "seed": builder.SEED})
    return rows


class StreamTests(unittest.TestCase):
    def test_determinism_rebuild_and_readonly_check(self):
        rows = fixture()
        raw = b"".join(builder.json_bytes(row) for row in rows)
        artifacts = builder.make_artifacts(rows, raw)
        self.assertEqual(artifacts, builder.make_artifacts(list(reversed(rows)), raw))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "output"
            builder.write_or_check(artifacts, directory)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            builder.write_or_check(artifacts, directory)
            builder.write_or_check(artifacts, directory, check=True)
            after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            self.assertEqual(before, after)
        streams = builder.build_streams(rows)
        self.assertEqual(len(streams), 512)
        self.assertEqual([r["question_id"] for r in streams[:32]], [f"code-{i}" for i in range(32)])
        self.assertEqual([r["question_id"] for r in streams[64:96]],
                         [f"code-{i}" for i in range(8)] + [f"math-{i}" for i in range(8, 16)] +
                         [f"code-{i}" for i in range(16, 24)] + [f"math-{i}" for i in range(24, 32)])
        self.assertEqual(artifacts["preview_R1_STABLE_CODE.md"].count(b"## Turn "), 32)

    def test_duplicate_original_and_normalized_text(self):
        for duplicate in ("exact", "whitespace"):
            with self.subTest(duplicate=duplicate):
                rows = fixture()
                rows[1]["prompt"] = rows[0]["prompt"] if duplicate == "exact" else " ".join(rows[0]["prompt"].split())
                rows[1]["prompt_sha256"] = builder.sha256(rows[1]["prompt"].encode("utf-8"))
                with self.assertRaisesRegex(builder.ValidationError, "duplicate (prompt|normalized)"):
                    builder.build_streams(rows)

    def test_support_evaluation_leakage(self):
        rows = fixture()
        streams = builder.build_streams(rows)
        streams[16]["phase"] = "adaptation_support"
        with self.assertRaisesRegex(builder.ValidationError, "phase mismatch"):
            builder.validate_streams(rows, streams)
        streams = builder.build_streams(rows)
        for field in builder.SELECTION_FIELDS:
            streams[16][field] = streams[8][field]
        with self.assertRaisesRegex(builder.ValidationError, "incorrect block order"):
            builder.validate_streams(rows, streams)

    def test_incorrect_block_and_order(self):
        rows = fixture()
        rows[0]["block"] = "M1"
        with self.assertRaisesRegex(builder.ValidationError, "block/label mismatch"):
            builder.build_streams(rows)
        rows = fixture()
        streams = builder.build_streams(rows)
        streams[64], streams[65] = streams[65], streams[64]
        with self.assertRaisesRegex(builder.ValidationError, "turns not exactly"):
            builder.validate_streams(rows, streams)

    def test_disjoint_replicas_and_shared_text(self):
        rows = fixture()
        rows[64]["source_id"] = rows[0]["source_id"]
        with self.assertRaisesRegex(builder.ValidationError, "duplicate source_id"):
            builder.build_streams(rows)
        rows = fixture()
        streams = builder.build_streams(rows)
        streams[64]["prompt"] += "changed"
        with self.assertRaisesRegex(builder.ValidationError, "metadata/text changed"):
            builder.validate_streams(rows, streams)

    def test_no_overwrite_mismatch_and_missing_check(self):
        artifacts = builder.make_artifacts(fixture(), b"synthetic-selection")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "output"
            with self.assertRaisesRegex(builder.ValidationError, "missing artifact"):
                builder.write_or_check(artifacts, directory, check=True)
            self.assertFalse(directory.exists())
            directory.mkdir()
            (directory / "validation.json").write_bytes(b"do not replace")
            with self.assertRaisesRegex(builder.ValidationError, "existing artifact differs"):
                builder.write_or_check(artifacts, directory)
            self.assertEqual(list(directory.iterdir()), [directory / "validation.json"])
            self.assertEqual((directory / "validation.json").read_bytes(), b"do not replace")

    def test_answers_missing_revision_and_bad_hash_rejected(self):
        cases = [({"assistant": "answer"}, "unexpected or missing fields"),
                 ({"source_revision": "main"}, "pinned"),
                 ({"prompt_sha256": "0" * 64}, "SHA256 mismatch"),
                 ({"replica": True}, "invalid replica")]
        for update, message in cases:
            with self.subTest(update=update):
                rows = copy.deepcopy(fixture())
                rows[0].update(update)
                with self.assertRaisesRegex(builder.ValidationError, message):
                    builder.build_streams(rows)

    def test_unicode_prompt_and_metadata_type_preservation(self):
        rows = fixture()
        rows[0]["prompt"] += "\u2028Unicode line separator\u0085and next-line character."
        rows[0]["prompt_sha256"] = builder.sha256(rows[0]["prompt"].encode("utf-8"))
        raw = b"".join(builder.json_bytes(row) for row in rows)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.jsonl"
            path.write_bytes(raw)
            loaded, loaded_raw = builder.read_selection(path)
            self.assertEqual(loaded, rows)
            self.assertEqual(loaded_raw, raw)
        streams = builder.build_streams(rows)
        streams[0]["replica"] = True
        with self.assertRaisesRegex(builder.ValidationError, "metadata/text changed"):
            builder.validate_streams(rows, streams)


if __name__ == "__main__":
    unittest.main()
