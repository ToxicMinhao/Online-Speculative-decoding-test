"""CPU-only synthetic regressions; never load research questions or models."""

from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_spec = importlib.util.spec_from_file_location("two_update_builder", Path(__file__).with_name("build_streams.py"))
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


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
                                 "prompt_sha256": builder.sha256(prompt.encode()), "replica": replica,
                                 "block": f"{prefix}{block}", "position_in_block": position,
                                 "sampling_rank": index + 1, "seed": builder.SEED})
    return rows


class TwoUpdateTests(unittest.TestCase):
    def setUp(self):
        self.selection = fixture()
        self.streams = builder.build_streams(self.selection)

    def mixed(self, replica=1):
        return [row for row in self.streams
                if row["replica"] == replica and row["stream_type"] == "RANDOM_MIX"]

    def replace_source(self, placement, source):
        for key in builder.SELECTION_FIELDS:
            placement[key] = source[key]

    def test_original_four_streams_and_shared_blocks_are_unchanged(self):
        before = builder.legacy.build_streams(self.selection)
        after = [row for row in self.streams if row["stream_type"] != "RANDOM_MIX"]
        self.assertEqual([{k: v for k, v in row.items() if k != "phase"} for row in before],
                         [{k: v for k, v in row.items() if k != "phase"} for row in after])
        self.assertEqual(Counter(row["question_id"] for row in before),
                         Counter({row["question_id"]: 2 for row in self.selection}))
        self.assertEqual([row["question_id"] for row in after[:8]],
                         [row["question_id"] for row in after[64:72]])

    def test_counts_multiplicity_and_phase_chronology(self):
        self.assertEqual(len(self.streams), 640)
        self.assertEqual(len({row["stream_id"] for row in self.streams}), 20)
        self.assertEqual(Counter(Counter(row["question_id"] for row in self.streams).values()),
                         {2: 128, 3: 128})
        for row in self.streams:
            expected = ("context", "support", "evaluation_then_support", "final_evaluation")[(row["turn"]-1)//8]
            self.assertEqual(row["phase"], expected)
            self.assertEqual(int(row["block"][1:]), (row["turn"]-1)//8+1)

    def test_seeded_golden_schedule_and_slots_allow_unbalanced_phases(self):
        schedules = ["MMCCMCMCMCCCMMCMCMMCMMMMMCMCCCCC", "CCMCMCMMMMCCCCCMMCMMCCCCMCMMMMMC",
                     "CCCMMMMMCMCCCCCCMMMCMMMMMCCMCCCM", "MMCCCMCMCCMMMCMMMCMCMCCCMMCMCMCC"]
        for replica, expected in enumerate(schedules, 1):
            rows = self.mixed(replica)
            self.assertEqual("".join(row["block"][0] for row in rows), expected)
            self.assertEqual(Counter(row["label"] for row in rows), {"code": 16, "math": 16})
        self.assertEqual([f"{row['block']}:{row['position_in_block']}" for row in self.mixed()], [
            "M1:1", "M1:5", "C1:3", "C1:7", "M1:4", "C1:1", "M1:6", "C1:6",
            "M2:8", "C2:3", "C2:1", "C2:6", "M2:4", "M2:1", "C2:5", "M2:6",
            "C3:2", "M3:4", "M3:5", "C3:7", "M3:1", "M3:8", "M3:2", "M3:7",
            "M4:4", "C4:1", "M4:8", "C4:4", "C4:5", "C4:2", "C4:3", "C4:7"])
        self.assertEqual([sum(row["label"] == "code" for row in self.mixed()[start:start+8])
                          for start in (0, 8, 16, 24)], [4, 4, 2, 6])

    def test_determinism_input_order_and_no_mutation(self):
        original = copy.deepcopy(self.selection)
        reversed_rows = builder.build_streams(list(reversed(self.selection)))
        self.assertEqual(self.streams, reversed_rows)
        self.assertEqual(self.selection, original)
        raw = b"".join(builder.json_bytes(row) for row in self.selection)
        self.assertEqual(builder.make_artifacts(self.selection, raw),
                         builder.make_artifacts(list(reversed(self.selection)), raw))

    def test_wrong_phase_and_premature_support_rejected(self):
        for phase in ("support", "evaluation", "adaptation_support"):
            with self.subTest(phase=phase):
                streams = copy.deepcopy(self.streams)
                streams[16]["phase"] = phase
                with self.assertRaisesRegex(builder.ValidationError, "chronological phase mismatch"):
                    builder.validate_streams(self.selection, streams)
        self.replace_source(self.mixed()[16], self.mixed()[8])
        with self.assertRaisesRegex(builder.ValidationError, "wrong phase pool"):
            builder.validate_streams(self.selection, self.streams)

    def test_random_mix_repeated_question_rejected(self):
        self.replace_source(self.mixed()[1], self.mixed()[0])
        with self.assertRaisesRegex(builder.ValidationError, "question repeats"):
            builder.validate_streams(self.selection, self.streams)

    def test_random_mix_changed_balance_or_seed_order_rejected(self):
        original = copy.deepcopy(self.streams)
        replacement = next(row for row in self.selection if row["replica"] == 1
                           and row["block"] == "C1" and row["question_id"] not in
                           {mixed["question_id"] for mixed in self.mixed()})
        self.replace_source(self.mixed()[0], replacement)
        with self.assertRaisesRegex(builder.ValidationError, "globally balanced"):
            builder.validate_streams(self.selection, self.streams)
        self.streams = original
        first, second = copy.deepcopy(self.mixed()[:2])
        self.replace_source(self.mixed()[0], second)
        self.replace_source(self.mixed()[1], first)
        with self.assertRaisesRegex(builder.ValidationError, "seeded RANDOM_MIX"):
            builder.validate_streams(self.selection, self.streams)

    def test_cross_replica_and_modified_shared_prompt_rejected(self):
        original = copy.deepcopy(self.streams)
        other_replica = next(row for row in self.selection if row["replica"] == 2 and row["block"] == "M1")
        self.replace_source(self.mixed()[0], other_replica)
        with self.assertRaisesRegex(builder.ValidationError, "stream_id mismatch"):
            builder.validate_streams(self.selection, self.streams)
        self.streams = original
        self.streams[64]["prompt"] += "changed"
        with self.assertRaisesRegex(builder.ValidationError, "metadata/text changed"):
            builder.validate_streams(self.selection, self.streams)

    def test_legacy_shared_order_and_stream_order_rejected(self):
        original = copy.deepcopy(self.streams)
        first, second = copy.deepcopy(self.streams[64:66])
        self.replace_source(self.streams[64], second)
        self.replace_source(self.streams[65], first)
        with self.assertRaisesRegex(builder.ValidationError, "incorrect within-block order"):
            builder.validate_streams(self.selection, self.streams)
        self.streams = original
        self.streams[128], self.streams[129] = self.streams[129], self.streams[128]
        with self.assertRaisesRegex(builder.ValidationError, "turns not exactly"):
            builder.validate_streams(self.selection, self.streams)

    def test_reference_answers_contaminated_schema_and_source_duplicates_rejected(self):
        self.streams[0]["reference_answer"] = "forbidden"
        with self.assertRaisesRegex(builder.ValidationError, "unexpected or missing placement fields"):
            builder.validate_streams(self.selection, self.streams)
        for update, error in (({"assistant": "answer"}, "unexpected or missing fields"),
                              ({"source_revision": "main"}, "pinned"),
                              ({"source_id": "source-code-0"}, "duplicate source_id")):
            with self.subTest(update=update):
                selection = copy.deepcopy(self.selection)
                selection[64].update(update)
                with self.assertRaisesRegex(builder.ValidationError, error):
                    builder.build_streams(selection)

    def test_manifests_previews_and_non_destructive_rebuild(self):
        raw = b"".join(builder.json_bytes(row) for row in self.selection)
        artifacts = builder.make_artifacts(self.selection, raw)
        self.assertEqual(len(artifacts), 23)
        manifest = json.loads(artifacts["manifest.json"])
        self.assertEqual(manifest["selection_sha256"], builder.sha256(raw))
        self.assertEqual(manifest["random_mix"]["seeds"], {str(i): 20261014+i for i in range(1,5)})
        self.assertEqual(manifest["support_chronology"][1]["support_turns"], [17, 24])
        for name, data in artifacts.items():
            if name.startswith("preview_"):
                self.assertEqual(data.count(b"## Turn "), 32)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            builder.write_or_check(artifacts, output)
            before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
            builder.write_or_check(artifacts, output, check=True)
            builder.write_or_check(artifacts, output)
            self.assertEqual(before, {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                                     for path in output.iterdir()})
            (output / "manifest.json").write_bytes(b"preserve me")
            with self.assertRaisesRegex(builder.ValidationError, "existing artifact differs"):
                builder.write_or_check(artifacts, output)
            self.assertEqual((output / "manifest.json").read_bytes(), b"preserve me")

    def test_public_checkout_output_forbidden(self):
        with self.assertRaisesRegex(builder.ValidationError, "outside the public code checkout"):
            builder.write_or_check({"streams.jsonl": b"synthetic"}, Path(__file__).parent / "data")


if __name__ == "__main__":
    unittest.main()
