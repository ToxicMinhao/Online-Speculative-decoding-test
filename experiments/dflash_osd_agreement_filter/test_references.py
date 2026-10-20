import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dflash_per_turn.experiment import EXPECTED_STREAMS, digest
from experiments.dflash_per_turn.test_experiment import stream
from .references import prepare_references, text_sha256, validate_references


class ReferenceTests(unittest.TestCase):
    def prepare(self, directory):
        streams = [stream(name) for name in EXPECTED_STREAMS]
        files = {}
        for source in streams:
            rows = [{"episode_id": source["stream_id"], "canonical": True, "turn": r["turn"],
                     "source_user_text": r["prompt"], "source_user_text_sha256": text_sha256(r["prompt"]),
                     "source_message_ids": r.get("source_message_ids"), "input_ids": [100, r["turn"]],
                     "input_ids_sha256": digest([100, r["turn"]]), "output_ids": [200, r["turn"]]}
                    for r in source["requests"][:3]]
            path = Path(directory) / (source["stream_id"] + ".json")
            path.write_text(json.dumps({"episode_id": source["stream_id"], "status": "complete", "canonical": rows}))
            files[source["stream_id"]] = path
        return streams, files

    def test_existing_tokens_text_order_and_complete_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            streams, files = self.prepare(directory)
            bundle = prepare_references(streams, files)
            refs = validate_references(bundle, streams)
            self.assertEqual(sum(len(r) for r in refs.values()), 12)
            self.assertEqual(refs["wildchat_00"][2]["input_ids"], [100, 2])
            self.assertEqual(refs["wildchat_00"][2]["source_user_text"], streams[0]["requests"][1]["prompt"])
            self.assertEqual(bundle["streams"][0]["source_binding"]["bytes"], files["wildchat_00"].stat().st_size)

    def test_whitespace_modification_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            streams, files = self.prepare(directory)
            streams[0]["requests"][1]["prompt"] = streams[0]["requests"][1]["prompt"].strip()
            with self.assertRaisesRegex(ValueError, "whitespace"):
                prepare_references(streams, files)

    def test_missing_or_duplicate_canonical_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            streams, files = self.prepare(directory)
            path = files["wildchat_00"]
            raw = json.loads(path.read_text())
            raw["canonical"].append(copy.deepcopy(raw["canonical"][1]))
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                prepare_references(streams, files)

    def test_reference_tamper_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            streams, files = self.prepare(directory)
            bundle = prepare_references(streams, files)
            bundle["streams"][0]["contexts"][0]["output_ids"][0] += 1
            with self.assertRaisesRegex(ValueError, "bundle changed"):
                validate_references(bundle, streams)

    def test_events_file_supported_without_replay_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            streams, files = self.prepare(directory)
            path = files["wildchat_00"]
            raw = json.loads(path.read_text())
            event_path = path.with_suffix(".jsonl")
            event_path.write_text("\n".join(json.dumps({"event": "canonical", "data": r}) for r in raw["canonical"]) +
                                 "\n" + json.dumps({"event": "score", "data": raw["canonical"][1]}) + "\n")
            files["wildchat_00"] = event_path
            self.assertEqual(len(prepare_references(streams, files)["streams"]), 6)


if __name__ == "__main__":
    unittest.main()
