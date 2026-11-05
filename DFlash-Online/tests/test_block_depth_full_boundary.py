"""Bounded full-campaign boundary tests; no runtime/model imports or execution."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.block_depth.full_campaign import lean_record
from experiments.block_depth.endpoint import validate_execution
from experiments.block_depth.reconstruct import validate_source, replay_request
from types import SimpleNamespace


def request():
    return dict(request_id="fresh", panel_ordinal=4, prompt_ids=[7, 8],
                stop_token_ids=[99], max_new_tokens=2048)


def native_result(trace=False):
    observation = dict(status="passed", observed_rounds=1, verification_calls=1,
                       accepted_draft_tokens=0, committed_progress=1,
                       shallow_rounds=0, deep_rounds=1, cache_maintenance_s=0.01,
                       decode_round_wall_s=0.05, feedback_capture=False)
    if trace:
        observation["trace"] = [dict(output_start=0, positions=list(range(2, 18)),
                                     draft_tokens=list(range(15)), committed_progress=1)]
    return dict(output_ids=[10, 99], meta_info={"finish_reason": {"type": "stop"}},
                native_generate_s=0.15, request_latency_s=0.2, trace=observation)


class FullBoundaryTests(unittest.TestCase):
    def test_reconstruction_binds_native_vocabulary_and_termination_configuration(self):
        class Sampling(SimpleNamespace):
            def normalize(self, tokenizer):
                self.stop_strs = []
                self.stop_regex_strs = []
        record = dict(prompt_ids=[7, 8], stop_token_ids=[99], max_new_tokens=2048)
        req = replay_request(SimpleNamespace, Sampling, record, SimpleNamespace(vocab_size=100))
        self.assertEqual(req.vocab_size, 100)
        self.assertFalse(99 > req.vocab_size or 99 < 0)
        self.assertEqual(req.eos_token_ids, {99})
        self.assertEqual(req.origin_input_ids, [7, 8])
        self.assertEqual(req.sampling_params.max_new_tokens, 2048)
        self.assertFalse(req.sampling_params.ignore_eos)
        self.assertEqual(req.sampling_params.stop_strs, [])
        self.assertEqual(req.sampling_params.stop_regex_strs, [])

    def test_final_uses_lean_api_outer_timer_and_exact_saved_tokens(self):
        backend = mock.Mock()
        backend._call.return_value = native_result()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("experiments.block_depth.full_campaign.time.perf_counter", side_effect=[10., 10.3]):
            path = Path(directory) / "raw.json"
            record = lean_record(backend, request(), "FIVE", 1, path)
            self.assertEqual(json.loads(path.read_text()), record)
        call = backend._call.call_args
        self.assertEqual(call.args, ("generate_lean",))
        self.assertEqual(call.kwargs["trace_mode"], "none")
        self.assertEqual(call.kwargs["prompt_ids"], [7, 8])
        self.assertEqual(call.kwargs["max_new_tokens"], 2048)
        self.assertAlmostEqual(record["metadata"]["request_latency_s"], 0.3)
        self.assertEqual(record["metadata"]["native_generate_s"], 0.15)
        self.assertEqual(record["metadata"]["worker_request_latency_s"], 0.2)
        self.assertEqual(record["finish_reason"], "eos")
        self.assertEqual(validate_execution(record, request(), record), [])
        wrong = dict(record, response_ids=[11, 99])
        self.assertIn("canonical_parity", validate_execution(record, request(), wrong))

    def test_gate_preserves_replay_inputs_and_completed_progress(self):
        backend = mock.Mock()
        backend._call.return_value = native_result(trace=True)
        with tempfile.TemporaryDirectory() as directory:
            record = lean_record(backend, request(), "FIVE", 0, Path(directory) / "gate.json", gate=True)
        self.assertEqual(backend._call.call_args.kwargs["trace_mode"], "gate")
        self.assertEqual(record["request_index"], 4)
        self.assertEqual(record["metadata"]["behavior"], "FIVE")
        self.assertEqual(record["metadata"]["trace"][0]["output_start"], 0)
        self.assertEqual(record["metadata"]["committed_progress"], 1)

    def test_invalid_native_tokens_are_saved_before_final_validation(self):
        backend = mock.Mock()
        result = native_result()
        result["output_ids"] = [10]  # Neither EOS nor the cap.
        backend._call.return_value = result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            record = lean_record(backend, request(), "FIVE", 0, path)
            self.assertEqual(json.loads(path.read_text())["response_ids"], [10])
        self.assertIn("termination", validate_execution(record, request(), record))

    def source(self):
        return dict(prompt_ids=[7, 8], response_ids=[10, 11, 99], finish_reason="eos",
                    max_new_tokens=2048, stop_token_ids=[99], metadata={"trace": [
                        dict(output_start=0, anchor_token=10, committed_progress=2,
                             committed_tokens=[11, 99], positions=list(range(2, 18)),
                             draft_tokens=list(range(15)), target_argmax=[11, 99]+[1]*14)]})

    def test_reconstruction_requires_complete_retained_history_and_termination(self):
        source = self.source()
        prompt, response, blocks = validate_source(source)
        self.assertEqual(prompt, [7, 8])
        self.assertEqual(response, [10, 11, 99])
        self.assertEqual(len(blocks), 1)
        malformed = deepcopy(source)
        malformed["metadata"]["trace"][0]["committed_tokens"] = [12, 99]
        with self.assertRaisesRegex(RuntimeError, "committed tokens"):
            validate_source(malformed)
        malformed = dict(source, finish_reason="length")
        with self.assertRaisesRegex(ValueError, "termination"):
            validate_source(malformed)
        malformed = deepcopy(source)
        malformed["metadata"]["trace"] = []
        with self.assertRaisesRegex(RuntimeError, "verification progress"):
            validate_source(malformed)


if __name__ == "__main__":
    unittest.main()
