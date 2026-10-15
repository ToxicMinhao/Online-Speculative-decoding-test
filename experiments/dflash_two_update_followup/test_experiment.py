"""Synthetic CPU tests of chronology, source custody and evaluation isolation."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_spec = importlib.util.spec_from_file_location("followup_experiment", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


def episode_rows(replica=1, scenario="STABLE_CODE"):
    return [{"replica": replica, "stream_type": scenario, "stream_id": f"R{replica}_{scenario}",
             "split": "development" if replica <= 2 else "confirmation", "turn": turn,
             "phase": experiment.PHASES[(turn - 1) // 8],
             "question_id": f"question-{replica}-{scenario}-{turn}", "prompt": f"task {turn}"}
            for turn in range(1, 33)]


def identity(checkpoint):
    return experiment.digest({"fixed_weights": checkpoint})


def rebind(reference):
    reference.pop("source_binding", None)
    payload = json.dumps(reference).encode()
    reference["source_binding"] = {"file_sha256": hashlib.sha256(payload).hexdigest(),
                                   "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
                                   "result_sha256": experiment.digest(reference)}
    return reference


def original_reference(rows=None):
    rows = rows or episode_rows()
    canonical, messages = [], [42]
    for row in rows:
        turn = row["turn"]
        messages.append(500 + turn)
        checkpoint = "F" if turn <= 16 else ("U1" if turn <= 24 else "U2")
        output = [1000 + turn, 2000 + turn]
        canonical.append({"episode_id": row["stream_id"], "turn": turn, "question_id": row["question_id"],
                          "phase": row["phase"], "checkpoint_id": checkpoint,
                          "input_ids": list(messages), "input_ids_sha256": experiment.digest(messages),
                          "output_ids": output, "checkpoint_identity_sha256": identity(checkpoint.replace("U", "C"))})
        messages.extend(output)
    updates = [{"status": "passed", "checkpoint_id": f"U{event}", "optimizer_steps": 8 * event,
                "checkpoint_path": f"/retained/{rows[0]['stream_id']}/U{event}",
                "checkpoint": {"checkpoint_identity_sha256": identity(f"C{event}"),
                               "files": {name: {"bytes": 123, "sha256": identity(name)} for name in
                                         ("config.json", "model.safetensors", "version.txt")}}}
               for event in (1, 2)]
    scores = [{"turn": turn, "repeat": repeat, "checkpoint_id": checkpoint,
               "input_ids_sha256": canonical[turn - 1]["input_ids_sha256"],
               "output_ids": list(canonical[turn - 1]["output_ids"]), "canonical_output_match": True}
              for turn in range(17, 33) for repeat in range(3)
              for checkpoint in ("F", "U1" if turn <= 24 else "U2", "AR")]
    return rebind({"schema_version": 1, "status": "complete", "episode_id": rows[0]["stream_id"],
                   "scenario": rows[0]["stream_type"], "replica": rows[0]["replica"],
                   "canonical": canonical, "updates": updates, "scores": scores})


class FakeBackend:
    def __init__(self, *, divergent=False, score_fault=None, support_fault=None, overflow_turn=None):
        self.events, self.encoded, self.optimizers = [], [], []
        self.divergent, self.score_fault = divergent, score_fault
        self.support_fault, self.overflow_turn = support_fault, overflow_turn

    def start_episode(self, episode, *, reference):
        self.episode, self.steps, self.reference = episode, 0, reference
        self.optimizer = object()
        self.optimizers.append(self.optimizer)
        self.events.append({"kind": "start", "episode": episode, "steps": self.steps})

    def encode(self, messages):
        ids = [42]
        for message in messages:
            if message["role"] == "user":
                ids.append(500 + int(message["content"].split()[-1]))
            else:
                ids.extend(message["output_ids"])
        turn = int(messages[-1]["content"].split()[-1])
        if turn == self.overflow_turn:
            ids.extend([99] * (experiment.INPUT_LIMIT + 1 - len(ids)))
        self.encoded.append({"episode": self.episode, "turn": turn,
                             "messages": copy.deepcopy(messages), "input_ids": list(ids)})
        return ids

    def serve(self, input_ids, checkpoint, *, capture, turn_index, repeat=None):
        self.events.append({"kind": "serve", "checkpoint": checkpoint, "turn": turn_index,
                            "capture": capture, "repeat": repeat, "input_ids": list(input_ids), "steps": self.steps})
        output = [1000 + turn_index, 2000 + turn_index]
        if self.divergent and repeat is None and turn_index >= 17:
            output[-1] += 3000
        result = {"output_ids": output, "text": f"answer {turn_index}", "request_s": .01,
                  "accepted_draft_tokens": 1, "verification_calls": 1,
                  "input_ids_sha256": experiment.digest(input_ids),
                  "checkpoint_identity_sha256": identity(checkpoint), "feedback": None, "capture_costs": None}
        if self.score_fault and (checkpoint, turn_index, repeat) == self.score_fault[:3]:
            fault = self.score_fault[3]
            if fault == "output":
                result["output_ids"] = [0]
            elif fault == "input":
                result["input_ids_sha256"] = identity("wrong input")
            elif fault == "identity":
                result["checkpoint_identity_sha256"] = identity("changed weights")
            elif fault == "capture":
                result["feedback"] = {"capture_path": "/forbidden/tensors.pt"}
        return result

    def prepare_support(self, item):
        self.events.append({"kind": "prepare", "turn": item["turn"], "item": copy.deepcopy(item)})
        prepared = {"support_turn": item["turn"], "capture_checkpoint": item["checkpoint_id"],
                    "input_ids": list(item["input_ids"]), "output_ids": list(item["output_ids"]),
                    "loss_owner_turn": item["turn"], "conditioning_only_prefix_length": len(item["input_ids"])}
        if self.support_fault == "original" and item["turn"] == 18:
            prepared["output_ids"] = self.reference["canonical"][17]["output_ids"]
        if self.support_fault == "tensor":
            prepared["feature_path"] = "/forbidden/capture.pt"
        return prepared

    def train_event(self, event, support):
        self.events.append({"kind": "train", "event": event, "support": copy.deepcopy(support),
                            "optimizer": self.optimizer, "steps_before": self.steps})
        self.steps += 8
        return {"checkpoint_id": f"A{event}", "optimizer_steps": self.steps,
                "checkpoint": {"checkpoint_identity_sha256": identity(f"A{event}")},
                "training_s": .1, "activation_s": .02}

    def finish_episode(self):
        self.events.append({"kind": "finish"})
        return {"status": "passed"}


class FollowupProtocolTests(unittest.TestCase):
    def run_case(self, mode="author-path", backend=None, rows=None, reference=None):
        rows = rows or episode_rows()
        reference = reference or original_reference(rows)
        backend = backend or FakeBackend()
        events = []

        def emit(kind, data):
            copied = {"kind": kind, "data": copy.deepcopy(data)}
            events.append(copied)
            backend.events.append({"kind": "emitted", "event": copied})

        run = experiment.run_author_episode if mode == "author-path" else experiment.run_checkpoint_comparison
        result = run(rows, reference, backend, emit=emit)
        return result, backend, events

    def test_checkpoint_only_never_generates_prepares_or_trains(self):
        result, backend, _ = self.run_case("checkpoints-only")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["scores"]), 72)
        self.assertEqual((result["canonical"], result["updates"], result["support"]), ([], [], []))
        self.assertEqual(backend.encoded, [])
        self.assertFalse(any(e["kind"] in ("train", "prepare") for e in backend.events))
        calls = [e for e in backend.events if e["kind"] == "serve"]
        self.assertEqual({e["checkpoint"] for e in calls}, {"F", "C1", "C2"})
        self.assertEqual({e["turn"] for e in calls}, set(range(25, 33)))
        self.assertEqual({e["steps"] for e in calls}, {0})
        self.assertFalse(any(e["capture"] for e in calls))

    def test_original_cost_and_exposure_snapshot_is_historical_and_compact(self):
        reference = original_reference()
        reference["updates"][0].update(observed_update_call_s=99., training_s=70.,
            accounting={"recipe": {"normalization": "mean_slots_then_blocks_then_responses"},
                        "eligible_position_presentations_this_event": 8000,
                        "response_audits": [{"eligible_positions": 500}],
                        "batches": [{"loss": 1.23, "large_trace": [0] * 1000}]})
        reference["canonical"][8]["capture_costs"] = {"append_capture_s": .03, "transfer_s": .01}
        rebind(reference)
        result, _, _ = self.run_case(reference=reference)
        self.assertEqual(result["status"], "complete")
        snapshot = result["original_update_costs"][0]
        self.assertEqual(snapshot["observed_update_call_s"], 99.)
        self.assertEqual(snapshot["actual_exposure"]["eligible_position_presentations_this_event"], 8000)
        self.assertNotIn("batches", snapshot["actual_exposure"])
        self.assertNotIn("loss", snapshot["actual_exposure"])
        self.assertEqual(result["original_capture_costs"], [{"turn": 9, "costs": {"append_capture_s": .03, "transfer_s": .01}}])
        self.assertIn("not fresh paired timing", result["original_cost_scope"])
        self.assertNotEqual(result["updates"][0]["observed_update_call_s"], 99.)

    def test_two_updates_and_all_early_scores_precede_second_support_admission(self):
        result, backend, _ = self.run_case()
        self.assertEqual(result["status"], "complete")
        self.assertEqual((len(result["canonical"]), len(result["scores"]), len(result["updates"])), (32, 168, 2))
        trains = [e for e in backend.events if e["kind"] == "train"]
        self.assertEqual([[r["support_turn"] for r in e["support"]] for e in trains],
                         [list(range(9, 17)), list(range(17, 25))])
        prepare17 = next(i for i, e in enumerate(backend.events) if e["kind"] == "prepare" and e["turn"] == 17)
        prior_scores = [e["event"]["data"] for e in backend.events[:prepare17]
                        if e["kind"] == "emitted" and e["event"]["kind"] == "score"]
        self.assertEqual(len(prior_scores), 72)
        self.assertEqual({r["turn"] for r in prior_scores}, set(range(17, 25)))
        self.assertEqual({r["checkpoint_id"] for r in prior_scores}, {"F", "C1", "A1"})
        self.assertEqual({r["checkpoint_id"] for r in result["scores"] if r["turn"] >= 25}, {"F", "C1", "C2", "A2"})
        self.assertIs(trains[0]["optimizer"], trains[1]["optimizer"])
        self.assertEqual([e["steps_before"] for e in trains], [0, 8])

    def test_weights_fixed_within_each_window_and_capture_disabled_everywhere(self):
        result, backend, _ = self.run_case()
        self.assertEqual(result["status"], "complete")
        calls = [e for e in backend.events if e["kind"] == "serve"]
        canonical = [e for e in calls if e["repeat"] is None]
        self.assertEqual([e["checkpoint"] for e in canonical], ["F"] * 16 + ["A1"] * 8 + ["A2"] * 8)
        self.assertEqual({e["steps"] for e in calls if 17 <= e["turn"] <= 24}, {8})
        self.assertEqual({e["steps"] for e in calls if e["turn"] >= 25}, {16})
        self.assertFalse(any(e["capture"] for e in calls))
        self.assertTrue(all(e["checkpoint_identity_match"] for e in result["scores"]))

    def test_new_history_divergence_is_evidence_and_benchmarks_use_original_inputs(self):
        result, backend, events = self.run_case(backend=FakeBackend(divergent=True))
        self.assertEqual(result["status"], "complete")
        comparisons = result["trajectory_comparisons"]
        self.assertEqual(len(comparisons), 32)
        self.assertTrue(all(r["input_match"] and r["output_match"] for r in comparisons[:16]))
        self.assertTrue(comparisons[16]["input_match"])
        self.assertFalse(comparisons[16]["output_match"])
        self.assertTrue(all(not r["input_match"] and not r["output_match"] for r in comparisons[17:]))
        for call in (e for e in backend.events if e["kind"] == "serve" and e["repeat"] is not None):
            self.assertEqual(call["input_ids"], backend.reference["canonical"][call["turn"] - 1]["input_ids"])
        self.assertEqual(len([e for e in events if e["kind"] == "trajectory_comparison"]), 32)
        self.assertTrue(all(r["same_generation_valid"] for r in result["scores"]))

    def test_training_uses_own_answers_and_each_current_response_owner_only(self):
        result, backend, _ = self.run_case(backend=FakeBackend(divergent=True))
        self.assertEqual(result["status"], "complete")
        trains = [e for e in backend.events if e["kind"] == "train"]
        for event in trains:
            for record in event["support"]:
                turn = record["support_turn"]
                own = result["canonical"][turn - 1]
                self.assertEqual(record["input_ids"], own["input_ids"])
                self.assertEqual(record["output_ids"], own["output_ids"])
                self.assertEqual(record["loss_owner_turn"], turn)
                self.assertEqual(record["conditioning_only_prefix_length"], len(own["input_ids"]))
        response18 = trains[1]["support"][1]
        self.assertIn(5017, response18["input_ids"])  # own turn17 is conditioning only
        self.assertNotIn(5017, response18["output_ids"])
        self.assertEqual(response18["output_ids"], [1018, 5018])
        self.assertEqual(response18["loss_owner_turn"], 18)

    def test_actual_output_tokens_accumulate_once_without_text_round_trip(self):
        result, backend, _ = self.run_case(backend=FakeBackend(divergent=True))
        for turn, encoded in enumerate(backend.encoded, 1):
            self.assertEqual(len(encoded["messages"]), 2 * turn - 1)
            expected = [42]
            for previous in range(1, turn):
                expected.extend([500 + previous, *result["canonical"][previous - 1]["output_ids"]])
            expected.append(500 + turn)
            self.assertEqual(encoded["input_ids"], expected)

    def test_three_rotating_method_orders_are_episode_counterbalanced(self):
        for replica, scenario in ((1, "STABLE_CODE"), (1, "STABLE_MATH"), (2, "STABLE_CODE")):
            rows = episode_rows(replica, scenario)
            result, backend, events = self.run_case(rows=rows)
            self.assertEqual(result["status"], "complete")
            orders = [e["data"] for e in events if e["kind"] == "method_order"]
            index = (replica - 1) * 5 + experiment.SCENARIOS.index(scenario)
            for window, methods in ((orders[:3], ["F", "C1", "A1"]), (orders[3:], ["F", "C1", "C2", "A2"])):
                self.assertEqual([r["methods"] for r in window],
                                 [experiment.method_order(methods, repeat, index) for repeat in range(3)])
                self.assertEqual(len({r["methods"][0] for r in window}), 3)
            self.assertEqual([e["checkpoint"] for e in backend.events if e["kind"] == "serve" and
                              e["repeat"] == 0 and 17 <= e["turn"] <= 24],
                             [method for method in orders[0]["methods"] for _ in range(8)])

    def test_common_input_output_or_weight_discrepancies_are_saved_before_failure(self):
        for fault, field in (("output", "canonical_output_match"), ("input", "input_match"),
                             ("identity", "checkpoint_identity_match")):
            with self.subTest(fault=fault):
                result, backend, events = self.run_case(backend=FakeBackend(score_fault=("F", 19, 0, fault)))
                self.assertEqual(result["status"], "failed")
                self.assertEqual(len(result["updates"]), 1)
                self.assertEqual([r["turn"] for r in result["scores"]], [17, 18, 19])
                self.assertFalse(result["scores"][-1][field])
                score_index = max(i for i, e in enumerate(events) if e["kind"] == "score")
                failure_index = next(i for i, e in enumerate(events) if e["kind"] == "failure")
                self.assertLess(score_index, failure_index)
                self.assertFalse(any(e["kind"] == "prepare" and e["turn"] >= 17 for e in backend.events))
                self.assertEqual(backend.events[-1]["kind"], "finish")

    def test_unexpected_capture_in_score_is_retained_then_fails(self):
        result, _, _ = self.run_case(backend=FakeBackend(score_fault=("F", 17, 0, "capture")))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["scores"]), 1)
        self.assertIsNotNone(result["scores"][0]["feedback"])
        self.assertIn("capture is forbidden", result["error"])

    def test_wrong_original_training_answer_and_tensor_support_are_rejected(self):
        for fault in ("original", "tensor"):
            result, backend, _ = self.run_case(backend=FakeBackend(divergent=True, support_fault=fault))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len([e for e in backend.events if e["kind"] == "train"]), 1 if fault == "original" else 0)

    def test_full_context_overflow_stops_before_serving_without_truncation(self):
        result, backend, _ = self.run_case(backend=FakeBackend(overflow_turn=17))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["canonical"]), 16)
        self.assertEqual(len(backend.encoded[-1]["input_ids"]), experiment.INPUT_LIMIT + 1)
        self.assertFalse(any(e["kind"] == "serve" and e["turn"] >= 17 for e in backend.events))

    def test_reset_between_episodes_includes_weights_optimizer_and_history(self):
        backend = FakeBackend()
        one, _, _ = self.run_case(backend=backend)
        two, _, _ = self.run_case(backend=backend, rows=episode_rows(2, "RANDOM_MIX"))
        self.assertEqual((one["status"], two["status"]), ("complete", "complete"))
        self.assertIsNot(backend.optimizers[0], backend.optimizers[1])
        self.assertEqual([r["input_ids"] for r in backend.encoded if r["turn"] == 1], [[42, 501], [42, 501]])
        self.assertEqual([r["checkpoint_id"] for r in two["canonical"][:16]], ["F"] * 16)

    def test_reserved_or_malformed_episode_never_starts_backend(self):
        cases = [episode_rows(3), episode_rows(4), episode_rows()[:-1], episode_rows()]
        cases[-1][17]["phase"] = "support"
        for rows in cases:
            backend = FakeBackend()
            with self.assertRaises(ValueError):
                experiment.run_author_episode(rows, original_reference(), backend)
            self.assertEqual(backend.events, [])

    def test_reference_source_binding_required_and_mutation_rejected(self):
        reference = original_reference()
        for mutation in (lambda r: r.pop("source_binding"), lambda r: r["canonical"][0]["output_ids"].append(77)):
            changed = copy.deepcopy(reference)
            mutation(changed)
            backend = FakeBackend()
            with self.assertRaisesRegex(ValueError, "source binding"):
                experiment.run_checkpoint_comparison(episode_rows(), changed, backend)
            self.assertEqual(backend.events, [])

    def test_reference_completeness_and_checkpoint_binding_checked_after_hash(self):
        for mutation in (lambda r: r["canonical"].pop(), lambda r: r["updates"].pop(),
                         lambda r: r["scores"].pop(),
                         lambda r: r["canonical"][0].update(input_ids_sha256=identity("bad")),
                         lambda r: r["updates"][0]["checkpoint"]["files"].pop("model.safetensors"),
                         lambda r: r["scores"][0].update(output_ids=[999])):
            reference = original_reference()
            mutation(reference)
            rebind(reference)
            with self.assertRaises(ValueError):
                experiment.validate_reference(reference, episode_rows())

    def test_json_and_gzip_references_have_explicit_file_byte_hashes(self):
        reference = original_reference()
        reference.pop("source_binding")
        plain = json.dumps(reference).encode()
        with tempfile.TemporaryDirectory() as directory:
            for name, payload in (("RESULT.json", plain), ("RESULT.json.gz", gzip.compress(plain))):
                path = Path(directory) / name
                path.write_bytes(payload)
                loaded = experiment.load_reference(path, hashlib.sha256(payload).hexdigest())
                experiment.validate_reference(loaded, episode_rows())
                self.assertEqual(loaded["source_binding"]["uncompressed_sha256"], hashlib.sha256(plain).hexdigest())
                with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                    experiment.load_reference(path, "0" * 64)

    def test_plan_is_metadata_only_and_excludes_reserved_replicas(self):
        streams = [row for replica in (1, 2, 3, 4) for scenario in experiment.SCENARIOS
                   for row in episode_rows(replica, scenario)]
        for mode, expected in (("checkpoints-only", (0, 720, 0)), ("author-path", (320, 0, 1680)),
                               ("both", (320, 0, 1680))):
            planned = experiment.plan(streams, mode)
            self.assertEqual(planned["status"], "review_only")
            self.assertFalse(planned["gpu_launch_authorized"])
            self.assertEqual(len(planned["episodes"]), 10)
            self.assertEqual({r["replica"] for r in planned["episodes"]}, {1, 2})
            self.assertEqual(tuple(planned[k] for k in ("canonical_request_executions",
                             "checkpoint_comparison_executions", "author_path_score_executions")), expected)
        with self.assertRaises(ValueError):
            experiment.plan(streams[32:])
        streams[-1]["replica"] = 5
        with self.assertRaises(ValueError):
            experiment.plan(streams)


if __name__ == "__main__":
    unittest.main()
