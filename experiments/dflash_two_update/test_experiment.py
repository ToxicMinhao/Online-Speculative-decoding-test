"""Independent, synthetic CPU protocol checks; no native/model imports."""

import copy
import importlib.util
from pathlib import Path
import unittest


_spec = importlib.util.spec_from_file_location("two_update_experiment", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


def episode_rows(replica=1, scenario="STABLE_CODE"):
    phases = ("context", "support", "evaluation_then_support", "final_evaluation")
    return [{"replica": replica, "stream_type": scenario, "stream_id": f"R{replica}_{scenario}",
             "split": "development" if replica <= 2 else "confirmation", "turn": turn,
             "phase": phases[(turn - 1) // 8], "question_id": f"R{replica}_{scenario}_{turn}",
             "prompt": f"task {turn}"} for turn in range(1, 33)]


class FakeBackend:
    """Record all external actions, with deliberately distinct text/token histories."""

    def __init__(self, *, mismatch=None, overflow_turn=None):
        self.events = []
        self.encoded = []
        self.mismatch = mismatch
        self.overflow_turn = overflow_turn
        self.optimizer_instances = []
        self.episode = None

    def start_episode(self, episode):
        self.episode = episode
        self.steps = 0
        self.optimizer = object()
        self.optimizer_instances.append(self.optimizer)
        self.events.append({"kind": "start", "episode": episode, "optimizer_steps": self.steps})

    def encode(self, messages):
        # A text round trip cannot reconstruct these IDs: output IDs are used
        # directly, matching the native conversation boundary contract.
        ids = [42]
        for message in messages:
            if message["role"] == "user":
                ids.append(500 + int(message["content"].split()[-1]))
            else:
                ids.extend(message["output_ids"])
        turn = int(messages[-1]["content"].split()[-1])
        if turn == self.overflow_turn:
            ids.extend([99] * (experiment.INPUT_LIMIT + 1 - len(ids)))
        self.encoded.append({"episode": self.episode, "messages": copy.deepcopy(messages),
                             "turn": turn, "input_ids": list(ids)})
        return ids

    def serve(self, input_ids, checkpoint, *, capture, turn_index, repeat):
        self.events.append({"kind": "serve", "episode": self.episode, "turn": turn_index,
                            "checkpoint": checkpoint, "capture": capture, "repeat": repeat,
                            "input_ids": list(input_ids), "optimizer_steps": self.steps})
        output_ids = [1000 + turn_index, 2000 + turn_index]
        if self.mismatch == (checkpoint, turn_index, repeat):
            output_ids[-1] += 1
        result = {"output_ids": output_ids, "text": f"generated answer {turn_index}",
                  "request_s": 0.01 + turn_index / 10000,
                  "accepted_draft_tokens": 0 if checkpoint == "AR" else 1,
                  "verification_calls": 0 if checkpoint == "AR" else 1,
                  "input_ids_sha256": experiment.digest(input_ids)}
        if capture:
            result["feedback"] = {"support_turn": turn_index, "capture_checkpoint": checkpoint,
                                  "input_ids": list(input_ids), "output_ids": list(output_ids)}
        return result

    def train_event(self, event, support):
        self.events.append({"kind": "train", "episode": self.episode, "event": event,
                            "support": copy.deepcopy(support), "optimizer": self.optimizer,
                            "steps_before": self.steps})
        self.steps += 8
        return {"checkpoint_id": f"U{event}", "optimizer_steps": self.steps,
                "training_s": 0.1, "activation_s": 0.02}

    def finish_episode(self):
        self.events.append({"kind": "finish", "episode": self.episode})
        return {"status": "closed"}


class ExperimentProtocolTests(unittest.TestCase):
    def run_case(self, backend=None, rows=None):
        backend = backend or FakeBackend()
        events = []

        def emit(kind, data):
            event = {"kind": kind, "data": copy.deepcopy(data)}
            events.append(event)
            backend.events.append({"kind": "emitted", "event": event})

        result = experiment.run_episode(rows or episode_rows(), backend, emit=emit)
        return result, backend, events

    def test_update_two_follows_all_evaluations_and_uses_only_new_responses(self):
        result, backend, events = self.run_case()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["canonical"]), 32)
        self.assertEqual(len(result["scores"]), 144)
        updates = [event for event in backend.events if event["kind"] == "train"]
        self.assertEqual([[row["support_turn"] for row in event["support"]] for event in updates],
                         [list(range(9, 17)), list(range(17, 25))])
        self.assertEqual([[row["capture_checkpoint"] for row in event["support"]] for event in updates],
                         [["F"] * 8, ["U1"] * 8])
        second_update_index = next(i for i, event in enumerate(backend.events)
                                   if event["kind"] == "train" and event["event"] == 2)
        before = backend.events[:second_update_index]
        scores = [event["event"]["data"] for event in before
                  if event["kind"] == "emitted" and event["event"]["kind"] == "score"]
        self.assertEqual(len(scores), 72)
        self.assertEqual(sum(row["checkpoint_id"] == "AR" for row in scores), 24)
        self.assertEqual({row["turn"] for row in scores}, set(range(17, 25)))
        self.assertTrue(any(event["kind"] == "emitted" and
                            event["event"]["kind"] == "evaluation_window_complete" for event in before))
        first_final_call = next(i for i, event in enumerate(backend.events)
                                if event["kind"] == "serve" and event["turn"] == 25)
        self.assertLess(second_update_index, first_final_call)

    def test_weights_and_optimizer_stay_fixed_within_each_evaluation_window(self):
        result, backend, _ = self.run_case()
        self.assertEqual(result["status"], "complete")
        calls = [event for event in backend.events if event["kind"] == "serve"]
        self.assertEqual({event["optimizer_steps"] for event in calls if 17 <= event["turn"] <= 24}, {8})
        self.assertEqual({event["optimizer_steps"] for event in calls if 25 <= event["turn"] <= 32}, {16})
        canonical = [event for event in calls if event["repeat"] is None]
        self.assertEqual([event["checkpoint"] for event in canonical], ["F"] * 16 + ["U1"] * 8 + ["U2"] * 8)
        self.assertEqual([event["turn"] for event in calls if event["capture"]], list(range(9, 25)))
        updates = [event for event in backend.events if event["kind"] == "train"]
        self.assertIs(updates[0]["optimizer"], updates[1]["optimizer"])
        self.assertEqual([event["steps_before"] for event in updates], [0, 8])

    def test_actual_generated_answers_and_ids_accumulate_once_in_history(self):
        result, backend, _ = self.run_case()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(backend.encoded), 32)
        for turn, encoded in enumerate(backend.encoded, 1):
            messages = encoded["messages"]
            self.assertEqual(len(messages), 2 * turn - 1)
            expected_ids = [42]
            for prior in range(1, turn):
                self.assertEqual(messages[2 * prior - 2], {"role": "user", "content": f"task {prior}"})
                self.assertEqual(messages[2 * prior - 1], {"role": "assistant",
                                 "content": f"generated answer {prior}", "output_ids": [1000 + prior, 2000 + prior]})
                expected_ids.extend([500 + prior, 1000 + prior, 2000 + prior])
            expected_ids.append(500 + turn)
            self.assertEqual(encoded["input_ids"], expected_ids)
            self.assertEqual(result["canonical"][turn - 1]["input_ids"], expected_ids)

    def test_every_timing_method_receives_identical_saved_inputs(self):
        result, backend, _ = self.run_case()
        for canonical in result["canonical"][16:]:
            turn = canonical["turn"]
            calls = [event for event in backend.events if event["kind"] == "serve" and event["turn"] == turn]
            self.assertEqual(len(calls), 10)  # canonical + three repeats of each of F/U/AR
            self.assertTrue(all(event["input_ids"] == canonical["input_ids"] for event in calls))
            scores = [row for row in result["scores"] if row["turn"] == turn]
            self.assertEqual(len(scores), 9)
            self.assertEqual({row["input_ids_sha256"] for row in scores},
                             {experiment.digest(canonical["input_ids"])})

    def test_three_alternating_pair_orders_and_episode_counterbalance(self):
        for replica, scenario, first in ((1, "STABLE_CODE", "F"),
                                         (1, "STABLE_MATH", "U1"), (2, "STABLE_CODE", "U1")):
            with self.subTest(replica=replica, scenario=scenario):
                result, backend, _ = self.run_case(rows=episode_rows(replica, scenario))
                self.assertEqual(result["status"], "complete")
                calls = [event for event in backend.events if event["kind"] == "serve"
                         and 17 <= event["turn"] <= 24 and event["repeat"] is not None]
                for repeat in range(3):
                    sd = [event for event in calls if event["repeat"] == repeat and event["checkpoint"] != "AR"]
                    lead = first if repeat % 2 == 0 else ("U1" if first == "F" else "F")
                    tail = "U1" if lead == "F" else "F"
                    self.assertEqual([event["checkpoint"] for event in sd], [lead] * 8 + [tail] * 8)
                    self.assertEqual([event["turn"] for event in sd], list(range(17, 25)) * 2)

    def test_sd_mismatch_is_retained_and_stops_before_second_update(self):
        result, backend, events = self.run_case(FakeBackend(mismatch=("F", 19, 0)))
        self.assertEqual(result["status"], "failed")
        self.assertIn("Native SD output mismatch", result["error"])
        self.assertEqual(len(result["updates"]), 1)
        self.assertEqual([event["event"] for event in backend.events if event["kind"] == "train"], [1])
        emitted_scores = [event["data"] for event in events if event["kind"] == "score"]
        self.assertEqual([row["turn"] for row in emitted_scores], [17, 18, 19])
        self.assertFalse(emitted_scores[-1]["canonical_output_match"])
        self.assertEqual(result["scores"], emitted_scores)
        self.assertEqual(backend.events[-1]["kind"], "finish")
        self.assertFalse(any(event["kind"] == "serve" and event["turn"] >= 25 for event in backend.events))

    def test_ar_mismatch_is_reported_without_claiming_same_generation(self):
        result, backend, _ = self.run_case(FakeBackend(mismatch=("AR", 20, 1)))
        self.assertEqual(result["status"], "complete")
        mismatches = [row for row in result["scores"] if not row["canonical_output_match"]]
        self.assertEqual([(row["checkpoint_id"], row["turn"], row["repeat"]) for row in mismatches],
                         [("AR", 20, 1)])
        self.assertFalse(mismatches[0]["same_generation_valid"])
        self.assertEqual(len(result["updates"]), 2)

    def test_overflow_fails_before_serving_without_truncating_history(self):
        result, backend, _ = self.run_case(FakeBackend(overflow_turn=17))
        self.assertEqual(result["status"], "failed")
        self.assertIn("truncation prohibited", result["error"])
        self.assertEqual(len(result["canonical"]), 16)
        self.assertEqual(len(backend.encoded[-1]["input_ids"]), experiment.INPUT_LIMIT + 1)
        self.assertEqual(len(backend.encoded[-1]["messages"]), 33)
        self.assertFalse(any(event["kind"] == "serve" and event["turn"] >= 17 for event in backend.events))
        self.assertEqual(len(result["updates"]), 1)

    def test_episode_boundaries_reset_draft_optimizer_and_history(self):
        backend = FakeBackend()
        first, _, _ = self.run_case(backend)
        second, _, _ = self.run_case(backend, episode_rows(1, "RANDOM_MIX"))
        self.assertEqual((first["status"], second["status"]), ("complete", "complete"))
        self.assertEqual(len(backend.optimizer_instances), 2)
        self.assertIsNot(backend.optimizer_instances[0], backend.optimizer_instances[1])
        starts = [event for event in backend.events if event["kind"] == "start"]
        self.assertEqual([event["optimizer_steps"] for event in starts], [0, 0])
        first_inputs = [row for row in backend.encoded if row["turn"] == 1]
        self.assertEqual([row["input_ids"] for row in first_inputs], [[42, 501], [42, 501]])
        self.assertEqual([len(row["messages"]) for row in first_inputs], [1, 1])
        self.assertEqual([row["checkpoint_id"] for row in second["canonical"][:16]], ["F"] * 16)
        self.assertEqual([row["optimizer_steps"] for row in second["updates"]], [8, 16])

    def test_confirmation_and_invalid_chronology_never_start_backend(self):
        for rows in (episode_rows(3), episode_rows()):
            if rows[0]["replica"] == 1:
                rows[16]["phase"] = "support"
            backend = FakeBackend()
            with self.assertRaises(ValueError):
                experiment.run_episode(rows, backend)
            self.assertEqual(backend.events, [])


if __name__ == "__main__":
    unittest.main()
