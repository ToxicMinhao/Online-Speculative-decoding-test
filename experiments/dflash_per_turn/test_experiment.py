"""CPU protocol tests: no model weights, training environment or network."""
import copy
import tempfile
from pathlib import Path
import unittest

from . import experiment as e


def stream(name="wildchat_00"):
    return dict(stream_id=name, source="personamem" if name.startswith("personamem") else "wildchat",
        synthetic=name.startswith("personamem"), requests=[dict(turn=t, prompt=f"  request {t}\n    code\n")
        for t in range(1, e.EXPECTED_STREAMS[name] + 1)])


class FakeBackend:
    def __init__(self, config=None):
        self.config = config or {}
        self.checkpoints = {"F": self.checkpoint("F")}
        self.calls, self.events, self.retirements = [], [], []
        self.training_state = 0
        self.mismatch = False
        self.identity_mismatch = False
        self.restore_failure = False
        self.encoded = []

    @staticmethod
    def checkpoint(label):
        return dict(checkpoint_id=label, checkpoint_identity_sha256=e.digest(label), path="/fixture/" + label)

    def start_episode(self, episode_id):
        self.episode = episode_id
        return dict(status="passed")

    def encode(self, messages):
        self.encoded.append(copy.deepcopy(messages))
        return [100] + [ord(c) for m in messages if m["role"] == "user" for c in m["content"]] + [101]

    def serve(self, ids, checkpoint_id, capture=False, *, turn_index, repeat=None):
        self.calls.append((turn_index, repeat, checkpoint_id, list(ids), capture))
        self.events.append(("serve", turn_index, repeat))
        output = [11, 12, turn_index, 128009]
        if self.mismatch and repeat is not None and turn_index == 3 and checkpoint_id == "F":
            output = [11, 13, turn_index, 128009]
        identity = self.checkpoints[checkpoint_id]["checkpoint_identity_sha256"]
        if self.identity_mismatch and repeat is not None:
            identity = e.digest("wrong")
        return dict(output_ids=output, input_ids=list(ids), input_ids_sha256=e.digest(ids),
            text="our answer", request_s=1., activation_s=.2, native_activation_s=.1,
            cache_maintenance_s=.01, verification_calls=2, accepted_draft_tokens=2,
            verification_committed_tokens=3, checkpoint_identity_sha256=identity,
            feedback=None, capture_costs=None, status="passed")

    def prepare_support(self, item):
        return dict(record_id=f'{self.episode}:t{item["turn"]}', turn=item["turn"],
            input_ids=list(item["input_ids"]), output_ids=list(item["output_ids"]),
            capture_checkpoint_sha256=item["checkpoint_identity_sha256"])

    def train_event(self, event, responses):
        self.events.append(("train", event))
        assert event == self.training_state + 1 and len(responses) == 1
        self.training_state = event
        response = responses[0]
        start, count = len(response["input_ids"]), len(response["output_ids"])
        presentation = dict(turn=event, record_id=response["record_id"],
            unique_supervised_position_ids=list(range(start + 1, start + count)),
            unique_supervised_positions=count - 1, eligible_positions=count - 1, zero_supervision=False)
        account = dict(optimizer_steps_this_event=2, optimizer_steps_total=2 * event,
            response_presentations_this_event=2, support_turns=[event],
            recipe=dict(responses_per_step=1, passes=2, learning_rate=3e-4, loss="forward_KL"),
            response_audits=[dict(response_start=start, response_end_exclusive=start + count,
                loss_mask_sha256=e.digest([0] * start + [1] * count + [0] * 9))],
            batches=[dict(presentations=[copy.deepcopy(presentation)]) for _ in range(2)])
        checkpoint = self.checkpoint(f"D{event}")
        self.checkpoints[checkpoint["checkpoint_id"]] = checkpoint
        return dict(event=event, checkpoint=checkpoint, checkpoint_id=f"D{event}",
                    optimizer_steps=2 * event, accounting=account)

    def diagnostic_event(self, event, original_update):
        self.events.append(("diagnostic", event))
        state = self.training_state
        checkpoint = self.checkpoint(f"L{event}")
        self.checkpoints[checkpoint["checkpoint_id"]] = checkpoint
        assert self.training_state == state
        return dict(checkpoint=checkpoint, main_state_restored=not self.restore_failure)

    def retire(self, *, keep_ids, protected_event_ids):
        self.retirements.append((set(keep_ids), set(protected_event_ids)))
        return dict(status="passed", removed=[])

    def finish_episode(self):
        return dict(status="passed")


class ProtocolTests(unittest.TestCase):
    def run_one(self, backend=None, **kwargs):
        backend = backend or FakeBackend()
        kwargs.setdefault("classify", lambda result, turn: {"kind": None})
        return e.run_stream(stream(), backend, **kwargs), backend

    def test_complete_schedule_masks_and_no_final_update(self):
        result, backend = self.run_one()
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(result["implementation_checks"]["status"], "passed")
        self.assertEqual(len(result["canonical"]), 20)
        self.assertEqual(len(result["scores"]), 171)
        self.assertEqual(len(result["updates"]), 19)
        self.assertEqual(result["updates"][-1]["optimizer_steps"], 38)
        for turn in range(1, 20):
            train = backend.events.index(("train", turn))
            last_score = max(i for i, event in enumerate(backend.events)
                if event[:2] == ("serve", turn))
            self.assertGreater(train, last_score)
        self.assertFalse(any(capture for *_, capture in backend.calls))

    def test_exact_saved_inputs_history_and_source_whitespace(self):
        result, backend = self.run_one()
        for item in result["canonical"]:
            for row in result["scores"]:
                if row["turn"] == item["turn"]:
                    self.assertEqual(row["input_ids"], item["input_ids"])
        for turn, messages in enumerate(backend.encoded, 1):
            self.assertEqual(len(messages), 2 * turn - 1)
            self.assertTrue(messages[-1]["content"].startswith("  request"))
            self.assertTrue(messages[-1]["content"].endswith("\n"))
            self.assertTrue(all(m["content"] == "our answer" for m in messages if m["role"] == "assistant"))

    def test_beginning_checkpoint_deduplication(self):
        self.assertEqual(e.methods_for_turn(1), ["F"])
        self.assertEqual(e.methods_for_turn(2), ["F", "D1"])
        self.assertEqual(e.methods_for_turn(3), ["F", "D2", "D1"])
        orders = [e.method_order(["F", "D2", "D1"], r, 0, 3) for r in range(3)]
        self.assertEqual(len({tuple(order) for order in orders}), 3)

    def test_all_six_counts(self):
        counts = [0, 0, 0]
        for name in e.EXPECTED_STREAMS:
            result = e.run_stream(stream(name), FakeBackend(), classify=lambda r, t: {"kind": None})
            self.assertEqual(result["status"], "complete")
            for i, key in enumerate(("canonical", "updates", "scores")):
                counts[i] += len(result[key])
        self.assertEqual(counts, [116, 110, 990])
        self.assertEqual(e.plan([stream(name) for name in e.EXPECTED_STREAMS])["optimizer_steps"], 220)

    def test_mismatches_preserved_and_gate_fails(self):
        backend = FakeBackend(); backend.mismatch = True
        result, _ = self.run_one(backend)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["implementation_checks"]["status"], "failed")
        self.assertEqual(sum(not r["same_generation_valid"] for r in result["scores"]), 3)
        self.assertEqual(len(result["scores"]), 171)

    def test_diagnostic_budget_and_isolation(self):
        budget = {"harmful": 0, "beneficial": 0}
        classifier = lambda result, turn: {"kind": "beneficial" if turn == 2 else "harmful"}
        result, backend = self.run_one(case_budget=budget, classify=classifier)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(budget, {"harmful": 2, "beneficial": 1})
        self.assertEqual(len(result["diagnostics"]), 3)
        self.assertEqual(sum(len(d["scores"]) for d in result["diagnostics"]), 54)
        self.assertEqual([r["checkpoint_id"] for r in result["canonical"]], ["F"] + [f"D{i}" for i in range(1, 20)])
        self.assertEqual(backend.training_state, 19)
        self.assertTrue(all(d["status"] == "complete" for d in result["diagnostics"]))
        self.assertTrue({1, 2, 3}.issubset(backend.retirements[-1][1]))

    def test_failed_diagnostic_restore_prevents_next_main_update(self):
        backend = FakeBackend(); backend.restore_failure = True
        result, _ = self.run_one(backend, classify=lambda r, t: {"kind": "harmful"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["updates"]), 1)
        self.assertEqual(backend.training_state, 1)

    def test_incomplete_diagnostic_evidence_fails_implementation_gate(self):
        result, _ = self.run_one(classify=lambda r, t: {"kind": "harmful"})
        self.assertEqual(result["implementation_checks"]["status"], "passed")
        for mutation in (lambda case: case.update(status="running"),
                         lambda case: case["scores"].pop(),
                         lambda case: case["branch"].update(main_state_restored=False)):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(result)
                mutation(changed["diagnostics"][0])
                checks = e.implementation_checks(changed)
                self.assertEqual(checks["status"], "failed")
                self.assertEqual(checks["incomplete_diagnostic_cases"], 1)

    def test_old_answer_loss_mask_rejected(self):
        class BadMask(FakeBackend):
            def train_event(self, event, responses):
                update = super().train_event(event, responses)
                update["accounting"]["response_audits"][0]["loss_mask_sha256"] = e.digest([1])
                return update
        result, _ = self.run_one(BadMask())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["canonical"]), 1)
        self.assertIn("History/prefix received loss", result["error"])

    def test_first_stream_gate_stops_campaign(self):
        from .run import run_campaign
        def make(config):
            backend = FakeBackend(config); backend.mismatch = True
            return backend
        with tempfile.TemporaryDirectory() as tmp:
            campaign = run_campaign([stream(name) for name in e.EXPECTED_STREAMS], make, Path(tmp), config={})
            self.assertEqual(campaign["status"], "stopped")
            self.assertEqual(len(campaign["results"]), 1)
            self.assertEqual(campaign["first_stream_gate"]["status"], "failed")

    def test_diagnostic_output_mismatch_stops_campaign_with_valid_main_scores(self):
        from .run import run_campaign
        class BadDiagnostic(FakeBackend):
            def serve(self, ids, checkpoint_id, capture=False, *, turn_index, repeat=None):
                row = super().serve(ids, checkpoint_id, capture, turn_index=turn_index, repeat=repeat)
                if checkpoint_id == "D1":
                    row["verification_calls"] = 3  # Select the first harmful event.
                if checkpoint_id.startswith("L"):
                    row["output_ids"] = [11, 99, turn_index, 128009]
                return row
        with tempfile.TemporaryDirectory() as tmp:
            campaign = run_campaign([stream(name) for name in e.EXPECTED_STREAMS], BadDiagnostic,
                                    Path(tmp), config={})
            self.assertEqual(campaign["status"], "stopped")
            self.assertEqual(len(campaign["results"]), 1)
            checks = campaign["first_stream_gate"]
            self.assertEqual(checks["status"], "failed")
            self.assertEqual(checks["invalid_scores"], 0)
            self.assertGreater(checks["invalid_diagnostic_scores"], 0)
            self.assertEqual(checks["diagnostic_timing_replays"], checks["expected_diagnostic_timing_replays"])

    def test_complete_campaign_continues_all_six(self):
        from .run import run_campaign
        with tempfile.TemporaryDirectory() as tmp:
            campaign = run_campaign([stream(name) for name in e.EXPECTED_STREAMS], FakeBackend, Path(tmp), config={})
            self.assertEqual(campaign["status"], "complete")
            self.assertEqual(len(campaign["results"]), 6)


if __name__ == "__main__":
    unittest.main()
