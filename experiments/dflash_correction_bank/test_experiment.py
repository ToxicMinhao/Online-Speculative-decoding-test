"""CPU protocol tests with isolated synthetic backend states; no model execution."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from . import experiment as e


def stream(name="wildchat_00"):
    return dict(stream_id=name, source="personamem" if name.startswith("personamem") else "wildchat",
        synthetic=name.startswith("personamem"), requests=[dict(turn=t, prompt=f"  request {t}\n    x = 1\n")
            for t in range(1, e.EXPECTED_STREAMS[name]+1)])


class FakeBackend:
    def __init__(self, config=None):
        self.config = config or {}; self.events = []; self.histories = []
        self.progress = {arm: 0 for arm in e.ARMS}
        self.bad_output = False; self.bad_input = False; self.bad_state = False
        self.bad_mask = False; self.qualification_failure = False; self.diverge_history = False
        self.restart_states = []

    def start_episode(self, episode_id, seed=17, qualification=False):
        self.episode = episode_id
        self.restart_states.append(copy.deepcopy(self.progress))
        assert seed == 17 and not any(self.progress.values())
        self.events.append(("start", qualification))
        return dict(status="passed", loading_s=.7)

    def encode(self, messages):
        self.histories.append(copy.deepcopy(messages))
        ids = [100]
        for message in messages:
            ids.extend([ord(c) for c in message["content"]] if message["role"] == "user" else message["output_ids"])
        return ids+[101]

    def snapshot_state(self, arm, turn):
        return dict(arm=arm, turn=turn, after_turn=self.progress[arm], before_read=arm == "BANK",
                    state_sha256=e.digest([self.episode, arm, self.progress[arm]]))

    def serve(self, ids, arm, state, capture=False, *, turn_index, repeat=None, canonical=False):
        assert not capture
        self.events.append(("serve", turn_index, arm, repeat, canonical))
        output = [11, 12, turn_index, 128009]
        if self.diverge_history and canonical and arm == "OSD":
            output[1] = 13
        if self.bad_output and not canonical and arm == "OSD" and turn_index == 3 and repeat == 0:
            output[1] = 14
        result = dict(input_ids=list(ids), input_ids_sha256=e.digest(ids), output_ids=output, text="generated answer",
            request_s=1.1 if arm == "BANK" else 1., activation_s=.2, native_activation_s=.1,
            deployment_activation_s=.2,
            cache_maintenance_s=.01, verification_calls=2, accepted_draft_tokens=2,
            verification_committed_tokens=3, state_sha256=state["state_sha256"], capture=False,
            feedback=None, capture_costs=None)
        if self.bad_input and not canonical:
            result["input_ids"] = [123]
        if self.bad_state and not canonical:
            result["state_sha256"] = e.digest("wrong")
        if arm == "BANK":
            result["read"] = dict(forecast_saved_before_response=True, descriptor_extracted=True,
                routing_recomputed=True, fusion_recomputed=True, snapshot_sha256=state["state_sha256"],
                coefficients=[0., 0., 0., 0.] if turn_index <= 2 else [.5, 0., 0., 0.],
                cost_in_request_s=True, predicted_gain=.001, descriptor_s=.01, routing_s=.03, fusion_s=.02)
        return result

    def learn(self, arm, item, state, *, turn_index, final=False, qualification=False):
        self.events.append(("learn", turn_index, arm, final))
        assert item["arm"] == arm and item["turn"] == turn_index
        assert state["after_turn"] == turn_index-1
        assert self.progress[arm] == turn_index-1
        if not final:
            self.progress[arm] = turn_index
        costs = dict(teacher_replay_s=.2, candidate_gradient_s=.3, solver_s=.4, training_s=.3)
        if arm == "BANK":
            return dict(arm=arm, event=turn_index, state_before_sha256=state["state_sha256"],
                scheduled_steps=0 if final else 2, accepted_steps=0 if final else 1,
                current_response_only=not self.bad_mask, frozen_reference=True,
                protection_records_immutable=True, served_record_precedes_trials=True,
                input_ids_sha256=e.digest(item["input_ids"]), output_ids_sha256=e.digest(item["output_ids"]),
                served_gain=-.01, post_write_gain=.1, owner=0, blocked_writes=1,
                recurring_adaptation_s=1.2, costs=costs, accounting=dict(anchor_count=5))
        start, length = len(item["input_ids"]), len(item["output_ids"])
        exposure = dict(turn=turn_index, record_id=item["record_id"],
            unique_supervised_position_ids=list(range(start+1, start+length)),
            unique_supervised_positions=length-1, eligible_positions=length-1, zero_supervision=False)
        accounting = dict(optimizer_steps_this_event=2, optimizer_steps_total=2*turn_index,
            response_presentations_this_event=2, support_turns=[turn_index],
            recipe=dict(responses_per_step=1, passes=2, learning_rate=3e-4, loss="forward_KL"),
            response_audits=[dict(response_start=start, response_end_exclusive=start+length,
                loss_mask_sha256=e.digest([0]*start+[1]*length+[0]*9))],
            batches=[dict(presentations=[copy.deepcopy(exposure)]) for _ in range(2)])
        return dict(arm=arm, event=turn_index, optimizer_steps=2*turn_index,
            checkpoint=dict(checkpoint_id=f"D{turn_index}"), accounting=accounting,
            recurring_adaptation_s=1.2, costs=costs)

    def qualify(self, result):
        checks = {key: True for key in e.QUALIFICATION_CHECKS}
        checks["direct_retention_bounds"] = not self.qualification_failure
        checks["fit_loss_change"] = .05  # Unfavorable fit is reported, never an efficacy gate.
        return checks

    def close_episode(self, result):
        self.events.append(("close",))
        return dict(status="passed", deleted=[])


class ProtocolTests(unittest.TestCase):
    def run_one(self, backend=None, **kwargs):
        backend = backend or FakeBackend()
        return e.run_stream(stream(), backend, **kwargs), backend

    def test_schedule_and_no_checkpoint_dedup(self):
        result, backend = self.run_one()
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(result["implementation_checks"]["status"], "passed")
        self.assertEqual(len(result["canonical"]), 40)
        self.assertEqual(len(result["scores"]), 180)
        self.assertEqual(len(result["updates"]), 38)
        self.assertEqual(len(result["feedback_summaries"]), 20)
        self.assertEqual(sum(row["scheduled_steps"] for row in result["feedback_summaries"]), 38)
        for turn in range(1, 21):
            first_learn = min(i for i, row in enumerate(backend.events) if row[0] == "learn" and row[1] == turn)
            last_serve = max(i for i, row in enumerate(backend.events) if row[0] == "serve" and row[1] == turn)
            self.assertGreater(first_learn, last_serve)
        self.assertEqual([r for r in backend.events if r[:2] == ("learn", 20)], [("learn", 20, "BANK", True)])
        self.assertEqual(len([r for r in result["scores"] if r["turn"] == 1]), 9)

    def test_same_bank_inputs_and_independent_generated_histories(self):
        backend = FakeBackend(); backend.diverge_history = True
        result, _ = self.run_one(backend)
        self.assertEqual(result["status"], "complete", result.get("error"))
        for item in result["canonical"]:
            self.assertTrue(item["source_user_text"].startswith("  "))
            self.assertTrue(item["source_user_text"].endswith("\n"))
        for turn in range(1, 21):
            bank = next(r for r in result["canonical"] if r["arm"] == "BANK" and r["turn"] == turn)
            for scored in [r for r in result["scores"] if r["turn"] == turn]:
                self.assertEqual(scored["input_ids"], bank["input_ids"])
        self.assertFalse(result["history_comparisons"][0]["outputs_match"])
        self.assertFalse(result["history_comparisons"][1]["inputs_match"])
        self.assertEqual(backend.histories[2][1]["output_ids"][1], 12)
        self.assertEqual(backend.histories[3][1]["output_ids"][1], 13)

    def test_exact_campaign_scope(self):
        streams = [stream(name) for name in e.EXPECTED_STREAMS]
        specification = e.plan(streams)
        self.assertEqual(specification["request_contexts"], 116)
        self.assertEqual(specification["timed_replays"], 1044)
        self.assertEqual(specification["update_events_per_learning_arm"], 110)
        totals = [0, 0, 0]
        for s in streams:
            result = e.run_stream(s, FakeBackend())
            self.assertEqual(result["status"], "complete", result.get("error"))
            for index, key in enumerate(("canonical", "scores", "updates")):
                totals[index] += len(result[key])
        self.assertEqual(totals, [232, 1044, 220])

    def test_mismatch_retained_not_claimed(self):
        backend = FakeBackend(); backend.bad_output = True
        result, _ = self.run_one(backend)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["scores"]), 180)
        self.assertEqual(sum(not r["same_generation_valid"] for r in result["scores"]), 1)
        self.assertEqual(result["implementation_checks"]["status"], "failed")

    def test_pre_read_state_and_forecast_required_for_every_repeat(self):
        class BadRead(FakeBackend):
            def serve(self, *args, **kwargs):
                row = super().serve(*args, **kwargs)
                if "read" in row and not kwargs["canonical"]:
                    row["read"]["routing_recomputed"] = False
                return row
        result, _ = self.run_one(BadRead())
        self.assertEqual(result["status"], "failed")
        self.assertIn("routing_recomputed", result["error"])
        self.assertEqual(len(result["updates"]), 0)

    def test_mask_failure_stops_without_next_turn(self):
        backend = FakeBackend(); backend.bad_mask = True
        result, _ = self.run_one(backend)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["canonical"]), 2)
        self.assertIn("current_response_only", result["error"])

    def test_qualification_fit_has_no_performance_gate(self):
        result, _ = self.run_one(qualification=True)
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertGreater(result["qualification_checks"]["fit_loss_change"], 0)

    def test_qualification_direct_retention_failure(self):
        backend = FakeBackend(); backend.qualification_failure = True
        result, _ = self.run_one(backend, qualification=True)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["qualification_checks"]["direct_retention_bounds"])

    def test_clean_restart_and_automatic_campaign(self):
        from .run import run_campaign
        backends = []
        def factory(config):
            backend = FakeBackend(config); backends.append(backend); return backend
        with tempfile.TemporaryDirectory() as temporary:
            campaign = run_campaign([stream(name) for name in e.EXPECTED_STREAMS], factory, Path(temporary), config={})
        self.assertEqual(campaign["status"], "complete")
        self.assertEqual(len(backends), 7)
        self.assertTrue(backends[0].config["qualification"])
        self.assertFalse(backends[1].config["qualification"])
        self.assertEqual(backends[0].episode, backends[1].episode)
        self.assertTrue(all(not any(b.restart_states[0].values()) for b in backends))
        self.assertEqual(len(campaign["results"]), 6)

    def test_failed_qualification_does_not_start_campaign(self):
        from .run import run_campaign
        backends = []
        def factory(config):
            backend = FakeBackend(config); backend.qualification_failure = True; backends.append(backend); return backend
        with tempfile.TemporaryDirectory() as temporary:
            campaign = run_campaign([stream(name) for name in e.EXPECTED_STREAMS], factory, Path(temporary), config={})
        self.assertEqual(campaign["status"], "stopped")
        self.assertEqual(len(backends), 1)
        self.assertFalse(campaign["results"])

    def test_cli_storage_failure_precedes_any_campaign_or_backend(self):
        from . import run
        # run.main imports this absolute namespace. Legacy test directories add
        # themselves to sys.path, so pytest may also load a short package alias.
        from experiments.dflash_correction_bank import native
        inventory = {"fixture.py": "reviewed"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root/"config.json"
            config.write_text(json.dumps(dict(output_cap=1024, block_size=10)))
            out = root/"output"
            with patch.object(run, "load_streams", return_value=[stream(name) for name in e.EXPECTED_STREAMS]), \
                 patch.object(run, "source_inventory", return_value=inventory), \
                 patch.object(run, "load_pinned_tokenizer", return_value=object()), \
                 patch.object(run, "qualify_text", return_value={"status": "passed"}), \
                 patch.object(native, "admit_campaign_storage", side_effect=ValueError("insufficient complete campaign storage")), \
                 patch.object(run, "run_campaign") as campaign:
                with self.assertRaisesRegex(ValueError, "insufficient complete campaign"):
                    run.main(["run", "--dataset", str(root/"dataset"), "--out", str(out),
                        "--tokenizer", str(root/"tokenizer"), "--runtime-config", str(config),
                        "--allow-gpu-execution", "--reviewed-code-sha256", e.digest(inventory)])
                campaign.assert_not_called()
            receipt = json.loads((out/"STORAGE_ADMISSION.json").read_text())
            self.assertEqual(receipt["status"], "blocked")
            self.assertFalse(receipt["gpu_started"])


if __name__ == "__main__":
    unittest.main()
