"""CPU controller tests with distinct arm histories and no model execution."""
import copy
import unittest

from experiments.dflash_per_turn.test_experiment import FakeBackend as QualifiedFake, stream
from . import experiment as e
from .controls import norm_match


def references(source):
    rows = {}
    for turn in (2, 3):
        ids, output = [700 + turn], [201, 202, turn]
        rows[turn] = {"episode_id": source["stream_id"], "turn": turn,
            "source": source["source"], "synthetic": source["synthetic"],
            "source_user_text": source["requests"][turn - 1]["prompt"],
            "input_ids": ids, "output_ids": output,
            "input_ids_sha256": e.digest(ids), "output_ids_sha256": e.digest(output)}
    return rows


class Backend(QualifiedFake):
    def __init__(self, arm, *, mismatch=False, training_failure=False, qualifier=True):
        super().__init__()
        self.arm, self.bad_score, self.training_failure = arm, mismatch, training_failure
        self.qualifier, self.finalized = qualifier, False
        self.supports = []

    def encode(self, messages):
        self.encoded.append(copy.deepcopy(messages))
        return [100, ord(self.arm)] + [v for m in messages for v in (m.get("output_ids") or [len(m["content"])])]

    def serve(self, ids, checkpoint_id, capture=False, *, turn_index, repeat=None):
        result = super().serve(ids, checkpoint_id, capture, turn_index=turn_index, repeat=repeat)
        result["output_ids"] = ([201, 202, turn_index] if repeat is not None else [500 + ord(self.arm), 502, turn_index])
        result["text"] = "own " + self.arm
        if self.bad_score and repeat is not None:
            result["output_ids"][0] += 1
        return result

    def prepare_support(self, item):
        self.supports.append(copy.deepcopy(item))
        return super().prepare_support(item)

    def train_event(self, event, responses):
        if self.training_failure:
            raise RuntimeError("test failure")
        result = super().train_event(event, responses)
        result["accounting"]["recipe"]["learning_rate"] = e.LEARNING_RATES[self.arm]
        result["accounting"]["capture_checkpoint_sha256"] = e.digest("F" if event == 1 else "D1")
        if self.arm in ("G", "N"):
            account = result["accounting"]
            start = len(responses[0]["input_ids"])
            group = {"a": [start + 1], "b": [start + 2], "identity_sha256": "fixture"}
            account.update(agreement_arm=self.arm, filter_enabled=True, fixed_groups=group)
            for index, batch in enumerate(account["batches"]):
                batch["pass_index"] = index
                batch["agreement"] = {"mode": self.arm, "optimizer_step_calls": 1,
                    "optimizer_steps_before": 2 * (event - 1) + index,
                    "optimizer_steps_after": 2 * (event - 1) + index + 1,
                    "proposed_step_norm": 1., "hypothetical_filtered_step_norm": .5,
                    "applied_step_norm": .5, "mask_keep_fraction": .5}
                if self.arm == "N":
                    batch["agreement"]["uniform_control"] = norm_match(.5, .5)
                batch["presentations"][0].update(normalization_denominator=1., grouping={
                    "identity_sha256": "fixture", "group_a_positions": group["a"], "group_b_positions": group["b"],
                    "shared_denominator": 1., "group_a_presentations": 1, "group_b_presentations": 1})
        result["qualification_checks"] = {"status": "passed" if self.qualifier else "failed"}
        return result

    def finalize_retention(self, *, result):
        assert result["status"] == "complete" and result["cleanup"]["status"] == "passed"
        assert all(s["same_generation_valid"] for s in result["scores"])
        self.finalized = True
        return {"status": "passed", "removed": []}


class ControllerTests(unittest.TestCase):
    def run_one(self, **options):
        source = stream()
        backends = {}
        def factory(arm, stream_id):
            backends[arm] = Backend(arm, **options.get(arm, {}))
            return backends[arm]
        result = e.run_stream(source, references(source), factory,
                              qualification=options.get("qualification", False))
        return result, backends

    def test_exact_bounded_schedule_and_primary_baseline(self):
        result, backends = self.run_one()
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(result["implementation_checks"]["status"], "passed")
        self.assertEqual(len(result["scores"]), 10)
        self.assertEqual(len(result["updates"]), 8)
        self.assertEqual(sum(len(v["canonical"]) for v in result["arms"].values()), 12)
        self.assertEqual([r["arm"] for r in result["scores"]].count("F"), 2)
        for backend in backends.values():
            self.assertEqual(backend.training_state, 2)
            self.assertTrue(backend.finalized)
            self.assertFalse(any(capture for *_, capture in backend.calls))

    def test_own_history_and_feedback_cannot_come_from_benchmark(self):
        result, backends = self.run_one()
        for arm, backend in backends.items():
            for index, messages in enumerate(backend.encoded):
                self.assertEqual(len(messages), 2 * index + 1)
                for message in messages:
                    if message["role"] == "assistant":
                        self.assertEqual(message["output_ids"][0], 500 + ord(arm))
                        self.assertEqual(message["content"], "own " + arm)
                self.assertTrue(messages[-1]["content"].startswith("  request"))
                self.assertTrue(messages[-1]["content"].endswith("\n"))
            self.assertEqual([r["turn"] for r in backend.supports], [1, 2])
            self.assertTrue(all(r["output_ids"][0] == 500 + ord(arm) for r in backend.supports))
            train2 = backend.events.index(("train", 2))
            score2 = max(i for i, event in enumerate(backend.events) if event[:2] == ("serve", 2))
            self.assertGreater(train2, score2)
        self.assertTrue(all(r["input_ids"] == [700 + r["turn"]] for r in result["scores"]))

    def test_mismatch_is_retained_and_blocks_retirement(self):
        result, backends = self.run_one(H={"mismatch": True})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(list(backends), ["H"])
        self.assertFalse(backends["H"].finalized)
        self.assertTrue(result["scores"])
        self.assertTrue(all(not r["same_generation_valid"] for r in result["scores"]))

    def test_failure_has_no_retry_or_later_arm(self):
        result, backends = self.run_one(L={"training_failure": True})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(list(backends), ["H", "L"])
        self.assertEqual(len(result["arms"]["L"]["canonical"]), 1)
        self.assertFalse(backends["L"].finalized)

    def test_qualification_requires_receipt(self):
        result, backends = self.run_one(qualification=True, G={"qualifier": False})
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("N", backends)
        self.assertFalse(backends["G"].finalized)

    def test_six_stream_counts(self):
        totals = [0, 0, 0]
        for name in e.EXPECTED_STREAMS:
            source = stream(name)
            result = e.run_stream(source, references(source), lambda arm, name: Backend(arm))
            totals[0] += len(result["scores"])
            totals[1] += len(result["updates"])
            totals[2] += sum(len(a["canonical"]) for a in result["arms"].values())
        self.assertEqual(totals, [60, 48, 72])
        plan = e.plan([stream(name) for name in e.EXPECTED_STREAMS])
        self.assertEqual(plan["optimizer_steps"], 96)
        self.assertEqual(plan["qualification"]["scored_replays"], 10)

    def test_reference_drift_rejected_before_backend(self):
        source = stream()
        refs = references(source)
        refs[2]["input_ids"].append(3)
        with self.assertRaisesRegex(ValueError, "token hash"):
            e.run_stream(source, refs, lambda *_: self.fail("must not start backend"))


if __name__ == "__main__":
    unittest.main()
