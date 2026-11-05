"""CPU integration stubs for the first native qualification/pricing milestone."""

from collections import Counter
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from online_feedback.storage import write_json
from online_feedback.sglang_backend import SGLangDFlashBackend
from experiments.block_depth import qualify as Q, pricing as P, worker as W


def requests():
    return [dict(request_id=f"readout-{i}", prompt_ids=[i + 1, 88],
                 max_new_tokens=2048, stop_token_ids=[151645, 151643])
            for i in range(16)]


def generated(request, mode):
    response = [42] + list(range(100, 115)) + [151645]
    progress = 1 if mode in ("THREE", "ALTERNATE") else 2
    blocks = []
    for start in range(0, len(response) - 1, progress):
        committed = response[start + 1:start + 1 + progress]
        anchor = len(request["prompt_ids"]) + start
        blocks.append(dict(
            output_start=start, anchor_token=response[start],
            positions=list(range(anchor, anchor + 16)),
            draft_tokens=(committed + [999] * 15)[:15],
            committed_tokens=committed, committed_progress=len(committed),
            accepted_draft_tokens=len(committed),
        ))
    return dict(
        prompt_ids=request["prompt_ids"], response_ids=response, finish_reason="eos",
        max_new_tokens=2048, stop_token_ids=request["stop_token_ids"],
        metadata=dict(
            trace=blocks, verification_calls=len(blocks), accepted_draft_tokens=16,
            committed_progress=16, request_latency_s=0.1, native_generate_s=0.09,
            capture=dict(cache_maintenance_s=0.001), feedback_capture_in_timed_request=False,
        ),
    )


RESTORATION = dict.fromkeys(("request_row_ids", "request_mapping", "allocator",
                             "native_graph_inputs", "attention_metadata", "cache_bytes"), True)


class FakeBackend:
    """No model/runtime construction: only the experiment's process contract."""

    def __init__(self, *, scenario=None):
        self.scenario, self.mode = scenario, None
        self.loading_s = 0.5
        self.generations, self.calls, self.weight_checks = [], [], 0
        self.closed = False

    def check_weights(self, path):
        self.weight_checks += 1
        bad = self.scenario == f"weights_{self.weight_checks}"
        return dict(status="passed", all_equal=not bad, checkpoint_path=path)

    def _call(self, operation, name, arguments):
        self.calls.append(name)
        path = Path(arguments["outputfile"])
        if name == "depth_mode":
            self.mode = arguments["mode"]
            result = dict(status="passed", mode=self.mode)
        elif name == "depth_status":
            result = dict(status="failed" if self.scenario == "route_status" else "passed")
        elif name == "depth_profile":
            record, ordinal = arguments["record"], arguments["prompt_ordinal"]
            assert self.mode == "FIVE"
            assert self.generations[-1]["mode"] == "FIVE"
            cohort = [dict(request_id=f"readout-{i}", observed_rounds=8) for i in range(16)]
            schedule = [row for row in P.pricing_schedule(cohort) if row["prompt_ordinal"] == ordinal]
            times = dict(FIVE=5.0, THREE=3.0, ADAPTIVE3=3.5, ADAPTIVE5=5.5)
            rows = [dict(row, valid=True, elapsed_ms=times[row["route"]]) for row in schedule]
            result = dict(status="passed", measurements=rows,
                          restoration_status="passed", restoration_checks=dict(RESTORATION))
            if self.scenario == "invalid_measurement":
                rows[0]["valid"] = False
            if self.scenario == "invalid_restoration":
                result["restoration_checks"]["cache_bytes"] = False
            if self.scenario in ("profile_partial", "profile_missing"):
                if self.scenario == "profile_partial":
                    write_json(path, dict(status="failed", measurements=rows[:1],
                                          accounting=dict(planned_timed_measurements=128,
                                                          returned_timed_measurements=1,
                                                          unknown_timed_measurements=1,
                                                          unattempted_timed_measurements=126)))
                raise RuntimeError("profile interrupted")
        else:
            raise AssertionError(name)
        write_json(path, result)
        return dict(result=result)

    def generate(self, prompt_ids, **kwargs):
        if self.scenario == "generation_exception" and self.generations:
            raise RuntimeError("generation transport interrupted")
        request = dict(request_id=kwargs["request_id"], prompt_ids=prompt_ids,
                       max_new_tokens=kwargs["max_new_tokens"], stop_token_ids=[151645, 151643])
        record = generated(request, self.mode)
        if self.scenario == "forced_proposal" and self.mode == "FORCED5":
            record["metadata"]["trace"][0]["draft_tokens"][0] += 1
        if self.scenario == "invalid_counts":
            record["metadata"]["accepted_draft_tokens"] = 999
        self.generations.append(dict(mode=self.mode, purpose=kwargs["purpose"],
                                     request_id=kwargs["request_id"], record=record))
        write_json(kwargs["save_to"], record)
        return record

    def close(self):
        self.closed = True


class QualificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        manifest = self.root / "MANIFESTS.json"
        write_json(manifest, dict(manifests=dict(READOUT=dict(requests=requests()))))
        self.plan = dict(manifest_file=str(manifest), target_path="/target", checkpoint_path="/KPLUS58",
                         worker_python="/qualified/python")

    def run_stub(self, scenario=None):
        self.backend = FakeBackend(scenario=scenario)
        with patch("experiments.block_depth.preflight.preflight", return_value=dict(status="passed")), \
             patch.object(Q, "SGLangDFlashBackend", return_value=self.backend) as constructor:
            Q.run(self.plan, self.root / "run")
        self.assertEqual(constructor.call_args.kwargs["worker_module"], "experiments.block_depth.worker")
        return json.loads((self.root / "run/STATUS.json").read_text())

    def status(self):
        return json.loads((self.root / "run/STATUS.json").read_text())

    def test_complete_milestone_has_exact_scope_and_preserves_both_depth_paths(self):
        status = self.run_stub()
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["returned_generations"], 32)
        self.assertEqual(status["valid_generations"], 32)
        self.assertEqual(status["readout_generations"], 16)
        self.assertEqual(status["qualification_generations"], 16)
        self.assertEqual(status["returned_route_measurements"], 2048)
        self.assertEqual((status["optimizer_steps"], status["fitting_presentations"], status["final_generations"]), (0, 0, 0))
        self.assertEqual(Counter(row["mode"] for row in self.backend.generations),
                         dict(FIVE=16, FORCED5=4, THREE=4, ALTERNATE=8))
        self.assertEqual(set(row["purpose"] for row in self.backend.generations), {"evaluation"})
        self.assertTrue(status["stopped_before_fitting"])
        self.assertTrue(self.backend.closed)
        self.assertEqual(self.backend.weight_checks, 2)
        pricing = json.loads((self.root / "run/PRICING.json").read_text())
        self.assertEqual(pricing["status"], "complete")
        self.assertEqual(pricing["decision"], "continue_to_readout_fit")
        # Passing the economic gate still stops at this authorized milestone.
        self.assertEqual(len(self.backend.generations), 32)

    def test_forced_five_checks_all_proposals_and_preserves_invalid_raw_record(self):
        with self.assertRaisesRegex(RuntimeError, "parity"):
            self.run_stub("forced_proposal")
        status = self.status()
        self.assertEqual((status["returned_generations"], status["invalid_generations"], status["unattempted_generations"]), (2, 1, 30))
        self.assertEqual(status["unknown_generations"], 0)
        self.assertTrue((self.root / "run/00_qualification_FORCED5.json").exists())
        parity = json.loads((self.root / "run/00_qualification_FORCED5_PARITY.json").read_text())
        self.assertIn("round_0_draft_tokens_mismatch", parity["errors"])

    def test_shallow_route_may_change_calls_but_must_match_output(self):
        request = requests()[0]
        five, three = generated(request, "FIVE"), generated(request, "THREE")
        self.assertEqual(Q.parity_errors(five, three, forced_five=False), [])
        self.assertIn("verification_count_mismatch", Q.parity_errors(five, three, forced_five=True))
        three["response_ids"][2] += 1
        self.assertIn("response_ids_mismatch", Q.parity_errors(five, three, forced_five=False))

    def test_invalid_readout_measurement_stops_before_any_profile(self):
        with self.assertRaises(RuntimeError):
            self.run_stub("invalid_counts")
        status = self.status()
        self.assertEqual(status["invalid_generations"], 1)
        self.assertEqual(status["returned_route_measurements"], 0)
        self.assertEqual(status["unattempted_generations"], 31)
        self.assertTrue((self.root / "run/00_READOUT_FIVE.json").exists())

    def test_weight_receipt_failure_is_checked_before_first_generation(self):
        with self.assertRaisesRegex(RuntimeError, "weight"):
            self.run_stub("weights_1")
        self.assertEqual(self.status()["returned_generations"], 0)
        self.assertEqual(self.status()["unattempted_generations"], 32)
        self.assertFalse(json.loads((self.root / "run/WEIGHTS_BEFORE.json").read_text())["all_equal"])

    def test_postrun_weight_failure_prevents_completed_result(self):
        with self.assertRaisesRegex(RuntimeError, "weight"):
            self.run_stub("weights_2")
        self.assertEqual(self.status()["status"], "blocked")
        self.assertEqual(self.status()["returned_generations"], 32)

    def test_invalid_profile_stops_before_next_generation(self):
        for scenario in ("invalid_measurement", "invalid_restoration"):
            with self.subTest(scenario=scenario):
                # Each scenario owns its separate failed-run destination.
                existing = self.root / "run"
                if existing.exists():
                    existing.rename(self.root / scenario)
                with self.assertRaisesRegex(RuntimeError, "reconstruction"):
                    self.run_stub(scenario)
                self.assertEqual(self.status()["returned_generations"], 1)
                self.assertTrue((self.root / "run/00_PROFILE.json").exists())
                self.assertTrue((self.root / "run/PRICING_MEASUREMENTS.json").exists())

    def test_route_status_failure_does_not_pass_as_absent_errors(self):
        with self.assertRaisesRegex(RuntimeError, "audit"):
            self.run_stub("route_status")
        self.assertEqual(self.status()["returned_generations"], 2)
        self.assertEqual(self.status()["status"], "blocked")

    def test_partial_profile_preserves_one_unknown_measurement(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_stub("profile_partial")
        status = self.status()
        self.assertEqual(status["returned_generations"], 1)
        self.assertEqual(status["unknown_generations"], 0)
        self.assertEqual(status["unattempted_generations"], 31)
        self.assertEqual(status["returned_route_measurements"], 1)
        work = status["active_profile"]["partial_work_accounting"]
        self.assertEqual(work["unknown_timed_measurements"], 1)
        self.assertEqual(work["unattempted_timed_measurements"], 126)
        self.assertEqual(work["planned_timed_measurements"], sum(work[key] for key in
                         ("returned_timed_measurements", "unknown_timed_measurements", "unattempted_timed_measurements")))

    def test_profile_without_receipt_remains_unknown_not_zero_work(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_stub("profile_missing")
        self.assertEqual(self.status()["active_profile"]["unknown_measurement_count_bounds"], [0, 128])

    def test_generation_transport_failure_keeps_unknown_separate(self):
        with self.assertRaisesRegex(RuntimeError, "transport"):
            self.run_stub("generation_exception")
        status = self.status()
        self.assertEqual((status["returned_generations"], status["unknown_generations"], status["unattempted_generations"]), (1, 1, 30))


class WorkerBoundaryTests(unittest.TestCase):
    def test_standard_worker_stays_default(self):
        self.assertEqual(inspect.signature(SGLangDFlashBackend.__init__).parameters["worker_module"].default,
                         "online_feedback.sglang_worker")

    def test_profile_calls_idle_then_native_controller_without_generation(self):
        scheduler, routes = object(), Mock()
        record = dict(prompt_ids=[1], response_ids=[2])
        with patch.object(W.sglang_hooks, "strict_idle") as idle, patch.object(W, "controller", return_value=routes):
            W.depth_profile(scheduler, record, 3, "/tmp/profile-receipt.json")
        idle.assert_called_once_with(scheduler)
        routes.profile_completed_request.assert_called_once_with(
            record, prompt_ordinal=3, output_path=Path("/tmp/profile-receipt.json"))

    def test_worker_does_not_offer_training_or_arbitrary_rpc(self):
        worker = object.__new__(W.DepthWorker)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            worker.depth_rpc(dict(name="train", arguments={}))


if __name__ == "__main__":
    unittest.main()
