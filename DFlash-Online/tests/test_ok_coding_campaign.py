"""CPU orchestration guards: sealed scope, fresh exec caches and partial work."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from online_feedback.storage import write_json
from experiments.ok_coding import campaign as C
from experiments.ok_coding.contract import (
    NEW_BUDGET, data_binding, evaluation_schedule, training_contract, validate_plan,
)


def requests(count=128, *, training=False):
    return [{"request_id": ("train" if training else "eval") + str(index),
             "request_index": index, "panel_ordinal": index,
             "original_index": index + (0 if training else 20000),
             "prompt_ids": [1, index + (1000 if training else 2000)],
             "stop_token_ids": [151645, 151643], "max_new_tokens": 2048}
            for index in range(count)]


def plan():
    training, evaluation = requests(512, training=True), requests()
    return {"configs": {"17": "/seed17.yaml", "29": "/seed29.yaml"},
            "windows": [{"event_index": event, "event_id": f"event_{event:03d}",
                         "requests": training[(event - 1) * 128:event * 128]} for event in range(1, 5)],
            "evaluation_requests": evaluation, "data_binding": data_binding(training, evaluation),
            "evaluation_schedule": evaluation_schedule(evaluation), "new_budget": dict(NEW_BUDGET),
            "training_seeds": [17, 29], "stop_step": 512, "inherited_horizon": 7424,
            "no_automatic_retry": True, "execution_source": {"commit": "a" * 40},
            "draft": "/F", "target": "/target", "worker_python": "/native/python"}


class ContractTests(unittest.TestCase):
    def test_fixed_contract_and_pinned_complete_schedule(self):
        fixed = plan()
        self.assertEqual(validate_plan(fixed)["status"], "passed")
        self.assertEqual(len(fixed["evaluation_schedule"]), 4608)
        self.assertEqual(training_contract(fixed, 29)["training_seed"], 29)
        self.assertEqual(training_contract(fixed, 17)["data_binding"], training_contract(fixed, 29)["data_binding"])

    def test_scope_labels_duplicates_and_wrong_order_fail(self):
        for mutation in (
            lambda p: p.update(stop_step=640),
            lambda p: p["new_budget"].update(qualification_model_work=1),
            lambda p: p["evaluation_schedule"].reverse(),
            lambda p: p["evaluation_requests"][0].update(output="answer"),
            lambda p: p["windows"][0]["requests"][0].update(prompt_ids=[1] * 225),
            lambda p: p["windows"][0]["requests"][0].update(request_index=1),
            lambda p: p["evaluation_requests"][0].update(prompt_ids=p["windows"][0]["requests"][0]["prompt_ids"]),
        ):
            changed = plan()
            mutation(changed)
            with self.assertRaises(ValueError):
                validate_plan(changed)


class WorkTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_unknown_action_is_charged_and_not_retried(self):
        ledger = C.WorkLedger(self.root / "WORK.jsonl", {"backwards": 1})
        with self.assertRaisesRegex(RuntimeError, "native failed"):
            ledger.call("backwards", lambda: (_ for _ in ()).throw(RuntimeError("native failed")))
        observed = C.read_work(ledger.path, ledger.limits)
        self.assertEqual(observed, {"attempts": {"backwards": 1}, "completed": {"backwards": 0},
                                   "unknown": {"backwards": 1}})
        with self.assertRaisesRegex(RuntimeError, "ceiling"):
            ledger.call("backwards", lambda: None)
        with self.assertRaises(ValueError):
            C.require_complete_work(observed, ledger.limits)
        with self.assertRaises(FileExistsError):
            C.WorkLedger(ledger.path, ledger.limits)

    def test_completed_actions_have_exact_starts_and_completion(self):
        ledger = C.WorkLedger(self.root / "WORK.jsonl", {"optimizer_steps": 2})
        for _ in range(2):
            self.assertEqual(ledger.call("optimizer_steps", lambda: 42), 42)
        C.require_complete_work(C.read_work(ledger.path, ledger.limits), ledger.limits)
        with ledger.path.open("a") as stream:
            stream.write(json.dumps({"kind": "optimizer_steps", "phase": "completed", "ordinal": 2}) + "\n")
        with self.assertRaisesRegex(ValueError, "completion"):
            C.read_work(ledger.path, ledger.limits)

    def test_complete_campaign_rejects_unknown_even_when_completed_counters_match(self):
        expected = {"forwards": 4096, "backwards": 4096, "optimizer_steps": 2048,
                    "feedback": 1024, "terminal_teacher_forwards": 1024}
        status = {"work": {"attempts": expected, "completed": expected,
                            "unknown": {key: 0 for key in expected}},
                  "arm_counters": {f"{seed}/{arm}": 512 for seed in (17, 29) for arm in ("O", "K")},
                  "arm_presentations": {f"{seed}/{arm}": 1024 for seed in (17, 29) for arm in ("O", "K")},
                  "evaluations": {f"{seed}/{event}": {} for seed, event in C.CELLS},
                  **{"evaluation_" + key: value for key, value in {
                      "attempted": 4608, "returned": 4608, "valid": 4608,
                      "unknown": 0, "invalid": 0, "pending": 0}.items()}}
        C.require_complete_campaign(status)
        changed = deepcopy(status)
        changed["work"]["unknown"]["optimizer_steps"] = 1
        with self.assertRaises(ValueError):
            C.require_complete_campaign(changed)

    def test_fresh_process_cache_policy_rejects_reuse_and_multiple_devices(self):
        environment = {"CUDA_VISIBLE_DEVICES": "1", "HF_HOME": "/immutable/models"}
        first = C.fresh_worker_environment(self.root / "O", environment)
        second = C.fresh_worker_environment(self.root / "K", environment)
        self.assertEqual(first["HF_HOME"], "/immutable/models")
        for name in C.CACHE_VARIABLES:
            self.assertNotEqual(first[name], second[name])
            self.assertEqual(list(Path(first[name]).iterdir()), [])
        with self.assertRaises(FileExistsError):
            C.fresh_worker_environment(self.root / "O", environment)
        for visible in ("", "0,1"):
            with self.assertRaises(ValueError):
                C.fresh_worker_environment(self.root / "bad", {"CUDA_VISIBLE_DEVICES": visible})

    def test_explicit_recovery_refuses_partial_scientific_work_before_backend(self):
        fixed = plan()
        output = self.root / "campaign_001"
        (output / "seed_17" / "event_001").mkdir(parents=True)
        write_json(output / "STATUS.json", {"status": "failed"})
        write_json(output / "PLAN_CHECK.json", validate_plan(fixed))
        ledger = C.WorkLedger(output / "seed_17" / "event_001_WORK.jsonl",
                             {"feedback": 128, "terminal_teacher_forwards": 128})
        ledger.call("feedback", lambda: None)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            C.admit_recovery(self.root, fixed)
        self.assertFalse((output / "recovery").exists())


if __name__ == "__main__":
    unittest.main()
