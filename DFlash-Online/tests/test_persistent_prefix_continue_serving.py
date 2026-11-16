"""CPU checks for the serving-only continuation; no native child is run."""
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import continue_serving as M
from experiments.persistent_prefix_withholding import serving_qualification as S
from experiments.persistent_prefix_withholding import admission as A
from experiments.persistent_prefix_withholding import campaign as C
from online_feedback.storage import write_json
from test_persistent_prefix_launch import training_receipts, serving_receipts


class ServingContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root, self.old = base/"new", base/"prior"
        self.root.mkdir()
        self.plan = dict(root=str(self.root), source_root=str(base), execution_commit="new",
            target="target", origin_export="K6", training_python="train_python", worker_python="serve_python",
            devices={"A":"GPU-A", "B":"GPU-B"}, runtime_contract={"fixed":True}, compiler_root="cache",
            compiler_baseline_allocated_bytes=5022478336, compiler_growth_reserved_bytes=4*2**30,
            warmups=str(base/"warmups.json"), warmup_references=str(base/"references.json"))
        write_json(self.plan["warmups"], dict(requests=["short", "long"]))
        write_json(self.plan["warmup_references"], dict(canonical=True))
        write_json(self.old/"sealed/PLAN.json", dict(self.plan, root=str(self.old), execution_commit="old"))
        self.clock = dict(first_model_load_epoch=time.time()-2000, completed_card_s=367.07596135139465,
                         active={}, finished=[dict(occupied_card_s=367.07596135139465)])
        write_json(self.old/"CLOCK.json", self.clock)
        write_json(self.old/"REQUALIFICATION.json", dict(status="stopped", execution_commit="old",
            production_launched=False, completed_phases=["cpu_preflight", "training_A", "training_B"]))
        for slot, report in training_receipts().items():
            write_json(self.old/"qualification"/slot/"STATUS.json", dict(report, execution_commit="old"))
        write_json(self.old/"qualification/CROSS_CARD.json", dict(status="passed", execution_commit="old",
                                                                 attempted_optimizer_applications=8))
        write_json(self.old/"serving_qualification/A/STATUS.json", dict(status="failed", execution_commit="old",
            error="SGLang startup failed", traceback="backend = SGLangDFlashBackend(\n__init__ failed"))
        runtime = self.old/"serving_qualification/A/runtime"
        runtime.mkdir(parents=True)
        (runtime/"worker.log").write_text("/bin/sh: c++: not found")
        write_json(runtime/"PHYSICAL_WORK.json", dict(initialization=[dict(model_forward_calls=1,
                    model_forward_completions=0, model_forward_failures=1)], physical={}))
        counters = {key: 0 for key in M.LIMITS}
        counters.update(adam=10, response_forward=26, response_backward=26, training_initialization=3)
        write_json(self.old/"CUMULATIVE_WORK.json", dict(cumulative=dict(attempted=counters, completed=counters,
            failed={key:0 for key in counters}, unknown={key:0 for key in counters})))
        authorization = base/"authority.json"
        write_json(authorization, dict(scope="serving_qualification_only", prior_root=str(self.old),
            new_root=str(self.root), new_generations=8, new_adam_steps=0, new_response_forwards=0,
            new_response_backwards=0, new_training_initializations=0, preserve_original_clock=True,
            main_campaign_authorized=False))
        self.plan["serving_continuation"] = dict(prior_root=str(self.old), execution_commit="old",
            authorization=dict(path=str(authorization), identity=M.file_identity(authorization)))
        self.plan_path = self.root/"sealed/PLAN.json"
        self.reseal()
        storage = patch.object(M, "storage_boundary", return_value=dict(status="passed"))
        self.storage = storage.start()
        self.addCleanup(storage.stop)

    def reseal(self):
        self.plan["serving_continuation"]["files"] = {
            name:M.file_identity(self.old/name) for name in M.PRIOR_FILES}
        write_json(self.plan_path, self.plan)

    def cpu_pass(self, *args, **kwargs):
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        write_json(self.root/"CPU_PREFLIGHT.json", dict(status="passed", execution_commit="new"))

    def serving_pass(self, path, plan, kind, slot):
        self.assertEqual(kind, "serving")
        receipt = serving_receipts()[slot]
        write_json(self.root/"serving_qualification"/slot/"calls/RESULT.json", receipt["qualification"])
        return receipt

    def test_only_eight_serving_calls_and_retained_training_used_for_pricing(self):
        original = {name:(self.old/name).read_bytes() for name in M.PRIOR_FILES}
        with patch.object(M.subprocess, "run", side_effect=self.cpu_pass), \
                patch.object(M, "run_qualification_child", side_effect=self.serving_pass) as child, \
                patch.object(S, "compare_serving_cards", return_value=dict(status="passed")), \
                patch.object(A, "qualify_resources", return_value=dict(status="passed")) as pricing, \
                patch.object(C, "run_campaign") as production:
            result = M.run(self.plan_path)
        self.assertEqual(result["status"], "qualified")
        self.assertEqual([call.args[-2:] for call in child.call_args_list], [("serving","A"),("serving","B")])
        self.assertTrue(all(row["execution_commit"] == "old" for row in pricing.call_args.args[1].values()))
        production.assert_not_called()
        self.assertFalse((self.root/"qualification").exists())
        self.assertEqual(M.read(self.root/"CLOCK.json"), self.clock)
        work = M.read(self.root/"CUMULATIVE_WORK.json")
        self.assertEqual(work["attempt"]["completed"], M.SERVING_BUDGET)
        self.assertEqual(work["cumulative"]["completed"]["adam"], 10)
        self.assertEqual(work["cumulative"]["completed"]["response_forward"], 26)
        self.assertEqual(work["cumulative"]["completed"]["qualification_generation"], 8)
        self.assertTrue(all(value == 0 for value in work["unattempted"].values()))
        self.assertEqual({name:(self.old/name).read_bytes() for name in M.PRIOR_FILES}, original)
        self.assertEqual(M.read(self.root/"PRIOR_QUALIFICATION.json")["prior_graph_work"]["initialization"][0]["model_forward_failures"], 1)

    def test_tampered_prior_or_unpassed_training_rejected(self):
        path = self.old/"qualification/A/STATUS.json"
        row = M.read(path)
        row["status"] = "failed"
        write_json(path, row)
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            M.authenticate(self.plan)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "training work"):
            M.authenticate(self.plan)

    def test_generations_cannot_be_assumed_zero_for_later_failure(self):
        path = self.old/"serving_qualification/A/STATUS.json"
        row = M.read(path)
        row["loading_s"] = 12
        write_json(path, row)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "precede request"):
            M.authenticate(self.plan)

    def test_active_lease_and_compiler_baseline_change_rejected(self):
        self.plan["compiler_baseline_allocated_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "qualified setting"):
            M.authenticate(self.plan)
        self.plan["compiler_baseline_allocated_bytes"] -= 1
        write_json(self.old/"CLOCK.json", dict(self.clock, active={"worker":{}}))
        self.reseal()
        with self.assertRaisesRegex(ValueError, "active or unaccounted"):
            M.authenticate(self.plan)

    def test_cpu_failure_calls_no_native_child(self):
        with patch.object(M.subprocess, "run", side_effect=subprocess.CalledProcessError(1,"cpu")), \
                patch.object(M, "run_qualification_child") as child:
            with self.assertRaises(subprocess.CalledProcessError):
                M.run(self.plan_path)
        child.assert_not_called()
        self.assertEqual(M.read(self.root/"CUMULATIVE_WORK.json")["cumulative"]["completed"]["adam"], 10)

    def test_failed_first_serving_child_stops_without_training_or_retry(self):
        def fail(path, plan, kind, slot):
            self.assertEqual((kind,slot), ("serving","A"))
            write_json(self.root/"serving_qualification/A/calls/RESULT.json",
                       dict(attempted_generations=1, returned_generations=0, unknown_generations=1))
            raise RuntimeError("serving failure")
        with patch.object(M.subprocess, "run", side_effect=self.cpu_pass), \
                patch.object(M, "run_qualification_child", side_effect=fail) as child:
            with self.assertRaisesRegex(RuntimeError, "serving failure"):
                M.run(self.plan_path)
        child.assert_called_once()
        work = M.read(self.root/"CUMULATIVE_WORK.json")
        self.assertEqual(work["cumulative"]["completed"]["adam"], 10)
        self.assertEqual(work["attempt"]["unknown"]["qualification_generation"], 1)
        self.assertEqual(work["unattempted"]["qualification_generation"], 7)
        with self.assertRaises(FileExistsError):
            M.run(self.plan_path)


if __name__ == "__main__":
    unittest.main()
