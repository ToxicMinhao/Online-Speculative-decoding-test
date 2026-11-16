"""CPU scheduling and accounting checks; no model construction or CUDA use."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import campaign as C, task_runner as T
from experiments.persistent_prefix_withholding.resources import LIMITS, HOST_HEADROOM_BYTES


def costs():
    result = {name: 1.0 for name in C.REQUIRED_COSTS}
    result.update(training_event=400., training_startup=20., training_load=30., publication=15., serving_load=20.,
                  activation=.2, scored_request=.5, warmup_request=.5, resident_wait=5.)
    return result


class SchedulingTests(unittest.TestCase):
    def test_exact_task_budget_and_own_dependencies(self):
        tasks = C.task_graph(costs(), 24)
        indexed = {task.name: task for task in tasks}
        self.assertEqual(set(indexed), {"E7", "S7", "N_collect", "E_collect", "S_collect", "N8", "E8", "S8", "references", "final"})
        self.assertEqual(indexed["N_collect"].dependencies, ())
        self.assertEqual(indexed["E_collect"].dependencies, ("E7",))
        self.assertEqual(indexed["E8"].dependencies, ("E_collect",))
        self.assertEqual(set(indexed["final"].dependencies), {"N8", "E8", "S8", "references"})
        self.assertEqual(indexed["final"].components["scored"], 1152 * .5)
        self.assertEqual(indexed["final"].components["warmups"], 72 * .5)
        self.assertAlmostEqual(indexed["final"].components["activations"], 36 * .2)
        self.assertEqual(indexed["references"].components["references"], 24.)
        self.assertEqual([task.name for task in C.ordered_ready(tasks, set(), set())][:2], ["E7", "S7"])
        self.assertIn("E_collect", [task.name for task in C.ordered_ready(tasks, {"E7"}, {"S7"})])

    def test_missing_cost_and_nonfinite_cost_fail_before_launch(self):
        for key in C.REQUIRED_COSTS:
            value = costs()
            del value[key]
            with self.assertRaisesRegex(ValueError, "category"):
                C.task_graph(value, 24)
        value = costs()
        value["activation"] = float("nan")
        with self.assertRaises(ValueError):
            C.task_graph(value, 24)
        with self.assertRaises(ValueError):
            C.task_graph(costs(), 129)

    def test_production_counts_include_discarded_startup_but_no_extra_adam(self):
        work = C.production_expectations(24)
        self.assertEqual(work["adam"], 640)
        self.assertEqual(work["response_forward"], 1280 + 10)
        self.assertEqual(work["response_backward"], 1280 + 10)
        self.assertEqual(work["training_initialization"], 5)
        self.assertEqual(work["qualification_generation"], 0)
        self.assertEqual(sum(work[k] for k in ("feedback_generation", "reference_generation", "warmup_generation", "scored_generation")), 1632)

    def test_dry_schedule_uses_two_workers_and_final_barrier(self):
        tasks = [C.Task("A", (), "training", {"compute": 10.}),
                 C.Task("B", (), "training", {"compute": 20.}),
                 C.Task("C", ("A",), "collection", {"compute": 5.}),
                 C.Task("final", ("B", "C"), "final", {"compute": 7.})]
        self.assertEqual(C.planned_elapsed(tasks), 27.)

    def test_memory_reserves_remaining_peak_of_an_active_worker(self):
        class Process:
            pid = 123
        plan = dict(task_memory_bytes={"training": 50, "collection": 30})
        task = C.Task("N_collect", (), "collection", {})
        active = {"E7": dict(task=C.Task("E7", (), "training", {}), process=Process())}
        with patch.object(C, "process_tree_rss", return_value=20), patch.object(C, "current_memory", return_value=HOST_HEADROOM_BYTES+59):
            receipt = C.memory_admission(plan, task, active)
            self.assertEqual(receipt["status"], "serialize")
            self.assertEqual(receipt["active_future_growth_bytes"], 30)
            self.assertEqual(receipt["required_host_bytes"], HOST_HEADROOM_BYTES+60)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.initial = dict(attempted={key: 0 for key in LIMITS}, completed={key: 0 for key in LIMITS},
                            pending={}, next_operation=0)
        T.write_json(self.root/"WORK.json", self.initial)
        T.write_json(self.root/"CLOCK.json", dict(first_model_load_epoch=100., completed_card_s=20., active={}, finished=[]))

    def test_shared_counts_do_not_reset_between_workers(self):
        first = T.SharedWork(self.root, "E7")
        second = T.SharedWork(self.root, "S7")
        self.assertEqual(first.call({"adam": 1}, "step", lambda: 7), 7)
        second.call({"adam": 1}, "step", lambda: None)
        state = T.read(self.root/"WORK.json")
        self.assertEqual(state["attempted"]["adam"], 2)
        self.assertEqual(state["completed"]["adam"], 2)
        self.assertEqual(state["pending"], {})

    def test_failed_call_keeps_unknown_work_and_cap_before_call(self):
        ledger = T.SharedWork(self.root, "E7")
        def fails():
            raise RuntimeError("native failure")
        with self.assertRaises(RuntimeError):
            ledger.call({"response_forward": 1}, "forward", fails)
        state = T.read(self.root/"WORK.json")
        self.assertEqual(state["attempted"]["response_forward"], 1)
        self.assertEqual(state["completed"]["response_forward"], 0)
        self.assertEqual(len(state["pending"]), 1)
        called = []
        with self.assertRaisesRegex(RuntimeError, "cap"):
            ledger.call({"adam": LIMITS["adam"]+1}, "oversized", lambda: called.append(True))
        self.assertEqual(called, [])

    def test_occupied_time_sums_cards_and_preserves_prior_qualification(self):
        with patch.object(T.time, "time", return_value=110.):
            with T.occupied_card(self.root, "E7", "GPU-A"):
                with patch.object(T.time, "time", return_value=115.):
                    with T.occupied_card(self.root, "S7", "GPU-B"):
                        now = T.clock_status(self.root, now=120.)
                        self.assertEqual(now["occupied_s"], 35.)
                        self.assertEqual(now["elapsed_s"], 20.)
                        with self.assertRaisesRegex(RuntimeError, "already owns"):
                            with T.occupied_card(self.root, "other", "GPU-A"):
                                pass
        self.assertEqual(T.read(self.root/"CLOCK.json")["active"], {})

    def test_native_original_n7_is_never_a_new_owned_endpoint(self):
        plan = dict(root=str(self.root), native_n7_state="/original/state", native_n7_export="/original/export")
        self.assertEqual(T.endpoint(plan, "N", 7), dict(state="/original/state", export="/original/export", historical=True))
        with self.assertRaises(FileNotFoundError):
            T.endpoint(plan, "E", 7)

    def test_wrong_physical_binding_stops_before_creating_task(self):
        plan = dict(root=str(self.root), devices={"A": "GPU-A", "B": "GPU-B"})
        T.write_json(self.root/"PLAN.json", plan)
        with patch.dict(T.os.environ, CUDA_VISIBLE_DEVICES="0"):
            with self.assertRaisesRegex(ValueError, "physical GPU"):
                T.run_task(self.root/"PLAN.json", "E7", "GPU-A")
        self.assertFalse((self.root/"tasks").exists())

    def test_training_pins_runtime_before_constructing_model(self):
        plan = dict(root=str(self.root), origin_state="source.pt", origin_export="export",
            event7_feature_paths=["feature.pt"]*128, configs={"E":"config.yaml"}, binding="sealed",
            origin_config=str(self.root/"original.yaml"), event7_schedule=str(self.root/"schedule.json"))
        Path(plan["origin_config"]).write_text("training: {}\n")
        T.write_json(plan["event7_schedule"], {})
        observed = []
        class Ledger:
            def call(self, counts, label, function, *args, **kwargs):
                observed.append(label)
                raise RuntimeError("construction stopped in CPU fixture")
        with patch("experiments.prefix_guard.qualify.pin_retained_graph_runtime", autospec=True,
                   side_effect=lambda root: observed.append(("pin", root))):
            with self.assertRaisesRegex(RuntimeError, "construction stopped"):
                T.train_task(plan, "E7", self.root, Ledger())
        self.assertEqual(observed, [("pin", self.root), "trainer_initialization"])

    def test_compiler_cache_charges_growth_against_original_baseline(self):
        plan = dict(compiler_root=str(self.root), compiler_baseline_allocated_bytes=100,
                    compiler_growth_reserved_bytes=40)
        with patch.object(T.subprocess, "check_output", return_value="140\tcache\n"):
            self.assertEqual(T.compiler_cache_growth(plan)["growth_bytes"],40)
        with patch.object(T.subprocess, "check_output", return_value="141\tcache\n"):
            with self.assertRaisesRegex(RuntimeError,"growth exceeded"):
                T.compiler_cache_growth(plan)
        with patch.object(T.subprocess, "check_output", return_value="99\tcache\n"):
            self.assertEqual(T.compiler_cache_growth(plan)["growth_bytes"],0)

    def test_storage_preserves_internal_source_symlinks_but_rejects_run_links(self):
        source=self.root/"source"
        (source/"docs/web").mkdir(parents=True)
        (source/"docs/data").mkdir()
        (source/"docs/web/data").symlink_to("../data")
        manifest=self.root/"SOURCE.json"
        T.write_json(manifest,dict(commit="current",dirty=False,
            files={"docs/web/data":dict(kind="symlink",target="../data")}))
        plan=dict(root=str(self.root),source_root=str(source),allocation_roots=[str(self.root)],
                  source_manifest=str(manifest),execution_commit="current",
                  storage={"peak_new_bytes":1},compiler_growth_reserved_bytes=0)
        from types import SimpleNamespace
        with patch.object(T.shutil,"disk_usage",return_value=SimpleNamespace(free=10**12)):
            self.assertEqual(T.storage_boundary(plan,check_compiler=False)["status"],"passed")
            (self.root/"run_state").symlink_to(source/"docs/data")
            with self.assertRaisesRegex(ValueError,"internal source"):
                T.storage_boundary(plan,check_compiler=False)
            (self.root/"run_state").unlink()
            (source/"external").symlink_to(self.root)
            with self.assertRaisesRegex(ValueError,"internal source"):
                T.storage_boundary(plan,check_compiler=False)

    def test_reserved_cache_bytes_still_require_future_free_capacity(self):
        from types import SimpleNamespace
        plan=dict(root=str(self.root),source_root=str(self.root/"source"),allocation_roots=[str(self.root)],
                  storage={"peak_new_bytes":1_000_000},compiler_growth_reserved_bytes=40)
        with patch.object(T.shutil,"disk_usage",return_value=SimpleNamespace(free=10**12)):
            receipt=T.storage_boundary(plan,check_compiler=False)
        physical=receipt["allocated_new_bytes"]-40
        self.assertEqual(receipt["projected_remaining_peak_bytes"],1_000_000-physical)

    def test_collection_prepares_and_retires_each_response_before_the_next(self):
        T.write_json(self.root/"requests.json", dict(requests=[dict(request_id=f"q{i}",
            request_index=i, prompt_ids=[1, 2]) for i in range(128)]))
        plan = dict(root=str(self.root), event8_inputs=str(self.root/"requests.json"),
                    native_n7_state="state", native_n7_export="export",
                    storage={"temporary_feature_bytes":1})
        events = []
        class Backend:
            loading_s = 1.
            failed = False
            def generate(self, prompt, **kwargs):
                events.append(("generate", kwargs["request_id"]))
                return dict(record_id=kwargs["request_id"])
            def prepare_feedback(self, record, path):
                events.append(("prepare", record["record_id"]))
                return dict(status="prepared", feature_path=str(path))
            def _call(self, operation):
                events.append((operation, None))
                return dict(status="passed")
            def close(self):
                events.append(("close", None))
        base = self.root/"collection"
        base.mkdir()
        with patch.object(T, "create_backend", return_value=Backend()), patch.object(T, "work_boundary"), \
             patch("experiments.full_k_gsm8k.campaign.online_record_check"), \
             patch("experiments.kplus_full_gsm8k.storage.retire_redundant_features",
                   side_effect=lambda record,*args: events.append(("retire", record["record_id"]))):
            result = T.collect_task(plan, "N_collect", base, T.SharedWork(self.root,"N_collect"), "GPU-A")
        self.assertEqual(len(result["feature_paths"]), 128)
        self.assertEqual(events[:6], [(kind, f"q{i}") for i in range(2) for kind in ("generate","prepare","retire")])
        self.assertEqual(events[-2:], [("work_receipt",None),("close",None)])
        work = T.read(self.root/"WORK.json")
        self.assertEqual(work["completed"]["feedback_generation"],128)
        self.assertEqual(work["completed"]["teacher_preparation"],128)
        self.assertEqual(work["pending"],{})


if __name__ == "__main__":
    unittest.main()
