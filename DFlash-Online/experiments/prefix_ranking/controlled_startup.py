"""Reuse the completed controlled qualification and its exact initialization.

This module never reruns qualification. A single ordinary process warmup is
discarded before independently constructing the two main arms.
"""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

import torch

from .qualify import RankingObserver, WorkLog, override, run_window
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

TRIALS = (("warmup", "osd", 0.), ("Oa", "osd", 0.), ("Ob", "osd", 0.),
          ("U0", "osd_uniform", 0.), ("P0", "osd_prefix", 0.),
          ("U1", "osd_uniform", 1.), ("P1", "osd_prefix", 1.))
EXPECTED_WORK = dict(current_training_forward_completed=14, backward_completed=14,
                     temporary_optimizer_step_completed=7, response_presentations=14,
                     reference_training_forward_completed=4, detached_comparison_forward_completed=4)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_qualification(config, ordinary_receipt_path, qualification_receipt_path,
                         qualification_seal_path, qualification_runner_path,
                         diagnostic_runner_path, pinned_initial_path):
    result_path = Path(qualification_receipt_path)
    result = json.loads(result_path.read_text())
    seal = json.loads(Path(qualification_seal_path).read_text())
    if (result.get("status") != "passed" or result.get("baseline_tolerance") != "exact equality"
            or result.get("main_campaign_started") is not False
            or result.get("main_optimizer_steps") != 0 or result.get("generations") != 0):
        raise ValueError("A completed controlled qualification with unchanged exact gates is required")
    for key, expected in EXPECTED_WORK.items():
        if result.get("work_counts", {}).get(key) != expected:
            raise ValueError(f"Controlled qualification work differs: {key}")
        if key.endswith("_completed") and result["work_counts"].get(key.replace("_completed", "_attempted")) != expected:
            raise ValueError(f"Controlled qualification has incomplete work: {key}")
    trials = result.get("trials", [])
    if [(r.get("label"), r.get("objective"), r.get("ranking_lambda")) for r in trials] != list(TRIALS):
        raise ValueError("Controlled qualification trial sequence differs")
    gates = []
    for row in trials:
        label = row["label"]
        if row.get("status") != ("discarded_warmup" if label == "warmup" else "passed"):
            raise ValueError(f"Incomplete controlled qualification trial: {label}")
        if len(row.get("records", [])) != 2 or row.get("frozen_target_version_layout_unchanged") is not True:
            raise ValueError(f"Missing trial coverage: {label}")
        gates.append(f"{label}/initial_vs_pinned.json")
        if label in {"Ob", "U0", "P0"}:
            if row.get("exact_baseline_compatibility") is not True:
                raise ValueError(f"Missing exact compatibility: {label}")
            gates.extend(f"{label}/{stage}_vs_Oa.json" for stage in
                         ("initial", "after_backward_0", "after_backward_1", "pre_optimizer", "final", "records"))
        if label in {"U1", "P1"}:
            if (len(row.get("prefix_checks", [])) != 2 or len(row.get("ranking_gradient_checks", [])) != 2
                    or row.get("reference_rng_isolation") is not True):
                raise ValueError(f"Missing active-ranking coverage: {label}")
            gates.append(f"{label}/rng_vs_Oa.json")
    gate_hashes = {}
    for relative in gates:
        path = result_path.parent / relative
        if json.loads(path.read_text()).get("exact_equal") is not True:
            raise ValueError(f"Controlled qualification exact gate failed: {relative}")
        gate_hashes[relative] = sha256(path)
    bindings = ((config, result["config_sha256"]),
                (ordinary_receipt_path, result["ordinary_receipt_sha256"]),
                (pinned_initial_path, result["pinned_initial_sha256"]),
                (diagnostic_runner_path, result["diagnostic_runner_sha256"]),
                (qualification_runner_path, seal["diagnostic_sha256"]["runner.py"]))
    for path, expected in bindings:
        if sha256(path) != expected:
            raise ValueError(f"Qualified artifact hash differs: {path}")
    for key, digest in (("pinned_initial", result["pinned_initial_sha256"]),
                        ("diagnostic_runner", result["diagnostic_runner_sha256"])):
        if seal["dependency_sha256"].get(result[key]) != digest:
            raise ValueError(f"Qualified dependency and seal disagree: {key}")
    return dict(status="passed", reused=True, qualification_receipt=str(result_path),
                qualification_receipt_sha256=sha256(result_path),
                qualification_seal_sha256=sha256(qualification_seal_path),
                qualification_runner_sha256=bindings[-1][1], exact_gate_sha256=gate_hashes,
                pinned_initial_sha256=result["pinned_initial_sha256"],
                prior_qualification_wall_s=result["wall_s"], prior_work=result["work_counts"],
                historical_O_limit="Retained O is immutable; this startup does not establish equality to its full historical 128-step trajectory")


class ControlledStartup:
    def __init__(self, config, ordinary_receipt_path, qualification_receipt_path,
                 qualification_seal_path, qualification_runner_path,
                 diagnostic_runner_path, pinned_initial_path):
        self.receipt = verify_qualification(config, ordinary_receipt_path, qualification_receipt_path,
            qualification_seal_path, qualification_runner_path, diagnostic_runner_path, pinned_initial_path)
        spec = importlib.util.spec_from_file_location("passed_controlled_qualification", qualification_runner_path)
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)
        self.driver.load_helpers(diagnostic_runner_path)
        self.pinned = self.driver.load_state(pinned_initial_path)
        self.config = config
        self.ordinary = json.loads(Path(ordinary_receipt_path).read_text())
        self._warmup_started = self._warmup_done = False
        self._prepared = []

    def prepare(self, trainer, root):
        if not self._warmup_done:
            raise RuntimeError("Complete the single discarded process warmup before main training")
        expected = ("osd_uniform", "osd_prefix")
        if (len(self._prepared) >= 2 or trainer.objective != expected[len(self._prepared)]
                or trainer.ranking_lambda != 1. or not trainer.bundle.model.ranking_enabled):
            raise RuntimeError("Each main arm must be freshly prepared once, in U/P order")
        started = time.perf_counter()
        initial = self.driver.validate_fresh(trainer, self.pinned, Path(root))
        del initial
        self._prepared.append(trainer.objective)
        receipt = dict(exact_pinned_initial_state=True, full_rng_restored_after_construction=True,
                       state_path=str(Path(root) / "initial.pt"), wall_s=time.perf_counter() - started,
                       reference_created=False, inherited_warmup_state=False)
        write_json(Path(root) / "RESULT.json", receipt)
        return receipt

    def warmup(self, root):
        if self._warmup_started:
            raise RuntimeError("Process warmup may run only once; no automatic retries")
        self._warmup_started = True
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        log = WorkLog(root)
        log.trial = "process_warmup"
        receipt = dict(status="running", records=[], generations=0, main_optimizer_steps=0,
                       scientific_state_inherited=False, snapshots="Initial only; failure state retained if needed")
        trainer = None
        model = observer = original_backward = original_step = backward = step = None
        started = time.perf_counter()
        try:
            if torch.cuda.is_initialized():
                torch.cuda.reset_peak_memory_stats()
            trainer = SpecForgeEventTrainer(self.config, root / "trainer", objective="osd", ranking_lambda=0.)
            receipt["loading_s"] = time.perf_counter() - started
            initial = self.driver.validate_fresh(trainer, self.pinned, root)
            del initial
            model = trainer.bundle.model
            rng = capture_rng()
            target = self.driver.frozen_tables(model)
            model.begin_event()
            assert_state_equal(capture_rng(), rng, "ordinary_warmup_rng")
            if model.reference is not None or model.ranking_enabled:
                raise AssertionError("Ordinary warmup entered ranking")
            observer = RankingObserver(model, log)
            original_backward, original_step = trainer.backend.backward, trainer.backend.optimizer.step

            def backward(*args, **kwargs):
                if log.counts["backward_attempted"] >= 2:
                    raise AssertionError("Process warmup backward budget exhausted")
                with log.work("backward"):
                    return original_backward(*args, **kwargs)

            def step(*args, **kwargs):
                if log.counts["temporary_optimizer_step_attempted"]:
                    raise AssertionError("Process warmup optimizer budget exhausted")
                with log.work("temporary_optimizer_step"):
                    return original_step(*args, **kwargs)

            with observer.installed(), override(trainer.backend, "backward", backward), \
                    override(trainer.backend.optimizer, "step", step), self.driver.check_full_shuffle(self.ordinary, log):
                for row in run_window(trainer, self.ordinary, log):
                    receipt["records"].append(row)
                    write_json(root / "RESULT.json", receipt)
            self.driver.assert_finite(self.driver.snapshot(trainer))
            assert_state_equal(self.driver.frozen_tables(model), target, "warmup_frozen_tables")
            for key, expected in dict(backward_completed=2, temporary_optimizer_step_completed=1,
                                      response_presentations=2, current_training_forward_completed=2).items():
                assert_state_equal(log.counts[key], expected, "warmup." + key)
            receipt["objective_work"] = model.end_event()
            receipt["status"] = "passed"
            self._warmup_done = True
            return receipt
        except BaseException as error:
            receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            if trainer is not None:
                try:
                    save_torch(root / "failure_state.pt", dict(state=self.driver.snapshot(trainer),
                        gradients=self.driver.helpers().named_gradients(trainer)))
                except BaseException as capture_error:
                    receipt["failure_capture_error"] = repr(capture_error)
            raise
        finally:
            receipt["wall_s"] = time.perf_counter() - started
            receipt["work_counts"] = dict(log.counts)
            receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
            receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
            write_json(root / "RESULT.json", receipt)
            if trainer is not None:
                trainer.failed = receipt["status"] != "passed"
                trainer.close()
            # No warmup model, bound backend method or observer survives into
            # the next constructor, including cycles held by the sampler.
            trainer = model = observer = original_backward = original_step = backward = step = None
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
