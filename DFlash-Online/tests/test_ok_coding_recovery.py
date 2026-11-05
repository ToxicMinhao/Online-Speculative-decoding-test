"""Tiny CPU fixtures exercise real admission and recovery without model work."""

from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import transformers

from online_feedback.storage import write_json
from experiments.ok_coding import campaign as C, preflight as P
from experiments.ok_coding.evaluation import evaluate
from experiments.ok_coding.storage import GIB, StorageLifecycle, full_campaign_budget, identity
from experiments.ok_continuation.test_resume import event as training_event
from test_ok_coding_campaign import plan
from test_ok_coding_evaluation import Backend as EvaluationBackend
from test_ok_coding_report import complete_work


class StopAfterContinuation(RuntimeError):
    """Stop the CPU orchestration fixture at the next unstarted K job."""


def read(path):
    return json.loads(Path(path).read_text())


def write_pool(seed_root, window, checkpoint, native):
    """Retain tiny captures, prepared features and their absolute identities."""
    event_root = seed_root / window["event_id"]
    features, sources = [], {}
    for request in window["requests"]:
        name = request["request_id"]
        feature = event_root / "features" / (name + ".pt")
        capture = native / "captures" / (name + ".pt")
        terminal = event_root / "features" / (name + ".terminal.json")
        response = event_root / "responses" / (name + ".json")
        feature.parent.mkdir(parents=True, exist_ok=True)
        capture.parent.mkdir(parents=True, exist_ok=True)
        feature.write_bytes(b"tiny prepared " + name.encode())
        capture.write_bytes(b"tiny capture " + name.encode())
        write_json(terminal, {"tensor_path": str(feature)})
        write_json(response, {"record_id": name, "checkpoint_id": checkpoint,
            "prompt_ids": request["prompt_ids"], "response_ids": [151645],
            "metadata": {"capture": {"feature_path": str(capture)}}})
        features.append(str(feature))
        sources.update({str(path): identity(path) for path in (response, capture, terminal)})
    pool = {"event_id": window["event_id"], "checkpoint": checkpoint,
        "record_ids": [row["request_id"] for row in window["requests"]],
        "features": features, "identities": {path: identity(path) for path in features},
        "immutable": True, "teacher_terminal_forwards": 128, "response_generations": 128,
        "retained_source_identities": sources}
    write_json(event_root / "POOL.json", pool)
    write_json(event_root / "COSTS.json", {"fixture": True})
    complete_work(seed_root / (window["event_id"] + "_WORK.jsonl"),
                  {"feedback": 128, "terminal_teacher_forwards": 128})
    return pool


def write_job(seed_root, arm, event, pool, parent, ordinary):
    """Build the same complete receipts used by the real recovery validator."""
    audit = seed_root / f"resume_{event:03d}_{arm}"
    folder = seed_root / arm / "events" / f"event_{event:03d}"
    export = folder / "export"
    export.mkdir(parents=True, exist_ok=False)
    state = folder / "training_state.pt"
    state.write_bytes(f"tiny {arm} state {event}".encode())
    (export / "model.safetensors").write_bytes(f"tiny {arm} weights {event}".encode())
    latest = seed_root / arm / "training_state.pt"
    if latest.exists():
        latest.unlink()
    os.link(state, latest)
    result = training_event(event, "osd" if arm == "O" else "osd_first_error")
    mapping = {row["record_id"]: pool["record_ids"][index]
               for index, row in enumerate(result["features"])}
    for row, record_id, feature in zip(result["features"], pool["record_ids"], pool["features"]):
        row.update(record_id=record_id, path=feature)
    for row in result["records"]:
        row["record_id"] = mapping[row["record_id"]]
    result.update(export_path=str(export), full_state_path=str(state), latest_state_path=str(latest))
    write_json(folder / "RESULT.json", result)
    complete_work(audit / "WORK.jsonl", C.TRAINING_LIMITS)
    write_json(audit / "JOB.json", {"seed": 17, "arm": arm, "event": event,
                                    "parent": parent, "ordinary": ordinary})
    write_json(audit / "WORKER.json", {"status": "completed", "result_path": str(folder / "RESULT.json")})
    write_json(audit / "EXEC.json", {"returncode": 0})
    write_json(audit / "CLOSED.json", {"global_step": event * 128, "presentations": event * 256})
    write_json(audit / "SAVED_ENDPOINT.json", {"status": "passed", "export_exact": True,
        "completed_event": event, "global_step": event * 128})
    write_json(audit / "LIVE_BOUNDARY.json", {"exact_boundary": True})
    write_json(audit / "COSTS.json", {"fixture": True})
    write_json(audit / "STATE_IDENTITIES.json", {
        str(path): identity(path) for path in (state, export / "model.safetensors")})
    if event == 1:
        initial = seed_root / arm / "initial_state.pt"
        initial.write_bytes(("tiny initial " + arm).encode())
        write_json(audit / "INITIAL.json", {"status": "passed", "path": str(initial),
            **identity(initial), "global_step": 0, "released_weights_exact": True})
        write_json(audit / "INITIAL_PARITY.json", {"exact_initial_state": True})
    return result, {"fixture": True}


def completed_boundary(root):
    """Seed17 events1/2 complete; event3 is entirely unstarted."""
    fixed = plan()
    fixed["storage"] = {"mode": "retain_local", "offload_assumed": False,
        "reserve_bytes": 8 * GIB, "campaign_capacity": full_campaign_budget(fixed)}
    sealed = root / "sealed"
    write_json(sealed / "PLAN.json", fixed)
    source = sealed / "source.py"
    source.write_text("# fixture execution source\n")
    write_json(sealed / "SEAL.json", {"execution_source": fixed["execution_source"],
        "files": {str(path): identity(path) for path in (source, sealed / "PLAN.json")}})
    write_json(sealed / "MODEL_PIN.json", {name: {"files": {}} for name in ("target", "draft")})
    write_json(sealed / "RUNTIME_PIN.json", {})
    write_json(sealed / "TRAINING_PIN.json", {"environment": {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__}})
    write_json(root / "STORAGE_ADMISSION.json", {"historical_fresh_admission": "preserved"})
    output = root / "campaign_001"
    seed_root = output / "seed_17"
    native = output / "native"
    native.mkdir(parents=True)
    (native / "request_000000.trace").write_bytes(b"original legacy trace\n")
    for number in (1, 3):
        old = output / "native_attempts" / f"attempt_{number:06d}" / "native"
        old.mkdir(parents=True)
        (old / "request_000000.trace").write_bytes(f"prior attempt {number}\n".encode())
    for folder, prefix in ((root / "resume_admission", "attempt_"), (output / "recovery", "admission_")):
        for number in (1, 3):
            write_json(folder / f"{prefix}{number:03d}.json", {"prior_receipt": number})
    seed_root.mkdir()
    storage = StorageLifecycle(fixed, seed_root, seed=17)
    parents = {arm: {"state": None, "export": fixed["draft"], "receipt": None} for arm in C.ARMS}
    for window in fixed["windows"][:2]:
        event = window["event_index"]
        pool = write_pool(seed_root, window, parents["O"]["export"], native)
        ordinary = None
        for arm in C.ARMS:
            result, _ = write_job(seed_root, arm, event, pool, parents[arm], ordinary)
            result_path = seed_root / arm / "events" / f"event_{event:03d}" / "RESULT.json"
            if arm == "O":
                ordinary = str(result_path)
            parents[arm] = {"state": result["full_state_path"], "export": result["export_path"],
                            "receipt": str(result_path)}
        if event == 2:
            evaluate(EvaluationBackend(fixed["evaluation_requests"]), fixed["evaluation_requests"],
                {"F": fixed["draft"], **{arm: parents[arm]["export"] for arm in C.ARMS}},
                seed_root / "event_002" / "evaluation", seed=17, event=2,
                canonical_path=output / "EVALUATION_CANONICAL.json")
        storage.after_event(event, parents, pool, evaluated=event == 2)
    write_json(seed_root / "CURRENT_PARENTS.json", parents)
    progress = {"status": "failed", "phase": "between_events", "events_completed": [[17, 1], [17, 2]]}
    C.reconcile_work(output, progress)
    write_json(output / "STATUS.json", progress)
    write_json(output / "PLAN_CHECK.json", C.validate_plan(fixed))
    write_json(output / "COSTS.json", {"execution_source": fixed["execution_source"]})
    return fixed, parents


def resource_fakes():
    """Only machine/model probes are faked; source/boundary/capacity gates run."""
    stack = ExitStack()
    original_read_text = Path.read_text

    def resource_text(path, *args, **kwargs):
        if path == Path("/proc/meminfo"):
            return "MemAvailable: 134217728 kB\n"
        return original_read_text(path, *args, **kwargs)

    stack.enter_context(patch.object(Path, "read_text", resource_text))
    stack.enter_context(patch.object(P, "hardware_admission", return_value={"status": "passed"}))
    stack.enter_context(patch.object(P, "check_configs", return_value={"fixture": True}))
    stack.enter_context(patch.object(P, "verify_files", return_value=None))
    stack.enter_context(patch.object(P, "verify_native_runtime", return_value={"status": "passed"}))
    stack.enter_context(patch.object(P.shutil, "disk_usage", return_value=SimpleNamespace(free=10**15)))
    return stack


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_completed_boundary_resumes_in_new_native_directory_and_reuses_science(self):
        fixed, parents = completed_boundary(self.root)
        output = self.root / "campaign_001"
        mutable = {output / "STATUS.json", output / "COSTS.json", output / "seed_17/CURRENT_PARENTS.json",
                   *(output / "seed_17" / arm / "training_state.pt" for arm in C.ARMS)}
        old_files = {path: path.read_bytes() for path in output.rglob("*")
                     if path.is_file() and path not in mutable}
        old_receipts = {path: path.read_bytes() for path in (self.root / "resume_admission").glob("*.json")}
        old_receipts[self.root / "STORAGE_ADMISSION.json"] = (self.root / "STORAGE_ADMISSION.json").read_bytes()
        starts, generations, preparations, jobs = [], [], [], []
        test = self

        class StrictBackend:
            def __init__(self, target, draft, **options):
                native = Path(options["output_root"])
                test.assertFalse(native.exists())
                test.assertEqual(native, output / "native_attempts" / "attempt_000004" / "native")
                test.assertTrue((self_root / "resume_admission" / "attempt_004.json").exists())
                test.assertTrue((output / "recovery" / "admission_004.json").exists())
                test.assertEqual(read(output / "STATUS.json")["optimizer_steps"], 512)
                native.mkdir(parents=True, exist_ok=False)
                self.native = native
                self.serial = 0
                self.loading_s = 0.0
                self.checkpoint_id = draft
                starts.append(self)

            def publish(self, checkpoint):
                from test_osd_one_event import publication
                self.checkpoint_id = checkpoint
                return publication(checkpoint)

            def check_weights(self, checkpoint):
                return EvaluationBackend(fixed["evaluation_requests"]).check_weights(checkpoint)

            def generate(self, prompt_ids, **options):
                request = fixed["windows"][2]["requests"][self.serial]
                test.assertEqual(prompt_ids, request["prompt_ids"])
                test.assertEqual(options["request_id"], request["request_id"])
                test.assertEqual(options["max_new_tokens"], 2048)
                test.assertEqual(self.checkpoint_id, parents["O"]["export"])
                trace = self.native / f"request_{self.serial:06d}.trace"
                capture = self.native / f"capture_{self.serial:06d}.pt"
                trace.write_bytes(b"new trace")
                capture.write_bytes(b"new capture")
                self.serial += 1
                record = {"record_id": options["request_id"], "prompt_ids": prompt_ids,
                    "response_ids": [151645], "checkpoint_id": self.checkpoint_id,
                    "metadata": {"request_latency_s": 0.0, "capture": {
                        "feature_path": str(capture), "trace_path": str(trace),
                        "capture_host_s": 0.0, "feature_persistence_s": 0.0, "cache_maintenance_s": 0.0}}}
                write_json(options["save_to"], record)
                generations.append(record["record_id"])
                return record

            def prepare_feedback(self, record, path):
                path.write_bytes(b"new prepared")
                write_json(path.with_suffix(".terminal.json"), {"tensor_path": str(path)})
                preparations.append(record["record_id"])
                return {"status": "prepared", "feature_path": str(path),
                        "teacher_preparation_s": 0.0, "feature_persistence_s": 0.0}

            def close(self):
                self.closed = True

        def dispatch(root, seed, arm, event, parent, ordinary):
            jobs.append((seed, arm, event, deepcopy(parent), ordinary))
            self.assertEqual((seed, event), (17, 3))
            self.assertEqual(parent, parents[arm])
            if arm == "K":
                self.assertEqual(ordinary, str(output / "seed_17/O/events/event_003/RESULT.json"))
                raise StopAfterContinuation("fixture reached next unstarted K job")
            self.assertIsNone(ordinary)
            pool = read(output / "seed_17/event_003/POOL.json")
            return write_job(output / "seed_17", arm, event, pool, parent, ordinary)

        self_root = self.root
        with resource_fakes(), patch("online_feedback.sglang_backend.SGLangDFlashBackend", StrictBackend), \
                patch.object(C, "dispatch_training", side_effect=dispatch), \
                patch("experiments.ok_coding.evaluation.evaluate", side_effect=AssertionError("completed evaluation reran")):
            with self.assertRaises(StopAfterContinuation):
                C.run(self.root, resume=True)
        self.assertEqual(len(starts), 1)
        self.assertTrue(starts[0].closed)
        self.assertEqual(starts[0].serial, 128)
        self.assertEqual(generations, [row["request_id"] for row in fixed["windows"][2]["requests"]])
        self.assertEqual(preparations, generations)
        self.assertEqual([(row[1], row[2]) for row in jobs], [("O", 3), ("K", 3)])
        for path, before in {**old_files, **old_receipts}.items():
            self.assertEqual(path.read_bytes(), before, str(path))
        status = read(output / "STATUS.json")
        self.assertEqual(status["arm_counters"], {"17/O": 384, "17/K": 256, "29/O": 0, "29/K": 0})
        attempt = read(output / "native_attempts/attempt_000004/STATUS.json")
        self.assertEqual(attempt["attempt_id"], "attempt_000004")
        self.assertEqual(attempt["native_path"], str(starts[0].native))
        self.assertEqual(attempt["mode"], "resume")
        self.assertEqual(attempt["execution_source"], fixed["execution_source"])
        self.assertEqual(attempt["startup"]["status"], "passed")
        self.assertEqual(attempt["closure"]["status"], "closed")
        self.assertTrue(attempt["admission_receipts"])

    def test_invalid_boundaries_stop_before_attempt_or_model_work(self):
        fixed, _ = completed_boundary(self.root)
        output = self.root / "campaign_001"
        cases = {
            "changed_source": (self.root / "sealed/source.py", b"# changed\n"),
            "changed_capture": (output / "native/captures/train0.pt", b"changed capture"),
            "missing_initial": (output / "seed_17/O/initial_state.pt", None),
            "missing_prepared_feature": (output / "seed_17/event_001/features/train0.pt", None),
            "unbound_extra_export": (output / "seed_17/O/events/event_002/export/unbound.safetensors", b"extra weights"),
            "missing_latest": (output / "seed_17/O/training_state.pt", None),
            "changed_latest": (output / "seed_17/O/training_state.pt", b"changed own latest"),
            "incomplete_training": (output / "seed_17/resume_002_K/WORKER.json", b'{"status":"running"}'),
            "missing_endpoint": (output / "seed_17/K/events/event_002/training_state.pt", None),
            "unknown_feedback": (output / "seed_17/event_003_WORK.jsonl",
                b'{"kind":"feedback","ordinal":1,"phase":"started"}\n'),
            "retirement_intent_before_unlink": (output / "seed_17/retention/event_004_RETIREMENT_INTENT.json",
                b'{"candidates":[]}'),
            "retirement_progress_before_unlink": (output / "seed_17/retention/event_004_RETIREMENT_PROGRESS.json",
                b'{"retired":[]}'),
        }
        for name, (path, replacement) in cases.items():
            with self.subTest(name=name):
                before = path.read_bytes() if path.exists() else None
                if replacement is None:
                    path.unlink()
                else:
                    path.write_bytes(replacement)
                attempts = sorted((output / "native_attempts").iterdir())
                try:
                    with resource_fakes(), patch("online_feedback.sglang_backend.SGLangDFlashBackend") as backend, \
                            patch.object(C, "dispatch_training") as training, \
                            patch.object(C, "collect_pool") as collection, \
                            patch("experiments.ok_coding.evaluation.evaluate") as evaluation:
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            C.run(self.root, resume=True)
                        backend.assert_not_called()
                        training.assert_not_called()
                        collection.assert_not_called()
                        evaluation.assert_not_called()
                    self.assertEqual(sorted((output / "native_attempts").iterdir()), attempts)
                    self.assertEqual(read(output / "STATUS.json")["phase"], "between_events")
                finally:
                    if before is None:
                        path.unlink()
                    else:
                        path.write_bytes(before)
        # All invalid inputs are restored before testing independent resource and startup failures.
        attempts = sorted((output / "native_attempts").iterdir())
        with resource_fakes(), patch.object(P.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
                patch("online_feedback.sglang_backend.SGLangDFlashBackend") as backend, \
                patch.object(C, "collect_pool") as collection, patch.object(C, "dispatch_training") as training:
            with self.assertRaisesRegex(RuntimeError, "storage admission failed"):
                C.run(self.root, resume=True)
            backend.assert_not_called()
            collection.assert_not_called()
            training.assert_not_called()
        self.assertEqual(sorted((output / "native_attempts").iterdir()), attempts)
        failed_admissions = sorted((self.root / "resume_admission").glob("attempt_*.json"))
        self.assertEqual(read(failed_admissions[-1])["status"], "failed")
        self.assertFalse(read(failed_admissions[-1])["storage"]["admitted"])
        self.assertEqual(read(output / "STATUS.json")["phase"], "between_events")

        def startup_failure(target, draft, **options):
            Path(options["output_root"]).mkdir(parents=True, exist_ok=False)
            raise RuntimeError("fixture backend construction failure")

        with resource_fakes(), patch("online_feedback.sglang_backend.SGLangDFlashBackend", side_effect=startup_failure), \
                patch.object(C, "collect_pool") as collection, patch.object(C, "dispatch_training") as training:
            with self.assertRaisesRegex(RuntimeError, "fixture backend construction failure"):
                C.run(self.root, resume=True)
            collection.assert_not_called()
            training.assert_not_called()
        attempt = read(output / "native_attempts/attempt_000004/STATUS.json")
        self.assertEqual(attempt["startup"]["status"], "failed")
        self.assertEqual(attempt["closure"]["status"], "no_backend")
        self.assertEqual(attempt["campaign_status"], "failed")
        self.assertEqual(C.validate_plan(fixed)["status"], "passed")


class AttemptAllocationTests(unittest.TestCase):
    def test_unexpected_name_collision_is_exclusive_and_never_skipped(self):
        from experiments.ok_coding import attempts

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "attempt_000004"
            parent.mkdir()
            receipt = root / "attempt_004.json"
            receipt.write_bytes(b"prior admission")
            for reserve, candidate in ((attempts.reserve_directory, parent),
                                       (attempts.reserve_receipt, receipt)):
                with self.subTest(candidate=candidate.name):
                    with patch.object(attempts, "_next_path", return_value=candidate):
                        with self.assertRaises(FileExistsError):
                            reserve(root)
            self.assertEqual(receipt.read_bytes(), b"prior admission")
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ["attempt_000004", "attempt_004.json"])

    def test_gaps_reserved_names_and_all_rng_streams_are_preserved(self):
        from experiments.ok_coding.attempts import reserve_directory, reserve_receipt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in (1, 3):
                (root / f"attempt_{number:06d}").mkdir()
                (root / f"attempt_{number:06d}" / "old").write_bytes(str(number).encode())
                write_json(root / f"admission_{number:03d}.json", {"number": number})
            # A reserved empty receipt counts even before its body is written.
            (root / "admission_007.json").touch()
            before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            with patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA RNG touched")), \
                    patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA RNG seeded")):
                attempt = reserve_directory(root)
                receipt = reserve_receipt(root, prefix="admission_")
                write_json(receipt, {"status": "passed"})
            self.assertEqual(attempt.name, "attempt_000004")
            self.assertEqual(receipt.name, "admission_008.json")
            self.assertTrue(attempt.is_dir())
            self.assertEqual(read(receipt), {"status": "passed"})
            self.assertEqual((root / "admission_007.json").read_bytes(), b"")
            self.assertEqual(random.getstate(), before[0])
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), before[1])))
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
            for number in (1, 3):
                self.assertEqual((root / f"attempt_{number:06d}" / "old").read_bytes(), str(number).encode())
                self.assertEqual(read(root / f"admission_{number:03d}.json"), {"number": number})


if __name__ == "__main__":
    unittest.main()
