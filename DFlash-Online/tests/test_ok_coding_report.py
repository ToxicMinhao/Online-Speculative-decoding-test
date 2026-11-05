"""Complete synthetic4608-record audit and independently calculated paired intervals."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from online_feedback.storage import write_json
from experiments.ok_continuation.test_resume import event as training_event
from experiments.ok_coding import campaign as C
from experiments.ok_coding import report as R
from experiments.ok_coding.contract import CELLS, METHODS, evaluation_schedule
from experiments.osd_persistent.run import score_record
from test_ok_coding_campaign import plan
from test_ok_coding_evaluation import Backend
from test_osd_one_event import publication


def complete_work(path, limits):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, count in limits.items():
        for ordinal in range(1, count + 1):
            for phase in ("started", "completed"):
                rows.append({"kind": kind, "ordinal": ordinal, "phase": phase})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(root):
    fixed = plan()
    write_json(root / "sealed" / "PLAN.json", fixed)
    output = root / "campaign_001"
    for seed in (17, 29):
        seed_root = output / f"seed_{seed}"
        for event in range(1, 5):
            complete_work(seed_root / f"event_{event:03d}_WORK.jsonl",
                          {"feedback": 128, "terminal_teacher_forwards": 128})
            for arm in ("O", "K"):
                audit = seed_root / f"resume_{event:03d}_{arm}"
                complete_work(audit / "WORK.jsonl", C.TRAINING_LIMITS)
                folder = seed_root / arm / "events" / f"event_{event:03d}"
                result = training_event(event, "osd" if arm == "O" else "osd_first_error")
                result.update(export_path=str(folder / "export"), full_state_path=str(folder / "training_state.pt"))
                write_json(folder / "RESULT.json", result)
                write_json(audit / "CLOSED.json", {"global_step": event * 128, "presentations": event * 256})
                write_json(audit / "WORKER.json", {"status": "completed", "result_path": str(folder / "RESULT.json"),
                    "startup": {"pid": seed * 100 + event * 2 + (arm == "K"),
                        "cuda_visible_devices": "1", "device_uuid": "same-updater",
                        "cuda_initialized_on_entry": False, "warmup_model_work": 0,
                        "caches": {name: {"path": str(audit / name), "initially_empty": True}
                                   for name in C.CACHE_VARIABLES}}})
                write_json(audit / "COSTS.json", {"training_s": 1.0, "event_wall_s": 1.5})
                write_json(audit / "INITIAL.json", {"status": "passed", "global_step": 0,
                                                     "released_weights_exact": True})
                write_json(audit / "INITIAL_PARITY.json", {"exact_initial_state": True})
                write_json(audit / "LIVE.json", {"exact_resume": True})
                write_json(audit / "SAVED_ENDPOINT.json", {"status": "passed", "export_exact": True})
                write_json(audit / "LIVE_BOUNDARY.json", {"exact_boundary": True})
    canonical = {}
    for seed, event in CELLS:
        folder = output / f"seed_{seed}" / f"event_{event:03d}" / "evaluation"
        folder.mkdir(parents=True)
        checkpoints = {"F": "/F"}
        checkpoints.update({arm: str(output / f"seed_{seed}" / arm / "events" / f"event_{event:03d}" / "export")
                            for arm in ("O", "K")})
        backend = Backend(fixed["evaluation_requests"])
        schedule = evaluation_schedule(fixed["evaluation_requests"], seed=seed, event=event)
        rows = []
        for slot in schedule:
            ordinal, repeat, method = slot["panel_ordinal"], slot["repeat"], slot["method"]
            request = fixed["evaluation_requests"][ordinal]
            label = f"{ordinal:04d}_r{repeat}_{method}"
            path = folder / (label + ".json")
            backend.checkpoint_id = checkpoints[method]
            record = backend.generate(request["prompt_ids"], max_new_tokens=2048,
                request_id=label, request_index=ordinal, session_id="ok_coding_evaluation",
                purpose="evaluation", save_to=path)
            row = score_record(record, request_id=request["request_id"], repeat=repeat, method=method)
            row.update(**slot, label=label, record_path=str(path),
                finish_reason=record["finish_reason"], cache_maintenance_s=0.1,
                structural_errors=[], errors=[], valid_measurement=True, validation_status="valid")
            rows.append(row)
            if method == "F" and request["request_id"] not in canonical:
                canonical[request["request_id"]] = {
                    "response_ids": record["response_ids"], "finish_reason": record["finish_reason"],
                    "source_record_path": str(path), "source_method": "F", "source_repeat": 0,
                    "source_cell": [17, 2], "F_checkpoint": "/F",
                    "F_counts": {name: row[name] for name in R.COUNT_FIELDS}}
        (folder / "MEASUREMENTS.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(folder / "PROTOCOL.json", {"schedule": schedule, "checkpoints": checkpoints})
        write_json(folder / "QUALIFICATION.json", {"status": "passed", "requests": 8, "generations": 72,
            "methods_with_native_equality": ["F", "K", "O"], "performance_win_required": False, "extra_generations": 0})
        write_json(folder / "STATUS.json", {"status": "completed", "qualification": "passed",
            "attempted_generations": 1152, "returned_generations": 1152, "completed_valid_generations": 1152,
            "invalid_generations": 0, "pending_canonical_generations": 0,
            "unknown_generation_attempts": 0, "unattempted_generations": 0})
        activations = []
        for method in METHODS:
            write_json(folder / ("WEIGHTS_" + method + ".json"),
                {"native_reply": backend.check_weights(checkpoints[method]), "errors": []})
            activations.append({"method": method, **publication(checkpoints[method]), "errors": []})
        (folder / "ACTIVATIONS.jsonl").write_text("".join(json.dumps(row) + "\n" for row in activations))
    write_json(output / "EVALUATION_CANONICAL.json", canonical)
    status = {"status": "completed", "extra_generations": 0, "optimizer_previews": 0,
              "qualification_model_work": 0, "extra_optimizer_steps": 0}
    C.reconcile_work(output, status)
    write_json(output / "STATUS.json", status)
    return fixed


class ReportTests(unittest.TestCase):
    def test_shared_bootstrap_matches_independent_loop_and_keeps_numpy_rng(self):
        before = np.random.get_state()
        indices = R.bootstrap_indices(128)
        after = np.random.get_state()
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(before, after)))
        rng = np.random.Generator(np.random.PCG64(20261030))
        independent = np.stack([rng.integers(0, 128, 128) for _ in range(10000)])
        self.assertTrue(np.array_equal(indices, independent))
        rows = []
        for index in range(128):
            value = {}
            for method in METHODS:
                value[method] = {name: 128 + index for name in R.COUNT_FIELDS}
                value[method].update({name: 1.0 for name in R.TIMINGS})
                value[method]["cap_hit"] = False
            value["K"]["verification_calls"] -= index % 4
            rows.append(value)
        summary = R.summarize_cell(rows, indices)
        a = np.asarray([row["O"]["verification_calls"] for row in rows])
        b = np.asarray([row["K"]["verification_calls"] for row in rows])
        direct = [100 * (a[sample].sum() - b[sample].sum()) / a[sample].sum() for sample in independent]
        self.assertEqual(summary["comparisons"]["K_vs_O"]["bootstrap_95pct"],
                         np.quantile(direct, [0.025, 0.975], method="linear").tolist())

    def test_full_raw4608_audit_then_tamper_preserves_other_completed_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            summary = R.write_campaign_report(root, root / "report", execution_commit="a" * 40)
            self.assertEqual(summary["errors"], [])
            self.assertEqual(summary["status"], "valid")
            self.assertEqual(summary["outcome"], "Unsupported at this budget")
            self.assertEqual(len((root / "report" / "ALL_REPEATS.csv").read_text().splitlines()), 4609)
            self.assertEqual(len((root / "report" / "REPEAT_TOTALS.csv").read_text().splitlines()), 37)
            self.assertEqual(len((root / "report" / "PAIRED_REQUESTS.csv").read_text().splitlines()), 513)
            path = root / "campaign_001" / "seed_29" / "event_004" / "evaluation" / "0000_r0_F.json"
            record = json.loads(path.read_text())
            record["metadata"]["request_latency_s"] = 2.0
            write_json(path, record)
            changed = R.write_campaign_report(root, root / "changed")
            self.assertEqual(changed["status"], "invalid_or_incomplete")
            self.assertIn("raw_measurement_disagreement", " ".join(changed["errors"]))
            self.assertEqual(len(changed["cells"]), 3)


if __name__ == "__main__":
    unittest.main()
