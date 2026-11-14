"""CPU-only tests for immutable memory and outcome-blind probe preparation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.threatened_repair_replay.prepare import (
    CONSTANTS, RetainedInputs, authenticate_source, deduplicate_memory, identity,
    select_probe_keys, validate_disjoint, attempt_accounting,
)
from experiments.survival_confirmation.data import digest


def binding(pid, request="r", pass_index=0):
    return dict(presentation_id=pid, record_id=request, pass_index=pass_index,
        sampled_anchors=[dict(positions=[[10, 20]], keep_mask=[[True, True]])])


def occurrence(pid, request="r", start=0, slot=1, ordinal=0, pass_index=0):
    return dict(presentation_id=pid, request_id=request, canonical_start=start,
        proposal_slot=slot, anchor_ordinal=ordinal, canonical_token=7,
        pass_index=pass_index, weight=1.0)


def prior_attempt_fixture(root, *, steps=3, category="qualification", seconds=185.9867403432727):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    status = root / "STATUS.json"
    status.write_text(json.dumps(dict(status="failed", execution_commit="prior-source",
        phase="training_qualification_and_native_trajectory",
        budget=dict(gpu_seconds=seconds, gpu_seconds_limit=10800))))
    ledger = root / "PHYSICAL_WORK.jsonl"
    records = [dict(serial=i, kind="optimizer_step", category=category, phase=phase)
               for i in range(1, steps+1) for phase in ("started", "completed")]
    records.append(dict(phase="ledger_closed", physical={}))
    ledger.write_text("".join(json.dumps(row)+"\n" for row in records))
    return dict(status_path=str(status), status_identity=identity(status),
        training_work_path=str(ledger), training_work_identity=identity(ledger),
        approved_cumulative_adam_limit=171)


class RetryAccountingTests(unittest.TestCase):
    def test_original_first_attempt_keeps_168_and_full_clock(self):
        result = attempt_accounting()
        self.assertEqual(result["cumulative_adam_limit"], 168)
        self.assertEqual(result["prior_adam_steps"], 0)
        self.assertEqual(result["current_attempt_gpu_seconds_limit"], 10800)

    def test_exact_authenticated_carryover_and_unchanged_local_ceiling(self):
        with tempfile.TemporaryDirectory() as folder:
            result = attempt_accounting(prior_attempt_fixture(folder))
            self.assertEqual(result["prior_adam_steps"], 3)
            self.assertEqual(result["current_attempt_adam_limit"], 168)
            self.assertEqual(result["cumulative_adam_limit"], 171)
            self.assertEqual(result["current_attempt_gpu_seconds_limit"], 10614.013259656727)

    def test_both_artifacts_require_prior_pins_and_unchanged_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("status", "training_work"):
                with self.subTest(name=name):
                    prior = prior_attempt_fixture(folder)
                    Path(prior[name+"_path"]).write_text("{}")
                    with self.assertRaisesRegex(ValueError, "Retained artifact changed"):
                        attempt_accounting(prior)
                    prior = prior_attempt_fixture(folder)
                    prior[name+"_identity"] = None
                    with self.assertRaisesRegex(ValueError, "Missing retained identity"):
                        attempt_accounting(prior)

    def test_wrong_work_population_or_nonfinite_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for kwargs in ({"steps":2}, {"steps":4}, {"category":"native"},
                           {"category":"startup"}, {"seconds":float("nan")},
                           {"seconds":float("inf")}, {"seconds":10800}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    attempt_accounting(prior_attempt_fixture(folder, **kwargs))

    def test_completed_unknown_or_malformed_prior_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for mutation in ("completed", "unknown", "mismatched_kind", "wrong_cap"):
                with self.subTest(mutation=mutation):
                    prior = prior_attempt_fixture(folder)
                    if mutation == "completed":
                        path = Path(prior["status_path"])
                        value = json.loads(path.read_text())
                        value["status"] = "completed"
                        path.write_text(json.dumps(value))
                        prior["status_identity"] = identity(path)
                    elif mutation == "wrong_cap":
                        prior["approved_cumulative_adam_limit"] = 172
                    else:
                        path = Path(prior["training_work_path"])
                        values = [json.loads(line) for line in path.read_text().splitlines()]
                        if mutation == "unknown":
                            del values[-2]
                        else:
                            values[1]["kind"] = "response_forward"
                        path.write_text("".join(json.dumps(row)+"\n" for row in values))
                        prior["training_work_identity"] = identity(path)
                    with self.assertRaises(ValueError):
                        attempt_accounting(prior)


class MemoryPreparationTests(unittest.TestCase):
    def test_earliest_eligible_layout_and_singleton(self):
        schedule = dict(records=[binding(0), binding(1), binding(2, pass_index=1)])
        rows = [occurrence(2, pass_index=1), occurrence(0),
                occurrence(1, start=10, slot=3, ordinal=1)]
        tags, layouts, counts = deduplicate_memory(rows, schedule, require_population=False)
        self.assertEqual([t["reference_presentation_id"] for t in tags], [0, 1])
        self.assertEqual(tags[0]["eligible_presentation_ids"], [0, 2])
        self.assertEqual(tags[1]["eligible_presentation_ids"], [1])
        self.assertEqual(counts["eligibility_counts"], {1: 1, 2: 1})
        self.assertEqual(len(layouts), 2)
        self.assertEqual(layouts[0]["binding"], schedule["records"][0])
        layouts[0]["binding"]["sampled_anchors"][0]["positions"][0][0] = -1
        self.assertEqual(schedule["records"][0]["sampled_anchors"][0]["positions"][0][0], 10)

    def test_duplicate_and_changed_token_fail(self):
        schedule = dict(records=[binding(0), binding(1)])
        for rows in ([occurrence(0), occurrence(0)],
                     [occurrence(0), dict(occurrence(1), canonical_token=8)]):
            with self.assertRaises(ValueError):
                deduplicate_memory(rows, schedule, require_population=False)

    def test_seed_and_dropped_anchor_fail(self):
        schedule = dict(records=[binding(0)])
        with self.assertRaises(ValueError):
            deduplicate_memory([occurrence(0, slot=0)], schedule, require_population=False)
        schedule["records"][0]["sampled_anchors"][0]["keep_mask"][0][0] = False
        with self.assertRaises(ValueError):
            deduplicate_memory([occurrence(0)], schedule, require_population=False)

    def test_population_contract_is_required_by_default(self):
        with self.assertRaises(ValueError):
            deduplicate_memory([occurrence(0)], dict(records=[binding(0)]))


class ProbeSelectionTests(unittest.TestCase):
    def fixture(self):
        return [dict(request_id=rid, canonical_start=ordinal*5, round_ordinal=ordinal,
                     future_acceptance=ordinal) for rid in ("a", "b") for ordinal in range(9)]

    def test_hash_formula_and_four_per_request(self):
        selected = select_probe_keys(self.fixture(), require_population=False)
        self.assertEqual(len(selected), 8)
        for rid in ("a", "b"):
            expected = sorted(range(9), key=lambda i: hashlib.sha256(
                f"trr-v1-probe|{rid}|{i*5}|{i}".encode()).hexdigest())[:4]
            self.assertEqual([r["round_ordinal"] for r in selected if r["request_id"] == rid], expected)

    def test_outcomes_and_input_order_do_not_change_selection(self):
        rows = self.fixture()
        baseline = select_probe_keys(rows, require_population=False)
        for row in rows:
            row["future_acceptance"] = -1000
            row["adversarial_endpoint_outcome"] = "must never be read"
        self.assertEqual(baseline, select_probe_keys(list(reversed(rows)), require_population=False))
        stripped = [{k:r[k] for k in ("request_id", "canonical_start", "round_ordinal")} for r in rows]
        self.assertEqual(baseline, select_probe_keys(stripped, require_population=False))
        self.assertTrue(all("future_acceptance" not in row for row in baseline))

    def test_short_population_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            select_probe_keys(self.fixture())

    def test_duplicate_origin_fails(self):
        rows = self.fixture()
        with self.assertRaises(ValueError):
            select_probe_keys(rows + [rows[0]], require_population=False)

    def test_fixed_constants(self):
        self.assertEqual(CONSTANTS["windows"], list(range(8, 129, 8)))
        self.assertEqual(CONSTANTS["replay_coefficient"], 0.10)
        self.assertEqual(CONSTANTS["card_seconds_limit"], 10800)


class DisjointnessTests(unittest.TestCase):
    def fixture(self):
        training, exclusions = [], []
        for i in range(896):
            row = dict(request_id=f"r{i}", dataset_id=f"d{i}", prompt_ids=[i],
                collect_feedback=True, event_id=f"event_{i//128+1:03d}")
            training.append(row)
            exclusions.append(dict(request_id=row["request_id"], dataset_id=row["dataset_id"],
                prompt_ids_sha256=digest(row["prompt_ids"]), normalized_question_sha256=f"text{i}",
                reason="events_1_6_training" if i < 768 else "PR42_discovery_batch"))
        return dict(train_requests=training), dict(rows=exclusions)

    def test_all_events_and_normalized_text_checked(self):
        manifest, exclusions = self.fixture()
        panel = [dict(request_id="new", source_id="test:1", prompt_ids_sha256=digest([999]),
                      normalized_question_sha256="fresh")]
        self.assertEqual(validate_disjoint(panel, manifest, exclusions)["training_requests"], 896)
        panel[0]["normalized_question_sha256"] = "text768"
        with self.assertRaises(ValueError):
            validate_disjoint(panel, manifest, exclusions)

    def test_stale_exclusion_fails(self):
        manifest, exclusions = self.fixture()
        exclusions["rows"][0]["prompt_ids_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            validate_disjoint([], manifest, exclusions)


class ProvenanceTests(unittest.TestCase):
    def test_missing_and_tampered_retained_input_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "PLAN.json"
            receipt.write_text('{"source": "original"}')
            inventory = root / "RETAINED.json"
            inventory.write_text(json.dumps({"files": {str(receipt): identity(receipt)}}))
            inputs = RetainedInputs(inventory, identity(inventory))
            self.assertEqual(inputs.verify(receipt), identity(receipt))
            missing = root / "UNPINNED.json"
            missing.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Missing retained identity"):
                inputs.verify(missing)
            receipt.write_text('{"source": "tampered"}')
            with self.assertRaisesRegex(ValueError, "Retained artifact changed"):
                inputs.verify(receipt)

    def test_inventory_itself_must_match_prior_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RETAINED.json"
            path.write_text('{"files": {}}')
            pin = identity(path)
            path.write_text('{"files": {}, "added": true}')
            with self.assertRaisesRegex(ValueError, "Retained artifact changed"):
                RetainedInputs(path, pin)
            with self.assertRaisesRegex(ValueError, "Missing retained identity"):
                RetainedInputs(path, None)

    def source_fixture(self, root):
        code = root / "code"
        code.mkdir()
        file = code / "train.py"
        file.write_text("print('bound source')\n")
        (code / "runtime").symlink_to("../external")
        manifest = root / "SOURCE.json"
        manifest.write_text(json.dumps(dict(commit="a"*40, dirty=False,
            source_path=str(code), files={"train.py": identity(file)},
            symlinks={"runtime": "../external"})))
        return manifest, code

    def test_actual_staged_source_bytes_and_commit_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, code = self.source_fixture(Path(tmp))
            pin = identity(manifest)
            self.assertEqual(authenticate_source(manifest, pin, "a"*40)["regular_files"], 1)
            with self.assertRaisesRegex(ValueError, "execution commit"):
                authenticate_source(manifest, pin, "b"*40)
            (code / "train.py").write_text("print('changed')\n")
            with self.assertRaisesRegex(ValueError, "Retained artifact changed"):
                authenticate_source(manifest, pin, "a"*40)

    def test_source_symlink_and_missing_pin_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, code = self.source_fixture(Path(tmp))
            pin = identity(manifest)
            with self.assertRaisesRegex(ValueError, "Missing retained identity"):
                authenticate_source(manifest, None, "a"*40)
            (code / "runtime").unlink()
            (code / "runtime").symlink_to("../changed")
            with self.assertRaisesRegex(ValueError, "symlink changed"):
                authenticate_source(manifest, pin, "a"*40)


if __name__ == "__main__":
    unittest.main()
