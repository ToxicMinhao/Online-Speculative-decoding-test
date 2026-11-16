"""The retained audit/prepare schema must reach the real reference consumer."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.persistent_prefix_withholding.data import token_digest
from experiments.persistent_prefix_withholding.references import (
    canonical_record, file_identity, resolve_missing_requests,
)
from experiments.persistent_prefix_withholding.task_runner import reference_task
from online_feedback.storage import write_json


def native_record(prompt, checkpoint="fixed-K6"):
    response = [151645]
    capture = dict(status="passed", native_capture_validated=True,
        label_binding_scope="native_prefill_and_committed_verification",
        prompt_ids=prompt, response_ids=response, blocks=[], target_argmax_ids=response,
        covered_token_ids=prompt, covered_rows=len(prompt), collect_features=False, feature_bytes=0)
    return dict(prompt_ids=prompt, response_ids=response, finish_reason="eos",
        max_new_tokens=2048, stop_token_ids=[151645, 151643], checkpoint_id=checkpoint,
        metadata=dict(capture=capture, trace=[], block_size=16, context_limit=4096,
            runtime="pinned_SGLang_FA3", cache_policy="idle_full_flush_before_every_request_and_publication",
            feedback_capture_in_timed_request=False, verification_calls=0, committed_progress=0))


def audit_schema():
    requests = [dict(request_id=f"persistent-v2-gsm8k-{i:04d}",
        source_id=f"gsm8k:main:test:{i}", panel_ordinal=i, block_ordinal=i // 32,
        prompt_ids=[90, 1000+i], prompt_ids_sha256=token_digest([90, 1000+i]),
        max_new_tokens=2048, stop_token_ids=[151645, 151643], collect_feedback=False)
        for i in range(128)]
    # Exactly the identity-only schema emitted by audit_reused and copied by prepare.
    missing = [dict(request_id=r["request_id"], source_id=r["source_id"], role="panel")
               for r in requests[104:]]
    reused = {r["request_id"]: canonical_record(r, native_record(r["prompt_ids"]),
        f"/retained/F_{i}.json", {"bytes": 1, "sha256": "a"*64}, {"receipt": "proof.json"})
        for i, r in enumerate(requests[:104])}
    return {"requests": requests}, missing, reused


class ImmediateLedger:
    def __init__(self):
        self.calls = []

    def call(self, counts, label, function, *args, **kwargs):
        self.calls.append((counts, label))
        return function(*args, **kwargs)


class ReferenceJoinTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.panel, self.missing, self.reused = audit_schema()
        self.plan = dict(origin_export="fixed-K6", execution_commit="published",
                         reference_provenance="sealed-proof.json", input_identities={})
        for key, value in (("data_panel", self.panel), ("missing_references", self.missing),
                           ("reused_references", self.reused), ("references", self.reused)):
            self.plan[key] = str(self.root / (key + ".json"))
            self.write_input(key, value)
        self.ledger = ImmediateLedger()

    def write_input(self, key, value):
        write_json(self.plan[key], value)
        if key != "references":
            self.plan["input_identities"][self.plan[key]] = file_identity(self.plan[key])

    def run_consumer(self, factory):
        with patch("experiments.persistent_prefix_withholding.task_runner.create_backend", factory), \
             patch("experiments.persistent_prefix_withholding.task_runner.work_boundary"), \
             patch("experiments.persistent_prefix_withholding.task_runner.close_backend"):
            return reference_task(self.plan, self.root / "generated", self.ledger, "GPU-A")

    def assert_rejected_before_backend(self):
        factory = Mock()
        with self.assertRaises(ValueError):
            self.run_consumer(factory)
        factory.assert_not_called()
        self.assertEqual(self.ledger.calls, [])

    def test_audit_identity_rows_join_before_real_consumer_allocates_backend(self):
        before = {key: Path(self.plan[key]).read_bytes() for key in
                  ("data_panel", "missing_references", "reused_references")}
        generated = []
        def generate(prompt, **kwargs):
            generated.append((prompt, kwargs))
            record = native_record(prompt)
            write_json(kwargs["save_to"], record)
            return record
        backend = Mock(loading_s=1.0, generate=generate)
        def construct(*args):
            bound = json.loads((self.root / "generated/BOUND_REFERENCE_REQUESTS.json").read_text())
            self.assertEqual(bound["status"], "validated_before_backend")
            self.assertEqual(bound["requests"], self.panel["requests"][104:])
            return backend
        result = self.run_consumer(Mock(side_effect=construct))
        self.assertEqual(result["new_reference_generations"], 24)
        self.assertEqual(len(generated), 24)
        self.assertEqual(len(self.ledger.calls), 24)
        for (prompt, kwargs), request in zip(generated, self.panel["requests"][104:]):
            self.assertEqual(prompt, request["prompt_ids"])
            self.assertEqual(kwargs["request_index"], request["panel_ordinal"])
            self.assertEqual(kwargs["request_id"], request["request_id"])
            self.assertEqual(kwargs["max_new_tokens"], 2048)
            self.assertEqual(kwargs["purpose"], "evaluation")
        merged = json.loads(Path(self.plan["references"]).read_text())
        self.assertEqual(len(merged), 128)
        self.assertEqual({key: merged[key] for key in self.reused}, self.reused)
        for key, contents in before.items():
            self.assertEqual(Path(self.plan[key]).read_bytes(), contents)

    def test_missing_identity_field_fails_before_backend(self):
        del self.missing[0]["source_id"]
        self.write_input("missing_references", self.missing)
        self.assert_rejected_before_backend()

    def test_duplicate_missing_row_fails_before_backend(self):
        self.missing[1] = deepcopy(self.missing[0])
        self.write_input("missing_references", self.missing)
        self.assert_rejected_before_backend()

    def test_contradictory_source_id_fails_before_backend(self):
        self.missing[0]["source_id"] = self.missing[1]["source_id"]
        self.write_input("missing_references", self.missing)
        self.assert_rejected_before_backend()

    def test_warmup_role_fails_before_backend(self):
        self.missing[0]["role"] = "warmup"
        self.write_input("missing_references", self.missing)
        self.assert_rejected_before_backend()

    def test_incomplete_panel_fields_fail_before_backend(self):
        for field in ("prompt_ids", "panel_ordinal", "stop_token_ids", "max_new_tokens"):
            with self.subTest(field=field):
                panel = deepcopy(self.panel)
                del panel["requests"][104][field]
                self.write_input("data_panel", panel)
                self.assert_rejected_before_backend()

    def test_wrong_reused_coverage_fails_before_backend(self):
        old = next(iter(self.reused))
        self.reused[self.missing[0]["request_id"]] = self.reused.pop(old)
        self.write_input("reused_references", self.reused)
        self.write_input("references", self.reused)
        self.assert_rejected_before_backend()

    def test_missing_count_fails_before_backend(self):
        self.write_input("missing_references", self.missing[:-1])
        self.assert_rejected_before_backend()

    def test_stale_enriched_fields_fail_before_backend(self):
        self.missing[0]["prompt_ids"] = [123]
        self.write_input("missing_references", self.missing)
        self.assert_rejected_before_backend()

    def test_changed_sealed_file_fails_before_backend(self):
        Path(self.plan["missing_references"]).write_text("[]")
        self.assert_rejected_before_backend()

    def test_partial_destination_is_not_implicitly_retried(self):
        extra = deepcopy(self.reused)
        extra[self.missing[0]["request_id"]] = deepcopy(next(iter(self.reused.values())))
        self.write_input("references", extra)
        self.assert_rejected_before_backend()

    def test_join_does_not_mutate_audit_or_panel(self):
        before = deepcopy((self.panel, self.missing, self.reused))
        resolved = resolve_missing_requests(self.panel, self.missing, self.reused)
        resolved[0]["prompt_ids"].append(123)
        self.assertEqual((self.panel, self.missing, self.reused), before)


if __name__ == "__main__":
    unittest.main()
