"""Canonical metadata is derived from retained native evidence, never asserted."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.functional_step_geometry.canonical_bindings import (
    bind_canonicals, load_pr32_proof,
)
from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from test_full_k_gsm8k_evaluation import FullBackend, requests


class CanonicalBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = requests(1)
        self.inputs[0]["inherited_panel_ordinal"] = 0
        self.source = self.root / "old.json"
        self.raw = FullBackend(self.inputs).generate(self.inputs[0]["prompt_ids"],
            request_id="old-record", session_id="old-panel", request_index=0,
            purpose="evaluation", save_to=self.source)
        self.refs = {"eval0": dict(response_ids=self.raw["response_ids"],
            finish_reason="eos", source_record_path="/remote/old.json")}
        self.proof = deepcopy(self.refs)
        self.proof["eval0"].update(prompt_ids=deepcopy(self.raw["prompt_ids"]),
            verified=True, binding_digest="a" * 64,
            source_record=dict(path="/remote/old.json", **identity(self.source)))
        self.identities = {"/remote/old.json": identity(self.source)}

    def bind(self):
        return bind_canonicals(self.inputs, self.refs, authenticated=self.proof,
            source_identities=self.identities, mirror=self.root, origin="/remote",
            binding="b" * 64)

    def reauthenticate_mutated_raw(self):
        write_json(self.source, self.raw)
        expected = identity(self.source)
        self.proof["eval0"]["source_record"].update(expected)
        self.identities["/remote/old.json"] = expected

    def test_exact_raw_derivation_preserves_inputs_and_source(self):
        before = deepcopy((self.inputs, self.refs, self.proof, self.identities))
        self.inputs[0]["panel_ordinal"] = 17
        before[0][0]["panel_ordinal"] = 17
        bound, receipt = self.bind()
        self.assertEqual((self.inputs, self.refs, self.proof, self.identities), before)
        for name, value in self.refs["eval0"].items():
            self.assertEqual(bound["eval0"][name], value)
        self.assertEqual(bound["eval0"]["prompt_ids"], self.raw["prompt_ids"])
        self.assertTrue(bound["eval0"]["verified"])
        self.assertEqual(receipt["records"][0]["identity"], identity(self.source))
        self.assertEqual(receipt["new_model_work"], 0)
        self.assertEqual(receipt["records"][0]["original_request_index"], 0)

    def test_raw_bytes_tampering_rejected_before_validation(self):
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source identity changed"):
            self.bind()

    def test_changed_prompt_cannot_inherit_reference(self):
        self.inputs[0]["prompt_ids"] = [7, 8]
        with self.assertRaisesRegex(ValueError, "source binding differs"):
            self.bind()

    def test_changed_raw_prompt_fails_even_with_fresh_byte_identity(self):
        self.raw["prompt_ids"] = [7, 8]
        self.reauthenticate_mutated_raw()
        with self.assertRaisesRegex(ValueError, "Invalid retained canonical"):
            self.bind()

    def test_changed_inherited_output_or_termination_rejected(self):
        for field, value in (("response_ids", [19, 151645]), ("finish_reason", "length")):
            with self.subTest(field=field):
                old = self.refs["eval0"][field]
                self.refs["eval0"][field] = value
                with self.assertRaisesRegex(ValueError, "changed since PR32"):
                    self.bind()
                self.refs["eval0"][field] = old

    def test_changed_raw_output_and_termination_fail_full_validation(self):
        original = deepcopy(self.raw)
        for field, value in (("response_ids", [19, 151645]), ("finish_reason", "length")):
            with self.subTest(field=field):
                self.raw = deepcopy(original)
                self.raw[field] = value
                self.reauthenticate_mutated_raw()
                with self.assertRaisesRegex(ValueError, "Invalid retained canonical"):
                    self.bind()

    def test_corrupt_proposal_cannot_pass_scalar_token_checks(self):
        self.raw["metadata"]["trace"][0]["draft_tokens"][0] += 1
        self.reauthenticate_mutated_raw()
        with self.assertRaisesRegex(ValueError, "Invalid retained canonical"):
            self.bind()

    def test_source_binding_and_original_index_are_checked(self):
        self.proof["eval0"]["source_record"]["path"] = "/remote/wrong.json"
        with self.assertRaisesRegex(ValueError, "source binding differs"):
            self.bind()
        self.proof["eval0"]["source_record"]["path"] = "/remote/old.json"
        self.inputs[0]["inherited_panel_ordinal"] = 1
        with self.assertRaisesRegex(ValueError, "request_identity"):
            self.bind()

    def test_membership_and_binding_required(self):
        self.refs["extra"] = deepcopy(self.refs["eval0"])
        with self.assertRaisesRegex(ValueError, "membership"):
            self.bind()


class OriginalProofTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "gsm8k").mkdir()
        write_json(self.root / "PLAN.json", dict(root="/original", execution_source={"commit": "abc"}))
        for name in ("MODEL_PIN.json", "RUNTIME_PIN.json"):
            write_json(self.root / name, {"original": name})
        self.sources = {f"/native/{index}.json": dict(bytes=1, sha256="a" * 64)
                        for index in range(512)}
        self.panel = requests(256)
        refs = {row["request_id"]: dict(prompt_ids=row["prompt_ids"], response_ids=[151645],
            finish_reason="eos", verified=True, binding_digest="b" * 64,
            source_record_path=f"/native/{index}.json",
            source_record=dict(path=f"/native/{index}.json", **self.sources[f"/native/{index}.json"]))
            for index, row in enumerate(self.panel)}
        write_json(self.root / "gsm8k/PANEL.json", dict(requests=self.panel))
        write_json(self.root / "gsm8k/CANONICALS.json", refs)
        write_json(self.root / "REFERENCE_AUDIT.json", dict(status="passed", records=512,
            extra_generations=0, files=self.sources))
        files = {str(Path("/original/sealed") / p.relative_to(self.root)): identity(p)
                 for p in self.root.rglob("*.json")}
        write_json(self.root / "SEAL.json", dict(files={**files, **self.sources}))

    def load(self, **changes):
        options = dict(model_pin=self.root / "MODEL_PIN.json", runtime_pin=self.root / "RUNTIME_PIN.json")
        return load_pr32_proof(self.root, "gsm8k", **{**options, **changes})

    def test_original_proof_and_current_pins_reproduce(self):
        refs, sources, receipt = self.load()
        self.assertEqual(len(refs), 256)
        self.assertEqual(len(sources), 512)
        self.assertEqual(len(receipt["files"]), 7)
        self.assertTrue(receipt["current_model_runtime_exact"])

    def test_tampered_proof_fails_original_identity(self):
        path = self.root / "gsm8k/CANONICALS.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "proof identity changed"):
            self.load()

    def test_current_runtime_cannot_be_substituted(self):
        other = self.root / "different.json"
        write_json(other, {"runtime": "different"})
        with self.assertRaisesRegex(ValueError, "model/runtime pins differ"):
            self.load(runtime_pin=other)


if __name__ == "__main__":
    unittest.main()
