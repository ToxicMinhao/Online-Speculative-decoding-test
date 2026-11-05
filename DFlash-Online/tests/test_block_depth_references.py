import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.block_depth.references import prepare_references


class ReferenceTests(unittest.TestCase):
    def fixture(self, root):
        request = dict(request_id="new", prompt_ids=[1, 2], stop_token_ids=[99], max_new_tokens=2048)
        old = dict(request, request_id="old", panel_ordinal=0)
        capture = dict(prompt_ids=[1, 2], response_ids=[3, 99])
        raw = dict(prompt_ids=[1, 2], response_ids=[3, 99], finish_reason="eos", checkpoint_id="frozen",
                   metadata=dict(native_trace_path="/remote/production_001/native/x.json", capture=capture))
        files = {
            "sealed/PLAN.json": dict(evaluation_requests=[old]),
            "campaign_001/canonicals/CANONICALS.json": {"old": dict(source="fresh_qualified_verifier_once",
                verified=True, binding_digest="target-runtime", source_record_path="/remote/production_001/campaign_001/canonicals/canonical_0000.json")},
            "campaign_001/canonicals/canonical_0000.json": raw,
            "native/x.json": capture,
        }
        inventory = {}
        for relative, value in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            payload = path.read_bytes()
            inventory[relative] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        index = root / "inventory.json"
        index.write_text(json.dumps(dict(files=inventory)))
        return request, index

    def old_audit(self):
        return {}, {}, dict(binding_digest="target-runtime", binding={}, reused_count=0)

    def test_exact_reference_and_missing_panel_requests_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.block_depth.references.reuse_pr25_references", return_value=self.old_audit()), patch("experiments.block_depth.references.validate_record", return_value=[]):
            root = Path(tmp)
            request, index = self.fixture(root)
            missing = dict(request, request_id="missing", prompt_ids=[4, 5])
            references, audit = prepare_references([request, missing], pr25_root="unused", pr25_inventory="unused", vfd_root=root, vfd_inventory=index)
            self.assertEqual(set(references), {"new"})
            self.assertEqual(audit["missing_request_ids"], ["missing"])
            self.assertEqual(audit["model_work"], 0)
            self.assertFalse(audit["endpoint_performance_analysis"])

    def test_modified_source_is_rejected_even_if_raw_validator_would_accept(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.block_depth.references.reuse_pr25_references", return_value=self.old_audit()), patch("experiments.block_depth.references.validate_record", return_value=[]):
            root = Path(tmp)
            request, index = self.fixture(root)
            (root / "native/x.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "differs from archive"):
                prepare_references([request], pr25_root="unused", pr25_inventory="unused", vfd_root=root, vfd_inventory=index)


if __name__ == "__main__":
    unittest.main()
