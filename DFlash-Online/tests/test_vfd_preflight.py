"""Immutable-start and reference-admission failures precede model work."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from experiments.vfd_second_pass import preflight as P
from experiments.vfd_second_pass.contract import ORIGINAL_EXECUTION, SERVING, digest


class VFDPreflightTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.original = self.root / "original"
        (self.original / "sealed").mkdir(parents=True)
        self.rows = [
            dict(
                request_id=f"train_{i:05d}",
                dataset_id=str(i),
                prompt_ids=[i + 1],
                event_id=f"event_{i//128+1:03d}",
            )
            for i in range(512)
        ]
        self.old_serving = {k: v for k, v in SERVING.items() if k != "capture"}
        self.old_serving.update(
            reasoning="off",
            cache_policy="idle_full_flush_before_every_request_and_publication",
        )
        self.write(
            "sealed/PLAN.json",
            dict(
                execution_source=dict(commit=ORIGINAL_EXECUTION),
                binding="a" * 64,
                serving=self.old_serving,
            ),
        )
        self.write("sealed/ORIGINAL_MANIFEST.json", dict(train_requests=self.rows))
        (self.original / "sealed/config_KPLUS.yaml").write_text("original K+ config")
        initial = self.original / "campaign_001/arms/KPLUS/trainer/initial_state.pt"
        initial.parent.mkdir(parents=True)
        initial.write_bytes(b"authenticated event zero")
        self.plan = dict(
            original_root=str(self.original),
            original_initial_state=str(initial),
            original_config=str(self.original / "sealed/config_KPLUS.yaml"),
            training_requests=deepcopy(self.rows),
            serving=dict(SERVING),
            evaluation_requests=[
                dict(request_id=f"eval_{i:03d}", prompt_ids=[i + 1]) for i in range(200)
            ],
        )
        self.audit = dict(
            status="passed",
            errors=[],
            model_forwards=0,
            optimizer_steps=0,
            cuda_initialized=False,
            initial_state=dict(
                global_step=0,
                presentations=0,
                events=[],
                adam_state_entries=0,
                released_weights_exact=True,
                rng_equal_original_K=True,
                parameter_tensors=58,
                cuda_rng_states=1,
                horizon=7424,
                accumulation_steps=2,
                reduction="response_mean",
                identity=dict(path=str(initial), **P.identity(initial)),
            ),
            seals={
                p.name: dict(path=str(p), **P.identity(p))
                for p in (self.original / "sealed").iterdir()
            },
            stream=dict(ordered_512_sha256=digest(self.rows)),
        )

    def write(self, relative, value):
        path = self.original / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_zero_order_and_initial_identity(self):
        self.assertEqual(P.verify_original(self.plan, self.audit)["status"], "passed")
        bad = deepcopy(self.plan)
        bad["training_requests"][0], bad["training_requests"][1] = (
            bad["training_requests"][1],
            bad["training_requests"][0],
        )
        with self.assertRaisesRegex(ValueError, "identity/order"):
            P.verify_original(bad, self.audit)
        bad = deepcopy(self.audit)
        bad["initial_state"]["global_step"] = 7424
        with self.assertRaisesRegex(ValueError, "Event-zero"):
            P.verify_original(self.plan, bad)
        bad = deepcopy(self.plan)
        bad["original_initial_state"] = bad["original_initial_state"].replace(
            "initial_state.pt", "training_state.pt"
        )
        with self.assertRaisesRegex(ValueError, "event-zero"):
            P.verify_original(bad, self.audit)
        Path(self.plan["original_initial_state"]).write_bytes(
            b"changed state same length"
        )
        with self.assertRaisesRegex(ValueError, "Authenticated input changed"):
            P.verify_original(self.plan, self.audit)

    def references(self):
        models = dict(
            target=dict(files={"config.json": dict(bytes=3, sha256="target")})
        )
        runtime = dict(
            native_runtime_sha256={"native.py": "native"},
            python_version="3.12.14",
            torch="2.9.1+cu130",
            transformers="4.57.1",
        )
        binding = dict(
            target_files=models["target"]["files"], **runtime, serving=self.old_serving
        )
        refs = {
            r["request_id"]: dict(
                prompt_ids=r["prompt_ids"],
                response_ids=[2],
                verified=True,
                binding_digest=digest(binding),
            )
            for r in self.plan["evaluation_requests"][:142]
        }
        missing = dict(
            request_ids=[
                r["request_id"] for r in self.plan["evaluation_requests"][142:]
            ],
            count=58,
            maximum_new_generations=58,
        )
        audit = dict(
            status="passed",
            model_work=0,
            binding=binding,
            binding_digest=digest(binding),
            reused_count=142,
            missing_count=58,
            checked_files={},
        )
        return [audit, models, runtime, refs, missing]

    def test_reference_partition_runtime_and_raw_subset(self):
        args = self.references()
        with patch(
            "experiments.vfd_second_pass.evaluation.audit_canonicals",
            return_value=dict(status="passed", references_checked=142),
        ) as raw:
            self.assertEqual(P.verify_references(self.plan, *args)["reused"], 142)
            self.assertEqual(len(raw.call_args.args[0]), 142)
            self.assertEqual(set(raw.call_args.args[1]), set(args[3]))
            for index in (1, 2):
                changed = deepcopy(args)
                if index == 1:
                    changed[1]["target"]["files"]["config.json"]["sha256"] = "changed"
                else:
                    changed[2]["native_runtime_sha256"]["native.py"] = "changed"
                with self.assertRaisesRegex(ValueError, "target/runtime"):
                    P.verify_references(self.plan, *changed)
            changed = deepcopy(args)
            changed[4]["request_ids"][0] = "eval_000"
            with self.assertRaisesRegex(ValueError, "partition"):
                P.verify_references(self.plan, *changed)
        with patch(
            "experiments.vfd_second_pass.evaluation.audit_canonicals",
            return_value=dict(status="invalid_or_incomplete", errors=["changed token"]),
        ):
            with self.assertRaisesRegex(ValueError, "raw validation failed"):
                P.verify_references(self.plan, *args)

    def test_serving_change_or_reference_path_escape(self):
        args = self.references()
        bad = deepcopy(self.plan)
        bad["serving"]["max_new_tokens"] = 4096
        with self.assertRaisesRegex(ValueError, "serving setting"):
            P.verify_references(bad, *args)
        args[0]["checked_files"]["../foreign.json"] = {}
        with self.assertRaisesRegex(ValueError, "escapes"):
            P.verify_references(self.plan, *args)

    def test_failed_source_gate_records_zero_model_work(self):
        with patch.object(P, "verify_seal", side_effect=ValueError("changed source")):
            with self.assertRaisesRegex(ValueError, "changed source"):
                P.verify(self.root)
        receipt = json.loads((self.root / "PREFLIGHT.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        for key in ("model_forwards", "optimizer_steps", "generations"):
            self.assertEqual(receipt[key], 0)


if __name__ == "__main__":
    unittest.main()
