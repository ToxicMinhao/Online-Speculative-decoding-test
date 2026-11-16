"""Exercise the actual package builder, archive unpacker and reference consumer."""
from copy import deepcopy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.persistent_prefix_withholding import staging as S
from experiments.persistent_prefix_withholding.references import file_identity
from experiments.persistent_prefix_withholding.task_runner import reference_task
from online_feedback.storage import write_json
from test_persistent_prefix_reference_join import audit_schema, ImmediateLedger, native_record


class PersistentStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.package, self.destination = self.base/"package", self.base/"execution"
        self.package.mkdir()
        self.archive = self.base/"package.tar.gz"
        panel, missing, reused = audit_schema()
        self.plan = dict(root=str(self.destination), references=str(self.destination/S.REFERENCE_NAME),
            input_identities={}, origin_export="fixed-K6", execution_commit="published",
            reference_provenance="sealed-proof.json")
        # Only immutable source artifacts exist here. The tested builder must
        # create the mutable seed; the fixture never supplies that destination.
        for key, value in (("data_panel", panel), ("missing_references", missing),
                           ("reused_references", reused)):
            path = self.base/"immutable"/(key+".json")
            write_json(path, value)
            self.plan[key] = str(path)
            self.plan["input_identities"][str(path)] = file_identity(path)
        write_json(self.package/"sealed/PLAN.json", self.plan)
        self.panel, self.reused = panel, reused

    def assemble(self):
        S.seed_reference_destination(self.plan, package_root=self.package)
        return S.build_package_archive(self.plan, self.package, self.archive)

    def invoke_consumer(self, plan, backend):
        ledger = ImmediateLedger()
        with patch("experiments.persistent_prefix_withholding.task_runner.create_backend", return_value=backend) as factory, \
             patch("experiments.persistent_prefix_withholding.task_runner.work_boundary"), \
             patch("experiments.persistent_prefix_withholding.task_runner.close_backend"):
            result = reference_task(plan, self.destination/"calls", ledger, "CPU-no-device")
        return result, ledger, factory

    def test_build_unpack_readiness_then_real_consumer_without_fixture_seed(self):
        self.assertFalse((self.package/S.REFERENCE_NAME).exists())
        self.assertFalse(Path(self.plan["references"]).exists())
        receipt = self.assemble()
        self.assertEqual(receipt["reference_seed"]["reused_references"], 104)
        with tarfile.open(self.archive) as archive:
            self.assertEqual([row.name for row in archive if row.name==str(S.REFERENCE_NAME)], [str(S.REFERENCE_NAME)])
        unpacked = S.unpack_package_archive(self.archive, self.destination)
        self.assertEqual(unpacked["reference_seed"]["missing_references"], 24)
        self.assertEqual(Path(self.plan["references"]).read_bytes(), Path(self.plan["reused_references"]).read_bytes())
        self.assertEqual(S.reference_readiness(unpacked["plan"])["panel_requests"], 128)
        actual = []
        def generate(prompt, **kwargs):
            actual.append((prompt,kwargs))
            record = native_record(prompt)
            write_json(kwargs["save_to"], record)
            return record
        backend = Mock(generate=generate, loading_s=0.0)
        result, ledger, factory = self.invoke_consumer(unpacked["plan"], backend)
        factory.assert_called_once()
        self.assertEqual(result["new_reference_generations"],24)
        self.assertEqual(len(actual),len(ledger.calls))
        for (prompt,kwargs),request in zip(actual,self.panel["requests"][104:]):
            self.assertEqual(prompt,request["prompt_ids"])
            self.assertEqual(kwargs["request_index"],request["panel_ordinal"])
            self.assertEqual(kwargs["purpose"],"evaluation")
        self.assertEqual(len(json.loads(Path(self.plan["references"]).read_text())),128)

    def test_original_omitted_seed_cannot_be_packaged(self):
        with self.assertRaises(FileNotFoundError):
            S.build_package_archive(self.plan,self.package,self.archive)
        self.assertFalse(self.archive.exists())

    def test_archive_missing_seed_fails_unpack_before_backend(self):
        # Reproduce the historical packaging omission, rather than repairing it
        # in the test fixture before exercising the consumer.
        with tarfile.open(self.archive,"w:gz") as archive:
            archive.add(self.package/"sealed/PLAN.json",arcname="sealed/PLAN.json")
        backend=Mock()
        with patch("experiments.persistent_prefix_withholding.task_runner.create_backend",backend):
            with self.assertRaises(FileNotFoundError):
                S.unpack_package_archive(self.archive,self.destination)
        backend.assert_not_called()

    def test_existing_partial_seed_is_never_overwritten(self):
        destination=self.package/S.REFERENCE_NAME
        write_json(destination,{"partial":"preserve"})
        before=destination.read_bytes()
        with self.assertRaises(FileExistsError):
            S.seed_reference_destination(self.plan,package_root=self.package)
        self.assertEqual(destination.read_bytes(),before)

    def test_even_identical_seed_cannot_be_reinitialized(self):
        S.seed_reference_destination(self.plan,package_root=self.package)
        before=(self.package/S.REFERENCE_NAME).read_bytes()
        with self.assertRaises(FileExistsError):
            S.seed_reference_destination(self.plan,package_root=self.package)
        self.assertEqual((self.package/S.REFERENCE_NAME).read_bytes(),before)

    def test_changed_seed_fails_archive_and_staged_readiness(self):
        self.assemble()
        S.unpack_package_archive(self.archive,self.destination)
        write_json(self.destination/S.REFERENCE_NAME,{})
        with self.assertRaises(ValueError):
            S.reference_readiness(self.plan)
        write_json(self.package/S.REFERENCE_NAME,{})
        with self.assertRaises(ValueError):
            S.build_package_archive(self.plan,self.package,self.base/"changed.tar.gz")

    def test_same_json_with_changed_seed_bytes_is_rejected(self):
        S.seed_reference_destination(self.plan,package_root=self.package)
        (self.package/S.REFERENCE_NAME).write_text(json.dumps(self.reused))
        with self.assertRaisesRegex(ValueError,"exact immutable"):
            S.reference_readiness(self.plan,package_root=self.package)

    def test_wrong_destination_cannot_seed_or_package(self):
        altered=deepcopy(self.plan)
        altered["references"]=str(self.base/"previous-run/CANONICALS.json")
        with self.assertRaisesRegex(ValueError,"new run"):
            S.seed_reference_destination(altered,package_root=self.package)
        self.assertFalse((self.package/S.REFERENCE_NAME).exists())

    def test_destination_parent_symlink_cannot_escape_run(self):
        outside=self.base/"outside";outside.mkdir()
        (self.package/"references").symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"escapes"):
            S.seed_reference_destination(self.plan,package_root=self.package)
        self.assertFalse((outside/"CANONICALS.json").exists())

    def test_wrong_coverage_is_rejected_before_seed_creation(self):
        path=Path(self.plan["missing_references"])
        value=json.loads(path.read_text());value[0]=deepcopy(value[1])
        write_json(path,value)
        self.plan["input_identities"][str(path)]=file_identity(path)
        with self.assertRaises(ValueError):
            S.seed_reference_destination(self.plan,package_root=self.package)
        self.assertFalse((self.package/S.REFERENCE_NAME).exists())

    def test_unpacked_files_are_never_replaced(self):
        self.assemble()
        S.unpack_package_archive(self.archive,self.destination)
        before=Path(self.plan["references"]).read_bytes()
        with self.assertRaises(FileExistsError):
            S.unpack_package_archive(self.archive,self.destination)
        self.assertEqual(Path(self.plan["references"]).read_bytes(),before)

    def test_local_preparation_resolves_authenticated_remote_inputs(self):
        remote=deepcopy(self.plan);mapping={}
        for key in ("data_panel","missing_references","reused_references"):
            old=remote[key];name="/retained/server/"+Path(old).name
            mapping[name]=Path(old);remote[key]=name
            remote["input_identities"][name]=remote["input_identities"].pop(old)
        write_json(self.package/"sealed/PLAN.json",remote)
        receipt=S.seed_reference_destination(remote,package_root=self.package,input_resolver=mapping.__getitem__)
        self.assertEqual(receipt["immutable_source"],remote["reused_references"])
        S.build_package_archive(remote,self.package,self.archive,input_resolver=mapping.__getitem__)
        result=S.unpack_package_archive(self.archive,self.destination,input_resolver=mapping.__getitem__)
        self.assertEqual(result["reference_seed"]["panel_requests"],128)


if __name__=="__main__":
    unittest.main()
