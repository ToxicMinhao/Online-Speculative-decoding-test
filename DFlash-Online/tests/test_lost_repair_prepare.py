"""CPU-only sealing, provenance and private-data boundary tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.lost_repair import prepare


def fixture_manifest():
    return dict(train_requests=[{}] * 768 + [dict(
        request_id=f"train_{i:05d}", dataset_id=f"source_{i}", dataset_index=i,
        request_index=i + 1, event_id="event_007", collect_feedback=True,
        prompt_ids=[1000 + i], max_new_tokens=2048,
    ) for i in range(768, 896)] + [dict(request_id="unused_event_8")])


def git(repository, *args):
    return subprocess.check_output(["git", *args], cwd=repository, text=True,
                                   stderr=subprocess.DEVNULL).strip()


def commit(repository):
    git(repository, "add", ".")
    git(repository, "-c", "user.name=CPU test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "fixture")


class SelectionTests(unittest.TestCase):
    def test_exact_contiguous_event_and_original_rows_are_preserved(self):
        manifest = fixture_manifest()
        rows = prepare.select_requests(manifest)
        self.assertEqual(rows, manifest["train_requests"][768:896])
        self.assertEqual(len(rows), 128)
        rows[0]["prompt_ids"].append(7)
        self.assertNotEqual(rows[0], manifest["train_requests"][768])

    def test_duplicates_are_rejected_without_replacing_requests(self):
        for field in ("dataset_id", "prompt_ids"):
            with self.subTest(field=field):
                manifest = fixture_manifest()
                manifest["train_requests"][769][field] = deepcopy(manifest["train_requests"][768][field])
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    prepare.select_requests(manifest)

    def test_identity_order_cap_and_context_are_required(self):
        for field, value in (("request_id", "wrong"), ("event_id", "event_008"),
                             ("request_index", 0), ("max_new_tokens", 1024),
                             ("prompt_ids", [1] * 2049)):
            with self.subTest(field=field):
                manifest = fixture_manifest()
                manifest["train_requests"][768][field] = value
                with self.assertRaises(ValueError):
                    prepare.select_requests(manifest)

    def test_short_manifest_is_not_silently_shrunk(self):
        manifest = fixture_manifest()
        manifest["train_requests"] = manifest["train_requests"][:895]
        with self.assertRaisesRegex(ValueError, "complete 128"):
            prepare.select_requests(manifest)


class SealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.code = self.repo / "DFlash-Online"
        self.code.mkdir(parents=True)
        self.runtime_file = self.code / "external_runtime/sglang/srt/speculative/dflash_worker.py"
        self.runtime_file.parent.mkdir(parents=True)
        self.runtime_file.write_text("# Pinned runtime fixture\n")
        (self.code / "module.py").write_text("# Tracked experimental code\n")
        (self.code / "module_link.py").symlink_to("module.py")
        (self.repo / "unrelated.txt").write_text("outside DFlash-Online\n")
        git(self.repo, "init", "-q")
        commit(self.repo)
        self.output = self.root / "private_package"
        self.remote = Path("/data/sources/minhao/runs/lost_repair_test/execution_001")
        self.manifest = self.root / "manifest.json"
        prepare.write(self.manifest, fixture_manifest())
        manifest_identity = prepare.identity(self.manifest)
        rows = fixture_manifest()["train_requests"][768:896]
        rows_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.patch_manifest = patch.object(prepare, "MANIFEST_IDENTITY", manifest_identity)
        self.patch_rows = patch.object(prepare, "ROWS_SHA256", rows_hash)
        self.patch_manifest.start()
        self.patch_rows.start()
        self.addCleanup(self.patch_manifest.stop)
        self.addCleanup(self.patch_rows.stop)
        self.config = self.root / "config_KPLUS.yaml"
        self.config.write_bytes(b"training:\n  seed: 17\n  learning_rate: 0.0003\n")
        small_identity = dict(bytes=17, sha256="a" * 64)
        runtime_hashes = {"speculative/dflash_worker.py": prepare.identity(self.runtime_file)["sha256"]}
        model = dict(target=dict(files={"config.json": small_identity}),
                     draft=dict(files={"config.json": small_identity}))
        runtime = dict(python="/worker/bin/python", python_version="3.12.14",
                       torch="2.9.1+cu130", transformers="4.57.1", cuda="13.0",
                       native_runtime_sha256=runtime_hashes)
        training = dict(environment=dict(python="3.12.15", torch="2.13.0+cu130",
                                         transformers="5.12.1"))
        pins = {}
        for name, document in (("MODEL_PIN.json", model), ("RUNTIME_PIN.json", runtime),
                               ("TRAINING_PIN.json", training)):
            path = self.root / name
            prepare.write(path, document)
            pins[name] = dict(local=str(path), sha256=prepare.identity(path)["sha256"])
        self.k6 = "/data/sources/minhao/runs/native/trainer/events/event_006/export"
        self.inventory_data = dict(
            manifest=dict(manifest_identity, original_rows_canonical_json_sha256=rows_hash),
            stop_token_ids=[151645, 151643], pin_files=pins, config_path=str(self.config),
            target_path="/data/sources/minhao/models/target",
            draft_foundation_path="/data/sources/minhao/models/draft", k6_export=self.k6,
            k6_recovery_path_reported_by_parent_remote_audit="/data/sources/minhao/runs/native/trainer/training_state.pt",
            worker_python="/worker/bin/python", training_python="/trainer/bin/python",
            serving=dict(block_size=16, max_new_tokens=2048, context_limit=4096, concurrency=1),
        )
        self.inventory = self.root / "inventory.json"
        prepare.write(self.inventory, self.inventory_data)
        self.runtime_plan = self.root / "runtime_plan.json"
        prepare.write(self.runtime_plan, dict(
            shared=dict(target_path=self.inventory_data["target_path"],
                        target_files=model["target"]["files"], runtime=runtime),
            checkpoints=dict(gsm8k={
                "KPLUS4": dict(export="/data/sources/minhao/runs/native4/export",
                    weights_identity=dict(bytes=2097259104, sha256=prepare.WEIGHT_HASHES["K4"])),
                "KPLUS6": dict(export=self.k6,
                    weights_identity=dict(bytes=2097259104, sha256=prepare.WEIGHT_HASHES["K6"])),
            }),
        ))

    def seal(self):
        return prepare.seal(self.manifest, self.inventory, self.runtime_plan,
                            self.output, self.remote, code=self.code)

    def test_seal_binds_actual_remote_paths_and_preserves_config_bytes(self):
        plan = self.seal()
        self.assertEqual(plan["execution_commit"], git(self.repo, "rev-parse", "HEAD"))
        self.assertEqual(plan["source_manifest_path"], str(self.remote / "SOURCE.json"))
        self.assertEqual(plan["config_path"], str(self.remote / "sealed/config_KPLUS.yaml"))
        self.assertEqual(plan["output_root"], str(self.remote / "campaign_001"))
        self.assertEqual((self.output / "sealed/config_KPLUS.yaml").read_bytes(), self.config.read_bytes())
        self.assertEqual(plan["requests"][0]["stop_token_ids"], [151645, 151643])
        self.assertNotIn("stop_token_ids", prepare.read(self.manifest)["train_requests"][768])
        self.assertEqual(plan["pinned_files"][plan["recovery_path"]], prepare.RECOVERY_IDENTITY)
        self.assertEqual(plan["storage"]["required_free_bytes"], 48 * 1024**3)
        self.assertEqual(plan["gpu_seconds_limit"], 14400)
        self.assertEqual(plan["scope"]["optimizer_steps"], 0)
        self.assertEqual(plan["versions"]["training"]["python"], "3.12.15")
        self.assertEqual(plan["versions"]["serving"]["python"], "3.12.14")
        self.assertTrue(all(path.startswith("/data/") for path in plan["pinned_files"]))
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")
        seal = prepare.read(self.output / "sealed/SEAL.json")
        self.assertEqual(seal["files"][str(self.remote / "sealed/PLAN.json")],
                         prepare.identity(self.output / "sealed/PLAN.json"))

    def test_source_manifest_covers_only_tracked_code_and_keeps_symlinks(self):
        source = prepare.source_manifest(self.code)
        self.assertIn("module.py", source["files"])
        self.assertNotIn("unrelated.txt", source["files"])
        self.assertEqual(source["symlinks"], {"module_link.py": "module.py"})

    def test_dirty_checkout_stops_before_writing_private_package(self):
        (self.code / "module.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "clean"):
            self.seal()
        self.assertFalse(self.output.exists())

    def test_manifest_tamper_is_rejected_before_copying(self):
        self.manifest.write_text(self.manifest.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "Pinned local"):
            self.seal()
        self.assertFalse(self.output.exists())

    def test_later_event_recovery_path_is_not_silently_accepted(self):
        self.inventory_data["k6_recovery_path_reported_by_parent_remote_audit"] = self.k6 + "/training_state.pt"
        prepare.write(self.inventory, self.inventory_data)
        with self.assertRaisesRegex(ValueError, "trainer-boundary"):
            self.seal()

    def test_changed_native_runtime_is_not_hidden_by_a_new_commit(self):
        self.runtime_file.write_text("# changed runtime\n")
        commit(self.repo)
        with self.assertRaisesRegex(ValueError, "Bundled native runtime"):
            self.seal()

    def test_private_package_cannot_be_written_inside_repository(self):
        self.output = self.repo / "private_requests"
        with self.assertRaisesRegex(ValueError, "Private request"):
            self.seal()

    def test_existing_package_is_not_overwritten(self):
        self.seal()
        with self.assertRaises(FileExistsError):
            self.seal()

    def test_prepare_import_requires_only_standard_library(self):
        script = ("import sys; import experiments.lost_repair.prepare; "
                  "assert not any(n in sys.modules for n in ('torch','numpy','transformers','yaml'))")
        subprocess.check_call([sys.executable, "-S", "-c", script],
                              cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
