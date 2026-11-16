"""CPU-only accounting of authenticated old source trees, without following links."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import task_runner as T


class PriorSourceAccountingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.current, self.prior = self.root / "source002", self.root / "source001"
        for source in (self.current, self.prior):
            (source / "docs/web").mkdir(parents=True)
            (source / "docs/data").mkdir()
            (source / "docs/data/content").write_bytes(b"retained source" * 500)
            (source / "docs/web/data").symlink_to("../data")
        self.current_manifest = self.write_manifest("SOURCE002.json", "new")
        self.prior_manifest = self.write_manifest("SOURCE001.json", "old")
        self.plan = dict(root=str(self.root), source_root=str(self.current),
            source_manifest=str(self.current_manifest), execution_commit="new",
            allocation_roots=[str(self.current), str(self.prior)],
            storage={"peak_new_bytes": 1}, compiler_growth_reserved_bytes=0,
            prior_source_manifests=[dict(source_root=str(self.prior),
                source_manifest=str(self.prior_manifest), execution_commit="old",
                manifest_identity=self.identity(self.prior_manifest))])
        self.free = patch.object(T.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12))
        self.free.start()
        self.addCleanup(self.free.stop)

    def write_manifest(self, name, commit):
        path = self.root / name
        T.write_json(path, dict(commit=commit, dirty=False,
            files={"docs/web/data": dict(kind="symlink", target="../data")}))
        return path

    @staticmethod
    def identity(path):
        payload = path.read_bytes()
        return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def boundary(self):
        return T.storage_boundary(self.plan, check_compiler=False)

    def test_old_and_current_sources_count_once_per_inode_without_following_links(self):
        # A hardlink and repeated allocation root must not charge the same inode twice.
        os.link(self.prior / "docs/data/content", self.current / "retained_hardlink")
        self.plan["allocation_roots"].append(str(self.prior))
        expected, seen = 0, set()
        for source in (self.current, self.prior):
            paths = [source]
            for directory, folders, files in os.walk(source, followlinks=False):
                paths.extend(Path(directory) / name for name in folders + files)
            for path in paths:
                state = path.lstat()
                inode = state.st_dev, state.st_ino
                if inode not in seen:
                    expected += state.st_blocks * 512
                    seen.add(inode)
        receipt = self.boundary()
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["allocated_new_bytes"], expected)
        self.assertTrue((self.prior / "docs/web/data").is_symlink())

    def test_old_source_requires_explicit_pinned_manifest(self):
        self.plan.pop("prior_source_manifests")
        with self.assertRaisesRegex(ValueError, "internal source"):
            self.boundary()

    def test_changed_manifest_or_wrong_commit_cannot_authorize_old_links(self):
        self.prior_manifest.write_text(self.prior_manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.boundary()
        entry = self.plan["prior_source_manifests"][0]
        entry["manifest_identity"] = self.identity(self.prior_manifest)
        entry["execution_commit"] = "wrong"
        with self.assertRaisesRegex(ValueError, "clean published"):
            self.boundary()

    def test_equivalent_resolved_target_still_requires_literal_link_match(self):
        for source in (self.current, self.prior):
            with self.subTest(source=source.name):
                link = source / "docs/web/data"
                link.unlink()
                link.symlink_to("../data/.")
                with self.assertRaisesRegex(ValueError, "internal source"):
                    self.boundary()
                link.unlink()
                link.symlink_to("../data")

    def test_manifested_external_or_dangling_link_still_fails(self):
        link = self.prior / "docs/web/data"
        link.unlink()
        link.symlink_to(self.root)
        manifest = json.loads(self.prior_manifest.read_text())
        manifest["files"]["docs/web/data"]["target"] = str(self.root)
        T.write_json(self.prior_manifest, manifest)
        self.plan["prior_source_manifests"][0]["manifest_identity"] = self.identity(self.prior_manifest)
        with self.assertRaisesRegex(ValueError, "internal source"):
            self.boundary()
        link.unlink()
        link.symlink_to("../missing")
        manifest["files"]["docs/web/data"]["target"] = "../missing"
        T.write_json(self.prior_manifest, manifest)
        self.plan["prior_source_manifests"][0]["manifest_identity"] = self.identity(self.prior_manifest)
        with self.assertRaisesRegex(ValueError, "internal source"):
            self.boundary()

    def test_run_state_links_are_not_exempted_by_old_source_authorization(self):
        run = self.root / "run001"
        run.mkdir()
        (run / "state").symlink_to(self.prior / "docs/data")
        self.plan["allocation_roots"].append(str(run))
        with self.assertRaisesRegex(ValueError, "internal source"):
            self.boundary()

    def test_unknown_internal_source_link_is_rejected(self):
        (self.prior / "undeclared").symlink_to("docs/data")
        with self.assertRaisesRegex(ValueError, "internal source"):
            self.boundary()


if __name__ == "__main__":
    unittest.main()
