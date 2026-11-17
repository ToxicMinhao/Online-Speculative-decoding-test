"""CPU-only exact source/metadata assembly and real reference-consumer checks."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments.bounded_prefix_allocation import staging as S
from experiments.bounded_prefix_allocation.prepare import _write
from test_bounded_prefix_prepare import fixture


class FullBundleTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.source=self.root/'checkout';self.source.mkdir()
        self.package=self.root/'package'
        self.plan=fixture(self.package)
        self.plan.update(execution_commit='c'*40,source_root='/data/sources/minhao/test_source/DFlash-Online',
                         source_manifest=self.plan['root']+'/SOURCE.json')
        _write(self.package/'sealed/PLAN.json',self.plan)
        (self.source/'train.py').write_text('print("not executed")\n')
        (self.source/'training/docs/data').mkdir(parents=True)
        (self.source/'training/docs/data/index.txt').write_text('documentation')
        (self.source/'training/docs/web').mkdir()
        (self.source/'training/docs/web/data').symlink_to('../data')
        self.files={str(path.relative_to(self.source)): (
            dict(kind='symlink',target=str(path.readlink())) if path.is_symlink() else S.identity(path))
            for path in self.source.rglob('*') if path.is_symlink() or path.is_file()}
        _write(self.package/'SOURCE.json',dict(commit='c'*40,dirty=False,files=self.files))
        self.archive=self.root/'bundle.tar.gz'

    @contextmanager
    def clean_git_fixture(self,dirty=''):
        # Git state is synthetic; the actual source bytes and packaged reader
        # below are real. The full published checkout is separately dry-staged.
        def git(command,**kwargs):
            if 'rev-parse' in command:return 'c'*40+'\n'
            if 'status' in command:return dirty
            if 'ls-files' in command:return ('\0'.join(self.files)+'\0').encode()
            raise AssertionError(command)
        with patch.object(S.subprocess,'check_output',side_effect=git):yield

    def build(self):
        with self.clean_git_fixture():
            return S.build_bundle(self.source,self.package,self.archive)

    def unpack(self,root=None,source=None,archive=None):
        path=archive or self.archive
        return S.unpack_bundle(path,root or self.root/'run',source or self.root/'staged',
                               expected_identity=S.identity(path))

    def test_full_namespaces_internal_link_and_actual_reference_payload(self):
        built=self.build()
        result=self.unpack()
        self.assertEqual(result['source']['files_checked'],len(self.files))
        self.assertEqual(result['readiness']['complete_references'],128)
        self.assertEqual(result['readiness']['new_reference_generations'],0)
        self.assertEqual(result['model_work'],0)
        self.assertTrue((self.root/'staged/training/docs/web/data').is_symlink())
        self.assertEqual((self.root/'staged/train.py').read_bytes(),(self.source/'train.py').read_bytes())
        self.assertEqual((self.root/'run/references/CANONICALS.json').read_bytes(),
                         (self.package/'references/CANONICALS.json').read_bytes())
        self.assertEqual(built['archive_identity'],S.identity(self.archive))
        self.assertFalse(result['automatic_launch'])
        with self.assertRaises(FileExistsError):self.unpack()

    def test_unpack_fresh_process_imports_no_torch(self):
        self.build()
        code='''import json,sys
from experiments.bounded_prefix_allocation.staging import unpack_bundle,identity
result=unpack_bundle(sys.argv[1],sys.argv[2],sys.argv[3],expected_identity=identity(sys.argv[1]))
assert 'torch' not in sys.modules
print(json.dumps(result))'''
        process=subprocess.run([sys.executable,'-c',code,str(self.archive),str(self.root/'run'),str(self.root/'staged')],
                               text=True,capture_output=True,timeout=30)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.assertFalse(json.loads(process.stdout)['torch_imported'])

    def test_dirty_source_or_missing_references_prevents_bundle(self):
        with self.clean_git_fixture(dirty=' M train.py\n'):
            with self.assertRaisesRegex(ValueError,'clean'):
                S.build_bundle(self.source,self.package,self.archive)
        self.assertFalse(self.archive.exists())
        (self.package/'references/CANONICALS.json').unlink()
        with self.assertRaises(FileNotFoundError):self.build()
        self.assertFalse(self.archive.exists())

    def test_modified_source_is_not_hidden_by_clean_git_claim(self):
        (self.source/'train.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Source bytes'):self.build()
        self.assertFalse(self.archive.exists())

    def test_checksum_and_existing_destination_fail_before_writing(self):
        self.build()
        with self.assertRaisesRegex(ValueError,'archive identity'):
            S.unpack_bundle(self.archive,self.root/'run',self.root/'staged',
                            expected_identity=dict(bytes=1,sha256='0'*64))
        self.assertFalse((self.root/'staged').exists())
        destination=self.root/'run/sealed/PLAN.json'
        destination.parent.mkdir(parents=True);destination.write_text('old evidence')
        with self.assertRaises(FileExistsError):self.unpack()
        self.assertEqual(destination.read_text(),'old evidence')
        self.assertFalse((self.root/'staged').exists())

    def test_escape_links_hardlinks_and_duplicate_members_are_rejected(self):
        self.build()
        for index,(name,kind,target) in enumerate([
                ('source/training/escape',tarfile.SYMTYPE,'../../../outside'),
                ('package/escape',tarfile.SYMTYPE,'../source/train.py'),
                ('source/hard',tarfile.LNKTYPE,'source/train.py'),
                ('../escaped',tarfile.REGTYPE,''),
                ('source/train.py',tarfile.REGTYPE,'')]):
            archive=self.root/f'bad_{index}.tar.gz'
            with tarfile.open(self.archive,'r:gz') as original,tarfile.open(archive,'w:gz') as output:
                for member in original.getmembers():
                    output.addfile(member,original.extractfile(member) if member.isfile() else None)
                bad=tarfile.TarInfo(name);bad.type=kind;bad.linkname=target
                output.addfile(bad,io.BytesIO(b''))
            with self.assertRaises(ValueError):self.unpack(archive=archive)
            self.assertFalse((self.root/'staged').exists())

    def test_cli_requires_approved_paths_and_helper_without_unpacking(self):
        with patch.object(sys,'argv',['staging','--archive','/tmp/input','--root','/tmp/root',
                '--source-root','/tmp/source','--sha256','0'*64,'--bytes','1']), \
                patch.object(S,'unpack_bundle') as unpack:
            with self.assertRaisesRegex(ValueError,'approved A800'):
                S.main()
            unpack.assert_not_called()


class SourceReuseTests(FullBundleTests):
    def setUp(self):
        super().setUp()
        self.old_source=self.root/'old_source'
        shutil.copytree(self.source,self.old_source,symlinks=True)
        self.old_manifest=self.root/'old_SOURCE.json'
        _write(self.old_manifest,dict(commit='b'*40,dirty=False,files=self.files))
        (self.source/'train.py').write_text('print("revised but not executed")\n')
        (self.source/'new.py').write_text('# new source\n')
        self.files={str(path.relative_to(self.source)): (
            dict(kind='symlink',target=str(path.readlink())) if path.is_symlink() else S.identity(path))
            for path in self.source.rglob('*') if path.is_symlink() or path.is_file()}
        _write(self.package/'SOURCE.json',dict(commit='c'*40,dirty=False,files=self.files))

    def build(self):
        with self.clean_git_fixture():
            return S.build_bundle(self.source,self.package,self.archive,
                base_source_manifest=dict(path=str(self.old_manifest),identity=S.identity(self.old_manifest)),
                base_source_root=str(self.old_source))

    def test_reuse_copies_unchanged_and_transfers_changed_and_new_source(self):
        built=self.build()
        self.assertEqual(built['source_reuse']['transferred'],['new.py','train.py'])
        with tarfile.open(self.archive) as archive:
            self.assertEqual({name for name in archive.getnames() if name.startswith('source/')},
                             {'source/new.py','source/train.py'})
            self.assertIn('package/references/CANONICALS.json',archive.getnames())
        result=self.unpack()
        self.assertEqual(result['source']['files_checked'],len(self.files))
        self.assertEqual(result['readiness']['complete_references'],128)
        old=self.old_source/'training/docs/data/index.txt'
        copied=self.root/'staged/training/docs/data/index.txt'
        self.assertNotEqual(old.stat().st_ino,copied.stat().st_ino)
        self.assertEqual(old.read_bytes(),copied.read_bytes())
        self.assertNotEqual((self.old_source/'train.py').read_bytes(),(self.root/'staged/train.py').read_bytes())
        self.assertTrue((self.root/'staged/training/docs/web/data').is_symlink())

    def test_missing_or_modified_base_fails_before_any_destination_write(self):
        self.build()
        old=self.old_source/'training/docs/data/index.txt'
        content=old.read_bytes();old.unlink()
        with self.assertRaises(FileNotFoundError):self.unpack()
        self.assertFalse((self.root/'staged').exists())
        old.write_bytes(content+b'changed')
        with self.assertRaisesRegex(ValueError,'Source bytes'):self.unpack()
        self.assertFalse((self.root/'run').exists())

    def test_bound_manifest_mismatch_fails_without_bundle(self):
        with self.clean_git_fixture():
            with self.assertRaisesRegex(ValueError,'manifest identity'):
                S.build_bundle(self.source,self.package,self.archive,
                    base_source_manifest=dict(path=str(self.old_manifest),identity={'bytes':0,'sha256':'0'*64}),
                    base_source_root=str(self.old_source))
        self.assertFalse(self.archive.exists())

    def test_local_override_preserves_remote_manifest_and_authenticates_mirror(self):
        remote='/data/sources/minhao/retained_test_source/DFlash-Online'
        with self.clean_git_fixture():
            S.build_bundle(self.source,self.package,self.archive,
                base_source_manifest=dict(path=str(self.old_manifest),identity=S.identity(self.old_manifest)),
                base_source_root=remote)
        original=(self.package/'SOURCE_REUSE.json').read_bytes()
        archive_identity=S.identity(self.archive)
        result=S.unpack_bundle(self.archive,self.root/'run',self.root/'staged',
            expected_identity=archive_identity,base_source_root_override=self.old_source)
        self.assertEqual(result['source_reuse_roots'],dict(original=remote,effective=str(self.old_source),local_override=True))
        self.assertEqual((self.root/'run/SOURCE_REUSE.json').read_bytes(),original)
        self.assertEqual(S.identity(self.archive),archive_identity)
        self.assertEqual(result['readiness']['complete_references'],128)
        self.assertEqual(result['source']['files_checked'],len(self.files))
        (self.old_source/'train.py').write_text('modified base')
        with self.assertRaisesRegex(ValueError,'Source bytes'):
            S.unpack_bundle(self.archive,self.root/'run2',self.root/'staged2',
                expected_identity=archive_identity,base_source_root_override=self.old_source)
        self.assertFalse((self.root/'run2').exists())

    def test_override_is_rejected_for_production_before_any_file_work(self):
        with patch.object(S,'identity') as checked:
            with self.assertRaisesRegex(ValueError,'forbidden'):
                S.unpack_bundle(self.archive,self.root/'run',self.root/'staged',
                    expected_identity={},require_plan_destinations=True,base_source_root_override=self.old_source)
            checked.assert_not_called()


if __name__=='__main__':unittest.main()
