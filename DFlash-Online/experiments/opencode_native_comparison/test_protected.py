"""Protected dependency fixtures exercise production checks without model files."""
import json
import os

import pytest

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from . import protected, storage


def seed_protection(root):
    """Explicit tiny authenticated keeper; no production missing-receipt bypass."""
    root.mkdir(parents=True,exist_ok=True)
    source=root/'SOURCE.json'
    write_json(source,dict(commit='fixture',dirty=False))
    keeper=root/'released_dependency.bin'
    keeper.write_bytes(b'released')
    protected.capture_authenticated(dict(root=str(root),source_manifest=str(source),
        input_identities={},authenticated_files=[dict(path=str(keeper),expected_identity=identity(keeper))],
        execution_commit='fixture',binding='fixture'))
    return keeper


def test_stat_checks_and_final_hash_checks_are_explicitly_different(tmp_path):
    keeper=seed_protection(tmp_path)
    before=keeper.stat()
    assert protected.check(tmp_path)['mode']=='stat_identity_only'
    keeper.write_bytes(b'changed!')
    os.utime(keeper,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert protected.check(tmp_path)['status']=='passed'
    with pytest.raises(ValueError,match='content changed'):
        protected.check(tmp_path,full_hash=True,purpose='postflight')
    rows=[json.loads(line) for line in (tmp_path/'PROTECTED_RECHECKS.jsonl').read_text().splitlines()]
    assert rows[-1]['status']=='failed' and rows[-1]['mode']=='full_content_hash_and_stat'


def test_missing_receipt_blocks_retirement_before_deletion(tmp_path):
    candidate=tmp_path/'disposable.pt'
    candidate.write_bytes(b'fixture')
    with pytest.raises(FileNotFoundError):
        storage._retire(tmp_path,tmp_path/'retire.json',[(candidate,'fixture')],[],reason='fixture',retained={})
    assert candidate.exists()


def test_outside_or_protected_candidate_cannot_be_deleted(tmp_path):
    root=tmp_path/'study'
    keeper=seed_protection(root)
    outsider=tmp_path/'outside.pt'
    outsider.write_bytes(b'keep')
    alias=root/'disposable_looking_hardlink.pt'
    os.link(keeper,alias)
    for candidate in (keeper,outsider,alias):
        with pytest.raises(ValueError,match='scope or remove a protected'):
            protected.require_retirement_paths(root,[candidate])
        assert candidate.exists()


def test_deleted_or_replaced_dependency_fails_stat_check(tmp_path):
    keeper=seed_protection(tmp_path)
    keeper.unlink()
    with pytest.raises(FileNotFoundError):
        protected.check(tmp_path)


def test_dependency_recheck_runs_even_when_a_deletion_batch_fails(tmp_path,monkeypatch):
    seed_protection(tmp_path)
    candidate=tmp_path/'disposable.pt'
    candidate.write_bytes(b'fixture')
    def partial_failure(*args,**kwargs):
        candidate.unlink()
        raise RuntimeError('retained partial failure')
    monkeypatch.setattr(storage,'native_retire',partial_failure)
    with pytest.raises(RuntimeError,match='partial failure'):
        storage._retire(tmp_path,tmp_path/'retire.json',[(candidate,'fixture')],[],reason='fixture',retained={})
    rows=[json.loads(line) for line in (tmp_path/'PROTECTED_RECHECKS.jsonl').read_text().splitlines()]
    assert rows[-1]['status']=='passed' and rows[-1]['purpose'].startswith('tensor_retirement:')


def test_native_feature_wrapper_rechecks_dependencies(tmp_path,monkeypatch):
    seed_protection(tmp_path)
    paths=[tmp_path/name for name in ('native.pt','terminal.pt')]
    for path in paths:path.write_bytes(b'fixture')
    record=dict(record_id='fixture',metadata=dict(capture=dict(feature_path=str(paths[0]))))
    prepared=dict(terminal_feature=dict(tensor_path=str(paths[1])))
    def retire(*args):
        for path in paths:path.unlink()
        return {'status':'completed'}
    monkeypatch.setattr(storage,'native_retire_redundant_features',retire)
    assert storage.retire_redundant_features(record,prepared,tmp_path,tmp_path)['status']=='completed'
    result=json.loads((tmp_path/'PROTECTED_RECHECKS.jsonl').read_text().splitlines()[-1])
    assert result['purpose']=='native_feature_retirement:fixture' and result['status']=='passed'
