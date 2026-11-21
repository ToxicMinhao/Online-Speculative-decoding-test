"""Lossless CPU packaging fixtures, including failures before any deletion."""
import json
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

import pytest

from online_feedback.storage import write_json
from . import compact
from .test_protected import seed_protection


def event_fixture(root, *, event=1):
    seed_protection(root)
    arm = 'C_CODE58'
    base = root / 'campaign_001' / 'arms' / arm
    event_root = base / 'events' / f'event_{event:03d}'
    trainer = base / 'trainer' / 'events' / f'event_{event:03d}'
    records = []
    for ordinal in range(128):
        path = base / 'native' / f'request_{ordinal:06d}.json'
        write_json(path, dict(response_ids=[ordinal, 99], prompt_ids=[1]))
        records.append(dict(record_id=f'record_{ordinal}', metadata=dict(native_trace_path=str(path))))
    write_json(event_root / 'loop/feedback_buffer/state.json', dict(records=records))
    write_json(event_root / 'COMMITTED.json', dict(status='passed'))
    write_json(trainer / 'RESULT.json', dict(event=event, status='completed'))
    write_json(trainer / 'export/config.json', dict(keep='required by export reload'))
    (trainer / 'training_state.pt').write_bytes(b'recovery keeper')
    (trainer / 'export/model.safetensors').write_bytes(b'weight keeper')
    write_json(base / 'boundaries' / f'event_{event:03d}.json', dict(status='passed', event=event))
    write_json(base / 'STATUS.json', dict(events_completed=event, phase='idle_boundary'))
    return arm, records, event_root, trainer


def test_exact_bytes_survive_archive_and_only_redundant_compact_is_removed(tmp_path):
    arm, records, event_root, trainer = event_fixture(tmp_path)
    member = event_root / 'COMMITTED.json'
    original = member.read_bytes()
    result = compact.archive_completed_event(tmp_path, arm, 1, records)
    assert result['status'] == 'completed'
    assert result['verification']['regular_members'] == 131
    assert not member.exists()
    assert compact.read_member(result['archive_path'], member.relative_to(tmp_path)) == original
    assert (trainer / 'training_state.pt').read_bytes() == b'recovery keeper'
    assert (trainer / 'export/model.safetensors').read_bytes() == b'weight keeper'
    assert (trainer / 'export/config.json').exists()
    assert result['removed_allocated_bytes'] > 0
    assert result['lost_scientific_capabilities'] == []


def test_no_retirement_before_full_archive_verification(tmp_path, monkeypatch):
    arm, records, event_root, trainer = event_fixture(tmp_path)
    original = compact.verify_archive

    def reject(*args, **kwargs):
        # Even the first member remains while the complete replacement is checked.
        assert (event_root / 'COMMITTED.json').exists()
        assert all(Path(record['metadata']['native_trace_path']).exists() for record in records)
        original(*args, **kwargs)
        raise ValueError('fixture verification rejection')

    monkeypatch.setattr(compact, 'verify_archive', reject)
    with pytest.raises(ValueError, match='verification rejection'):
        compact.archive_completed_event(tmp_path, arm, 1, records)
    assert (event_root / 'COMMITTED.json').exists()
    assert (trainer / 'RESULT.json').exists()
    receipts = list((tmp_path / 'compact').rglob('*.retirement.json'))
    receipt = json.loads(receipts[0].read_text())
    assert receipt['status'] == 'failed' and receipt['removed_paths'] == []


def test_changed_member_during_retirement_stops_with_verified_archive(tmp_path, monkeypatch):
    arm, records, event_root, trainer = event_fixture(tmp_path)
    original = compact.verify_archive
    changed = event_root / 'COMMITTED.json'

    def mutate_after_verification(*args, **kwargs):
        result = original(*args, **kwargs)
        changed.write_text('{"changed":true}')
        return result

    monkeypatch.setattr(compact, 'verify_archive', mutate_after_verification)
    with pytest.raises(ValueError, match='changed after'):
        compact.archive_completed_event(tmp_path, arm, 1, records)
    assert changed.exists()
    assert (trainer / 'training_state.pt').exists()
    archive = next((tmp_path / 'compact').rglob('*.tar.zst'))
    assert json.loads(compact.read_member(archive, changed.relative_to(tmp_path))) == {'status': 'passed'}


def test_tampered_archive_or_index_is_rejected(tmp_path):
    arm, records, _, _ = event_fixture(tmp_path)
    result = compact.archive_completed_event(tmp_path, arm, 1, records)
    archive, index_path = Path(result['archive_path']), Path(result['index_path'])
    original_index = index_path.read_bytes()
    index = json.loads(original_index)
    next(iter(index['members'].values()))['bytes'] += 1
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='index binding'):
        compact.verify_archive(archive)
    index_path.write_bytes(original_index)
    with archive.open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='archive identity'):
        compact.verify_archive(archive)


@pytest.mark.parametrize('mutation,expected_error',[
    ('missing','missing expected'),('duplicate','duplicate'),('bytes','member bytes')])
def test_member_validation_is_independent_of_archive_checksum(tmp_path,mutation,expected_error):
    arm, records, _, _ = event_fixture(tmp_path)
    result=compact.archive_completed_event(tmp_path,arm,1,records)
    archive=Path(result['archive_path'])
    unpacked=subprocess.check_output(['zstd','-q','-d','-c',str(archive)])
    with tarfile.open(fileobj=io.BytesIO(unpacked),mode='r:') as original:
        rows=[(member,original.extractfile(member).read()) for member in original]
    member,payload=rows[0]
    if mutation=='missing':rows=rows[1:]
    elif mutation=='duplicate':rows.append((member,payload))
    else:rows[0]=(member,bytes([payload[0]^1])+payload[1:])
    packed=io.BytesIO()
    with tarfile.open(fileobj=packed,mode='w:') as output:
        for member,payload in rows:output.addfile(member,io.BytesIO(payload))
    archive.write_bytes(subprocess.run(['zstd','-q','-3','-c'],input=packed.getvalue(),
        stdout=subprocess.PIPE,check=True).stdout)
    index=json.loads(Path(result['index_path']).read_text())
    index.update(archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    index['binding_digest']=compact.digest({k:v for k,v in index.items() if k!='binding_digest'})
    write_json(result['index_path'],index)
    with pytest.raises(ValueError,match=expected_error):
        compact.verify_archive(archive)


def test_missing_member_is_not_replaced_or_ignored(tmp_path):
    arm, records, _, _ = event_fixture(tmp_path)
    Path(records[0]['metadata']['native_trace_path']).unlink()
    with pytest.raises(ValueError, match='regular'):
        compact.archive_completed_event(tmp_path, arm, 1, records)
    assert not (tmp_path / 'compact').exists()


def test_outside_or_wrong_arm_member_cannot_be_retired(tmp_path):
    arm, records, event_root, _ = event_fixture(tmp_path)
    other = tmp_path / 'unrelated.json'
    write_json(other, {})
    records[0]['metadata']['native_trace_path'] = str(other)
    write_json(event_root / 'loop/feedback_buffer/state.json', dict(records=records))
    with pytest.raises(ValueError, match='same CODE arm'):
        compact.archive_completed_event(tmp_path, arm, 1, records)
    assert other.exists()


def test_record_set_must_match_committed_buffer(tmp_path):
    arm, records, _, _ = event_fixture(tmp_path)
    records[0]['record_id'] = 'changed'
    with pytest.raises(ValueError, match='committed event'):
        compact.archive_completed_event(tmp_path, arm, 1, records)


def test_active_or_unpublished_event_cannot_be_compacted(tmp_path):
    arm, records, _, _ = event_fixture(tmp_path)
    write_json(tmp_path / 'campaign_001/arms' / arm / 'STATUS.json',
               dict(events_completed=0, phase='training'))
    with pytest.raises(ValueError, match='idle boundary'):
        compact.archive_completed_event(tmp_path, arm, 1, records)


def test_symlink_member_cannot_escape_owned_event(tmp_path):
    arm, records, event_root, _ = event_fixture(tmp_path)
    (event_root / 'link.json').symlink_to(tmp_path / 'not_owned.json')
    with pytest.raises(ValueError, match='unsymlinked'):
        compact.archive_completed_event(tmp_path, arm, 1, records)
