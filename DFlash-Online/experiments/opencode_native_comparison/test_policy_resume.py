"""The policy amendment reuses evidence, not GPU calls or a new study clock."""
import copy
from pathlib import Path

import pytest

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from . import policy_resume as resume, protected
from .contract import OUTPUT_POLICY, digest
from .runtime import LIMITS


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'runs/run'
    root.mkdir(parents=True)
    old_source = tmp_path/'source_old/DFlash-Online'
    old_source.mkdir(parents=True)
    (old_source/'code.py').write_text('original')
    source_manifest = root/'SOURCE.json'
    write_json(source_manifest, dict(commit='a'*40, dirty=False,
        files={'code.py': identity(old_source/'code.py')}))
    plan = dict(root=str(root), source_root=str(old_source), source_manifest=str(source_manifest),
        execution_commit='a'*40, execution_source=dict(commit='a'*40, dirty=False),
        allocation_roots=[str(root), str(old_source.parent)], serving={'unchanged': True},
        exact_qualification_work=dict(resume.NORMAL_EXACT), input_identities={'original': 'unchanged'},
        compiler_baseline={'unchanged': 17}, startup_retry={'historical': 'preserve receipt'})
    plan['binding'] = digest(plan)
    path = root/'sealed/PLAN_RETRY_001.json'
    write_json(path, plan)
    work = dict(limits=copy.deepcopy(LIMITS), next_operation=14,
        attempted={key: 0 for key in LIMITS}, completed={key: 0 for key in LIMITS},
        pending=copy.deepcopy(resume.UNKNOWN), failures=[])
    work['attempted']['qualification.generation'] = 14
    work['completed']['qualification.generation'] = 13
    clock = dict(first_model_load_epoch=1234., completed_card_s=237.903253,
        active={}, finished=[{'retained_cost': 237.903253}])
    write_json(root/'WORK.json', work)
    write_json(root/'CLOCK.json', clock)
    write_json(root/'LAUNCH.json', dict(status='failed', completed=[], preflight=dict(status='passed')))
    write_json(root/'LAUNCH_PROCESS.json', {'pid': 'retained'})
    write_json(root/'PROTECTED_DEPENDENCIES.json', {'old': 'immutable'})
    write_json(root/'CPU_PREFLIGHT.json', {'status': 'passed'})
    write_json(root/'phases/qualification_serving_A/STATUS.json', {'status': 'interrupted'})
    write_json(root/'diagnostic_001/COMPLETE.json', {'generations': 2, 'not_qualification_slots': True})
    write_json(root/'failure_002/FAILURE.json', {'historical': 'failure'})
    write_json(root/'attempts/failed_startup_001/FAILURE.json', {'previous': 'startup'})
    for card, labels in resume.REUSED.items():
        base = root/'qualification'/card
        rows = {}
        write_json(base/'PROVENANCE.json', dict(execution_commit='a'*40))
        for label in labels:
            invalid = label == 'B_F_cross_card_0_2048_0'
            raw = base/'records'/(label+'.json')
            write_json(raw, dict(record_id=label, original='native record',
                metadata=dict(native_trace_path=str(base/'native/trace.json'))))
            rows[label] = dict(status='invalid' if invalid else 'valid', record_path=str(raw),
                errors=['Canonical mismatch: response_ids'] if invalid else [])
        if card == 'A':
            rows['A_AR_reserved_8_2048_0'] = dict(status='started')
        write_json(base/'WORK.json', rows)
        write_json(base/'native/trace.json', {'native': 'immutable'})
    authorization = root/'AUTHORIZATION_POLICY.json'
    write_json(authorization, dict(status='authorized', scope=resume.SCOPE,
        qualification_generation_ceiling=67))
    new_source = tmp_path/'source_new/DFlash-Online'
    new_source.mkdir(parents=True)
    (new_source/'code.py').write_text('new')
    manifest = root/'SOURCE_POLICY.json'
    write_json(manifest, dict(commit='b'*40, dirty=False,
        files={'code.py': identity(new_source/'code.py')}))
    monkeypatch.setattr(protected, 'check', lambda *args, **kwargs: dict(status='passed'))
    return root, path, new_source, manifest, authorization, plan, work, clock


def prepared(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    result = resume.prepare(*values[1:5])
    return values, result, resume.read(result['plan'])


def test_preserves_failed_evidence_clock_and_unknown_with_only_limit_amendment(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    root, path, source, manifest, auth, old, work, clock = values
    retained = {str(p): identity(p) for p in (root/'qualification').rglob('*') if p.is_file()}
    result = resume.prepare(path, source, manifest, auth)
    new = resume.read(result['plan'])
    receipt = resume.validate_ready(new, result['receipt'])
    assert retained == {str(p): identity(p) for p in (root/'qualification').rglob('*') if p.is_file()}
    assert resume.read(root/'CLOCK.json') == clock
    admitted = copy.deepcopy(work)
    admitted['limits']['qualification.generation'] = 67
    assert resume.read(root/'WORK.json') == admitted
    assert receipt['original_work'] == work and receipt['original_clock'] == clock
    assert receipt['remaining_normal_generations'] == 53
    assert receipt['repeated_completed_slots'] == 0
    assert receipt['historical_startup_retry'] == old['startup_retry']
    assert not (root/'LAUNCH.json').exists()
    assert not (root/'PROTECTED_DEPENDENCIES.json').exists()
    assert (root/'attempts/output_policy_resume_001/LAUNCH.json').is_file()
    assert resume.read(root/'qualification/B/WORK.json')['B_F_cross_card_0_2048_0']['status'] == 'invalid'
    assert new['qualification_directory'] == 'qualification_policy_001'
    assert new['output_policy'] == OUTPUT_POLICY and 'startup_retry' not in new
    changed = {'source_root', 'source_manifest', 'execution_commit', 'execution_source',
        'allocation_roots', 'binding', 'startup_retry'}
    for key in old.keys()-changed:
        assert new[key] == old[key]
    assert new['allocation_roots'][:-1] == old['allocation_roots']
    assert new['allocation_roots'][-1] == str(source.parent)
    with pytest.raises(FileExistsError):
        resume.prepare(path, source, manifest, auth)


def test_reusable_evidence_contains_eleven_original_records_and_no_diagnostic(tmp_path, monkeypatch):
    (root, *_), result, plan = prepared(tmp_path, monkeypatch)
    a, b = resume.reused_records(plan, 'A'), resume.reused_records(plan, 'B')
    assert len(a) == 8 and len(b) == 3
    assert b['B_F_cross_card_0_2048_0']['original_slot_status'] == 'invalid'
    assert all(row['source_execution_commit'] == 'a'*40 for row in [*a.values(), *b.values()])
    assert all('/qualification/' in row['record_path'] for row in [*a.values(), *b.values()])
    Path(a[next(iter(a))]['record_path']).write_text('changed')
    with pytest.raises(ValueError, match='reused raw'):
        resume.reused_records(plan, 'A')


def test_effective_limits_and_exact_totals_keep_historical_work_visible(tmp_path, monkeypatch):
    (root, *rest), result, plan = prepared(tmp_path, monkeypatch)
    limits = resume.effective_work_limits(root)
    assert limits == {**LIMITS, 'qualification.generation': 67}
    totals = resume.expected_qualification_totals(plan, resume.NORMAL_EXACT)
    assert totals == dict(attempted={**resume.NORMAL_EXACT, 'generation': 67},
        completed={**resume.NORMAL_EXACT, 'generation': 66})
    work = resume.read(root/'WORK.json')
    resume.assert_historical_work_only(plan, work)
    work['pending']['99'] = {'new': 'unknown'}
    with pytest.raises(ValueError, match='New unknown'):
        resume.assert_historical_work_only(plan, work)
    work['pending'].pop('99')
    work['failures'].append({'new': 'failure'})
    with pytest.raises(ValueError, match='New unknown'):
        resume.assert_historical_work_only(plan, work)
    work['failures'].clear()
    work['pending'].clear()
    with pytest.raises(ValueError, match='New unknown'):
        resume.assert_historical_work_only(plan, work)


@pytest.mark.parametrize('change', ['no_cap', 'not_authorized', 'scope', 'pending', 'counts',
    'active', 'failed_record', 'extra_return', 'bad_source'])
def test_ineligible_recovery_rejected_before_mutation(tmp_path, monkeypatch, change):
    root, path, source, manifest, auth, plan, work, clock = fixture(tmp_path, monkeypatch)
    if change in ('no_cap', 'not_authorized', 'scope'):
        approval = resume.read(auth)
        if change == 'no_cap': approval.pop('qualification_generation_ceiling')
        elif change == 'not_authorized': approval['status'] = 'pending'
        else: approval['scope'] = 'generic_retry'
        write_json(auth, approval)
    elif change == 'pending':
        work['pending']['11']['label'] = 'different_unknown'
        write_json(root/'WORK.json', work)
    elif change == 'counts':
        work['attempted']['qualification.generation'] = 15
        write_json(root/'WORK.json', work)
    elif change == 'active':
        clock['active']['worker'] = {'running': True}
        write_json(root/'CLOCK.json', clock)
    elif change in ('failed_record', 'extra_return'):
        p = root/'qualification/B/WORK.json'
        rows = resume.read(p)
        if change == 'failed_record': rows['B_F_cross_card_0_2048_0']['errors'].append('invalid_trace')
        else: rows['unapproved_extra'] = dict(status='valid')
        write_json(p, rows)
    else:
        (source/'code.py').write_text('unpublished')
    before = {str(p): identity(p) for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        resume.prepare(path, source, manifest, auth)
    assert before == {str(p): identity(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['clock', 'work', 'archived', 'raw', 'diagnostic', 'approval', 'source'])
def test_ready_revalidates_original_evidence_and_new_source(tmp_path, monkeypatch, change):
    values, result, plan = prepared(tmp_path, monkeypatch)
    root, _, source, _, auth, *_ = values
    target = dict(clock=root/'CLOCK.json', work=root/'WORK.json',
        archived=root/'attempts/output_policy_resume_001/LAUNCH.json',
        raw=root/'qualification/A/native/trace.json',
        diagnostic=root/'diagnostic_001/COMPLETE.json', approval=auth, source=source/'code.py')[change]
    if change in ('clock', 'work'):
        value = resume.read(target)
        value['unexpected'] = True
        write_json(target, value)
    else:
        target.write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        resume.validate_ready(plan, result['receipt'])


def test_unamended_run_has_no_allowance_or_historical_exception(tmp_path):
    assert resume.effective_work_limits(tmp_path) == LIMITS
    plan = {'root': str(tmp_path)}
    assert resume.reused_records(plan, 'A') == {}
    assert resume.expected_qualification_totals(plan, resume.NORMAL_EXACT) == dict(
        attempted=resume.NORMAL_EXACT, completed=resume.NORMAL_EXACT)
    with pytest.raises(ValueError):
        resume.assert_historical_work_only(plan, dict(pending=resume.UNKNOWN, failures=[]))
