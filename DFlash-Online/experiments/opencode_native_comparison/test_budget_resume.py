"""Budget amendments preserve all returned work and cannot waive new failures."""
import copy
from pathlib import Path

import pytest

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from . import budget_resume as resume, policy_resume
from .contract import ARMS, OUTPUT_POLICY, RESOURCE_LIMITS, digest, qualification_root
from .qualification import CASES
from .test_policy_resume import fixture as policy_fixture


def fixture(tmp_path, monkeypatch):
    values = policy_fixture(tmp_path, monkeypatch)
    root, *rest = values
    recovery = policy_resume.prepare(*values[1:5])
    path = Path(recovery['plan'])
    plan = resume.read(path)
    work = resume.read(root/'WORK.json')
    work['next_operation'] = 2231
    for field, generations in (('attempted', 67), ('completed', 66)):
        for key, count in policy_resume.NORMAL_EXACT.items():
            work[field]['qualification.'+key] = count
        work[field]['qualification.generation'] = generations
        work[field]['production.canonical_generation'] = 1024
        work[field]['production.frozen_precheck_generation'] = 1024
    write_json(root/'WORK.json', work)
    clock = resume.read(root/'CLOCK.json')
    clock['completed_card_s'] = 9269.05
    write_json(root/'CLOCK.json', clock)
    error = 'RuntimeError("The complete unchanged CODE study no longer fits: original caps")'
    write_json(root/'LAUNCH.json', dict(status='failed', completed=resume.COMPLETED,
        execution_commit=plan['execution_commit'], error=error))
    write_json(root/'LAUNCH_PROCESS.json', {'pid': 'finished'})
    write_json(root/'forecasts/AFTER_0_ARMS.json', dict(completed_arms=[], error=error))
    write_json(root/'CPU_PREFLIGHT.json', {'status': 'passed'})
    write_json(root/'PROTECTED_DEPENDENCIES.json', {'protected': 'unchanged'})
    (root/'launcher.log').write_text('retained budget stop')
    write_json(root/'logs/precheck.log', {'completed': True})
    write_json(root/'processes/precheck_A.json', {'completed': True})
    qualified = qualification_root(plan)
    for arm in ARMS:
        write_json(qualified/arm/'DETAILS.json', dict(status='passed',
            cases=[dict(case=case, status='completed', checks='passed') for case in CASES[arm]],
            core_adam=len(CASES[arm]), core_response_forward_backward_each=2*len(CASES[arm]),
            startup_response_forward_backward_each=2, origin_restored=True,
            exact_own_resume=True, exact_native_compatibility=True))
    write_json(qualified/'INITIAL_SERVING.json', dict(status='passed_initial_serving',
        executions=60, cross_card_parity=True, capture_on_off_parity=True))
    write_json(root/'phases/qualification_publication_A/STATUS.json', dict(status='completed',
        result=dict(status='passed', executions=4, exact_output_and_reload_counts=True,
                    no_additional_optimizer_steps=True)))
    write_json(qualified/'QUALIFICATION.json', dict(status='passed',
        execution_commit=plan['execution_commit'], work=policy_resume.NORMAL_EXACT,
        physical_work={field: {key.removeprefix('qualification.'): count
                              for key, count in work[field].items() if key.startswith('qualification.')}
                       for field in ('attempted', 'completed')},
        historical_unknown_work=policy_resume.UNKNOWN, output_policy=OUTPUT_POLICY,
        native_output_checks=64, qualification_states_are_disposable=True))
    canonicals = {str(i): {'AR': 'retained', 'F': 'retained'} for i in range(1024)}
    write_json(root/'precheck/CANONICALS.json', canonicals)
    write_json(root/'precheck/AR_F_PRECHECK.json', dict(status='passed', canonicals=1024,
        parity_generations=1024, output_policy=OUTPUT_POLICY, canonical_digest=digest(canonicals)))
    write_json(root/'precheck/A/RESULT.json', {'status': 'passed'})
    write_json(root/'precheck/A/native/trace.json', {'retained': True})
    new_source = tmp_path/'source_budget/DFlash-Online'
    new_source.mkdir(parents=True)
    (new_source/'code.py').write_text((Path(plan['source_root'])/'code.py').read_text())
    changed = 'experiments/opencode_native_comparison/budget_resume.py'
    added = new_source/changed
    added.parent.mkdir(parents=True)
    added.write_text('administrative continuation')
    manifest = root/'SOURCE_BUDGET.json'
    write_json(manifest, dict(commit='c'*40, dirty=False,
        files={'code.py': identity(new_source/'code.py'), changed: identity(added)}))
    auth = root/'AUTHORIZATION_BUDGET.json'
    write_json(auth, dict(status='authorized', scope=resume.SCOPE,
        elapsed_s=96*3600, occupied_card_s=120*3600,
        scientific_source_review=dict(status='passed', changed_files=[changed])))
    return root, path, new_source, manifest, auth, plan, work, clock


def prepared(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    result = resume.prepare(*values[1:5])
    return values, result, resume.read(result['plan'])


def test_reuses_all_passed_work_without_reset_or_qualification(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    root, path, source, manifest, auth, old, work, clock = values
    untouched = {str(p): identity(p) for name in
        ('qualification', 'qualification_policy_001', 'phases', 'logs', 'processes', 'precheck')
        for p in (root/name).rglob('*') if p.is_file()}
    clock_identity, work_identity = identity(root/'CLOCK.json'), identity(root/'WORK.json')
    forecast_identity = identity(root/'forecasts/AFTER_0_ARMS.json')
    result = resume.prepare(path, source, manifest, auth)
    plan = resume.read(result['plan'])
    receipt = resume.validate_ready(plan, result['receipt'])
    assert resume.effective_limits(plan) == resume.LIMITS
    assert resume.parent_plan(plan) == old
    assert plan['policy_resume'] == old['policy_resume']
    assert identity(root/'CLOCK.json') == clock_identity
    assert identity(root/'WORK.json') == work_identity
    assert receipt['original_clock'] == clock and receipt['original_work'] == work
    assert receipt['completed'] == resume.COMPLETED
    assert receipt['reused_qualification_generations'] == 66
    assert receipt['reused_precheck_generations'] == 2048
    assert receipt['additional_qualification_work'] == receipt['repeated_completed_slots'] == 0
    assert all(identity(path) == pin for path, pin in untouched.items())
    assert identity(root/'forecasts/AFTER_0_ARMS.json') == forecast_identity
    for name in ('LAUNCH.json', 'LAUNCH_PROCESS.json', 'CPU_PREFLIGHT.json',
                 'PROTECTED_DEPENDENCIES.json', 'launcher.log'):
        assert not (root/name).exists()
        assert (root/'attempts/budget_resume_001'/name).is_file()
    for key in old.keys()-resume.PLAN_CHANGES:
        assert plan[key] == old[key]
    with pytest.raises(FileExistsError):
        resume.prepare(path, source, manifest, auth)


def test_ordinary_plan_has_original_limits_and_identity():
    plan = {'root': '/unused'}
    assert resume.effective_limits(plan) == RESOURCE_LIMITS
    assert resume.parent_plan(plan) is plan
    with pytest.raises(ValueError, match='explicit valid budget'):
        resume.effective_limits(dict(plan, limits=resume.LIMITS))


@pytest.mark.parametrize('change', ['new_unknown', 'unknown_removed', 'failure', 'adam',
    'feedback', 'qualified_again', 'precheck_incomplete', 'active', 'wrong_launch',
    'missing_cap', 'too_large_cap', 'authorization_pending', 'review_missing',
    'science_changed', 'qualification_failed'])
def test_ineligible_continuation_rejected_without_mutation(tmp_path, monkeypatch, change):
    root, path, source, manifest, auth, plan, work, clock = fixture(tmp_path, monkeypatch)
    if change in ('new_unknown', 'unknown_removed', 'failure', 'adam', 'feedback',
                  'qualified_again', 'precheck_incomplete'):
        if change == 'new_unknown': work['pending']['2231'] = {'new': 'unknown'}
        elif change == 'unknown_removed': work['pending'] = {}
        elif change == 'failure': work['failures'].append({'new': 'failed'})
        else:
            key = dict(adam='production.adam', feedback='production.feedback_generation',
                       qualified_again='qualification.adam',
                       precheck_incomplete='production.canonical_generation')[change]
            work['attempted'][key] += 1
        write_json(root/'WORK.json', work)
    elif change == 'active':
        clock['active']['new_task'] = {'active': True}
        write_json(root/'CLOCK.json', clock)
    elif change == 'wrong_launch':
        launch = resume.read(root/'LAUNCH.json')
        launch['error'] = 'A real correctness failure'
        write_json(root/'LAUNCH.json', launch)
    elif change in ('missing_cap', 'too_large_cap', 'authorization_pending', 'review_missing'):
        approval = resume.read(auth)
        if change == 'missing_cap': approval.pop('occupied_card_s')
        elif change == 'too_large_cap': approval['elapsed_s'] += 1
        elif change == 'authorization_pending': approval['status'] = 'pending'
        else: approval.pop('scientific_source_review')
        write_json(auth, approval)
    elif change == 'science_changed':
        (source/'code.py').write_text('scientific change')
        value = resume.read(manifest)
        value['files']['code.py'] = identity(source/'code.py')
        write_json(manifest, value)
    else:
        value = resume.read(qualification_root(plan)/'QUALIFICATION.json')
        value['status'] = 'failed'
        write_json(qualification_root(plan)/'QUALIFICATION.json', value)
    before = {str(p): identity(p) for p in root.rglob('*') if p.is_file()}
    with pytest.raises((ValueError, KeyError)):
        resume.prepare(path, source, manifest, auth)
    assert {str(p): identity(p) for p in root.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('change', ['clock', 'work', 'qualification', 'precheck', 'archive',
    'authorization', 'source', 'source_manifest', 'parent_plan', 'receipt'])
def test_full_admission_rejects_changed_retained_evidence(tmp_path, monkeypatch, change):
    values, result, plan = prepared(tmp_path, monkeypatch)
    root, parent, source, manifest, auth, *_ = values
    target = dict(clock=root/'CLOCK.json', work=root/'WORK.json',
        qualification=qualification_root(plan)/'QUALIFICATION.json',
        precheck=root/'precheck/A/native/trace.json',
        archive=root/'attempts/budget_resume_001/LAUNCH.json',
        authorization=auth, source=source/'code.py', source_manifest=manifest,
        parent_plan=parent, receipt=Path(result['receipt']))[change]
    target.write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        resume.validate_ready(plan, result['receipt'])


def test_receipt_does_not_authorize_modified_scientific_plan(tmp_path, monkeypatch):
    values, result, plan = prepared(tmp_path, monkeypatch)
    plan['serving']['unchanged'] = False
    plan['binding'] = digest({k: v for k, v in plan.items() if k != 'binding'})
    with pytest.raises(ValueError):
        resume.effective_limits(plan)


def test_cheap_runtime_authentication_does_not_freeze_progress(tmp_path, monkeypatch):
    values, result, plan = prepared(tmp_path, monkeypatch)
    root = values[0]
    work = resume.read(root/'WORK.json')
    work['attempted']['production.adam'] = 1
    write_json(root/'WORK.json', work)
    # Routine budget checks must permit subsequent authorized production work.
    assert resume.effective_limits(plan) == resume.LIMITS
    assert resume.parent_plan(plan) == values[5]
    with pytest.raises(ValueError, match='Original clock or work'):
        resume.validate_ready(plan, result['receipt'])
