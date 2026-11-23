"""Resume admitted CODE work after a time-budget amendment, without replaying gates.

Qualification, precheck, failed launch and the original work/clock remain evidence.
Only elapsed and occupied-card limits change; this module performs no GPU work.
"""
import argparse
import copy
import math
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.bounded_prefix_allocation.preflight import verify_source
from experiments.persistent_prefix_withholding.task_runner import read
from .contract import ARMS, OUTPUT_POLICY, RESOURCE_LIMITS, digest, qualification_root
from . import policy_resume

SCOPE = 'resume_qualified_CODE_production_with_time_extension'
RECEIPT = 'BUDGET_RESUME.json'
LIMITS = dict(RESOURCE_LIMITS, elapsed_s=96*3600, occupied_card_s=120*3600)
COMPLETED = ['qualification', 'full_panel_precheck']
SOURCE_FILES = {
    'budget_resume.py', 'test_budget_resume.py', 'launch.py', 'runtime.py',
    'preflight.py', 'policy_resume.py', 'raw_audit.py', 'report.py',
    'test_launch.py', 'test_runtime.py', 'test_output_policy.py',
}
PLAN_CHANGES = {'source_root', 'source_manifest', 'execution_commit',
                'execution_source', 'allocation_roots', 'limits', 'binding', 'budget_resume'}


def _authorization(value):
    if (value.get('status') != 'authorized' or value.get('scope') != SCOPE
            or value.get('elapsed_s') != LIMITS['elapsed_s']
            or value.get('occupied_card_s') != LIMITS['occupied_card_s']):
        raise ValueError('Require explicit 96-hour/120-card-hour continuation authorization')


def _files(folder):
    return {str(p): identity(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def _source_changes(old, source_manifest, authorization):
    """Prove byte equality outside the reviewed administrative changes."""
    before, after = read(old['source_manifest'])['files'], read(source_manifest)['files']
    changed = sorted(name for name in before.keys() | after.keys()
                     if before.get(name) != after.get(name))
    for name in changed:
        path = Path(name)
        administrative = (path.parent == Path('experiments/opencode_native_comparison')
                          and path.name in SOURCE_FILES)
        report = name.startswith('reports/dflash_opencode_native_comparison_20261120/')
        if not (administrative or report):
            raise ValueError('Budget continuation changes scientific source: '+name)
    review = authorization.get('scientific_source_review', {})
    if review.get('status') != 'passed' or review.get('changed_files') != changed:
        raise ValueError('Require source review bound to the exact administrative diff')
    return dict(changed_files=changed, unchanged_files=len(before.keys() & after.keys())-
                sum(name in before and name in after for name in changed),
                scientific_files_unchanged=True, review=review)


def _stop(plan, launch, work, clock, forecast):
    """Accept this single known forecast stop, not an arbitrary failed campaign."""
    from .runtime import LIMITS as work_limits
    if plan['binding'] != digest({k: v for k, v in plan.items() if k != 'binding'}):
        raise ValueError('Original sealed plan changed')
    if plan.get('limits', RESOURCE_LIMITS) != RESOURCE_LIMITS:
        raise ValueError('Original resource limits differ from the qualified contract')
    policy_resume._receipt(plan['root'], plan)
    expected_limits = dict(work_limits, **{'qualification.generation': 67})
    attempted = {key: 0 for key in expected_limits}
    for key, count in policy_resume.NORMAL_EXACT.items():
        attempted['qualification.'+key] = count
    attempted.update({'qualification.generation': 67,
        'production.canonical_generation': 1024, 'production.frozen_precheck_generation': 1024})
    completed = dict(attempted, **{'qualification.generation': 66})
    if (work['limits'] != expected_limits or work['attempted'] != attempted
            or work['completed'] != completed or work['next_operation'] != 2231
            or work['pending'] != policy_resume.UNKNOWN or work['failures']):
        raise ValueError('Require completed qualification/precheck and zero production training; unknown work changed')
    if (clock['active'] or not math.isfinite(clock['completed_card_s'])
            or clock['completed_card_s'] < 0 or not math.isfinite(clock['first_model_load_epoch'])
            or clock['first_model_load_epoch'] <= 0):
        raise ValueError('Original clock must be finite with no active leases')
    if (launch['status'] != 'failed' or launch['completed'] != COMPLETED
            or launch['execution_commit'] != plan['execution_commit']
            or not launch.get('error', '').startswith('RuntimeError("The complete unchanged CODE study no longer fits: ')
            or forecast.get('error') != launch['error'] or forecast['completed_arms']):
        raise ValueError('Require only the retained full-campaign forecast failure')
    if plan['exact_qualification_work'] != policy_resume.NORMAL_EXACT:
        raise ValueError('Qualified scientific schedule changed')


def _passed_evidence(plan, work):
    """Read passed receipts without rewriting qualification or regenerating inputs."""
    from .qualification import CASES
    root, qual = Path(plan['root']), qualification_root(plan)
    result = read(qual/'QUALIFICATION.json')
    physical = {field: {key.removeprefix('qualification.'): value
                       for key, value in work[field].items() if key.startswith('qualification.')}
                for field in ('attempted', 'completed')}
    if (result['status'] != 'passed' or result['execution_commit'] != plan['execution_commit']
            or result['work'] != policy_resume.NORMAL_EXACT or result['physical_work'] != physical
            or result['historical_unknown_work'] != policy_resume.UNKNOWN
            or result['output_policy'] != OUTPUT_POLICY or result['native_output_checks'] != 64
            or result['qualification_states_are_disposable'] is not True):
        raise ValueError('Retained complete qualification is unavailable or changed')
    for arm in ARMS:
        detail = read(qual/arm/'DETAILS.json')
        if (detail['status'] != 'passed'
                or [r['case'] for r in detail['cases']] != list(CASES[arm])
                or any(r['checks'] != 'passed' or r['status'] != 'completed' for r in detail['cases'])
                or detail['core_adam'] != len(CASES[arm])
                or detail['core_response_forward_backward_each'] != 2*len(CASES[arm])
                or detail['startup_response_forward_backward_each'] != 2
                or any(detail[k] is not True for k in ('origin_restored', 'exact_own_resume', 'exact_native_compatibility'))):
            raise ValueError('Training qualification is incomplete: '+arm)
    serving = read(qual/'INITIAL_SERVING.json')
    publication = read(root/'phases/qualification_publication_A/STATUS.json')
    published = publication['result']
    if (serving['status'] != 'passed_initial_serving' or serving['executions'] != 60
            or serving['cross_card_parity'] is not True or serving['capture_on_off_parity'] is not True
            or publication['status'] != 'completed' or published['status'] != 'passed'
            or published['executions'] != 4 or published['exact_output_and_reload_counts'] is not True
            or published['no_additional_optimizer_steps'] is not True):
        raise ValueError('Serving/publication qualification is incomplete')
    precheck = read(root/'precheck/AR_F_PRECHECK.json')
    canonicals = read(root/'precheck/CANONICALS.json')
    if (precheck['status'] != 'passed' or precheck['canonicals'] != 1024
            or precheck['parity_generations'] != 1024 or precheck['output_policy'] != OUTPUT_POLICY
            or len(canonicals) != 1024 or precheck['canonical_digest'] != digest(canonicals)):
        raise ValueError('Full-panel precheck is incomplete or changed')
    return result, precheck


def _plan_delta(old, new):
    if ({k: v for k, v in old.items() if k not in PLAN_CHANGES}
            != {k: v for k, v in new.items() if k not in PLAN_CHANGES}):
        raise ValueError('Budget continuation changed the scientific plan or policy amendment')
    roots = list(old['allocation_roots'])
    source_parent = str(Path(new['source_root']).parent)
    if source_parent not in roots:
        roots.append(source_parent)
    if (new['limits'] != LIMITS or new['allocation_roots'] != roots
            or new['execution_source'] != dict(commit=new['execution_commit'], dirty=False)):
        raise ValueError('Budget continuation changed limits, allocations or source policy')


def prepare(plan_path, source_root, source_manifest, authorization):
    plan_path, source_root = Path(plan_path).resolve(), Path(source_root).resolve()
    source_manifest, authorization = Path(source_manifest).resolve(), Path(authorization).resolve()
    old, approval = read(plan_path), read(authorization)
    root = Path(old['root']).resolve()
    archive, destination = root/'attempts/budget_resume_001', root/'sealed/PLAN_BUDGET_RESUME_001.json'
    receipt_path = root/RECEIPT
    if any(p.exists() for p in (archive, destination, receipt_path)):
        raise FileExistsError('This budget continuation was already prepared or attempted')
    _authorization(approval)
    clock, work, launch = (read(root/name) for name in ('CLOCK.json', 'WORK.json', 'LAUNCH.json'))
    forecast_path = root/'forecasts/AFTER_0_ARMS.json'
    _stop(old, launch, work, clock, read(forecast_path))
    qualification, precheck = _passed_evidence(old, work)
    verify_source(old)
    source = read(source_manifest)
    verify_source(dict(source_root=str(source_root), source_manifest=str(source_manifest),
                       execution_commit=source['commit']))
    source_review = _source_changes(old, source_manifest, approval)
    from .protected import check
    check(root, full_hash=True, purpose='before_authorized_budget_continuation')
    retained = {old['source_manifest']: identity(old['source_manifest'])}
    for name in ('qualification', old.get('qualification_directory', 'qualification'),
                 'precheck', 'phases', 'logs', 'processes'):
        retained.update(_files(root/name))
    new = copy.deepcopy(old)
    new.update(source_root=str(source_root), source_manifest=str(source_manifest),
        execution_commit=source['commit'], execution_source=dict(commit=source['commit'], dirty=False),
        limits=dict(LIMITS))
    if str(source_root.parent) not in new['allocation_roots']:
        new['allocation_roots'].append(str(source_root.parent))
    new['budget_resume'] = dict(receipt=str(receipt_path), authorization=str(authorization),
        authorization_identity=identity(authorization), prior_plan=str(plan_path),
        prior_plan_identity=identity(plan_path), prior_execution_commit=old['execution_commit'],
        archive=str(archive))
    new['binding'] = digest({k: v for k, v in new.items() if k != 'binding'})
    _plan_delta(old, new)
    archive.mkdir(parents=True)
    administrative = {'CPU_PREFLIGHT.json', 'launcher.log'}
    administrative.update(p.name for pattern in ('LAUNCH*', 'PROTECTED*') for p in root.glob(pattern))
    for name in sorted(administrative):
        path = root/name
        if path.is_file():
            path.rename(archive/name)
    for name in ('CLOCK.json', 'WORK.json'):
        shutil.copy2(root/name, archive/name)
    shutil.copy2(plan_path, archive/'PLAN.json')
    shutil.copy2(old['source_manifest'], archive/'SOURCE.json')
    shutil.copy2(forecast_path, archive/'AFTER_0_ARMS.json')
    write_json(destination, new)
    receipt = dict(status='prepared_once', new_plan=str(destination),
        new_plan_identity=identity(destination), original_clock=clock, original_work=work,
        archive=str(archive), archived_files=_files(archive), retained_files=retained,
        authorization_identity=identity(authorization), execution_commit=source['commit'],
        prior_execution_commit=old['execution_commit'], original_limits=dict(RESOURCE_LIMITS),
        admitted_limits=dict(LIMITS),
        source_manifest_identity=identity(source_manifest), source_review=source_review,
        completed=COMPLETED, qualification=qualification, precheck=precheck,
        reused_qualification_generations=66, reused_precheck_generations=2048,
        repeated_completed_slots=0, additional_qualification_work=0,
        original_clock_preserved=True, original_work_preserved=True)
    receipt['binding'] = digest(receipt)
    write_json(receipt_path, receipt)
    return dict(plan=str(destination), receipt=str(receipt_path))


def _receipt(plan):
    """Authenticate small sealed records; full trace hashing belongs to admission."""
    root = Path(plan['root']).resolve()
    meta = plan['budget_resume']
    value = read(root/RECEIPT)
    if (value['status'] != 'prepared_once'
            or value['binding'] != digest({k: v for k, v in value.items() if k != 'binding'})
            or meta['receipt'] != str(root/RECEIPT)
            or identity(value['new_plan']) != value['new_plan_identity']
            or read(value['new_plan']) != plan
            or plan['binding'] != digest({k: v for k, v in plan.items() if k != 'binding'})
            or identity(meta['prior_plan']) != meta['prior_plan_identity']
            or identity(meta['authorization']) != meta['authorization_identity']
            or meta['authorization_identity'] != value['authorization_identity']
            or identity(plan['source_manifest']) != value['source_manifest_identity']):
        raise ValueError('Budget authorization, receipt or sealed plan changed')
    approval, old = read(meta['authorization']), read(meta['prior_plan'])
    _authorization(approval)
    _plan_delta(old, plan)
    if _source_changes(old, plan['source_manifest'], approval) != value['source_review']:
        raise ValueError('Budget source review changed')
    # Avoid a recursive policy->budget->policy call with the descendant plan.
    policy = policy_resume._receipt(root)
    if read(policy['new_plan']) != old or meta['prior_execution_commit'] != old['execution_commit']:
        raise ValueError('Budget parent is not the authenticated qualified policy plan')
    return value, old


def parent_plan(plan):
    return _receipt(plan)[1] if 'budget_resume' in plan else plan


def effective_limits(plan):
    if 'budget_resume' not in plan:
        if plan.get('limits', RESOURCE_LIMITS) != RESOURCE_LIMITS:
            raise ValueError('Changed resource limits require an explicit valid budget amendment')
        return dict(RESOURCE_LIMITS)
    _receipt(plan)
    return dict(LIMITS)


def validate_ready(plan, receipt_path):
    root = Path(plan['root']).resolve()
    if Path(receipt_path).resolve() != root/RECEIPT:
        raise ValueError('Unexpected budget continuation receipt path')
    value, old = _receipt(plan)
    for path, pin in {**value['archived_files'], **value['retained_files']}.items():
        if identity(path) != pin:
            raise ValueError('Retained qualification, precheck or archived evidence changed: '+path)
    if (read(root/'CLOCK.json') != value['original_clock']
            or read(root/'WORK.json') != value['original_work']):
        raise ValueError('Original clock or work changed before continuation')
    archive = Path(value['archive'])
    _stop(old, read(archive/'LAUNCH.json'), value['original_work'],
          value['original_clock'], read(archive/'AFTER_0_ARMS.json'))
    qualification, precheck = _passed_evidence(old, value['original_work'])
    if qualification != value['qualification'] or precheck != value['precheck']:
        raise ValueError('Passed qualification or precheck receipt changed')
    verify_source(old)
    verify_source(plan)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'source-root', 'source-manifest', 'authorization'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(prepare(args.plan, args.source_root, args.source_manifest, args.authorization))


if __name__ == '__main__':
    main()
