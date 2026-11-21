"""Resume only the retained CODE qualification after its explicit policy amendment.

The failed records and unknown operation remain evidence. This module does not
retry completed slots, forgive new failures, or reset the study clock.
"""
import argparse
import copy
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.bounded_prefix_allocation.preflight import verify_source
from .contract import OUTPUT_POLICY, digest
from experiments.persistent_prefix_withholding.task_runner import read

SCOPE = 'allow_AR_speculative_differences_resume_remaining_qualification_then_campaign'
RECEIPT = 'OUTPUT_POLICY_RESUME.json'
CEILING = 67
UNKNOWN = {'11': dict(task='qualification_serving_A',
    label='A_AR_reserved_8_2048_0', counts={'qualification.generation': 1})}
REUSED = {
    'A': tuple(f'A_AR_reserved_{i}_2048_0' for i in range(8)),
    'B': ('B_AR_cross_card_0_2048_0', 'B_AR_cross_card_2_2048_0',
          'B_F_cross_card_0_2048_0'),
}
NORMAL_EXACT = dict(generation=64, adam=17, response_forward=40,
    response_backward=40, teacher_preparation=16, training_initialization=3)


def _authorization(value):
    if (value.get('status') != 'authorized' or value.get('scope') != SCOPE
            or value.get('qualification_generation_ceiling') != CEILING):
        raise ValueError('Require explicit output-policy and 67-generation-ceiling authorization')


def _origin(plan, authorization, launch, work, clock):
    from .runtime import LIMITS
    _authorization(authorization)
    if plan['binding'] != digest({k: v for k, v in plan.items() if k != 'binding'}):
        raise ValueError('Original sealed plan changed')
    if (launch['status'] != 'failed' or launch['completed']
            or launch.get('preflight', {}).get('status') != 'passed'):
        raise ValueError('Require the failed initial serving qualification')
    attempted = {key: 0 for key in LIMITS}
    completed = dict(attempted)
    attempted['qualification.generation'] = 14
    completed['qualification.generation'] = 13
    if (work['limits'] != LIMITS or work['next_operation'] != 14
            or work['attempted'] != attempted or work['completed'] != completed
            or work['pending'] != UNKNOWN or work['failures']):
        raise ValueError('Recovery requires the exact retained 14-attempt/13-return ledger')
    if (clock['active'] or clock['completed_card_s'] < 0
            or clock['first_model_load_epoch'] <= 0):
        raise ValueError('Original clock must have no active leases')
    if plan['exact_qualification_work'] != NORMAL_EXACT:
        raise ValueError('The original scientific qualification schedule changed')


def _files(path):
    """Pin retained bytes without moving paths embedded in native records."""
    path = Path(path)
    return {str(p): identity(p) for p in sorted(path.rglob('*')) if p.is_file()}


def _reuse_inventory(plan):
    root = Path(plan['root'])
    reused = {}
    for card, labels in REUSED.items():
        base = root/'qualification'/card
        work = read(base/'WORK.json')
        provenance = read(base/'PROVENANCE.json')
        if provenance['execution_commit'] != plan['execution_commit']:
            raise ValueError('Returned qualification source differs from the original plan')
        returned = {label for label, row in work.items()
                    if row['status'] in ('completed', 'valid', 'invalid')}
        if returned != set(labels):
            raise ValueError('Returned qualification slots differ from the retained eleven')
        reused[card] = {}
        for label in labels:
            row = work[label]
            invalid = label == 'B_F_cross_card_0_2048_0'
            expected_errors = ['Canonical mismatch: response_ids'] if invalid else []
            if (row['status'] != ('invalid' if invalid else 'valid')
                    or row.get('errors', []) != expected_errors):
                raise ValueError('A reused slot has a failure outside the authorized policy change')
            path = base/'records'/(label+'.json')
            if Path(row['record_path']).resolve() != path.resolve():
                raise ValueError('Returned record is outside its original slot')
            record = read(path)
            if record['record_id'] != label:
                raise ValueError('Returned record identity differs from its slot')
            reused[card][label] = dict(record_path=str(path), identity=identity(path),
                original_slot_status=row['status'], original_errors=expected_errors,
                source_execution_commit=provenance['execution_commit'])
    pending = read(root/'qualification/A/WORK.json')['A_AR_reserved_8_2048_0']
    if pending['status'] not in ('started', 'failed_return_unknown'):
        raise ValueError('Original pending serving slot must remain unknown')
    return reused


def prepare(plan_path, source_root, source_manifest, authorization):
    plan_path, source_root = Path(plan_path).resolve(), Path(source_root).resolve()
    source_manifest, authorization = Path(source_manifest).resolve(), Path(authorization).resolve()
    old = read(plan_path)
    root = Path(old['root']).resolve()
    archive = root/'attempts/output_policy_resume_001'
    destination = root/'sealed/PLAN_POLICY_RESUME_001.json'
    receipt_path = root/RECEIPT
    if any(p.exists() for p in (archive, destination, receipt_path, root/'qualification_policy_001')):
        raise FileExistsError('This output-policy continuation was already prepared or attempted')
    clock, work, launch = (read(root/name) for name in ('CLOCK.json', 'WORK.json', 'LAUNCH.json'))
    _origin(old, read(authorization), launch, work, clock)
    reused = _reuse_inventory(old)
    source = read(source_manifest)
    verify_source(dict(source_root=str(source_root), source_manifest=str(source_manifest),
                       execution_commit=source['commit']))
    from .protected import check
    check(root, full_hash=True, purpose='before_authorized_output_policy_continuation')
    retained = _files(root/'qualification')
    for path in (root/'failure_002', root/'attempts/failed_startup_001'):
        if path.exists():
            retained.update(_files(path))
    for path in sorted(root.iterdir()):
        if path.is_dir() and 'diagnostic' in path.name.lower():
            retained.update(_files(path))
    new = copy.deepcopy(old)
    historical_retry = new.pop('startup_retry', None)
    new.update(source_root=str(source_root), source_manifest=str(source_manifest),
        execution_commit=source['commit'], execution_source=dict(commit=source['commit'], dirty=False),
        qualification_directory='qualification_policy_001', output_policy=OUTPUT_POLICY)
    # The diagnostic source is also this study's allocation, even though none of
    # its two instrumented generations may fill ordinary qualification slots.
    diagnostic_source = root.parent.parent/'opencode_native_source_20261120_003'
    for path in (diagnostic_source, source_root.parent):
        if path.exists() and str(path) not in new['allocation_roots']:
            new['allocation_roots'].append(str(path))
    new['policy_resume'] = dict(receipt=str(receipt_path), authorization=str(authorization),
        authorization_identity=identity(authorization), prior_plan=str(plan_path),
        prior_plan_identity=identity(plan_path), prior_execution_commit=old['execution_commit'],
        archive=str(archive), generation_ceiling=CEILING)
    new['binding'] = digest({k: v for k, v in new.items() if k != 'binding'})

    archive.mkdir(parents=True)
    administrative = {'CPU_PREFLIGHT.json', 'phases', 'logs', 'processes', 'launcher.log'}
    administrative.update(p.name for pattern in ('LAUNCH*', 'PROTECTED*') for p in root.glob(pattern))
    for name in sorted(administrative):
        path = root/name
        if path.exists():
            path.rename(archive/name)
    for name in ('CLOCK.json', 'WORK.json'):
        shutil.copy2(root/name, archive/name)
    shutil.copy2(plan_path, archive/'PLAN.json')
    shutil.copy2(old['source_manifest'], archive/'SOURCE.json')
    admitted_work = copy.deepcopy(work)
    admitted_work['limits']['qualification.generation'] = CEILING
    write_json(root/'WORK.json', admitted_work)
    write_json(destination, new)
    receipt = dict(status='prepared_once', new_plan=str(destination),
        new_plan_identity=identity(destination), original_clock=clock, original_work=work,
        admitted_work=admitted_work, archive=str(archive), archived_files=_files(archive),
        retained_files=retained, reused_records=reused,
        authorization_identity=identity(authorization), execution_commit=source['commit'],
        historical_startup_retry=historical_retry, normal_qualification_generations=64,
        diagnostic_generations=2, historical_unknown_generations=1,
        remaining_normal_generations=53, repeated_completed_slots=0,
        original_clock_preserved=True, original_failure_records_preserved=True)
    receipt['binding'] = digest(receipt)
    write_json(receipt_path, receipt)
    return dict(plan=str(destination), receipt=str(receipt_path))


def _receipt(root, plan=None):
    """Authenticate the narrow amendment without rehashing traces on every call."""
    root = Path(root).resolve()
    value = read(root/RECEIPT)
    if (value['status'] != 'prepared_once'
            or value['binding'] != digest({k: v for k, v in value.items() if k != 'binding'})
            or identity(value['new_plan']) != value['new_plan_identity']):
        raise ValueError('Output-policy recovery receipt or sealed plan changed')
    sealed = read(value['new_plan'])
    if plan is not None and sealed != plan:
        raise ValueError('Output-policy recovery does not bind this plan')
    meta = sealed['policy_resume']
    if (Path(sealed['root']).resolve() != root or meta['receipt'] != str(root/RECEIPT)
            or sealed.get('output_policy') != OUTPUT_POLICY
            or sealed.get('qualification_directory') != 'qualification_policy_001'
            or sealed['binding'] != digest({k: v for k, v in sealed.items() if k != 'binding'})
            or identity(meta['authorization']) != meta['authorization_identity']
            or meta['authorization_identity'] != value['authorization_identity']
            or identity(meta['prior_plan']) != meta['prior_plan_identity']):
        raise ValueError('Output-policy amendment identity changed')
    _origin(read(meta['prior_plan']), read(meta['authorization']),
        read(Path(value['archive'])/'LAUNCH.json'), value['original_work'], value['original_clock'])
    return value


def validate_ready(plan, receipt_path):
    root = Path(plan['root']).resolve()
    if Path(receipt_path).resolve() != root/RECEIPT:
        raise ValueError('Unexpected output-policy recovery receipt path')
    receipt = _receipt(root, plan)
    for path, pin in {**receipt['archived_files'], **receipt['retained_files']}.items():
        if identity(path) != pin:
            raise ValueError('Retained failure or diagnostic evidence changed: '+path)
    if (read(root/'CLOCK.json') != receipt['original_clock']
            or read(root/'WORK.json') != receipt['admitted_work']):
        raise ValueError('Original clock or admitted work changed before continuation')
    expected = copy.deepcopy(receipt['original_work'])
    expected['limits']['qualification.generation'] = CEILING
    if receipt['admitted_work'] != expected or _reuse_inventory(read(plan['policy_resume']['prior_plan'])) != receipt['reused_records']:
        raise ValueError('The recovery changed scientific work or reusable slots')
    verify_source(plan)
    return receipt


def effective_work_limits(root):
    from .runtime import LIMITS
    limits = dict(LIMITS)
    if (Path(root)/RECEIPT).exists():
        _receipt(root)
        limits['qualification.generation'] = CEILING
    return limits


def assert_historical_work_only(plan, state):
    expected = (_receipt(plan['root'], plan)['original_work']
                if 'policy_resume' in plan else dict(pending={}, failures=[]))
    if state['pending'] != expected['pending'] or state['failures'] != expected['failures']:
        raise ValueError('New unknown or failed work cannot be waived by the historical amendment')


def expected_qualification_totals(plan, normal_exact):
    attempted, completed = dict(normal_exact), dict(normal_exact)
    if 'policy_resume' in plan:
        _receipt(plan['root'], plan)
        if normal_exact != NORMAL_EXACT:
            raise ValueError('The original scientific qualification totals changed')
        attempted['generation'], completed['generation'] = 67, 66
    return dict(attempted=attempted, completed=completed)


def reused_records(plan, card):
    if card not in REUSED:
        raise ValueError('Unknown qualification card')
    if 'policy_resume' not in plan:
        return {}
    receipt = _receipt(plan['root'], plan)
    rows = receipt['reused_records'][card]
    for row in rows.values():
        if identity(row['record_path']) != row['identity']:
            raise ValueError('A reused raw qualification record changed')
    return copy.deepcopy(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan-path', 'source-root', 'source-manifest', 'authorization'):
        parser.add_argument('--'+key, required=True)
    print(prepare(**vars(parser.parse_args())))
