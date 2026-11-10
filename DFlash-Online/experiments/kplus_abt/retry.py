"""Explicitly sealed accounting for the authorized failed-checker retry.

This metadata-only path preserves the failed attempt and its four valid packets.
It neither resumes its missing endpoint nor grants an automatic retry.
"""
from copy import deepcopy
from pathlib import Path
import shutil

from experiments.ok_coding.campaign import read_work, require_complete_work
from experiments.ok_coding.storage import identity
from .contract import DATASETS, QUALIFICATION_BUDGET
from .data import read, feedback_requests


RECEIPTS = ('PLAN.json', 'SEAL.json', 'PREPARATION.json', 'RESULT.json',
            'WORK.jsonl', 'STATUS.json', 'EXIT_CODE', 'ADDITIONAL_STATUS.txt',
            'AUTHORIZATION.json')
REPAIRED_SOURCE = {
    'experiments/kplus_abt/qualification.py',
    'experiments/kplus_abt/seal.py',
    'experiments/kplus_abt/prepare_feedback.py',
    'experiments/kplus_abt/run_qualification.py',
    'experiments/kplus_abt/report.py',
    'tests/test_adversarial_block_qualification.py',
}


def prior_accounting(directory):
    """Reconcile the one consumed Adam step with its raw ledger and startup."""
    directory = Path(directory)
    result, status = read(directory/'RESULT.json'), read(directory/'STATUS.json')
    prior, preparation = read(directory/'PLAN.json'), read(directory/'PREPARATION.json')
    authorization = read(directory/'AUTHORIZATION.json')
    if not isinstance(authorization.get('user_authorization'), str) or not authorization['user_authorization'].strip():
        raise ValueError('An explicit user retry authorization must be retained')
    if (result.get('status') != 'failed' or result.get('dataset') != 'gsm8k'
            or result.get('case') != 'KPLUS' or status.get('status') != 'failed'
            or status.get('completed') != [] or status.get('attempted') !=
                dict(dataset='gsm8k', case='KPLUS', kind='disposable_update')
            or int((directory/'EXIT_CODE').read_text()) != 1
            or (directory/'ADDITIONAL_STATUS.txt').read_text().splitlines()[:2] !=
                ['DISPOSABLE_ENDPOINT_NOT_SAVED', 'MAIN_PILOT_NOT_STARTED']):
        raise ValueError('This authorization covers only the retained failed checker attempt')
    for receipt in (result, status):
        if (receipt.get('execution_commit') != prior['execution_source']['commit']
                or receipt.get('binding') != prior['binding']):
            raise ValueError('Prior failure receipts have inconsistent execution identities')
    case_work = dict(response_forwards=2, response_backwards=2, optimizer_steps=1)
    actual = read_work(directory/'WORK.jsonl', case_work)
    require_complete_work(actual, case_work)
    if result['work'] != actual:
        raise ValueError('Prior failure work differs from its raw ledger')
    startup = dict(response_forwards=2, response_backwards=2, optimizer_steps=0)
    require_complete_work(result['startup']['work'], startup)
    if (directory/'STARTUP_WORK.jsonl').is_file():
        require_complete_work(read_work(directory/'STARTUP_WORK.jsonl', startup), startup)
    if (preparation.get('status') != 'completed'
            or preparation.get('execution_commit') != prior['execution_source']['commit']
            or preparation.get('shared_feedback_generations') != 4
            or preparation.get('teacher_preparations') != 4
            or preparation.get('extra_qualification_generations') != 0
            or preparation.get('main_updates') != 0):
        raise ValueError('Prior feedback preparation is incomplete or changed')
    return dict(status='failed', execution_source=prior['execution_source'],
        root=prior['root'], endpoint_saved=False, automatic_retry=False,
        counts=dict(optimizer_steps=1, response_forwards=4, response_backwards=4,
            case_forwards=2, case_backwards=2, startup_forwards=2, startup_backwards=2,
            serving_generations=0, main_optimizer_steps=0,
            feedback_generations=4, teacher_preparations=4),
        costs=dict(qualification_wall_s=status['wall_s'],
            feedback_preparation_wall_s=preparation['wall_s'],
            failed_case_wall_s=result['wall_s'], startup_wall_s=result['startup']['wall_s']),
        timer_overlap='Feedback preparation and failed case are inside qualification; startup is inside failed case.')


def seal_prior_attempt(directory, output, plan, code):
    """Copy raw receipts, bind packet bytes, and reject changed scientific inputs."""
    directory, output, code = Path(directory), Path(output), Path(code)
    for name in RECEIPTS:
        if not (directory/name).is_file():
            raise ValueError('Missing prior-attempt receipt: '+name)
    accounting = prior_accounting(directory)
    old, old_seal = read(directory/'PLAN.json'), read(directory/'SEAL.json')
    old_root, remote_root = Path(old['root']), Path(plan['root'])
    old_code = Path(old['execution_source']['source_path'])
    if old_root == remote_root or old.get('prior_attempt') is not None:
        raise ValueError('Use a new execution root for this single explicitly authorized retry')
    if (old_seal['binding'] != old['binding'] or old_seal['execution_source'] != old['execution_source']
            or old_seal['files'].get(str(old_root/'sealed/PLAN.json')) != identity(directory/'PLAN.json')):
        raise ValueError('Prior plan is not authenticated by its retained seal')
    for name in ('schema', 'arms', 'events', 'rho', 'budgets', 'analysis', 'target', 'draft',
                 'worker_python', 'training_python', 'training_gpu'):
        if plan[name] != old[name]:
            raise ValueError('Retry changed the scientific/runtime plan: '+name)
    for name in ('ORIGINAL_CONFIG.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
        if identity(output/'sealed'/name) != old_seal['files'].get(str(old_root/'sealed'/name)):
            raise ValueError('Retry model/configuration pin changed: '+name)
    for dataset in DATASETS:
        item, previous = plan['datasets'][dataset], old['datasets'][dataset]
        if {k:v for k,v in item.items() if k.startswith('origin_')} != {
                k:v for k,v in previous.items() if k.startswith('origin_')}:
            raise ValueError('Retry must restore the same complete original state')
        for name in ('TRAINING.json', 'PANEL.json'):
            if identity(output/'sealed'/dataset/name) != old_seal['files'].get(str(old_root/'sealed'/dataset/name)):
                raise ValueError('Retry training requests or panel changed')
        if item['evaluation_schedule'] != previous['evaluation_schedule']:
            raise ValueError('Retry endpoint schedule changed')
    source_files = 0
    for name, expected in old_seal['files'].items():
        try:
            relative = Path(name).relative_to(old_code)
        except ValueError:
            continue
        if relative.suffix != '.py' or str(relative) in REPAIRED_SOURCE or str(relative).startswith('tests/kplus_abt/'):
            continue
        if identity(code/relative) != expected:
            raise ValueError('Retry changed source outside the explicit checker/retry repair: '+str(relative))
        source_files += 1
    if source_files == 0:
        raise ValueError('Prior feedback-producing source identities are missing')
    prepared = read(directory/'PREPARATION.json')
    external_files = {}
    for dataset in DATASETS:
        requests = read(output/'sealed'/dataset/'TRAINING.json')['train_requests']
        requests = [r for r in requests if r['event_id'] == 'event_005'][:2]
        rows = prepared['datasets'][dataset]
        if len(rows) != 2 or [r['request_id'] for r in rows] != [r['request_id'] for r in requests]:
            raise ValueError('Retained packets are not the prescribed event-five prefix')
        for row in rows:
            if row['status'] != 'completed':
                raise ValueError('Retained packet is incomplete')
            for kind in ('record', 'feature'):
                path = Path(row[kind+'_path'])
                path.relative_to(old_root/'qualification_feedback'/dataset)
                if str(path) in external_files:
                    raise ValueError('Duplicate retained packet path')
                external_files[str(path)] = row[kind+'_identity']
    destination = output/'sealed/prior_attempt'
    destination.mkdir()
    receipts = {}
    names = list(RECEIPTS)
    if (directory/'STARTUP_WORK.jsonl').is_file():
        names.append('STARTUP_WORK.jsonl')
    for name in names:
        shutil.copy2(directory/name, destination/name)
        receipts[name] = dict(path=str(remote_root/'sealed/prior_attempt'/name),
                              **identity(destination/name))
    accounting.update(schema='kplus_abt_explicit_retry_v1', receipts=receipts,
        unchanged_source_files=source_files, retained_packet_files=external_files,
        retry_qualification_budget=dict(deepcopy(QUALIFICATION_BUDGET),
            feedback_generations=0, teacher_preparations=0, reused_feedback_packets=4),
        retry_new_feedback_generations=0, retry_new_teacher_preparations=0,
        reused_feedback_packets=4)
    return accounting, external_files


def reuse_feedback(plan):
    """Validate retained bytes before allowing zero-generation packet reuse."""
    from experiments.full_k_gsm8k.campaign import online_record_check

    prior = plan['prior_attempt']
    for row in prior['receipts'].values():
        if identity(row['path']) != {k:row[k] for k in ('bytes', 'sha256')}:
            raise ValueError('A sealed prior-attempt receipt changed')
    prepared = read(prior['receipts']['PREPARATION.json']['path'])
    old_root = Path(prior['root'])
    if (old_root/'qualification/gsm8k/KPLUS/ENDPOINT.pt').exists():
        raise ValueError('The missing prior endpoint status changed; do not infer a resume')
    for dataset in DATASETS:
        requests = feedback_requests(plan, dataset, 5)[:2]
        rows = prepared['datasets'][dataset]
        if [r['request_id'] for r in rows] != [r['request_id'] for r in requests]:
            raise ValueError('Retry requested different shared feedback')
        for row, request in zip(rows, requests):
            for kind in ('record', 'feature'):
                expected = prior['retained_packet_files'][row[kind+'_path']]
                if expected != row[kind+'_identity'] or identity(row[kind+'_path']) != expected:
                    raise ValueError('Retained feedback packet changed')
            record = read(row['record_path'])
            if record['record_id'] != request['request_id']:
                raise ValueError('Retained record identity differs from requested feedback')
            online_record_check(record, request, plan['datasets'][dataset]['origin_export'], capture=True)
    return deepcopy(prepared['datasets'])
