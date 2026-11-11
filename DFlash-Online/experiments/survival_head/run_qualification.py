"""Four one-window native cases and six serving checks; no implicit retries."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.ok_coding.campaign import require_complete_work, fresh_worker_environment
from experiments.ok_coding.storage import identity
from experiments.functional_step_geometry.pilot_storage import retire_exact
from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
from experiments.o_half_matched_lr.archive import validate_record
from online_feedback.storage import write_json
from .contract import ARMS, QUALIFICATION_CASES, QUALIFICATION_BUDGET
from .data import read


def limits(case):
    active = case in ('LOCAL', 'SURVIVAL')
    return dict(response_forwards=2, response_backwards=2, optimizer_steps=1,
                auxiliary_backwards=2 if active else 0, head_optimizer_steps=int(active))


def identity_or_retirement(path, expected, folder):
    """Require surviving bytes or a verified completed exact-retirement receipt."""
    path, folder = Path(path), Path(folder)
    if path.is_file():
        if identity(path) != expected:
            raise ValueError('Qualification artifact bytes changed: '+str(path))
        return 'retained'
    for name in ('RETIREMENT.json', 'BASELINE_RETIREMENT.json'):
        retirement = folder/name
        if retirement.is_file():
            receipt = read(retirement)
            if receipt.get('status') == 'completed' and any(
                    Path(row['path']).resolve() == path.resolve() and row['identity'] == expected
                    for row in receipt.get('removed', [])):
                return 'retired_not_archived'
    raise ValueError('Qualification artifact missing without authenticated retirement: '+str(path))


def validate_case(root, case, plan):
    root = Path(root); folder = root/'qualification'/case
    row = read(folder/'RESULT.json')
    require_complete_work(row['work'], limits(case))
    require_complete_work(row['startup']['work'],
                          dict(response_forwards=2, response_backwards=2, optimizer_steps=0))
    if (row['status'] != 'completed' or row['complete_state']['status'] != 'passed'
            or row.get('finite', {}).get('status') != 'passed'
            or row.get('binding') != plan['binding']
            or row.get('execution_commit') != plan['execution_source']['commit']
            or row.get('startup', {}).get('status') != 'completed'
            or read(folder/'startup/EXACT_RESTORATION.json').get('exact_equal') is not True):
        raise ValueError('Incomplete, wrong-source or nonfinite qualification state')
    identity_or_retirement(folder/'ENDPOINT.pt', row['endpoint_identity'], folder)
    if identity(folder/'COMPACT_ENDPOINT.pt') != row['compact_endpoint_identity']:
        raise ValueError('Compact qualification RNG/anchor/counter evidence changed')
    pool = read(root/'feedback/POOL.json')
    if (row.get('inputs', {}).get('pool_identity') != identity(root/'feedback/POOL.json')
            or row.get('inputs', {}).get('packets') != pool['records'][:2]):
        raise ValueError('Qualification did not consume the sealed shared prefix packets')
    if case != 'KPLUS' and row.get('comparison', {}).get('exact_equal') is not True:
        raise ValueError('Disabled endpoint or active sampling/RNG comparison failed')
    if case != 'KPLUS' and read(folder/'EXACT.json') != row['comparison']:
        raise ValueError('Exact comparison sidecar changed')
    if case in ('LOCAL', 'SURVIVAL'):
        checks = row.get('auxiliary_gradient_checks', [])
        if (len(checks) != 2 or any(not all(check.get(name) is True for name in
                ('finite','nonzero','native_gradient_buffer_unchanged'))
                or not isinstance(check.get('norm_sq'), (float,int)) or not math.isfinite(check['norm_sq'])
                or check['norm_sq'] <= 0 for check in checks)
                or row.get('head_steps') != 1
                or row.get('head_initial_check', {}).get('status') != 'passed'
                or row['head_initial_check'].get('nonzero_weight') is not True
                or row['head_initial_check'].get('zero_bias') is not True
                or row.get('head_final_check', {}).get('status') != 'passed'):
            raise ValueError('Auxiliary gradient/head boundary qualification failed')
        if (read(folder/'AUXILIARY_CHECKS.json') != checks
                or identity(folder/'HEAD_INITIAL.pt') != row.get('head_initial_identity')
                or identity(folder/'HEAD_FINAL.pt') != row.get('head_final_identity')):
            raise ValueError('Retained head/gradient evidence changed')
    elif row.get('auxiliary_gradient_checks'):
        raise ValueError('Disabled case unexpectedly charged auxiliary backward')
    if case == 'SURVIVAL' and row.get('head_initialization_exact', {}).get('exact_equal') is not True:
        raise ValueError('Active heads did not start from identical head weights')
    if case == 'SURVIVAL' and read(folder/'HEAD_INITIAL_EXACT.json') != row['head_initialization_exact']:
        raise ValueError('Head initialization equality sidecar changed')
    if case in ARMS:
        export = row['export_check']
        if export.get('status') != 'passed' or export.get('exact_tensor_values_and_membership') is not True:
            raise ValueError('Head-free export exactness was not established')
        for relative, expected in export['files'].items():
            identity_or_retirement(Path(row['export'])/relative, expected, folder)
    return row


def validate_serving(root, case, row, plan):
    folder = Path(root)/'qualification'/case
    serving = read(folder/'SERVING_CHECK.json')
    require_complete_work(serving['work'], dict(serving_generations=2))
    if (serving.get('status') != 'passed' or serving.get('binding') != plan['binding']
            or serving.get('execution_commit') != plan['execution_source']['commit']
            or serving.get('inputs') != row['inputs'] or len(serving.get('checks', [])) != 2
            or publication_errors(serving['publication'], row['export'])
            or weight_check_errors(serving['weights'], row['export'])):
        raise ValueError('Disposable native serving load/provenance checks failed')
    for index, check in enumerate(serving['checks']):
        if check.get('index') != index or check.get('exact_output_and_termination') is not True or check.get('errors'):
            raise ValueError('Disposable serving response failed')
        raw_path, native_path = Path(check['raw_path']), Path(check['native_path'])
        if identity(raw_path) != check['raw_identity'] or identity(native_path) != check['native_identity']:
            raise ValueError('Disposable serving raw bytes changed')
        raw = read(raw_path)
        canonical = read(row['inputs']['packets'][index]['canonical']['path'])
        errors = validate_record(raw, dict(canonical, panel_ordinal=index), row['export'], canonical,
            label=canonical['record_id'], session='survival_qualification')
        if errors or read(native_path) != raw['metadata']['capture']:
            raise ValueError('Disposable raw trace revalidation failed: '+repr(errors))
    return serving


def summarize(root):
    root = Path(root); plan = read(root/'sealed/PLAN.json')
    counts = dict(optimizer_steps=0, head_optimizer_steps=0, response_forwards=0,
                  response_backwards=0, auxiliary_backwards=0, serving_generations=0)
    cases = []
    for case in QUALIFICATION_CASES:
        row = validate_case(root, case, plan)
        for name in ('optimizer_steps','head_optimizer_steps','response_forwards','response_backwards','auxiliary_backwards'):
            counts[name] += row['work']['completed'][name]
        for name in ('response_forwards','response_backwards'):
            counts[name] += row['startup']['work']['completed'][name]
        if case in ARMS:
            serving = validate_serving(root, case, row, plan)
            counts['serving_generations'] += serving['work']['completed']['serving_generations']
        cases.append(dict(case=case, result=str(root/'qualification'/case/'RESULT.json'),
            result_identity=identity(root/'qualification'/case/'RESULT.json')))
    # autograd.grad is a separately classified response-backward equivalent,
    # never a native backward or optimizer update.
    counts['response_backwards'] += counts['auxiliary_backwards']
    if counts != {name:QUALIFICATION_BUDGET[name] for name in counts}:
        raise ValueError('Qualification work differs from the sealed exact budget')
    return dict(status='passed', execution_commit=plan['execution_source']['commit'],
        binding=plan['binding'], counts=counts, cases=cases, checks={name:True for name in
            ('complete_state','disabled_native_exact','same_feedback_and_anchors','same_native_rng',
             'identical_initial_heads','finite_nonzero_auxiliary_backbone_gradients',
             'native_gradients_unchanged_by_observer','separate_finite_head_optimizer',
             'one_native_step_per_window','head_free_native_serving','raw_serving_revalidation')},
        scope='Bounded native correctness qualification; no efficacy claim or automatic retry')


def retire_case(root, case, *, baseline=False):
    """Retire only a validated disposable episode and leave failures untouched."""
    root = Path(root); plan = read(root/'sealed/PLAN.json')
    row = validate_case(root, case, plan)
    if case in ARMS:
        validate_serving(root, case, row, plan)
    folder = root/'qualification'/case
    if baseline and case != 'KPLUS':
        raise ValueError('Only the shared KPLUS reference may use final retirement')
    candidates=[]
    if case != 'KPLUS' or baseline:
        candidates.append((folder/'ENDPOINT.pt', row['endpoint_identity'],
                           'Validated disposable endpoint; compact RNG/anchor/counter evidence retained'))
    if case in ARMS and not baseline:
        candidates.extend((Path(row['export'])/relative, expected, 'Passed two raw native serving checks')
                          for relative,expected in row['export_check']['files'].items())
    kept = [folder/'RESULT.json', folder/'COMPACT_ENDPOINT.pt', folder/'startup/EXACT_RESTORATION.json']
    kept += [p for p in (folder/'EXACT.json', folder/'HEAD_INITIAL.pt', folder/'HEAD_FINAL.pt',
             folder/'HEAD_INITIAL_EXACT.json', folder/'AUXILIARY_CHECKS.json', folder/'SERVING_CHECK.json') if p.is_file()]
    if case in ARMS:
        for check in read(folder/'SERVING_CHECK.json')['checks']:
            kept.extend((Path(check['raw_path']), Path(check['native_path'])))
    retained = {str(path):identity(path) for path in kept}
    return retire_exact(root, candidates,
        folder/('BASELINE_RETIREMENT.json' if baseline else 'RETIREMENT.json'), retained=retained,
        lost='Disposable backbone/optimizer/gradient and export replay is retired, not archived. '
             'Exact comparisons, RNG/anchors/counters, head states and raw serving remain; main starts from original KPLUS4.')


def run(root):
    root = Path(root); plan = read(root/'sealed/PLAN.json')
    preflight = read(root/'PREFLIGHT_QUALIFICATION.json')
    if preflight.get('status') != 'passed' or preflight.get('binding') != plan['binding']:
        raise ValueError('Require fresh successful source/runtime/storage preflight')
    output = root/'qualification'; output.mkdir(exist_ok=False)
    receipt = dict(status='running', binding=plan['binding'], execution_commit=plan['execution_source']['commit'],
        budgets=QUALIFICATION_BUDGET, completed=[], attempted=None, automatic_retry=False,
        feedback_generations=0, canonical_generations=0)
    started=time.perf_counter()
    def launch(command, log, case, serve=False):
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['serving_gpu'] if serve else plan['training_gpu']))
        if not serve:
            env=fresh_worker_environment(output/'worker_caches'/case, env)
        with Path(log).open('w') as stream:
            process=subprocess.run(command,cwd=plan['execution_source']['source_path'],env=env,
                stdout=stream,stderr=subprocess.STDOUT,check=False)
        if process.returncode:
            raise RuntimeError('Qualification worker failed: '+str(log))
    try:
        for case in QUALIFICATION_CASES:
            receipt['attempted']=dict(case=case,kind='disposable_update')
            write_json(output/'STATUS.json',receipt)
            command=[plan['training_python'],'-u','-m','experiments.survival_head.qualification',str(root),'--case',case]
            launch(command,output/f'{case}.log',case)
            if case in ARMS:
                receipt['attempted']=dict(case=case,kind='serving')
                write_json(output/'STATUS.json',receipt)
                launch(command+['--serve'],output/f'{case}_serving.log',case,True)
            retire_case(root,case)
            receipt['completed'].append(case)
            write_json(output/'STATUS.json',receipt)
        # Keep the reference until every disabled/active comparison is finished.
        retire_case(root,'KPLUS',baseline=True)
        write_json(output/'ADMISSION.json',summarize(root))
        receipt.update(status='completed',attempted=None)
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt['unattempted_cases']=[case for case in QUALIFICATION_CASES
            if case not in receipt['completed'] and case != (receipt.get('attempted') or {}).get('case')]
        receipt['wall_s']=time.perf_counter()-started
        write_json(output/'STATUS.json',receipt)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); args=parser.parse_args()
    run(args.root)
