"""One-card threatened-repair qualification, native trajectory and fixed probes.

No automatic retry. Training and the native evaluator never coexist. All exact
window exports are scored only after training closes, so held-out outcomes cannot
enter the selection rule. The three-card-hour bound includes both startup phases.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.verification_transition.budget import WorkBudget, BudgetService
from .prepare import read, identity, CONSTANTS, authenticate_source, attempt_accounting
from .work import read_ledger


def capacity(plan):
    free = shutil.disk_usage(plan['root']).free
    required = sum(plan['storage'].values())
    receipt = dict(free_bytes=free, required_bytes=required,
        status='passed' if free >= required else 'failed')
    if free < required:
        raise RuntimeError('Peak additional artifacts plus reserved storage unavailable: ' + repr(receipt))
    return receipt


def checked_accounting(plan):
    accounting = attempt_accounting(plan.get('prior_attempt'))
    if plan.get('budget_accounting', accounting) != accounting:
        raise ValueError('Sealed prior-attempt accounting changed')
    return accounting


def preflight(plan):
    """Read-only admission at the actual paths, before any model execution."""
    from experiments.lost_repair.run import preflight as original_preflight
    from experiments.survival_confirmation.runtime import physical_gpu
    from .probe_client import resource_estimate

    root = Path(plan['root'])
    accounting = checked_accounting(plan)
    if plan['constants'] != CONSTANTS or not root.resolve().is_relative_to('/data/sources/minhao/runs'):
        raise ValueError('Sealed experiment constants or destination differ')
    source = authenticate_source(plan['source_manifest'],
        plan['input_identities'][plan['source_manifest']], plan['execution_commit'])
    if (Path(plan['code_path']).resolve() != Path(source['source_path']).resolve()
            or Path(__file__).resolve().parents[2] != Path(source['source_path']).resolve()):
        raise ValueError('Controller and training checkout must be the authenticated execution source')
    for name, expected in plan['immutable_files'].items():
        if identity(root / 'sealed' / name) != expected:
            raise ValueError('Sealed population changed: ' + name)
    for path, expected in plan['input_identities'].items():
        if identity(path) != {key:expected[key] for key in ('bytes','sha256')}:
            raise ValueError('Retained input changed since sealing: ' + path)
    native = read(plan['native_plan_path'])
    old = native['original_plan']
    removed = old['checkpoints']['K4']['export'] + '/'
    legacy = dict(old, source_manifest_path=plan['source_manifest'],
        checkpoints={name:value for name,value in old['checkpoints'].items() if name in ('K5','K6')},
        pinned_files={path:value for path,value in old['pinned_files'].items() if not path.startswith(removed)},
        storage=dict(required_free_bytes=sum(plan['storage'].values())))
    original = original_preflight(legacy, root)
    estimate = resource_estimate(plan)
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    # Two rolling complete states plus native/autograd/gradient-copy overhead.
    training_ram = 60 * 1024**3
    required_ram = max(training_ram, estimate['conservative_precompaction_peak_bytes']) + plan['host_reserve_bytes']
    if available < required_ram:
        raise MemoryError('Host RAM cannot fit sequential training/probe phases plus reserved headroom')
    result = dict(status='passed', source=source, original_runtime=original,
        physical_gpu=physical_gpu(plan), capacity=capacity(plan),
        host_available_bytes=available, host_required_bytes=required_ram,
        probe_resources=estimate, budget_accounting=accounting, no_model_work=True)
    write_json(root / 'PREFLIGHT.json', result)
    return result


class CampaignBudget(WorkBudget):
    def check_time(self):
        if self.gpu_seconds >= self.gpu_seconds_limit:
            from experiments.verification_transition.budget import BudgetExceeded
            self._write(dict(kind='budget_stop', reason='gpu_time', gpu_seconds=self.gpu_seconds))
            raise BudgetExceeded('Three occupied GPU-card-hour limit reached')


def stop_owned_processes(budget):
    """Hard deadline only; terminate only native workers registered by this run."""
    for row in budget.processes():
        try:
            if os.getpgid(row['pid']) == row['pgid']:
                os.killpg(row['pgid'], signal.SIGTERM)
        except ProcessLookupError:
            pass


def training_environment(plan):
    """Honor the assigned card; never redirect a process assigned another GPU."""
    device = str(plan['serving_gpu'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, device):
        raise ValueError('Ambient CUDA assignment differs from the sealed physical GPU')
    environment = dict(os.environ)
    python = Path(plan['training_python'])
    environment.update(CUDA_VISIBLE_DEVICES=device, CONDA_PREFIX=str(python.parent.parent))
    environment['PATH'] = str(python.parent) + os.pathsep + environment.get('PATH', '')
    environment['PYTHONPATH'] = os.pathsep.join((plan['code_path'], str(Path(plan['code_path'])/'training')))
    return environment


def launch_training(plan, deadline):
    root = Path(plan['root'])
    command = [plan['training_python'], '-u', '-m',
        'experiments.threatened_repair_replay.runner', '--plan', str(root/'sealed/PLAN.json'),
        '--output', str(root/'training'), '--deadline', str(deadline)]
    began = time.monotonic()
    environment = training_environment(plan)
    receipt = dict(status='running', command=command, execution_commit=plan['execution_commit'],
        environment={name:environment[name] for name in ('CUDA_VISIBLE_DEVICES','CONDA_PREFIX','PATH','PYTHONPATH')})
    with (root/'TRAINING.log').open('x') as stream:
        process = subprocess.Popen(command, cwd=plan['code_path'], stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True, env=environment)
        receipt['pid'] = process.pid
        write_json(root/'TRAINING_PROCESS.json', receipt)
        try:
            code = process.wait(timeout=max(0.001, deadline-time.monotonic()))
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            receipt['interrupted'] = True
            raise
        finally:
            receipt.update(status='completed' if process.returncode==0 else 'failed',
                returncode=process.returncode, wall_s=time.monotonic()-began)
            write_json(root/'TRAINING_PROCESS.json', receipt)
    if code:
        raise RuntimeError('Training/qualification stopped; preserve evidence; no automatic retry')
    result = read(root/'training/RESULT.json')
    if result['status'] != 'completed':
        raise ValueError('Training process did not return a completed qualification/native trajectory')
    return result


def rows(root, name):
    """Read durably saved partial evidence without fabricating unattempted rows."""
    path = Path(root) / (name+'.jsonl')
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    path = Path(root) / (name+'.json')
    return read(path) if path.exists() else []


def native_limits(plan):
    """Count the sealed reconstructions and separate two-engine startup allowance."""
    first = plan['probe_requests'][0]['request_id']
    rounds = {rid: len(read(path)['metadata']['trace']) for rid, path in plan['parent_records'].items()}
    return dict(draft_block=16384 + 256 + sum(rounds.values()) + rounds[first] + 4 + 2 * 128,
                model_load=68)


def report(plan, receipt, budget):
    from .analysis import write_report
    root = Path(plan['root'])
    ledger_path = root/'training/PHYSICAL_WORK.jsonl'
    training_work = read_ledger(ledger_path) if ledger_path.exists() else {}
    accounting = checked_accounting(plan)
    local_steps = training_work.get('completed', {}).get('optimizer_step', 0)
    attempted_steps = training_work.get('attempted', {}).get('optimizer_step', 0)
    cumulative = dict(
        completed_adam_steps=accounting['prior_adam_steps'] + local_steps,
        attempted_adam_steps=accounting['prior_adam_steps'] + attempted_steps,
        gpu_seconds=accounting['prior_gpu_seconds'] + budget.summary()['gpu_seconds'],
        adam_limit=accounting['cumulative_adam_limit'],
        gpu_seconds_limit=accounting['cumulative_gpu_seconds_limit'])
    work = dict(native_budget=budget.summary(), training=training_work,
        prior_attempt=plan.get('prior_attempt'), budget_accounting=accounting,
        cumulative=cumulative,
        phases=receipt.get('phases', {}), campaign_wall_s=receipt['wall_s'],
        maximum_gpu_card_seconds=10800, new_complete_response_generations=0,
        clocks='Phase intervals include nested work; do not sum nested timers.',
        delayed_evaluation='Exact exported window weights; outcomes unavailable to training or selection.')
    write_json(root/'WORK_AND_COSTS.json', work)
    if (attempted_steps > accounting['current_attempt_adam_limit']
            or cumulative['attempted_adam_steps'] > cumulative['adam_limit']):
        raise ValueError('Physical Adam work exceeds the approved cumulative budget')
    if receipt['status'] == 'completed' and cumulative['gpu_seconds'] > cumulative['gpu_seconds_limit']:
        raise ValueError('Completed execution exceeds the cumulative occupied GPU-time budget')
    validity = dict(complete=receipt['status']=='completed',
        all_required_checks_passed=receipt['status']=='completed',
        errors=[] if receipt['status']=='completed' else [receipt.get('error','unfinished')])
    result = write_report(root/'final_report', probe_rows=rows(root/'probes','PROBE_OUTCOMES'),
        selection_rows=rows(root/'training','WINDOW_SELECTION'),
        memory_rows=rows(root/'training','MEMORY_OUTCOMES'),
        sealed_probes=read(plan['probe_keys']), expected_tags=read(plan['memory_tags']),
        validity=validity, work=work, execution_commit=plan['execution_commit'])
    for name in ('MEMORY_REFERENCE.csv','PROBE_KEYS.csv'):
        shutil.copyfile(root/'sealed'/name, root/'final_report'/name)
    for name in ('STATE_PARITY.json','UPDATE_RECEIPTS.csv'):
        if (root/'training'/name).exists():
            shutil.copyfile(root/'training'/name, root/'final_report'/name)
    shutil.copyfile(root/'WORK_AND_COSTS.json', root/'final_report/WORK_AND_COSTS.json')
    from .independent_audit import audit_report
    audit = audit_report(root/'final_report', root/'sealed')
    write_json(root/'final_report/INDEPENDENT_REDUCTION.json', audit)
    if audit['status'] != 'passed':
        raise ValueError('Independent CPU reduction disagrees; preserve report for review')
    if receipt['status'] == 'completed' and result['status'] != 'completed':
        raise ValueError('Completed execution has an incomplete or invalid scientific report')
    return result


def run(plan_path):
    from . import probe_client
    plan = read(plan_path)
    root = Path(plan['root'])
    if (root/'STATUS.json').exists():
        raise FileExistsError('Campaign destination already used; no automatic retry')
    accounting = checked_accounting(plan)
    remaining_seconds = accounting['current_attempt_gpu_seconds_limit']
    receipt = dict(status='preflight', phase='preflight', execution_commit=plan['execution_commit'],
        budget_accounting=accounting, phases={})
    write_json(root/'STATUS.json', receipt)
    began = time.monotonic()
    budget = CampaignBudget(root/'NATIVE_WORK.jsonl', gpu_seconds=remaining_seconds, limits=native_limits(plan))
    service = BudgetService(budget)
    execution_failure = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def deadline_handler(_signal, _frame):
        stop_owned_processes(budget)
        raise TimeoutError('Three occupied GPU-card-hour hard deadline')
    def phase(name, function):
        receipt['phase'] = name
        write_json(root/'STATUS.json', receipt)
        start = time.monotonic()
        try:
            value = function()
        except BaseException as error:
            receipt['phases'][name] = dict(status='failed', wall_s=time.monotonic()-start, error=repr(error))
            raise
        receipt['phases'][name] = dict(status='completed', wall_s=time.monotonic()-start)
        write_json(root/'STATUS.json', receipt)
        return value
    try:
        preflight(plan)
        budget.start_gpu(1)
        deadline = time.monotonic() + remaining_seconds
        signal.signal(signal.SIGALRM, deadline_handler)
        signal.setitimer(signal.ITIMER_REAL, remaining_seconds)
        phase('native_probe_qualification', lambda: probe_client.qualify(
            plan, root/'native_qualification', service.connection, serving_gpu=str(plan['serving_gpu'])))
        training = phase('training_qualification_and_native_trajectory', lambda: launch_training(plan, deadline))
        # Reauthenticate every published temporary checkpoint before native loading.
        for row in training['probe_exports']:
            if identity(Path(row['checkpoint'])/'model.safetensors') != row['weights_identity']:
                raise ValueError('Window export changed before delayed probe evaluation')
        phase('independent_native_probes', lambda: probe_client.run(plan, training['probe_exports'],
            root/'probes', service.connection, serving_gpu=str(plan['serving_gpu'])))
        receipt.update(status='completed', phase='report')
    except BaseException as error:
        execution_failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        write_json(root/'FAILURE.json', receipt)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if budget.gpu_seconds >= remaining_seconds:
            stop_owned_processes(budget)
        budget.end_gpu()
        receipt.update(wall_s=time.monotonic()-began, budget=budget.summary(),
            cumulative_gpu_seconds=accounting['prior_gpu_seconds'] + budget.summary()['gpu_seconds'],
            disk_free_bytes=shutil.disk_usage(root).free)
        write_json(root/'STATUS.json', receipt)
        try:
            report(plan, receipt, budget)
            if receipt['status'] == 'completed':
                receipt['phase'] = 'complete'
                write_json(root/'STATUS.json', receipt)
        except BaseException as error:
            receipt.update(status='failed', report_error=repr(error))
            write_json(root/'REPORT_FAILURE.json', dict(error=repr(error), traceback=traceback.format_exc()))
            write_json(root/'STATUS.json', receipt)
            if execution_failure is None:
                raise
        finally:
            service.close()
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
