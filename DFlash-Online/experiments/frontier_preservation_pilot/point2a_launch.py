"""One bounded Point2A subprocess, durable Adam reservations and one shared clock."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .point2a_prepare import CASES, HISTORICAL_CARD_S, LIMITS, SCHEMA, gpu_inventory, host_memory, resource_admission


def child_memory(pid, proc_root=Path('/proc')):
    """Read the owned child's resident and peak host memory without attaching."""
    path = Path(proc_root)/str(pid)/'status'
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return dict(status='exited', resident_bytes=None, peak_resident_bytes=None)
    values = {}
    for line in lines:
        key, value = line.split(':', 1)
        if key in ('VmRSS', 'VmHWM'):
            values[key] = int(value.split()[0])*1024
    return dict(status='measured', resident_bytes=values.get('VmRSS'),
                peak_resident_bytes=values.get('VmHWM'))


def completed_cases(ledger_value, attempt, duplicate_reuse=False):
    rows = [row for row in ledger_value['calls'] if row['attempt'] == attempt]
    expected = list(CASES[:-1] if duplicate_reuse else CASES)
    return ([row['case'] for row in rows] == expected
            and all(row['status'] == 'completed' for row in rows))


def completed_oracle(ledger_value, attempt):
    rows = [row for row in ledger_value['token_backwards'] if row['attempt'] == attempt]
    return len(rows) == 28 and all(row['status'] == 'completed' for row in rows)


class Point2AStepLedger:
    """Single-process reservations are durable before the actual optimizer call.

    A reserved call without completion remains charged. An implementation retry
    can reopen this ledger, but cannot erase cases, attempts or the six-call cap.
    """

    def __init__(self, path, contract_identity):
        self.path = Path(path)
        if self.path.exists():
            self.value = json.loads(self.path.read_text())
            if (self.value['contract_identity'] != contract_identity
                    or self.value['maximum_attempted_calls'] != 6):
                raise ValueError('Point2A step ledger belongs to another contract')
        else:
            self.value = dict(schema='point2a_native_step_ledger_v1', contract_identity=contract_identity,
                planned_cases=list(CASES), maximum_attempted_calls=6, maximum_attempts=2,
                attempts=[], calls=[], token_backwards=[])
            self.save()

    def save(self):
        write_json(self.path, self.value)

    def start_attempt(self, number, correction_note=None):
        attempts = self.value['attempts']
        if number != len(attempts)+1 or number > 2:
            raise ValueError('Point2A permits the initial attempt and one identified correction retry')
        if number == 2 and (not correction_note or attempts[-1]['status'] != 'failed'):
            raise ValueError('Retry requires a failed attempt and identified implementation correction')
        attempts.append(dict(number=number, status='running', correction_note=correction_note,
                             started_epoch=time.time()))
        self.save()

    def reserve(self, case):
        if case not in CASES:
            raise ValueError('Unplanned native Adam case')
        if not self.value['attempts'] or self.value['attempts'][-1]['status'] != 'running':
            raise ValueError('No running Point2A attempt')
        calls = self.value['calls']
        if len(calls) >= self.value['maximum_attempted_calls']:
            raise RuntimeError('Point2A six attempted Adam calls exhausted')
        attempt = self.value['attempts'][-1]['number']
        if any(row['attempt']==attempt and row['case']==case for row in calls):
            raise RuntimeError('Cannot repeat a native case inside an attempt')
        row = dict(number=len(calls)+1, attempt=attempt, case=case,
                   status='reserved', reserved_epoch=time.time())
        calls.append(row)
        self.save()
        return row['number']

    def complete(self, reservation):
        row = self.value['calls'][reservation-1]
        if row['number'] != reservation or row['status'] != 'reserved':
            raise ValueError('Optimizer completion has no unique pending reservation')
        row.update(status='completed', completed_epoch=time.time())
        self.save()

    def reserve_token(self, label):
        attempt = self.value['attempts'][-1]
        if attempt['status'] != 'running':
            raise ValueError('No active Point2A attempt')
        rows = self.value['token_backwards']
        count = sum(row['attempt'] == attempt['number'] for row in rows)
        if len(rows) >= 60 or count >= 30:
            raise RuntimeError('Point2A token-backward budget exhausted')
        row = dict(number=len(rows)+1,attempt=attempt['number'],label=label,
                   status='reserved',reserved_epoch=time.time())
        rows.append(row)
        self.save()
        return row['number']

    def complete_token(self, reservation):
        row = self.value['token_backwards'][reservation-1]
        if row['status'] != 'reserved':
            raise ValueError('Token backward has no pending reservation')
        row.update(status='completed',completed_epoch=time.time())
        self.save()

    def finish_attempt(self, number, success):
        # The child writes step entries, so reload before changing supervisor metadata.
        self.value = json.loads(self.path.read_text())
        current = self.value['attempts'][-1]
        if current['number'] != number or current['status'] != 'running':
            raise ValueError('Point2A attempt finalization differs')
        current.update(status='completed' if success else 'failed', finished_epoch=time.time())
        self.save()


def phase_clock(root, contract_identity, attempt, now):
    path = Path(root)/'CLOCK.json'
    if attempt == 1:
        value = dict(schema='point2a_clock_v1', contract_identity=contract_identity,
            first_gpu_action_epoch=now, elapsed_limit_s=3600, occupied_card_limit_s=3600,
            completed_occupied_card_s=0.0, historical_occupied_card_s=HISTORICAL_CARD_S,
            active_attempt=None, attempts=[])
        with path.open('x') as stream:
            json.dump(value, stream, indent=2)
    else:
        value = json.loads(path.read_text())
        if (value['contract_identity'] != contract_identity
                or value['elapsed_limit_s'] != 3600 or value['occupied_card_limit_s'] != 3600
                or value['active_attempt'] is not None):
            raise ValueError('Cannot reset or overlap the Point2A clock')
    return value


def remaining_time(clock, now, active_occupied_s=0.0):
    return min(clock['elapsed_limit_s']-(now-clock['first_gpu_action_epoch']),
               clock['occupied_card_limit_s']-clock['completed_occupied_card_s']-active_occupied_s)


def storage_usage(plan):
    from experiments.kplus_full_gsm8k.storage import compiler_snapshot
    from experiments.bounded_prefix_allocation.runtime import allocated_bytes
    current = compiler_snapshot()
    if current['directories'] != plan['compiler_baseline']['directories']:
        raise ValueError('Compiler cache roots changed')
    growth = max(0,current['allocated_bytes']-plan['compiler_baseline']['allocated_bytes'])
    remote = allocated_bytes(plan['charged_remote_roots'])
    return dict(remote_allocated_bytes=remote, local_allocated_bytes=plan['local_allocated_bytes'],
        compiler_growth_bytes=growth, all_locations_charged_bytes=remote+growth+plan['local_allocated_bytes'],
        free_bytes=shutil.disk_usage(plan['root']).free)


def run(contract, attempt=1, correction_note=None):
    from .qualify import STOP_GRACE_S, stop_child, verify_source
    contract = Path(contract)
    plan = json.loads(contract.read_text())
    if plan['schema'] != SCHEMA or plan['status'] != 'admitted' or plan['limits'] != LIMITS:
        raise ValueError('Require the sealed Point2A diagnostic contract')
    verify_source(plan)
    root = Path(plan['root'])
    attempt_root = root/f'attempt_{attempt:03d}'
    attempt_root.mkdir(exist_ok=False)
    usage = storage_usage(plan)
    memory, gpu = host_memory(), gpu_inventory()
    selected = next(row for row in gpu['cards'] if row['uuid']==plan['devices']['B'])
    admission = resource_admission(free_bytes=usage['free_bytes'],
        charged_bytes=usage['all_locations_charged_bytes'],
        remaining_write_bytes=plan['remaining_write_budget_bytes'],
        host_available_bytes=memory['available_bytes'],
        gpu_free_bytes=selected['total_bytes']-selected['used_bytes'],
        gpu_busy=any(plan['devices']['B'] in row for row in gpu['processes'].splitlines()))
    write_json(attempt_root/'ADMISSION.json',dict(**admission,host_memory=memory,gpu_inventory=gpu))
    if not admission['admitted']:
        raise RuntimeError('Live Point2A resource admission failed')
    ledger_path = root/'STEP_LEDGER.json'
    ledger = Point2AStepLedger(ledger_path, identity(contract))
    ledger.start_attempt(attempt, correction_note)
    clock = phase_clock(root, identity(contract), attempt, time.time())
    if remaining_time(clock,time.time()) <= STOP_GRACE_S:
        ledger.finish_attempt(attempt,False)
        raise RuntimeError('Point2A shared clock is exhausted')
    command = [plan['training_python'],'-u','-m','experiments.frontier_preservation_pilot.point2a_native',
        '--contract',str(contract),'--output',str(attempt_root/'native'),
        '--execution-commit',plan['execution_commit'],'--step-ledger',str(ledger_path)]
    source = Path(plan['source_root'])
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES=plan['devices']['B'],PYTHONDONTWRITEBYTECODE='1')
    environment['PYTHONPATH'] = os.pathsep.join(str(source/p) for p in ('.','training','inference'))
    started = time.perf_counter()
    clock['active_attempt'] = dict(number=attempt,started_epoch=time.time())
    write_json(root/'CLOCK.json',clock)
    stopped = None
    child = None
    with (attempt_root/'native.log').open('x') as log:
        try:
            child = subprocess.Popen(command,cwd=source,env=environment,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            write_json(attempt_root/'LAUNCH.json',dict(pid=child.pid,command=command,
                physical_gpu_uuid=plan['devices']['B'],first_gpu_action_epoch=clock['first_gpu_action_epoch']))
            last_sample = 0.0
            while child.poll() is None:
                elapsed = time.perf_counter()-started
                memory = host_memory()
                if remaining_time(clock,time.time(),elapsed) <= STOP_GRACE_S:
                    stopped = 'Point2A elapsed/card budget exhausted'
                if shutil.disk_usage(root).free < LIMITS['reserve_bytes']:
                    stopped = 'Original filesystem reserve crossed'
                if memory['available_bytes'] < admission['host_reserve_bytes']:
                    stopped = 'Point2A host memory reserve crossed'
                if time.perf_counter()-last_sample >= 30:
                    usage = storage_usage(plan)
                    resources = dict(epoch=time.time(),elapsed_s=elapsed,**usage,
                                     host_memory=memory,child_memory=child_memory(child.pid),
                                     gpu_inventory=gpu_inventory())
                    with (attempt_root/'RESOURCE_SAMPLES.jsonl').open('a') as stream:
                        stream.write(json.dumps(resources)+'\n')
                    if usage['all_locations_charged_bytes'] > LIMITS['allocated_bytes']:
                        stopped = 'Point2A all-locations disk cap exceeded'
                    last_sample = time.perf_counter()
                if stopped:
                    break
                time.sleep(5)
        except BaseException as error:
            stopped = 'Point2A supervisor failure: '+repr(error)
        finally:
            if child is not None:
                stop_child(child)
    elapsed = time.perf_counter()-started
    clock['completed_occupied_card_s'] += elapsed
    clock['active_attempt'] = None
    clock['attempts'].append(dict(number=attempt,occupied_card_s=elapsed,
        exit_code=None if child is None else child.returncode,stopped_reason=stopped))
    write_json(root/'CLOCK.json',clock)
    succeeded = child is not None and child.returncode == 0 and stopped is None
    native_result_path = attempt_root/'native/RESULT.json'
    native_result = json.loads(native_result_path.read_text()) if native_result_path.exists() else None
    ledger_value = json.loads(ledger_path.read_text())
    if succeeded and (not completed_cases(ledger_value,attempt, bool(native_result and native_result.get('duplicate_F_C_reused')))
                      or not completed_oracle(ledger_value, attempt)
                      or native_result is None or native_result['status'] != 'completed'):
        succeeded = False
        stopped = 'Native completion, three-case or 28-token oracle ledger is incomplete'
    storage_error = None
    try:
        usage = storage_usage(plan)
    except Exception as error:
        storage_error = repr(error)
        succeeded = False
    if (usage['all_locations_charged_bytes'] > LIMITS['allocated_bytes']
            or usage['free_bytes'] < LIMITS['reserve_bytes']):
        succeeded = False
        stopped = 'Final Point2A disk accounting exceeds its admitted limits'
    ledger.finish_attempt(attempt,succeeded)
    result = dict(status='completed' if succeeded else 'failed',
        exit_code=None if child is None else child.returncode,stopped_reason=stopped,
        elapsed_s=elapsed,occupied_card_s=elapsed,new_phase_elapsed_s=time.time()-clock['first_gpu_action_epoch'],
        new_phase_occupied_card_s=clock['completed_occupied_card_s'],
        historical_occupied_card_s=HISTORICAL_CARD_S,
        historical_plus_new_occupied_card_s=HISTORICAL_CARD_S+clock['completed_occupied_card_s'],
        final_storage_error=storage_error,**usage)
    write_json(attempt_root/'PROCESS.json',result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--attempt',type=int,choices=(1,2),default=1)
    parser.add_argument('--correction-note')
    result = run(**vars(parser.parse_args()))
    if result['status'] != 'completed':
        raise SystemExit(1)
