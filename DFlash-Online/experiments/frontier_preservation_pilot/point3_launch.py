"""Durable Point3 work ledger and admitted, serial phase supervision."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .point3_prepare import ARMS, CASES, HISTORICAL_CARD_S, LIMITS, SCHEMA


class Point3StepLedger:
    """Reserve before applying work, so an interrupted call remains charged."""

    def __init__(self, path, contract_identity):
        self.path = Path(path)
        if self.path.exists():
            self.value = json.loads(self.path.read_text())
            if self.value['contract_identity'] != contract_identity:
                raise ValueError('Point3 ledger belongs to another contract')
        else:
            self.value = dict(schema='point3_step_ledger_v1', contract_identity=contract_identity,
                              attempts=[], calls=[], token_backwards=[])
            self.save()

    def save(self):
        write_json(self.path, self.value)

    def start_attempt(self, number, correction_note=None):
        attempts = self.value['attempts']
        if number != len(attempts)+1 or number > 2:
            raise ValueError('Only initial qualification and one identified correction retry')
        if number == 2 and (not correction_note or attempts[-1]['status'] != 'failed'):
            raise ValueError('Retry requires a retained failure and implementation correction')
        if any(row['phase'] == 'event' for row in self.value['calls']):
            raise ValueError('Qualification cannot restart after production begins')
        attempts.append(dict(number=number, correction_note=correction_note,
                             status='running', started_epoch=time.time()))
        self.save()

    def reserve(self, case, *, window=None):
        calls = self.value['calls']
        if len(calls) >= LIMITS['attempted_adam_calls']:
            raise RuntimeError('Point3 total attempted Adam budget exhausted')
        if window is None:
            if case not in CASES or not self.value['attempts']:
                raise ValueError('Unplanned qualification case')
            attempt = self.value['attempts'][-1]
            previous = [r for r in calls if r['phase']=='qualification']
            current = [r for r in previous if r['attempt']==attempt['number']]
            if (attempt['status'] != 'running' or len(previous) >= 6
                    or len(current) >= 3 or case != CASES[len(current)]):
                raise RuntimeError('Qualification Adam order or budget changed')
            phase, attempt_number = 'qualification', attempt['number']
        else:
            if (case not in ARMS or not self.value['attempts']
                    or self.value['attempts'][-1]['status'] != 'completed'):
                raise ValueError('Event requires completed qualification')
            previous = [r for r in calls if r['phase']=='event' and r['case']==case]
            if window != len(previous)+1 or window > 128 or any(r['status']!='completed' for r in previous):
                raise RuntimeError('No duplicate, skipped or restarted production window')
            phase, attempt_number = 'event', None
        row = dict(number=len(calls)+1, phase=phase, attempt=attempt_number,
                   case=case, window=window, status='reserved', reserved_epoch=time.time())
        calls.append(row)
        self.save()
        return row['number']

    def complete(self, reservation):
        row = self.value['calls'][reservation-1]
        if row['number'] != reservation or row['status'] != 'reserved':
            raise ValueError('Adam completion lacks a pending reservation')
        row.update(status='completed', completed_epoch=time.time())
        self.save()

    def reserve_token(self, label):
        attempt = self.value['attempts'][-1]
        rows = self.value['token_backwards']
        this_attempt = [row for row in rows if row['attempt']==attempt['number']]
        if attempt['status'] != 'running' or len(rows)>=60 or len(this_attempt)>=30:
            raise RuntimeError('Point3 native oracle backward budget exhausted')
        row = dict(number=len(rows)+1, attempt=attempt['number'], label=label,
                   status='reserved', reserved_epoch=time.time())
        rows.append(row)
        self.save()
        return row['number']

    def complete_token(self, reservation):
        row = self.value['token_backwards'][reservation-1]
        if row['status'] != 'reserved':
            raise ValueError('Oracle completion lacks a pending reservation')
        row.update(status='completed', completed_epoch=time.time())
        self.save()

    def finish_attempt(self, number, success):
        self.value = json.loads(self.path.read_text())
        attempt = self.value['attempts'][-1]
        if attempt['number'] != number or attempt['status'] != 'running':
            raise ValueError('Qualification finalization differs')
        if success:
            calls = [row for row in self.value['calls'] if row['attempt']==number]
            tokens = [row for row in self.value['token_backwards'] if row['attempt']==number]
            if ([row['case'] for row in calls] != list(CASES) or len(tokens)!=28
                    or any(row['status']!='completed' for row in calls+tokens)):
                raise ValueError('Qualification cannot pass with missing/unknown work')
        attempt.update(status='completed' if success else 'failed', finished_epoch=time.time())
        self.save()


def boundary(plan):
    """Read-only child boundary; no new clock or omitted occupied interval."""
    from experiments.bounded_prefix_allocation.runtime import allocated_bytes
    root = Path(plan['root'])
    clock = json.loads((root/'CLOCK.json').read_text())
    now = time.time()
    current = clock['phases'][-1]
    active = now-current['started_epoch'] if current['status']=='running' else 0.
    occupied = clock['completed_card_s']+active
    elapsed = now-clock['first_gpu_action_epoch']
    charged_remote = allocated_bytes([str(root),str(Path(plan['source_manifest']).parent)])
    charged = charged_remote + plan['forecast']['storage']['local_allowance_bytes']
    remaining = max(0,plan['forecast']['storage']['remote_peak_bytes']-charged_remote)
    free = shutil.disk_usage(root).free
    if elapsed >= LIMITS['elapsed_s']-30 or occupied >= LIMITS['occupied_card_s']-30:
        raise TimeoutError('Point3 elapsed or occupied-card budget exhausted')
    if charged > LIMITS['allocated_bytes'] or free-remaining < LIMITS['reserve_bytes']:
        raise RuntimeError('Point3 peak storage/reserve no longer admitted')
    return dict(elapsed_s=elapsed,occupied_card_s=occupied,
        cumulative_card_s=HISTORICAL_CARD_S+occupied,charged_bytes=charged,
        free_bytes=free,remaining_peak_bytes=remaining)


def launch(plan_path):
    """No automatic retry. Native qualification precedes both events and serving."""
    from .point1_prepare import gpu_inventory, host_memory
    from .qualify import verify_source
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if (plan['schema'] != SCHEMA or plan['limits'] != LIMITS
            or not plan['production_admitted'] or not plan['resource_admission']['admitted']):
        raise ValueError('Require the complete admitted Point3 contract')
    verify_source(plan)
    root = Path(plan['root'])
    clock_path = root/'CLOCK.json'
    if clock_path.exists():
        raise FileExistsError('Existing Point3 clock: preserve work; no full-event restart')
    ledger = Point3StepLedger(root/'STEP_LEDGER.json', identity(plan_path))
    ledger.start_attempt(1)
    phases = [('qualification', None), *[('train', arm) for arm in ARMS], ('evaluate', None)]
    clock = dict(first_gpu_action_epoch=None, completed_card_s=0.,
                 historical_card_s=HISTORICAL_CARD_S, phases=[], status='preflight')
    for phase, arm in phases:
        inventory = gpu_inventory()
        if inventory['processes']:
            raise RuntimeError('Both cards must be quiet before Point3 phase launch')
        if host_memory()['available_bytes'] < 87_000_000_000:
            raise RuntimeError('Point3 host workspace plus32GB reserve unavailable')
        if shutil.disk_usage(root).free < LIMITS['reserve_bytes']:
            raise RuntimeError('A800 free-space reserve lost')
        if phase == 'qualification' and shutil.disk_usage(root).free < 420_000_000_000:
            raise RuntimeError('Whole comparison storage no longer admitted')
        now = time.time()
        if clock['first_gpu_action_epoch'] is None:
            clock['first_gpu_action_epoch'] = now
        remaining = min(LIMITS['elapsed_s']-(now-clock['first_gpu_action_epoch']),
                        LIMITS['occupied_card_s']-clock['completed_card_s']) - 30
        if remaining <= 0:
            raise TimeoutError('Point3 phase budget exhausted')
        card = 'A' if phase=='evaluate' else 'B'
        selected = next(row for row in inventory['cards'] if row['uuid']==plan['devices'][card])
        if selected['total_bytes']-selected['used_bytes'] < 60_000_000_000:
            raise RuntimeError('Point3 free VRAM requirement not met')
        source = Path(plan['source_root'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['devices'][card],
            PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(str(source/x) for x in ('.','training','inference')))
        command = [plan['training_python'], '-m',
            'experiments.frontier_preservation_pilot.point3_launch', '--child', phase,
            '--plan', str(plan_path)]
        if arm is not None:
            command += ['--arm',arm]
        label = phase if arm is None else phase+'_'+arm
        record = dict(phase=phase, arm=arm, card=card, started_epoch=now, status='running', command=command)
        clock['phases'].append(record)
        clock['status']='running'
        write_json(clock_path,clock)
        child = None
        try:
            boundary(plan)
            with (root/(label+'.log')).open('xb') as log:
                child = subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=child.pid
                write_json(clock_path,clock)
                while child.poll() is None:
                    time.sleep(5)
                    try:
                        boundary(plan)
                    except (RuntimeError,TimeoutError):
                        from experiments.repair_conditional_kl_pilot.launch import stop_owned
                        stop_owned(child)
                        raise
                    if time.time()-now >= remaining:
                        from experiments.repair_conditional_kl_pilot.launch import stop_owned
                        stop_owned(child)
                        raise TimeoutError('Point3 phase budget exhausted')
                record['exit_code']=child.returncode
                if child.returncode:
                    raise RuntimeError('Point3 phase failed; no automatic retry: '+label)
            record['status']='completed'
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            clock['status']='failed'
            if child is not None and child.poll() is None:
                from experiments.repair_conditional_kl_pilot.launch import stop_owned
                try:
                    stop_owned(child)
                except BaseException as shutdown_error:
                    record.update(shutdown_error=repr(shutdown_error), child_termination='unknown')
            raise
        finally:
            record['wall_s']=time.time()-now
            clock['completed_card_s']+=record['wall_s']
            clock['cumulative_card_s']=HISTORICAL_CARD_S+clock['completed_card_s']
            try:
                if phase=='qualification':
                    ledger.finish_attempt(1, record['status']=='completed')
            except BaseException as error:
                record.update(status='failed', ledger_error=repr(error))
                clock['status']='failed'
                try:
                    ledger.finish_attempt(1,False)
                except BaseException as ledger_error:
                    record['ledger_failure_record_error']=repr(ledger_error)
                raise
            finally:
                write_json(clock_path,clock)
    clock['status']='completed'
    write_json(clock_path,clock)
    return clock


def child(plan_path, phase, arm=None):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    root = Path(plan['root'])
    plan.update(step_ledger=str(root/'STEP_LEDGER.json'), contract_identity=identity(plan_path),
                qualification_result=str(root/'qualification/native/RESULT.json'))
    if phase=='qualification':
        from .point3_native import run
        return run(str(plan_path),root/'qualification/native',plan['execution_commit'],plan['step_ledger'])
    if phase=='train':
        from .point3_training import run_event
        return run_event(plan,arm)
    from .point3_evaluation import run_evaluation
    return run_evaluation(plan)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--child',choices=('qualification','train','evaluate'))
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    child(args.plan,args.child,args.arm) if args.child else launch(args.plan)
