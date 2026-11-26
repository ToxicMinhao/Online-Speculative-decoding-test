"""Fixed serial pilot supervisor and cumulative Adam/resource accounting."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from experiments.frontier_preservation_pilot.qualify import verify_source, storage_usage
from experiments.frontier_preservation_pilot.point1_prepare import gpu_inventory
from online_feedback.storage import write_json


class AdamLedger:
    """Reserve an actual optimizer application before calling it; never erase failures."""
    def __init__(self, path, *, phase, arm=None):
        self.path = Path(path)
        self.phase, self.arm = phase, arm
        self.value = json.loads(self.path.read_text()) if self.path.exists() else dict(calls=[])

    def reserve(self, arm):
        self.value = json.loads(self.path.read_text()) if self.path.exists() else self.value
        if self.phase not in ('qualification', 'training'):
            raise ValueError('Unknown optimizer budget')
        calls = [r for r in self.value['calls'] if r['phase'] == self.phase]
        cap = 6 if self.phase == 'qualification' else 128
        if len(calls) >= cap or (self.phase == 'training' and arm != self.arm):
            raise ValueError('Declared optimizer application budget exhausted')
        token = len(self.value['calls'])
        self.value['calls'].append(dict(token=token, phase=self.phase, arm=arm, status='attempted', epoch=time.time()))
        write_json(self.path, self.value)
        return token

    def complete(self, token):
        self.value['calls'][token].update(status='completed', completed_epoch=time.time())
        write_json(self.path, self.value)


def boundary(plan, *, remaining_write_bytes=0):
    """Current all-location storage and original clock, including inherited failed work."""
    root = Path(plan['root'])
    usage = storage_usage(plan)
    usage['all_locations_charged_bytes'] = usage['charged_bytes'] + plan['local_charged_bytes']
    usage['remaining_write_bytes'] = remaining_write_bytes
    clock_path = root/'CLOCK.json'
    if clock_path.exists():
        clock = json.loads(clock_path.read_text())
        elapsed = time.time() - clock['first_gpu_action_epoch']
        if elapsed >= plan['limits']['elapsed_s'] or clock['occupied_card_s'] >= plan['limits']['occupied_card_s']:
            raise RuntimeError('Original pilot time limit exhausted')
    write_json(root/'STORAGE.json', usage)
    if (usage['free_bytes'] - remaining_write_bytes < plan['limits']['reserve_bytes'] or
            usage['all_locations_charged_bytes'] + remaining_write_bytes > plan['limits']['allocated_bytes']):
        raise RuntimeError('Remaining pilot work violates storage budget or reserve')
    return usage


def stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute(plan):
    root = Path(plan['root'])
    if (root/'CLOCK.json').exists() or (root/'STATUS.json').exists():
        raise FileExistsError('Existing attempt is preserved; no automatic campaign restart')
    verify_source(plan)
    boundary(plan, remaining_write_bytes=plan['resource_admission']['remote_remaining_peak_bytes'])
    if gpu_inventory()['processes']:
        raise RuntimeError('Both cards must be available before this serial pilot begins')
    clock = dict(first_gpu_action_epoch=time.time(), occupied_card_s=0., elapsed_s=0., processes=[])
    write_json(root/'CLOCK.json', clock)
    status = dict(status='running', execution_commit=plan['execution_commit'], phases=[],
                  automatic_retry=False, new_feedback_generations=0, new_teacher_preparations=0)
    phase_specs = [('qualification', 'native', ['--phase','qualification'], plan['training_gpu']),
                   ('training', 'native', ['--phase','training'], plan['training_gpu']),
                   ('readout', 'readout', [], plan['training_gpu']),
                   ('evaluation', 'serving', [], plan['serving_gpu'])]
    try:
        for phase, module, extra, device in phase_specs:
            boundary(plan)
            if gpu_inventory()['processes']:
                raise RuntimeError('Background GPU work prevents the next pilot phase')
            command = [plan['training_python'], '-u', '-m', 'experiments.continuation_repair_pilot.'+module,
                       '--plan', str(root/'sealed/PLAN.json'), *extra]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            began = time.time()
            process = None
            record = dict(phase=phase, physical_gpu=device, start_epoch=began, end_epoch=None, exit_code=None)
            clock['processes'].append(record)
            try:
                with (root/(phase+'.log')).open('w') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                               cwd=plan['source_root'], start_new_session=True)
                    record['pid'] = process.pid
                    while process.poll() is None:
                        now = time.time()
                        clock['occupied_card_s'] = sum((p['end_epoch'] or now)-p['start_epoch'] for p in clock['processes'])
                        clock['elapsed_s'] = now-clock['first_gpu_action_epoch']
                        write_json(root/'CLOCK.json',clock)
                        boundary(plan)
                        if clock['occupied_card_s'] >= plan['limits']['occupied_card_s']-30 or clock['elapsed_s'] >= plan['limits']['elapsed_s']-30:
                            raise RuntimeError('Pilot time shutdown boundary reached')
                        time.sleep(1)
            finally:
                if process is not None:
                    stop(process)
                    record['exit_code'] = process.returncode
                record['end_epoch'] = time.time()
                clock['occupied_card_s'] = sum(p['end_epoch']-p['start_epoch'] for p in clock['processes'])
                clock['elapsed_s'] = time.time()-clock['first_gpu_action_epoch']
                write_json(root/'CLOCK.json',clock)
            if record['exit_code'] != 0:
                raise RuntimeError(f'{phase} failed; preserve evidence, no automatic retry')
            if phase == 'qualification':
                result = json.loads(Path(plan['qualification_result']).read_text())
                if result['status'] != 'passed' or not result['base_disabled_exact'] or not result['retained_BASE_reusable']:
                    raise ValueError('Required native qualification did not pass')
            status['phases'].append(dict(phase=phase,status='completed',wall_s=record['end_epoch']-began))
            write_json(root/'STATUS.json',status)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='blocked',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        status['clock'] = clock
        write_json(root/'STATUS.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    execute(json.loads(Path(args.plan).read_text()))


if __name__=='__main__':
    main()
