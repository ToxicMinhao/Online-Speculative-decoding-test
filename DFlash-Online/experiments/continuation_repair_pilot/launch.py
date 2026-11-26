"""Fixed serial pilot supervisor and cumulative Adam/resource accounting."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import traceback

from experiments.frontier_preservation_pilot.qualify import verify_source, storage_usage
from experiments.frontier_preservation_pilot.point1_prepare import gpu_inventory
from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity


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
    # Supervisor and GPU child both check storage; each owns its atomic file.
    usage['writer_pid'] = os.getpid()
    usage['observed_epoch'] = time.time()
    write_json(root / f'STORAGE.{os.getpid()}.json', usage)
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


def execute(plan, *, continuation=None):
    root = Path(plan['root'])
    if continuation is None and ((root/'CLOCK.json').exists() or (root/'STATUS.json').exists()):
        raise FileExistsError('Existing attempt is preserved; no automatic campaign restart')
    verify_source(plan)
    remaining_write_bytes = (2_000_000_000 if continuation is not None
                             else plan['resource_admission']['remote_remaining_peak_bytes'])
    boundary(plan, remaining_write_bytes=remaining_write_bytes)
    if gpu_inventory()['processes']:
        raise RuntimeError('Both cards must be available before this serial pilot begins')
    clock = (deepcopy(continuation['clock']) if continuation is not None else
             dict(first_gpu_action_epoch=time.time(), occupied_card_s=0., elapsed_s=0., processes=[]))
    write_json(root/'CLOCK.json', clock)
    status = dict(status='running', execution_commit=plan['execution_commit'],
                  training_execution_commit=plan.get('training_execution_commit', plan['execution_commit']),
                  phases=deepcopy(continuation['status']['phases']) if continuation is not None else [],
                  automatic_retry=False, new_feedback_generations=0, new_teacher_preparations=0)
    log_root = Path(continuation['directory']) if continuation is not None else root
    plan_path = log_root/'EXECUTION_PLAN.json' if continuation is not None else root/'sealed/PLAN.json'
    if continuation is not None:
        status.update(remaining_phase_continuation=True, repair_receipt=str(log_root/'REPAIR.json'),
                      original_failed_status=str(log_root/'prior/STATUS.json'))
    phase_specs = [('qualification', 'native', ['--phase','qualification'], plan['training_gpu']),
                   ('training', 'native', ['--phase','training'], plan['training_gpu']),
                   ('readout', 'readout', [], plan['training_gpu']),
                   ('evaluation', 'serving', [], plan['serving_gpu'])]
    if continuation is not None:
        phase_specs = [('readout', 'readout', ['--resume-unexecuted'], plan['training_gpu']),
                       ('evaluation', 'serving', [], plan['serving_gpu'])]
    try:
        for phase, module, extra, device in phase_specs:
            boundary(plan)
            if gpu_inventory()['processes']:
                raise RuntimeError('Background GPU work prevents the next pilot phase')
            command = [plan['training_python'], '-u', '-m', 'experiments.continuation_repair_pilot.'+module,
                       '--plan', str(plan_path), *extra]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            began = time.time()
            process = None
            record = dict(phase=phase, physical_gpu=device, start_epoch=began, end_epoch=None, exit_code=None,
                          execution_commit=plan['execution_commit'], continuation=continuation is not None)
            clock['processes'].append(record)
            try:
                with (log_root/(phase+'.log')).open('w') as log:
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
                try:
                    if process is not None:
                        stop(process)
                except BaseException as error:
                    record['termination_error'] = repr(error)
                    raise
                finally:
                    now = time.time()
                    record['exit_code'] = process.poll() if process is not None else None
                    if process is None or record['exit_code'] is not None:
                        record['end_epoch'] = now
                    else:
                        record['active_process_unresolved'] = True
                    clock['occupied_card_s'] = sum((p['end_epoch'] or now)-p['start_epoch']
                                                    for p in clock['processes'])
                    clock['elapsed_s'] = now-clock['first_gpu_action_epoch']
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



def allowed_repair_file(relative):
    """Admit only the demonstrated I/O defect and its bounded resume plumbing."""
    return (relative in ('experiments/continuation_repair_pilot/launch.py',
                         'experiments/continuation_repair_pilot/readout.py',
                         'experiments/continuation_repair_pilot/analysis.py',
                         'experiments/continuation_repair_pilot/README.md')
            or (relative.startswith('tests/test_continuation_repair_') and relative.endswith('.py'))
            or relative.startswith('reports/dflash_continuation_repair_pilot_20261126/'))


def prepare_continuation(plan_path, *, repair_source_root, repair_manifest, repair_commit):
    """Bind a single I/O repair to the unchanged experiment and completed state."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    root = Path(plan['root'])
    directory = root/'resume_001'
    if directory.exists():
        raise FileExistsError('The single localized continuation attempt is already recorded')
    if Path(repair_source_root).resolve() == Path(plan['source_root']).resolve():
        raise ValueError('Keep the original execution checkout intact; stage the repair separately')
    verify_source(plan)
    repaired = deepcopy(plan)
    repaired.update(source_root=str(Path(repair_source_root)), source_manifest=str(Path(repair_manifest)),
                    source_identity=identity(repair_manifest), execution_commit=repair_commit,
                    training_execution_commit=plan['execution_commit'], original_plan=str(plan_path),
                    original_plan_identity=identity(plan_path))
    old_manifest = json.loads(Path(plan['source_manifest']).read_text())
    new_manifest = json.loads(Path(repair_manifest).read_text())
    if new_manifest.get('clean') is not True or new_manifest['commit'] != repair_commit:
        raise ValueError('Repair requires a clean published source identity')
    verify_source(repaired)
    changed = sorted(path for path in old_manifest['files'].keys() | new_manifest['files'].keys()
                     if old_manifest['files'].get(path) != new_manifest['files'].get(path))
    if not changed or any(not allowed_repair_file(path) for path in changed):
        raise ValueError('Repair changes files outside orchestration/readout continuation scope: ' + repr(changed))
    repaired['prior_charged_roots'] = list(dict.fromkeys([
        *plan.get('prior_charged_roots', []), str(Path(plan['source_manifest']).parent)]))
    clock = json.loads((root/'CLOCK.json').read_text())
    status = json.loads((root/'STATUS.json').read_text())
    training = json.loads((root/'arms/TAIL/STATUS.json').read_text())
    qualification = json.loads(Path(plan['qualification_result']).read_text())
    mechanism = json.loads((root/'mechanism/STATUS.json').read_text())
    calls = json.loads((root/'ADAM_LEDGER.json').read_text())['calls']
    if (status['status'] != 'blocked' or [row['phase'] for row in status['phases']] != ['qualification','training']
            or training['status'] != 'completed'
            or [training['global_step'], training['presentations'], training['completed_event']] != [256,512,2]
            or qualification['status'] != 'passed' or qualification['binding'] != plan['binding']
            or not qualification['base_disabled_exact'] or not qualification['retained_BASE_reusable']
            or mechanism['status'] != 'failed' or (root/'evaluation').exists()):
        raise ValueError('Resume requires passed qualification, completed TAIL, failed partial readout and no serving')
    if (len([row for row in calls if row['phase']=='training']) != 128
            or any(row['status'] != 'completed' for row in calls)):
        raise ValueError('Resume cannot bypass incomplete/unknown optimizer work')
    # Native endpoints and their optimizer histories are kept exactly as completed.
    for path, expected in ((training['recovery'], training['recovery_identity']),
                           (Path(training['checkpoint'])/'model.safetensors', training['checkpoint_identity'])):
        if identity(path) != expected:
            raise ValueError('Completed TAIL state changed before continuation')
    if (any(row['end_epoch'] is None for row in clock['processes'])
            or abs(sum(row['end_epoch']-row['start_epoch'] for row in clock['processes'])
                   - clock['occupied_card_s']) > 1e-6):
        raise ValueError('Original occupied work is incomplete or inconsistent')
    boundary(repaired, remaining_write_bytes=2_000_000_000)
    if gpu_inventory()['processes']:
        raise RuntimeError('Both cards must be idle before restoring remaining phases')
    directory.mkdir()
    prior = directory/'prior'
    prior.mkdir()
    preserved = {}
    for relative in ('STATUS.json', 'CLOCK.json', 'EXIT', 'EXIT.json', 'EXIT_CODE', 'EXIT_CODE.txt',
                     'ADAM_LEDGER.json', 'PROCESS_EXIT_CODE', 'PROCESS_PID', 'RUN.sh',
                     'readout.log', 'launch.log', 'qualification/RESULT.json',
                     'arms/TAIL/STATUS.json', 'mechanism/STATUS.json', 'mechanism/ENDPOINT_STATE.json'):
        source = root/relative
        if source.is_file():
            destination = prior/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            preserved[relative] = identity(destination)
    receipt = dict(status='prepared', defect='Concurrent processes shared STORAGE.json.tmp',
                   fix='Per-process storage receipts; continue only unexecuted readout and serving',
                   original_plan=str(plan_path), original_plan_identity=identity(plan_path),
                   binding=plan['binding'], original_training_commit=plan['execution_commit'],
                   orchestration_execution_commit=repair_commit, changed_files=changed,
                   original_source_identity=plan['source_identity'], repair_source_identity=repaired['source_identity'],
                   original_clock_epoch=clock['first_gpu_action_epoch'], prior_occupied_card_s=clock['occupied_card_s'],
                   original_adam_applications=len(calls), additional_adam_applications=0,
                   preserved_failures=preserved, original_settings_unchanged=True)
    write_json(directory/'REPAIR.json', receipt)
    write_json(directory/'EXECUTION_PLAN.json', repaired)
    return repaired, dict(directory=str(directory), clock=clock, status=status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--resume-unexecuted', action='store_true')
    parser.add_argument('--repair-source-root')
    parser.add_argument('--repair-source-manifest')
    parser.add_argument('--repair-commit')
    args=parser.parse_args()
    if args.resume_unexecuted:
        if not all((args.repair_source_root, args.repair_source_manifest, args.repair_commit)):
            parser.error('Remaining-phase continuation requires all three repair source arguments')
        plan, continuation = prepare_continuation(args.plan, repair_source_root=args.repair_source_root,
            repair_manifest=args.repair_source_manifest, repair_commit=args.repair_commit)
        execute(plan, continuation=continuation)
    else:
        if any((args.repair_source_root, args.repair_source_manifest, args.repair_commit)):
            parser.error('Repair source arguments require --resume-unexecuted')
        execute(json.loads(Path(args.plan).read_text()))


if __name__=='__main__':
    main()
