"""CPU admission, fixed qualification, then the unchanged bounded campaign.

The launcher never retries scientific work. It keeps the first model-load clock
running across qualification, resource admission and the production queue.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from .resources import CARD_SECONDS, ELAPSED_SECONDS, LIMITS
from .task_runner import clock_status, occupied_card, read


def child_environment(plan, slot=None):
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = '' if slot is None else plan['devices'][slot]
    environment['PYTHONPATH'] = os.pathsep.join([
        plan['source_root'], str(Path(plan['source_root'])/'training'),
        str(Path(plan['source_root'])/'inference'), environment.get('PYTHONPATH', '')])
    for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        environment.pop(name, None)
    return environment


def run_qualification_child(plan_path, plan, kind, slot):
    """Lease a card and enforce both clocks even inside a long native call."""
    root = Path(plan['root'])
    name = kind+'_'+slot
    module = ('qualification_entry' if kind == 'training' else 'serving_qualification')
    command = [plan['training_python'], '-u', '-m',
               'experiments.persistent_prefix_withholding.'+module,
               '--plan', str(plan_path), '--slot', slot]
    with (root/'logs'/(name+'.log')).open('x') as log:
        with occupied_card(root, name, plan['devices'][slot]):
            process = subprocess.Popen(command, cwd=plan['source_root'],
                env=child_environment(plan, slot), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            write_json(root/'QUALIFICATION_PROCESS.json', dict(
                phase=kind, slot=slot, pid=process.pid, physical_gpu_uuid=plan['devices'][slot]))
            try:
                while process.poll() is None:
                    clock = clock_status(root)
                    if clock['occupied_s'] >= CARD_SECONDS or clock['elapsed_s'] >= ELAPSED_SECONDS:
                        raise TimeoutError('Shared qualification/campaign time cap reached')
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError('Qualification failed: '+name)
            finally:
                if process.poll() is None:
                    import psutil
                    try:
                        descendants = psutil.Process(process.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        descendants = []
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        for child in reversed(descendants):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
    directory = 'qualification' if kind == 'training' else 'serving_qualification'
    report = read(root/directory/slot/'STATUS.json')
    if report['status'] != 'passed':
        raise RuntimeError('Unpassed native qualification: '+name)
    return report


def seed_production_work(root, training, serving):
    """Carry actual completed qualification work into the shared main ledger."""
    counters = {key: 0 for key in LIMITS}
    for report in training.values():
        work = report['physical_work']
        if work['failed'] or work['attempted'] != work['completed']:
            raise ValueError('Qualification has failed or unknown physical work')
        for old, new in (('model_load', 'training_initialization'),
                         ('optimizer_step', 'adam'),
                         ('response_forward', 'response_forward'),
                         ('response_backward', 'response_backward')):
            counters[new] += work['completed'][old]
    for report in serving.values():
        result = report['qualification']
        if result['valid_generations'] != result['attempted_generations']:
            raise ValueError('Qualification has invalid or unknown serving work')
        counters['qualification_generation'] += result['attempted_generations']
    expected = dict(adam=8, response_forward=20, response_backward=20,
                    training_initialization=2, qualification_generation=8)
    if any(counters[key] != value for key, value in expected.items()):
        raise ValueError('The bounded native qualification work differs')
    write_json(Path(root)/'WORK.json', dict(attempted=counters, completed=dict(counters),
        next_operation=0, pending={}, origin='actual complete qualification ledgers'))


def run(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    root = Path(plan['root'])
    if (root/'LAUNCH.json').exists():
        raise FileExistsError('No automatic scientific retry or reused launch directory')
    (root/'logs').mkdir(exist_ok=True)
    receipt = dict(status='cpu_preflight', execution_commit=plan['execution_commit'],
                   began_epoch=time.time(), automatic_retry=False, completed_phases=[])
    write_json(root/'LAUNCH.json', receipt)
    try:
        with (root/'logs/CPU_PREFLIGHT.log').open('x') as log:
            subprocess.run([plan['training_python'], '-u', '-m',
                'experiments.persistent_prefix_withholding.preflight', str(root)],
                cwd=plan['source_root'], env=child_environment(plan),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt['completed_phases'].append('cpu_preflight')
        # The baseline includes all existing compilation caches; only growth is
        # charged against the separately reserved four GiB for this campaign.
        allocated = int(subprocess.check_output(
            ['du', '-s', '-B1', plan['compiler_root']], text=True).split()[0])
        plan['compiler_baseline_allocated_bytes'] = allocated
        plan['gpu_deadline_monotonic'] = time.monotonic()+ELAPSED_SECONDS
        write_json(plan_path, plan)
        write_json(root/'CLOCK.json', dict(first_model_load_epoch=time.time(),
            completed_card_s=0., active={}, finished=[]))
        receipt.update(status='qualification', clock=str(root/'CLOCK.json'))
        write_json(root/'LAUNCH.json', receipt)
        training = {}
        serving = {}
        # Measure the simpler persistent worker before admitting two workers.
        # These fixed qualification applications are never duplicated.
        for slot in ('A', 'B'):
            training[slot] = run_qualification_child(plan_path, plan, 'training', slot)
            receipt['completed_phases'].append('training_'+slot)
            write_json(root/'LAUNCH.json', receipt)
        from .qualification import compare_cards
        cross = compare_cards(training['A']['qualification'], training['B']['qualification'])
        cross['execution_commit'] = plan['execution_commit']
        write_json(root/'qualification/CROSS_CARD.json', cross)
        for slot in ('A', 'B'):
            serving[slot] = run_qualification_child(plan_path, plan, 'serving', slot)
            receipt['completed_phases'].append('serving_'+slot)
            write_json(root/'LAUNCH.json', receipt)
        from .serving_qualification import compare_serving_cards
        cross = compare_serving_cards(serving['A'], serving['B'])
        cross['execution_commit'] = plan['execution_commit']
        write_json(root/'serving_qualification/CROSS_CARD.json', cross)
        seed_production_work(root, training, serving)
        from .admission import qualify_resources
        admission = qualify_resources(plan, training, serving)
        write_json(root/'RESOURCE_ADMISSION.json', admission)
        if admission['status'] != 'passed':
            raise RuntimeError('Complete fixed production work does not fit resource admission')
        plan.update(costs=admission['costs'], task_memory_bytes=admission['task_memory_bytes'],
                    task_vram_bytes=admission['task_vram_bytes'])
        write_json(plan_path, plan)
        receipt.update(status='production', expected_remaining=admission['time'])
        write_json(root/'LAUNCH.json', receipt)
        from .campaign import run_campaign
        receipt['campaign'] = run_campaign(plan_path)
        receipt['completed_phases'].append('production')
        receipt['status'] = 'completed'
    except BaseException as error:
        receipt.update(status='stopped', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['wall_s'] = time.time()-receipt['began_epoch']
        if (root/'CLOCK.json').exists():
            receipt['clock'] = clock_status(root)
        write_json(root/'LAUNCH.json', receipt)
        from .report import assemble_report
        try:
            assemble_report(plan_path)
        except BaseException as error:
            write_json(root/'REPORT_FAILURE.json', dict(error=repr(error),
                traceback=traceback.format_exc(), scientific_records_preserved=True))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    run(args.plan)


if __name__ == '__main__':
    main()
