"""One clock, one bounded derivative child, no optimizer applications or retry."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from online_feedback.storage import write_json
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from experiments.ok_coding.storage import identity
from experiments.bounded_prefix_allocation.runtime import allocated_bytes
from .prepare import RESERVE

STOP_GRACE_S = 20  # five-second monitor interval plus the fifteen-second termination grace


def verify_source(plan):
    """Check actual staged bytes, not just the name of a source manifest."""
    if identity(plan['source_manifest']) != plan['source_identity']:
        raise ValueError('Source binding changed')
    manifest = json.loads(Path(plan['source_manifest']).read_text())
    if manifest['commit'] != plan['execution_commit']:
        raise ValueError('Execution commit differs from staged source')
    source = Path(plan['source_root'])
    for relative, expected in manifest['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid staged source path')
        actual = source / path
        if expected.get('kind') == 'symlink':
            valid = actual.is_symlink() and os.readlink(actual) == expected['target']
        else:
            valid = not actual.is_symlink() and identity(actual) == expected
        if not valid:
            raise ValueError('Staged source changed: ' + relative)


def storage_usage(plan):
    compiler = compiler_snapshot()
    if compiler['directories'] != plan['compiler_baseline']['directories']:
        raise ValueError('Compiler cache locations changed')
    growth = max(0, compiler['allocated_bytes'] - plan['compiler_baseline']['allocated_bytes'])
    allocated = allocated_bytes([
        plan['root'], str(Path(plan['source_manifest']).parent),
        *plan.get('prior_charged_roots', []),
    ])
    return dict(allocated_bytes=allocated, compiler_growth_bytes=growth,
                charged_bytes=allocated + growth, free_bytes=shutil.disk_usage(plan['root']).free)


def remaining_budget(clock, prior_card_s, now):
    """A repair inherits the original clock and every earlier occupied second."""
    return min(clock['elapsed_limit_s'] - (now - clock['first_gpu_action_epoch']),
               clock['occupied_card_limit_s'] - prior_card_s)


def verify_original_records(plan):
    required = {plan['original_clock'], plan['original_process']}
    required.update(str(Path(plan['original_probe']) / name)
                    for name in ('RESULT.json', 'WINDOW.json'))
    if not required <= plan['prior_record_identities'].keys():
        raise ValueError('Repair plan omits an original clock, process, result or window binding')
    for path, expected in plan['prior_record_identities'].items():
        if identity(path) != expected:
            raise ValueError('Original qualification evidence changed: ' + path)


def stop_child(child):
    """Terminate only our child group; tolerate a simultaneous normal exit."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def qualify(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan['root'])
    repair = plan['allowed_phase'] == 'native_derivative_repair_only'
    if plan['status'] != 'admitted' or plan['allowed_phase'] not in (
        'native_derivative_prerequisite_only', 'native_derivative_repair_only'
    ):
        raise ValueError('Require an admitted derivative-only plan')
    verify_source(plan)
    quiet = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'
    ], text=True).strip()
    if quiet or shutil.disk_usage(root).free - plan['additional_feasibility_peak_bytes'] < RESERVE:
        raise RuntimeError('GPU or peak-storage admission changed')
    storage = storage_usage(plan)
    if storage['charged_bytes'] > plan['additional_feasibility_peak_bytes']:
        raise RuntimeError('Feasibility storage ceiling already exceeded')
    prior_card_s = 0.
    if repair:
        verify_original_records(plan)
        clock_state = json.loads(Path(plan['original_clock']).read_text())
        previous = json.loads(Path(plan['original_process']).read_text())
        prior_card_s = previous['occupied_card_s']
    else:
        clock_state = dict(first_gpu_action_epoch=time.time(), elapsed_limit_s=14400,
                           occupied_card_limit_s=28800, scope='one-card native derivative child')
    if remaining_budget(clock_state, prior_card_s, time.time()) < plan['qualification_wall_limit_s']:
        raise RuntimeError('Original cumulative budget cannot admit the bounded child')
    with (root / 'CLOCK.json').open('x') as stream:
        json.dump(dict(**clock_state, inherited=repair, prior_occupied_card_s=prior_card_s), stream)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['devices']['B'], PYTHONDONTWRITEBYTECODE='1')
    source = Path(plan['source_root'])
    env['PYTHONPATH'] = os.pathsep.join(str(source / path) for path in ('.', 'training', 'inference'))
    module_name = 'repair_native' if repair else 'native'
    command = [plan['training_python'], '-u', '-m', 'experiments.frontier_preservation_pilot.' + module_name,
               '--parent-plan', plan['parent_plan'], '--output', str(root / 'derivative_support_001'),
               '--execution-commit', plan['execution_commit']]
    if repair:
        command.extend(['--original-probe', plan['original_probe']])
    started = time.perf_counter()
    reason = None
    last_storage_check = 0.
    with (root / 'native.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            write_json(root / 'LAUNCH.json', dict(pid=child.pid, command=command, gpu=plan['devices']['B']))
            while child.poll() is None:
                if time.perf_counter() - started >= plan['qualification_wall_limit_s'] - STOP_GRACE_S:
                    reason = 'bounded derivative feasibility timeout'
                if remaining_budget(clock_state, prior_card_s + time.perf_counter() - started, time.time()) <= STOP_GRACE_S:
                    reason = 'original cumulative time or card budget exhausted'
                if shutil.disk_usage(root).free < RESERVE:
                    reason = 'shared filesystem reserve crossed'
                if time.perf_counter() - last_storage_check >= 30:
                    storage = storage_usage(plan)
                    write_json(root / 'STORAGE_CURRENT.json', storage)
                    if storage['charged_bytes'] > plan['additional_feasibility_peak_bytes']:
                        reason = 'bounded feasibility storage ceiling exceeded'
                    last_storage_check = time.perf_counter()
                if reason:
                    break
                time.sleep(5)
        except Exception as error:
            reason = 'resource supervision failed: ' + repr(error)
        finally:
            stop_child(child)
    elapsed = time.perf_counter() - started
    final_storage_error = None
    try:
        storage = storage_usage(plan)
    except Exception as error:
        final_storage_error = repr(error)
        reason = reason or 'final storage measurement failed'
    result = dict(
        exit_code=child.returncode, stopped_reason=reason, elapsed_s=elapsed,
        occupied_card_s=elapsed, **storage,
        cumulative_occupied_card_s=prior_card_s + elapsed,
        cumulative_elapsed_s=time.time() - clock_state['first_gpu_action_epoch'],
        final_storage_error=final_storage_error,
        native_adam_steps=0, production_started=False,
    )
    write_json(root / 'QUALIFICATION_PROCESS.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    qualify(parser.parse_args().plan)
