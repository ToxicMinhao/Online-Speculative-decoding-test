"""Admit one disposable integration attempt under the existing allocation clock."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .allocation_launch import remaining_budget, total_storage
from .qualify import STOP_GRACE_S, stop_child, verify_source


def run(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan['root'])
    if plan['schema'] != 'frontier_native_integration_admission_v2' or plan['production_admitted']:
        raise ValueError('Require separate disposable integration admission')
    clock = json.loads(Path(plan['phase_clock']).read_text())
    audit = Path(plan['numerical_result'])
    numerical = json.loads(audit.read_text())
    if (identity(audit) != plan['numerical_result_identity']
            or numerical.get('status') != 'numerically_qualified'
            or numerical.get('passed') is not True
            or numerical.get('restoration', {}).get('status') != 'passed'):
        raise ValueError('Numerical admission is missing or changed')
    verify_source(plan)
    quiet = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader'], text=True).strip()
    usage = total_storage(plan)
    require_time = plan['qualification_wall_limit_s']
    if quiet or remaining_budget(clock, plan['prior_occupied_card_s'], 0, time.time()) < require_time:
        raise RuntimeError('Integration card or time admission failed')
    if (usage['free_bytes']-plan['additional_phase_peak_bytes'] < plan['reserve_bytes']
            or usage['all_locations_charged_bytes']+plan['additional_phase_peak_bytes'] >
            plan['limits']['additional_allocated_bytes']):
        raise RuntimeError('Integration storage peak cannot be admitted')
    write_json(root / 'STORAGE_ADMISSION.json', usage)
    command = [plan['training_python'], '-u', '-m',
        'experiments.frontier_preservation_pilot.allocation_integration',
        '--parent-plan', plan['parent_plan'], '--original-probe', plan['original_probe'],
        '--numerical-audit', str(audit.parent), '--output', str(root / 'native'),
        '--execution-commit', plan['execution_commit'],
        '--step-ledger', str(root.parent / 'ALLOCATION_STEP_LEDGER.json')]
    source = Path(plan['source_root'])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['devices']['B'],
        PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(
            str(source / part) for part in ('.', 'training', 'inference')))
    started = time.perf_counter()
    stopped = None
    initial_charge = usage['all_locations_charged_bytes']
    with (root / 'native.log').open('x') as log:
        child = subprocess.Popen(command, cwd=source, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        write_json(root / 'LAUNCH.json', dict(pid=child.pid, command=command,
            physical_gpu_uuid=plan['devices']['B'], existing_phase_clock=clock))
        last_storage = 0.
        try:
            while child.poll() is None:
                elapsed = time.perf_counter()-started
                if elapsed >= require_time-STOP_GRACE_S:
                    stopped = 'integration attempt timebox'
                if remaining_budget(clock, plan['prior_occupied_card_s'], elapsed, time.time()) <= STOP_GRACE_S:
                    stopped = 'shared phase time/card limit'
                if shutil.disk_usage(root).free < plan['reserve_bytes']:
                    stopped = 'shared filesystem reserve crossed'
                if time.perf_counter()-last_storage >= 30:
                    usage = total_storage(plan)
                    write_json(root / 'STORAGE_CURRENT.json', usage)
                    if usage['all_locations_charged_bytes'] > min(
                        plan['limits']['additional_allocated_bytes'],
                        initial_charge+plan['additional_phase_peak_bytes']):
                        stopped = 'integration storage envelope exceeded'
                    last_storage = time.perf_counter()
                if stopped:
                    break
                time.sleep(5)
        except Exception as error:
            stopped = 'resource supervision failed: '+repr(error)
        finally:
            stop_child(child)
    elapsed = time.perf_counter()-started
    final_storage_error = None
    try:
        usage = total_storage(plan)
    except Exception as error:
        final_storage_error = repr(error)
    result = dict(exit_code=child.returncode, stopped_reason=stopped,
        elapsed_s=elapsed, occupied_card_s=elapsed,
        prior_occupied_card_s=plan['prior_occupied_card_s'],
        cumulative_occupied_card_s=plan['prior_occupied_card_s']+elapsed,
        new_phase_elapsed_s=time.time()-clock['first_gpu_action_epoch'],
        maximum_native_adam_attempts=8, production_started=False,
        final_storage_error=final_storage_error, **usage)
    write_json(root / 'QUALIFICATION_PROCESS.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
