"""Supervise one admitted allocation audit; never restart a failed child."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .qualify import STOP_GRACE_S, stop_child, storage_usage, verify_source


def remaining_budget(clock, historical_card_s, new_card_s, now):
    return min(clock['elapsed_limit_s'] - (now-clock['first_gpu_action_epoch']),
               clock['occupied_card_limit_s'] - historical_card_s-new_card_s)


def validate_plan(plan):
    from .allocation_contract import numerical_contract
    if (plan['schema'] != 'frontier_allocation_audit_v2'
            or plan['allowed_phase'] != 'allocation_numerical_audit'
            or plan['production_admitted'] or plan['native_adam_steps_this_phase'] != 0
            or plan['old_hidden_gate_status'] != 'failed_retained_diagnostic'):
        raise ValueError('Require the zero-Adam allocation audit contract')
    path = Path(plan['numerical_contract'])
    if (identity(path) != plan['numerical_contract_identity']
            or json.loads(path.read_text()) != numerical_contract()):
        raise ValueError('Prospective numerical contract changed')
    required = {str(Path(plan['original_probe']) / name)
                for name in ('RESULT.json', 'WINDOW.json')}
    required.update(str(Path(plan['previous_root']) / name) for name in (
        'PLAN.json', 'CLOCK.json', 'QUALIFICATION_PROCESS.json',
        'derivative_support_001/MODEL.json', 'derivative_support_001/RESULT.json',
        'derivative_support_001/WINDOW.json'))
    if not required <= plan['prior_record_identities'].keys():
        raise ValueError('Missing historical failure and cost bindings')
    for path, expected in plan['prior_record_identities'].items():
        if identity(path) != expected:
            raise ValueError('Retained prior evidence changed: ' + path)


def initialize_clock(plan, now):
    """A new user-authorized elapsed phase retains the old interval and card cost."""
    path = Path(plan['root']).parent / 'ALLOCATION_PHASE_CLOCK.json'
    if path.exists():
        raise FileExistsError('An allocation phase already exists; do not duplicate it')
    prior_end = plan['prior_gpu_start_epoch'] + plan['prior_elapsed_s']
    clock = dict(
        schema='frontier_allocation_phase_clock_v2', first_gpu_action_epoch=now,
        elapsed_limit_s=plan['limits']['elapsed_s'],
        occupied_card_limit_s=plan['limits']['occupied_card_s'],
        prior_occupied_card_s=plan['prior_occupied_card_s'],
        historical_first_gpu_action_epoch=plan['prior_gpu_start_epoch'],
        historical_elapsed_interval_s=plan['prior_elapsed_s'],
        prior_completion_to_new_phase_s=now-prior_end,
        gap_scope='user review and CPU implementation/staging; no occupied GPU time claimed',
    )
    with path.open('x') as stream:
        json.dump(clock, stream, indent=2)
    write_json(Path(plan['root']) / 'CLOCK.json', clock)
    return clock


def total_storage(plan):
    # Include the shared phase clock/oracle ledger alongside all child attempts.
    # allocated_bytes deduplicates nested roots, so prior run paths are not added twice.
    usage = storage_usage(dict(plan, root=str(Path(plan['root']).parent)))
    usage['local_allocated_bytes'] = plan['local_allocated_bytes']
    usage['all_locations_charged_bytes'] = usage['charged_bytes'] + plan['local_allocated_bytes']
    return usage


def audit_command(plan):
    root = Path(plan['root'])
    return [
        plan['training_python'], '-u', '-m',
        'experiments.frontier_preservation_pilot.allocation_qualify',
        '--parent-plan', plan['parent_plan'], '--original-probe', plan['original_probe'],
        '--output', str(root / 'native'), '--execution-commit', plan['execution_commit'],
        '--reference-ledger', str(root.parent / 'ALLOCATION_REFERENCE_LEDGER.json'),
        '--attempt', '1', '--contract', plan['numerical_contract'],
    ]


def run(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan['root'])
    if plan['status'] != 'admitted':
        raise ValueError('Allocation numerical audit was not admitted')
    validate_plan(plan)
    verify_source(plan)
    quiet = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader',
    ], text=True).strip()
    usage = total_storage(plan)
    if quiet or usage['free_bytes']-plan['additional_phase_peak_bytes']<plan['reserve_bytes']:
        raise RuntimeError('Idle-card or peak-storage admission changed')
    if usage['all_locations_charged_bytes']+plan['additional_phase_peak_bytes']>plan['limits']['additional_allocated_bytes']:
        raise RuntimeError('Study-wide storage cannot admit the audit peak')
    write_json(root / 'STORAGE_ADMISSION.json', usage)
    if plan['limits']['occupied_card_s']-plan['prior_occupied_card_s']<=STOP_GRACE_S:
        raise RuntimeError('Historical work already exhausts the occupied-card allowance')
    clock = initialize_clock(plan, time.time())
    source = Path(plan['source_root'])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['devices']['B'],
                       PYTHONDONTWRITEBYTECODE='1')
    environment['PYTHONPATH'] = os.pathsep.join(str(source / p) for p in ('.', 'training', 'inference'))
    command = audit_command(plan)
    started = time.perf_counter()
    stopped = None
    last_storage_check = 0.
    initial_charge = usage['all_locations_charged_bytes']
    with (root / 'native.log').open('x') as log:
        child = subprocess.Popen(command, cwd=source, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            write_json(root / 'LAUNCH.json', dict(pid=child.pid, command=command,
                       physical_gpu_uuid=plan['devices']['B'], first_gpu_action_epoch=clock['first_gpu_action_epoch']))
            while child.poll() is None:
                elapsed = time.perf_counter()-started
                if elapsed >= plan['qualification_wall_limit_s']-STOP_GRACE_S:
                    stopped = 'allocation numerical investigation timebox'
                if remaining_budget(clock, plan['prior_occupied_card_s'], elapsed, time.time())<=STOP_GRACE_S:
                    stopped = 'new elapsed or historical-plus-new card limit'
                if shutil.disk_usage(root).free<plan['reserve_bytes']:
                    stopped = 'shared filesystem reserve crossed'
                if time.perf_counter()-last_storage_check>=30:
                    usage = total_storage(plan)
                    write_json(root / 'STORAGE_CURRENT.json', usage)
                    if usage['all_locations_charged_bytes']>min(
                        plan['limits']['additional_allocated_bytes'],
                        initial_charge+plan['additional_phase_peak_bytes'],
                    ):
                        stopped = 'admitted storage envelope exceeded'
                    last_storage_check=time.perf_counter()
                if stopped:
                    break
                time.sleep(5)
        except Exception as error:
            stopped='resource supervision failed: '+repr(error)
        finally:
            stop_child(child)
    elapsed = time.perf_counter()-started
    final_error=None
    try:
        usage=total_storage(plan)
    except Exception as error:
        final_error=repr(error)
    result=dict(exit_code=child.returncode, stopped_reason=stopped,
                elapsed_s=elapsed, occupied_card_s=elapsed,
                historical_occupied_card_s=plan['prior_occupied_card_s'],
                cumulative_occupied_card_s=plan['prior_occupied_card_s']+elapsed,
                new_phase_elapsed_s=time.time()-clock['first_gpu_action_epoch'],
                prior_elapsed_interval_s=plan['prior_elapsed_s'],
                native_adam_steps=0, production_started=False,
                final_storage_error=final_error, **usage)
    write_json(root / 'QUALIFICATION_PROCESS.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
