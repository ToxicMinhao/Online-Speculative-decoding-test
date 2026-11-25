"""Admit only the bounded native derivative prerequisite, without model work."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from .native import select_parent

RESERVE = 400_000_000_000
FEASIBILITY_BYTES = 6_000_000_000
FULL_INPUTS = Path('/data/sources/minhao/runs/dflash_opencode_native_comparison_20261120/sealed/INPUTS.json')
INPUT_IDENTITY = dict(bytes=32309622, sha256='df7194e667126f34f8e2ab2f6e22e2e5480cd7921728ceb4f946bd3a0cce6581')


def prepare(parent_plan, root, source_manifest, execution_commit, original_run=None,
            failed_repair=None, completed_localization=None, failed_transpose=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    parent, recovery, selected = select_parent(parent_plan)
    if identity(FULL_INPUTS) != INPUT_IDENTITY:
        raise ValueError('Full original PR56 inputs changed')
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'
    ], text=True).strip()
    free = shutil.disk_usage(root).free
    plan = dict(
        schema='frontier_preservation_derivative_feasibility_v1',
        epoch=time.time(), root=str(root), execution_commit=execution_commit,
        source_manifest=str(source_manifest), source_identity=identity(source_manifest),
        parent_plan=str(parent_plan), parent_binding=parent['binding'],
        parent_recovery=str(recovery), parent_features=selected,
        full_input_manifest=str(FULL_INPUTS), full_input_identity=INPUT_IDENTITY,
        source_root=str(Path(__file__).resolve().parents[2]),
        devices=parent['devices'], training_python=parent['training_python'],
        free_bytes=free, reserve_bytes=RESERVE,
        additional_feasibility_peak_bytes=FEASIBILITY_BYTES,
        qualification_wall_limit_s=900,
        limits=dict(elapsed_s=14400, occupied_card_s=28800, additional_allocated_bytes=40_000_000_000),
        compiler_baseline=compiler_snapshot(),
        allowed_phase='native_derivative_prerequisite_only', production_admitted=False,
        attempted_work_ceiling=dict(response_presentations=0, native_adam_steps=0,
                                    complete_response_generations=0, selected_training_responses=2),
        gpu_processes=processes,
        status='admitted' if not processes and free - FEASIBILITY_BYTES >= RESERVE else 'blocked',
    )
    if original_run is not None:
        original_run = Path(original_run)
        old_plan = json.loads((original_run / 'PLAN.json').read_text())
        previous = json.loads((original_run / 'QUALIFICATION_PROCESS.json').read_text())
        probe = original_run / 'derivative_support_001'
        result = json.loads((probe / 'RESULT.json').read_text())
        if (old_plan['parent_plan'] != str(parent_plan)
                or result['status'] != 'blocked_native_derivative'
                or previous['native_adam_steps'] != 0):
            raise ValueError('Repair must reuse the original blocked zero-update prerequisite')
        records = [original_run / name for name in
                   ('PLAN.json', 'CLOCK.json', 'QUALIFICATION_PROCESS.json')]
        records.extend(probe / name for name in ('RESULT.json', 'WINDOW.json'))
        plan.update(
            allowed_phase='native_derivative_repair_only',
            original_probe=str(probe), original_clock=str(original_run / 'CLOCK.json'),
            original_process=str(original_run / 'QUALIFICATION_PROCESS.json'),
            prior_record_identities={str(path): identity(path) for path in records},
            prior_charged_roots=[str(original_run), str(Path(old_plan['source_manifest']).parent)],
            compiler_baseline=old_plan['compiler_baseline'],
            prior_occupied_card_s=previous['occupied_card_s'],
            prospective_duality_criterion=dict(atol=1e-5, rtol=0.016),
            prospective_primal_and_ordinary_backward='exact equality',
        )
    if failed_repair is not None:
        if original_run is None:
            raise ValueError('Localization requires the original qualification clock')
        failed_repair = Path(failed_repair)
        failed_probe = failed_repair / 'derivative_support_001'
        failed_model = json.loads((failed_probe / 'MODEL.json').read_text())
        failed_result = json.loads((failed_probe / 'RESULT.json').read_text())
        failed_plan = json.loads((failed_repair / 'PLAN.json').read_text())
        failed_process = json.loads((failed_repair / 'QUALIFICATION_PROCESS.json').read_text())
        if (failed_result['status'] != 'failed' or failed_result['native_adam_steps'] != 0
                or not failed_model['native_primal_bit_equal']
                or not failed_model['ordinary_backward_bit_equal']
                or failed_model['duality']['passed']):
            raise ValueError('Localization requires the retained finite derivative discrepancy')
        records = [failed_repair / name for name in ('PLAN.json', 'QUALIFICATION_PROCESS.json')]
        records.extend(failed_probe / name for name in ('RESULT.json', 'MODEL.json', 'OPERATOR.json'))
        plan['prior_record_identities'].update({str(path): identity(path) for path in records})
        plan['prior_charged_roots'].extend(
            [str(failed_repair), str(Path(failed_plan['source_manifest']).parent)])
        plan.update(
            allowed_phase='native_derivative_localization_only',
            failed_repair=str(failed_probe),
            original_process=str(failed_repair / 'QUALIFICATION_PROCESS.json'),
            prior_occupied_card_s=failed_process['cumulative_occupied_card_s'],
            qualification_acceptance_changed=False,
        )
    if completed_localization is not None:
        if failed_repair is None:
            raise ValueError('The reverse route requires the localized original discrepancy')
        localized = Path(completed_localization)
        local_plan = json.loads((localized / 'PLAN.json').read_text())
        local_process = json.loads((localized / 'QUALIFICATION_PROCESS.json').read_text())
        local_result = json.loads((localized / 'derivative_support_001/RESULT.json').read_text())
        if (local_result['status'] != 'native_derivative_localization_completed'
                or local_result['native_adam_steps'] != 0
                or not local_result['localization']['global_reproduction_exact']):
            raise ValueError('Require completed, unchanged-input zero-update localization')
        records = [localized / name for name in ('PLAN.json', 'QUALIFICATION_PROCESS.json',
                   'derivative_support_001/RESULT.json', 'derivative_support_001/LOCALIZATION.json')]
        plan['prior_record_identities'].update({str(path): identity(path) for path in records})
        plan['prior_charged_roots'].extend(
            [str(localized), str(Path(local_plan['source_manifest']).parent)])
        plan.update(
            allowed_phase='native_derivative_transpose_only',
            completed_localization=str(localized),
            original_process=str(localized / 'QUALIFICATION_PROCESS.json'),
            prior_occupied_card_s=local_process['cumulative_occupied_card_s'],
        )
    if failed_transpose is not None:
        if completed_localization is None:
            raise ValueError('The single-use graph repair retains the completed localization')
        failed = Path(failed_transpose)
        failed_result = json.loads((failed / 'derivative_support_001/RESULT.json').read_text())
        previous = json.loads((failed / 'QUALIFICATION_PROCESS.json').read_text())
        previous_plan = json.loads((failed / 'PLAN.json').read_text())
        if (failed_result['status'] != 'failed' or failed_result['native_adam_steps'] != 0
                or 'donated buffers' not in failed_result.get('error', '')):
            raise ValueError('Require the preserved internal graph-retention failure')
        records = [failed / name for name in ('PLAN.json', 'QUALIFICATION_PROCESS.json',
                   'derivative_support_001/RESULT.json', 'derivative_support_001/MODEL.json')]
        plan['prior_record_identities'].update({str(path): identity(path) for path in records})
        plan['prior_charged_roots'].extend(
            [str(failed), str(Path(previous_plan['source_manifest']).parent)])
        plan.update(
            failed_transpose=str(failed),
            original_process=str(failed / 'QUALIFICATION_PROCESS.json'),
            prior_occupied_card_s=previous['cumulative_occupied_card_s'],
            implementation_fix='Consume each inner native backward graph exactly once',
        )
    write_json(root / 'PLAN.json', plan)
    if plan['status'] != 'admitted':
        raise RuntimeError('Derivative-only resource admission failed; production is not admitted')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--execution-commit', required=True)
    parser.add_argument('--original-run')
    parser.add_argument('--failed-repair')
    parser.add_argument('--completed-localization')
    parser.add_argument('--failed-transpose')
    args = parser.parse_args()
    plan = prepare(args.parent_plan, args.root, args.source_manifest, args.execution_commit,
                   args.original_run, args.failed_repair, args.completed_localization,
                   args.failed_transpose)
    print(json.dumps({key: plan[key] for key in ('status', 'free_bytes', 'production_admitted')}))
