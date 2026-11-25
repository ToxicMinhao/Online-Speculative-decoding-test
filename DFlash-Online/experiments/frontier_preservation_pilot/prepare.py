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


def prepare(parent_plan, root, source_manifest, execution_commit):
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
    args = parser.parse_args()
    plan = prepare(args.parent_plan, args.root, args.source_manifest, args.execution_commit)
    print(json.dumps({key: plan[key] for key in ('status', 'free_bytes', 'production_admitted')}))
