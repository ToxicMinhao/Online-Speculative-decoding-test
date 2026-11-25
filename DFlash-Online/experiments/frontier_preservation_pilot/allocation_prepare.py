"""Seal the v2 allocation audit, preserving the historical PR60 failure ledger."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .native import select_parent
from .prepare import FULL_INPUTS, INPUT_IDENTITY, RESERVE

PARENT_EXPORT_IDENTITY = dict(
    bytes=2097259104,
    sha256='1556409004f7c73c0007c1f01764d4fd1dd189827994596939597be43b2d21b2',
)
AUDIT_ADDITIONAL_BYTES = 2_000_000_000  # no tensor/checkpoint persistence
TOTAL_STORAGE_BYTES = 40_000_000_000
NUMERICAL_WALL_LIMIT_S = 3600


def prior_evidence(previous_root):
    """The new elapsed phase does not reset the old GPU/storage consumption."""
    previous_root = Path(previous_root)
    old = json.loads((previous_root / 'PLAN.json').read_text())
    process = json.loads((previous_root / 'QUALIFICATION_PROCESS.json').read_text())
    model = json.loads((previous_root / 'derivative_support_001/MODEL.json').read_text())
    result = json.loads((previous_root / 'derivative_support_001/RESULT.json').read_text())
    if (result['execution_commit'] != '65ae6f95bd73aef8f60542a3d2ed3f5a3e51b48c'
            or result['native_adam_steps'] != 0
            or model['duality']['passed']
            or not model['ordinary_backward_bit_equal']
            or not model['native_primal_bit_equal']):
        raise ValueError('Require the unchanged final PR60 repair evidence')
    records = dict(old['prior_record_identities'])
    for name in ('PLAN.json', 'CLOCK.json', 'QUALIFICATION_PROCESS.json',
                 'derivative_support_001/MODEL.json', 'derivative_support_001/RESULT.json',
                 'derivative_support_001/WINDOW.json'):
        path = previous_root / name
        records[str(path)] = identity(path)
    for path, expected in records.items():
        if identity(path) != expected:
            raise ValueError('Historical PR60 evidence changed: ' + path)
    clock = json.loads(Path(old['original_clock']).read_text())
    return dict(
        previous_plan=old, previous_process=process,
        records=records, original_probe=old['original_probe'],
        prior_card_s=process['cumulative_occupied_card_s'],
        prior_start_epoch=clock['first_gpu_action_epoch'],
        prior_elapsed_s=process['cumulative_elapsed_s'],
        prior_roots=list(dict.fromkeys([
            *old['prior_charged_roots'], str(previous_root),
            str(Path(old['source_manifest']).parent),
        ])),
    )


def prepare(parent_plan, root, source_manifest, execution_commit, previous_root,
            local_allocated_bytes):
    from .allocation_contract import numerical_contract
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    parent, recovery, selected = select_parent(parent_plan)
    prior = prior_evidence(previous_root)
    export = Path(parent_plan).parents[1] / 'arms/PAIR/trainer/events/event_001/export/model.safetensors'
    if identity(export) != PARENT_EXPORT_IDENTITY or identity(FULL_INPUTS) != INPUT_IDENTITY:
        raise ValueError('Parent export or full ordered input manifest changed')
    source_manifest = Path(source_manifest)
    source = json.loads(source_manifest.read_text())
    if source['commit'] != execution_commit:
        raise ValueError('Source manifest commit differs')
    quiet = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader',
    ], text=True).strip()
    free = shutil.disk_usage(root).free
    contract_path = root / 'NUMERICAL_CONTRACT.json'
    write_json(contract_path, numerical_contract())
    plan = dict(
        schema='frontier_allocation_audit_v2', epoch=time.time(), root=str(root),
        source_root=str(Path(__file__).resolve().parents[2]),
        source_manifest=str(source_manifest), source_identity=identity(source_manifest),
        execution_commit=execution_commit, parent_plan=str(parent_plan),
        parent_binding=parent['binding'], parent_recovery=str(recovery), parent_features=selected,
        parent_export=str(export), parent_export_identity=PARENT_EXPORT_IDENTITY,
        full_input_manifest=str(FULL_INPUTS), full_input_identity=INPUT_IDENTITY,
        original_probe=prior['original_probe'], prior_record_identities=prior['records'],
        previous_root=str(Path(previous_root)),
        prior_charged_roots=prior['prior_roots'],
        prior_occupied_card_s=prior['prior_card_s'],
        prior_gpu_start_epoch=prior['prior_start_epoch'], prior_elapsed_s=prior['prior_elapsed_s'],
        compiler_baseline=prior['previous_plan']['compiler_baseline'],
        devices=parent['devices'], training_python=parent['training_python'],
        numerical_contract=str(contract_path), numerical_contract_identity=identity(contract_path),
        new_phase_clock=True, allowed_phase='allocation_numerical_audit',
        old_hidden_gate_status='failed_retained_diagnostic',
        maximum_serial_reference_backwards_this_attempt=60,
        maximum_serial_reference_backwards_total=120,
        maximum_native_qualification_adam_steps=8,
        native_adam_steps_this_phase=0, production_admitted=False,
        maximum_implementation_correction_retries=1,
        qualification_wall_limit_s=NUMERICAL_WALL_LIMIT_S,
        reserve_bytes=RESERVE, additional_phase_peak_bytes=AUDIT_ADDITIONAL_BYTES,
        local_allocated_bytes=local_allocated_bytes,
        limits=dict(elapsed_s=14400, occupied_card_s=28800,
                    additional_allocated_bytes=TOTAL_STORAGE_BYTES),
        gpu_processes=quiet, free_bytes=free,
        status='admitted' if not quiet and free-AUDIT_ADDITIONAL_BYTES>=RESERVE else 'blocked',
    )
    write_json(root / 'PLAN.json', plan)
    if plan['status'] != 'admitted':
        raise RuntimeError('Allocation audit resource admission failed')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-plan', 'root', 'source-manifest', 'execution-commit', 'previous-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--local-allocated-bytes', type=int, required=True)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(json.dumps({key: result[key] for key in ('status', 'free_bytes', 'production_admitted')}))
