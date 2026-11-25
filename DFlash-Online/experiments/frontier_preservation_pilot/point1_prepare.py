"""Authenticate the retained Point1 window and seal its small diagnostic budget.

This module performs CPU/file checks only. It never samples replacement inputs,
loads a model, creates a recovery, or invokes the completed event coordinator.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json

SCHEMA = 'point1_update_fidelity_v1'
PRODUCTION_PLAN_IDENTITY = dict(bytes=29897, sha256='dd9bfed29119c6178c767781c618c1a49488562a2bb8bd6cc0196e21879c9945')
WINDOW_IDENTITY = dict(bytes=25388, sha256='a4245d34875ecc349e04cf34de24ef1d626d9cbd03ee30176f637d189d431d3d')
RECOVERY_IDENTITY = dict(bytes=6296473051, sha256='728770a7abf0ba3ef96805ef760997976ad7bedbbee1d8debe165e785a4d98a2')
EXPORT_IDENTITY = dict(bytes=2097259104, sha256='1556409004f7c73c0007c1f01764d4fd1dd189827994596939597be43b2d21b2')
ARCHIVE_MANIFEST_IDENTITY = dict(bytes=211418, sha256='dd65e3f371b4c5b6a4f9385447fb3f02ceca68642d79a9940076f0ede3576cc5')
PARENT_BINDING = 'b120c76b0036b4786eb6b9d11b1594ec1dc6e57e3d20c9ed931924262b1c072a'
HISTORICAL_CARD_S = 5175.047027264
CASES = ('BETA', 'C', 'F', 'C_REPEAT')
LIMITS = dict(elapsed_s=3600, occupied_card_s=3600, allocated_bytes=2_000_000_000,
              attempted_adam_calls=6, planned_adam_calls=4, correction_retries=1,
              reserve_bytes=400_000_000_000)


def read_bound_json(path, expected):
    if identity(path) != expected:
        raise ValueError('Retained artifact identity changed: ' + str(path))
    return json.loads(Path(path).read_text())


def host_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            values[key] = int(value.split()[0]) * 1024
    return dict(total_bytes=values['MemTotal'], available_bytes=values['MemAvailable'])


def gpu_inventory():
    rows = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    cards = []
    for line in rows.splitlines():
        ordinal, uuid, total, used = (part.strip() for part in line.split(','))
        cards.append(dict(ordinal=int(ordinal), uuid=uuid,
                          total_bytes=int(total)*2**20, used_bytes=int(used)*2**20))
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True).strip()
    return dict(cards=cards, processes=processes)


def resource_admission(*, free_bytes, charged_bytes, remaining_write_bytes,
                       host_available_bytes, gpu_free_bytes, gpu_busy):
    """Keep disk and RAM reserves separate; neither is GPU memory."""
    result = dict(free_bytes=free_bytes, charged_bytes=charged_bytes,
                  remaining_write_bytes=remaining_write_bytes,
                  host_available_bytes=host_available_bytes, gpu_free_bytes=gpu_free_bytes,
                  host_workspace_bytes=55_000_000_000, host_reserve_bytes=32_000_000_000,
                  gpu_required_free_bytes=60_000_000_000)
    if min(charged_bytes, remaining_write_bytes) < 0:
        raise ValueError('Storage accounting cannot be negative')
    checks = dict(
        disk_reserve=free_bytes-remaining_write_bytes >= LIMITS['reserve_bytes'],
        all_locations=charged_bytes+remaining_write_bytes <= LIMITS['allocated_bytes'],
        host_memory=host_available_bytes >= result['host_workspace_bytes']+result['host_reserve_bytes'],
        gpu_memory=gpu_free_bytes >= result['gpu_required_free_bytes'],
        gpu_idle=not gpu_busy,
    )
    result.update(checks=checks, admitted=all(checks.values()))
    return result


def authenticate_witness(production_root, archive_manifest, selected_features, window):
    """Authenticate the original new-buffer set, never substitute another set."""
    archive = read_bound_json(archive_manifest, ARCHIVE_MANIFEST_IDENTITY)
    if archive.get('execution_commit') != '225802657d630c0985abd68d6e7453f3e47c5b74':
        raise ValueError('Wrong retained production archive')
    entries = {row['path']: dict(bytes=row['bytes'], sha256=row['sha256']) for row in archive['files']}
    paths = {key: Path(production_root) / key for key in (
        'witness/BINDINGS.json', 'witness/parent.json', 'shared/FEATURE_IDENTITIES.json')}
    binding = read_bound_json(paths['witness/BINDINGS.json'], entries['witness/BINDINGS.json'])
    origin = read_bound_json(paths['witness/parent.json'], entries['witness/parent.json'])
    features = read_bound_json(paths['shared/FEATURE_IDENTITIES.json'], entries['shared/FEATURE_IDENTITIES.json'])
    if binding.get('status') != 'passed' or origin.get('status') != 'passed':
        raise ValueError('Witness was not valid at the retained parent')
    if (origin.get('endpoint') != 'parent' or origin.get('counters') != [128, 256, 1]
            or origin.get('binding_identity') != entries['witness/BINDINGS.json']):
        raise ValueError('Witness origin is not the identically bound PAIR1 parent')
    new = binding['sets']['new_buffer']
    if len(new) != 8 or len(origin['sets']['new_buffer']) != 8:
        raise ValueError('The original eight new-buffer witnesses are incomplete')
    feature_map = {row['path']: row for row in features}
    update_ids = {row['record_id'] for row in window['responses']}
    update_paths = {str(Path(row['path']).resolve()) for row in selected_features}
    for item, first in zip(new, origin['sets']['new_buffer']):
        source = feature_map[item['feature_path']]
        if (item['feature_identity'] != source['identity']
                or item['record_id'] != source['record_id']
                or first['record_id'] != item['record_id']
                or first['anchors'] != item['anchors']):
            raise ValueError('Witness source, origin or anchor binding changed')
        if item['record_id'] in update_ids or str(Path(item['feature_path']).resolve()) in update_paths:
            raise ValueError('New-buffer witness overlaps the qualification window')
        if identity(item['feature_path']) != item['feature_identity']:
            raise ValueError('Witness feature bytes changed')
    return dict(status='authenticated', set_name='new_buffer', responses=8,
                binding_path=str(paths['witness/BINDINGS.json']),
                binding_identity=entries['witness/BINDINGS.json'],
                parent_path=str(paths['witness/parent.json']), parent_identity=entries['witness/parent.json'],
                feature_manifest_path=str(paths['shared/FEATURE_IDENTITIES.json']),
                feature_manifest_identity=entries['shared/FEATURE_IDENTITIES.json'],
                archive_manifest_identity=ARCHIVE_MANIFEST_IDENTITY, items=new,
                qualification_record_ids_disjoint=True, qualification_feature_paths_disjoint=True)


def prepare(production_plan, root, source_manifest, execution_commit,
            local_allocated_bytes, archive_manifest, remaining_write_budget_bytes=1_000_000_000):
    from experiments.kplus_full_gsm8k.storage import compiler_snapshot
    from experiments.bounded_prefix_allocation.runtime import allocated_bytes
    from .native import select_parent
    from .qualify import verify_source

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    production = read_bound_json(production_plan, PRODUCTION_PLAN_IDENTITY)
    source_root = Path(__file__).resolve().parents[2]
    report = source_root / 'reports/dflash_frontier_preservation_20261125/allocation_qualification_001/IDENTITIES.json'
    pins = json.loads(report.read_text())
    for label in ('numerical', 'integration'):
        result = read_bound_json(production[label+'_result'], production[label+'_result_identity'])
        if result.get('passed') is not True or result['execution_commit'] != pins[label+'_execution_commit']:
            raise ValueError('Prior qualification receipt changed')
    integration_root = Path(production['integration_result']).parent
    imports = read_bound_json(integration_root / 'SOURCE_INPUTS.json', pins['integration_raw_receipt_identities']['native/SOURCE_INPUTS.json'])
    old_plan = read_bound_json(integration_root.parent / 'PLAN.json', pins['integration_raw_receipt_identities']['PLAN.json'])
    parent, recovery, selected = select_parent(production['parent_plan'])
    if parent['binding'] != PARENT_BINDING or selected != imports['selected_features']:
        raise ValueError('Parent contract or exact qualification features changed')
    if imports['parent_plan_identity'] != identity(production['parent_plan']):
        raise ValueError('Parent plan changed since qualification')
    export = Path(production['parent_export'])
    if identity(recovery) != RECOVERY_IDENTITY or identity(export) != EXPORT_IDENTITY:
        raise ValueError('Protected parent state/export changed')
    window_path = Path(old_plan['original_probe']) / 'WINDOW.json'
    window = read_bound_json(window_path, WINDOW_IDENTITY)
    if [len(row['anchors'][0]) for row in window['responses']] != [512, 269]:
        raise ValueError('Original 512/269 qualification anchors changed')
    witness = dict(status='unavailable', scope='qualification_window_only')
    try:
        witness = authenticate_witness(production['root'], archive_manifest, selected, window)
    except (OSError, ValueError, KeyError) as error:
        witness['reason'] = repr(error)
    source_manifest = Path(source_manifest)
    plan = dict(
        schema=SCHEMA, root=str(root), execution_commit=execution_commit,
        source_root=str(source_root), source_manifest=str(source_manifest), source_identity=identity(source_manifest),
        production_plan=str(production_plan), production_plan_identity=PRODUCTION_PLAN_IDENTITY,
        parent_plan=production['parent_plan'], parent_plan_identity=identity(production['parent_plan']),
        parent_binding=PARENT_BINDING, parent_recovery=str(recovery), parent_recovery_identity=RECOVERY_IDENTITY,
        parent_export=str(export), parent_export_identity=EXPORT_IDENTITY,
        parent_features=selected, original_probe=old_plan['original_probe'],
        original_window=str(window_path), original_window_identity=WINDOW_IDENTITY,
        numerical_result=production['numerical_result'], numerical_result_identity=production['numerical_result_identity'],
        integration_result=production['integration_result'], integration_result_identity=production['integration_result_identity'],
        witness=witness, cases=list(CASES), primary_contrast='F_minus_C', secondary_contrast='F_minus_BETA',
        native_steps_independently_restored=True, parent_counters=[128,256,1], case_counters=[129,258,1],
        live_precision='native BF16', diagnostic_product_reduction='FP64', shadow_precision='FP32',
        objective_changes=False, full_events=0, generation_calls=0, serial_token_oracle_calls=0,
        coefficients='frozen once from current native roles; C from actual native path',
        clip_jvp_test_atol=2e-12, clip_jvp_test_rtol=2e-12,
        native_beta_g0_equality='exact', native_C_repeatability='exact complete state and gradients',
        vector_zero_policy='report exact zero; no denominator floor; undefined cosine/ratio is null',
        prior_occupied_card_s=HISTORICAL_CARD_S, limits=dict(LIMITS),
        devices=production['devices'], training_python=production['training_python'],
        compiler_baseline=compiler_snapshot(), local_allocated_bytes=int(local_allocated_bytes),
        charged_remote_roots=[str(root),str(source_manifest.parent)],
        remaining_write_budget_bytes=int(remaining_write_budget_bytes),
        created_epoch=time.time(), status='preparing', old_hidden_gate_status='failed_retained_diagnostic',
    )
    verify_source(plan)
    memory, gpu = host_memory(), gpu_inventory()
    card = next(row for row in gpu['cards'] if row['uuid'] == plan['devices']['B'])
    charged = allocated_bytes(plan['charged_remote_roots']) + plan['local_allocated_bytes']
    admission = resource_admission(free_bytes=shutil.disk_usage(root).free, charged_bytes=charged,
        remaining_write_bytes=plan['remaining_write_budget_bytes'], host_available_bytes=memory['available_bytes'],
        gpu_free_bytes=card['total_bytes']-card['used_bytes'],
        gpu_busy=any(plan['devices']['B'] in row for row in gpu['processes'].splitlines()))
    plan.update(status='admitted' if admission['admitted'] else 'resource_blocked',
                resource_admission=admission, host_memory=memory, gpu_inventory=gpu)
    write_json(root/'CONTRACT.json', plan)
    if not admission['admitted']:
        raise RuntimeError('Point1 bounded resource admission failed')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('production-plan','root','source-manifest','execution-commit','archive-manifest'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--local-allocated-bytes', required=True, type=int)
    parser.add_argument('--remaining-write-budget-bytes', default=1_000_000_000, type=int)
    result = prepare(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'], witness=result['witness']['status'], admission=result['resource_admission'])))
