"""Artifact-derived P8 peak-write budgets and read-only destination admission.

This module creates no directories, removes no artifacts, and starts no model.
The older 256 MiB/pool allowance excluded repeated full evidence configurations.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from experiments.dflash_baseline_learning_curve.experiment import digest, require


SCHEMA = 'dflash_p8_storage_plan_v1'
PARAMETERS, PARAMETER_BYTES = 1048626432, 2
CHECKPOINT_METADATA_BYTES = 64 * 2**20
MIN_RESERVE_BYTES = 8 * 2**30
LEGACY_POOL_EVIDENCE_BYTES = 256 * 2**20
NATIVE_POOL_DYNAMIC_ALLOWANCE_BYTES = 512 * 2**20
CONTROLLER_FIXED_ALLOWANCE_BYTES = 512 * 2**20
CONTROLLER_SERIALIZED_POOL_MULTIPLIER = 12


def serialized_bytes(value):
    """Exact UTF-8 bytes for the native pretty-JSON writer, including newline."""
    encoder = json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False)
    return 1 + sum(len(chunk.encode('utf-8')) for chunk in encoder.iterencode(value))


def _seal(plan):
    return digest({key: value for key, value in plan.items() if key != 'plan_sha256'})


def build_storage_plan(pools):
    """Bind conservative peak writes to the three actual sealed pool payloads.

Native: three full configuration copies per pool plus 512 MiB for dynamic
training/loss/native evidence, failure receipts, logs, and JSON replacement.
Controller: 512 MiB plus twelve serialized configurations covers results,
events, raw paired/training reports, and atomic progress-file overlap. Reserve
one additional controller-evidence copy for retrieval staging. Three final
weights and one full checkpoint export/retrieval overlap are separate.
"""
    require(isinstance(pools, list) and len(pools) == 3 and
            len({p['pool_id'] for p in pools}) == 3, 'Exactly three sealed P8 pools required')
    sizes, bindings, pool_evidence = {}, {}, {}
    for pool in pools:
        require(pool.get('status') == 'prepared' and pool['pool_id'] == 'p8_' + pool['domain'] and
                isinstance(pool.get('pool_payload_sha256'), str) and len(pool['pool_payload_sha256']) == 64,
                'Storage sizing requires a prepared sealed pool')
        require(pool['pool_payload_sha256'] == digest({key: value for key, value in pool.items()
            if key != 'pool_payload_sha256'}), 'Pool payload changed before storage sizing')
        # Nesting adds indentation to every line, so plain pool JSON length is
        # insufficient. Other configuration fields fit in the dynamic allowance.
        count = serialized_bytes({'original_pool': pool})
        sizes[pool['pool_id']] = count
        bindings[pool['pool_id']] = pool['pool_payload_sha256']
        pool_evidence[pool['pool_id']] = max(LEGACY_POOL_EVIDENCE_BYTES,
            3 * count + NATIVE_POOL_DYNAMIC_ALLOWANCE_BYTES)
    checkpoint = PARAMETERS * PARAMETER_BYTES + CHECKPOINT_METADATA_BYTES
    native_evidence = sum(pool_evidence.values())
    native_write = 4 * checkpoint + native_evidence
    controller_primary = (CONTROLLER_FIXED_ALLOWANCE_BYTES +
        CONTROLLER_SERIALIZED_POOL_MULTIPLIER * sum(sizes.values()))
    controller_staging = controller_primary
    controller_write = controller_primary + controller_staging
    plan = dict(schema=SCHEMA, gpu_started=False,
        pool_order=[pool['pool_id'] for pool in pools], pool_payload_sha256=bindings,
        serialized_pool_config_bytes=sizes, pool_evidence_bytes=pool_evidence,
        native_pool_dynamic_allowance_bytes=NATIVE_POOL_DYNAMIC_ALLOWANCE_BYTES,
        controller_fixed_allowance_bytes=CONTROLLER_FIXED_ALLOWANCE_BYTES,
        controller_serialized_pool_multiplier=CONTROLLER_SERIALIZED_POOL_MULTIPLIER,
        parameter_count=PARAMETERS, parameter_bytes=PARAMETER_BYTES,
        checkpoint_metadata_bytes=CHECKPOINT_METADATA_BYTES,
        model_tensor_bytes=PARAMETERS * PARAMETER_BYTES,
        final_checkpoints=3, final_checkpoint_bytes=3 * checkpoint,
        export_staging_overlap_bytes=checkpoint,
        native_evidence_bytes=native_evidence, native_write_bytes=native_write,
        controller_primary_write_bytes=controller_primary,
        controller_retrieval_staging_bytes=controller_staging,
        controller_write_bytes=controller_write,
        total_write_bytes=native_write + controller_write,
        minimum_reserve_bytes=MIN_RESERVE_BYTES,
        same_filesystem_required_bytes=native_write + controller_write + MIN_RESERVE_BYTES,
        separate_filesystem_required_bytes=dict(native=native_write + MIN_RESERVE_BYTES,
                                                controller=controller_write + MIN_RESERVE_BYTES),
        optimizer_state_exports=0, intermediate_checkpoints=0,
        rationale=dict(
            native_config_copies='NATIVE_CONFIG plus training and serving WORKER_CONFIG; each nests original_pool.',
            native_dynamic='512 MiB per pool for sampled training/KL receipts, repeated worker/controller results, failures, native traces, logs and atomic JSON overlap.',
            controller='512 MiB plus 12 times actual nested pool JSON for result/progress/event/report copies and atomic overlap.',
            retrieval='One additional controller-evidence copy is reserved on the controller filesystem; the separate full-checkpoint overlap covers at most one staged checkpoint at a time.',
            reserve='8 GiB per distinct filesystem; native/controller writes are summed before one reserve when st_dev is shared.',
            legacy='The historical approximately 18.05 GB estimate is superseded by these actual-artifact-derived bounds.',
            limits='Additional simultaneous archive/checkpoint copies or unbounded logs require a new capacity budget; no compression is assumed.'))
    plan['plan_sha256'] = _seal(plan)
    return plan


def _probe(path):
    requested = Path(path).resolve()
    existing = requested
    while not existing.exists():
        require(existing.parent != existing, 'No existing storage ancestor')
        existing = existing.parent
    require(existing.is_dir(), 'Storage destination ancestor is not a directory: ' + str(existing))
    return dict(destination=str(requested), existing_ancestor=str(existing),
        filesystem_device=existing.stat().st_dev, free_bytes=shutil.disk_usage(existing).free)


def preflight_paths(plan, native_root, controller_root, reserve_bytes=MIN_RESERVE_BYTES):
    """Check actual destination devices without creating any path.

On failure, ``ValueError.storage_preflight`` preserves the complete read-only
receipt so the caller can record both destination checks and exact shortfalls.
"""
    require(plan.get('schema') == SCHEMA and plan.get('gpu_started') is False and
            plan.get('plan_sha256') == _seal(plan), 'Storage plan is missing or changed')
    require(type(reserve_bytes) is int and reserve_bytes >= MIN_RESERVE_BYTES,
            'Preserve at least 8 GiB on every storage filesystem')
    probes = dict(native=_probe(native_root), controller=_probe(controller_root))
    by_device = {}
    for label, probe in probes.items():
        device = probe['filesystem_device']
        require(type(device) is int and type(probe['free_bytes']) is int and probe['free_bytes'] >= 0,
                'Invalid storage filesystem/free-space observation')
        filesystem = by_device.setdefault(device, dict(filesystem_device=device, destinations=[],
            write_bytes=0, free_bytes=probe['free_bytes'], reserve_bytes=reserve_bytes))
        filesystem['destinations'].append(label)
        filesystem['write_bytes'] += plan[label + '_write_bytes']
        # Two observations may differ while the filesystem is in use. Admit
        # against the smaller observation, never sum the same free space twice.
        filesystem['free_bytes'] = min(filesystem['free_bytes'], probe['free_bytes'])
    filesystems = list(by_device.values())
    for filesystem in filesystems:
        filesystem['required_bytes'] = filesystem['write_bytes'] + reserve_bytes
        filesystem['shortfall_bytes'] = max(0, filesystem['required_bytes'] - filesystem['free_bytes'])
        filesystem['status'] = 'failed' if filesystem['shortfall_bytes'] else 'passed'
    failed = [filesystem for filesystem in filesystems if filesystem['status'] == 'failed']
    receipt = dict(status='failed' if failed else 'passed', gpu_started=False,
        storage_plan_sha256=plan['plan_sha256'], reserve_bytes=reserve_bytes,
        same_filesystem=len(filesystems) == 1, destinations=probes, filesystems=filesystems,
        native_write_bytes=plan['native_write_bytes'], controller_write_bytes=plan['controller_write_bytes'],
        total_write_bytes=plan['total_write_bytes'],
        total_required_bytes=sum(filesystem['required_bytes'] for filesystem in filesystems),
        scope='Read-only capacity admission before training; no directory creation, cleanup, offload or model work.')
    if failed:
        detail = '; '.join(','.join(item['destinations']) + ': shortfall ' + str(item['shortfall_bytes']) +
            ' bytes (required ' + str(item['required_bytes']) + ', free ' + str(item['free_bytes']) + ')'
            for item in failed)
        error = ValueError('P8 planned writes, retrieval staging and reserve exceed free space: ' + detail)
        error.storage_preflight = receipt
        raise error
    return receipt
