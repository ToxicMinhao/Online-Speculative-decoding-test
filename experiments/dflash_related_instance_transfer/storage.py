"""Read-only artifact sizing for 24 new requests; no model/state copies.

Capacity admission checks both actual destinations and sums writes when they
share a filesystem. Sizing binds the complete actual group/runtime payloads,
including repeated historical evidence in native/controller configurations.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from experiments.dflash_baseline_learning_curve.experiment import digest, require

SCHEMA = 'dflash_related_instance_storage_v1'
MIN_RESERVE_BYTES = 8 * 2**30
OUTPUT_CAP = 1024
# Token-only native traces: at most 1024 verification rounds. This allowance
# covers proposal/commit IDs, offsets, provenance, reconciliation and diagnostics
# for every round; it explicitly does not admit hidden states or dense logits.
ROUND_JSON_BYTES = 16 * 2**10
FIXED_REQUEST_BYTES = 2 * 2**20
GROUP_LOG_METADATA_BYTES = 32 * 2**20
CONFIG_COPIES = 3  # native config, worker config, atomic-write overlap
CONTROLLER_GROUP_COPIES = 8
RESULT_COPIES = 6  # raw native, normalized, failure/progress, atomic overlap


def serialized_bytes(value):
    encoder = json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False)
    return 1 + sum(len(chunk.encode('utf-8')) for chunk in encoder.iterencode(value))


def _seal(plan):
    return digest({k: v for k, v in plan.items() if k != 'plan_sha256'})


def runtime_payload(config):
    """Remove only generated fields that cannot be inputs to their own sizing."""
    return {k: v for k, v in config.items() if k not in (
        'storage_budget', 'output_root', 'controller_root', 'related_instance_source_sha256')}


def build_storage_plan(groups, runtime_config=None):
    require(isinstance(groups, list) and len(groups) == 6 and
            len({g['group_id'] for g in groups}) == 6, 'Exactly six sealed original groups required')
    config = runtime_payload(runtime_config or {})
    group_sizes, request_bounds, bindings = {}, {}, {}
    for group in groups:
        gid = group['group_id']
        require(group.get('group_payload_sha256') == digest({k: v for k, v in group.items()
            if k != 'group_payload_sha256'}), 'Group payload changed before storage sizing')
        require(len(group.get('children', [])) == 2, 'Exactly two new children per group')
        bindings[gid] = group['group_payload_sha256']
        group_sizes[gid] = serialized_bytes(dict(config, original_group=group))
        for child in group['children']:
            require(child.get('role') == 'related_instance' and child.get('group_id') == gid and
                    child.get('input_ids_sha256') == digest(child.get('input_ids')),
                    'Storage sizing requires sealed new-request token bindings')
            rid = child['request_id']
            require(rid not in request_bounds, 'New request IDs must be unique')
            # Prompt appears in trace/raw/normalized records. Size its exact text
            # and token serialization instead of assuming a typical prompt.
            request_bounds[rid] = (serialized_bytes(child) * 4 +
                OUTPUT_CAP * ROUND_JSON_BYTES + FIXED_REQUEST_BYTES)
    request_total = 2 * sum(request_bounds.values())  # one F and one own S8
    native_configs = CONFIG_COPIES * sum(group_sizes.values())
    native_dynamic = RESULT_COPIES * request_total + 6 * GROUP_LOG_METADATA_BYTES
    native_write = native_configs + native_dynamic
    controller_primary = (CONTROLLER_GROUP_COPIES * sum(group_sizes.values()) +
        RESULT_COPIES * request_total + GROUP_LOG_METADATA_BYTES)
    # Retrieval may include the entire native tree plus controller evidence;
    # do not rely on compression, deduplication, or deleting the source.
    staging = native_write + controller_primary
    controller_write = controller_primary + staging
    plan = dict(schema=SCHEMA, gpu_started=False, group_order=[g['group_id'] for g in groups],
        group_payload_sha256=bindings, runtime_payload_sha256=digest(config),
        serialized_group_config_bytes=group_sizes, request_artifact_bound_bytes=request_bounds,
        round_json_bound_bytes=ROUND_JSON_BYTES, maximum_rounds_per_request=OUTPUT_CAP,
        maximum_generations=24, native_config_copies=CONFIG_COPIES,
        native_result_copies=RESULT_COPIES, controller_group_copies=CONTROLLER_GROUP_COPIES,
        group_log_metadata_bytes=GROUP_LOG_METADATA_BYTES,
        native_configuration_bytes=native_configs, native_dynamic_bytes=native_dynamic,
        native_write_bytes=native_write, controller_primary_write_bytes=controller_primary,
        controller_retrieval_staging_bytes=staging, controller_write_bytes=controller_write,
        total_write_bytes=native_write + controller_write,
        minimum_reserve_bytes=MIN_RESERVE_BYTES,
        same_filesystem_required_bytes=native_write + controller_write + MIN_RESERVE_BYTES,
        separate_filesystem_required_bytes=dict(native=native_write + MIN_RESERVE_BYTES,
            controller=controller_write + MIN_RESERVE_BYTES),
        checkpoint_copies=0, checkpoint_exports=0, optimizer_state_exports=0,
        feature_tensor_exports=0, model_downloads=0,
        rationale=dict(config='Three actual nested native configuration copies; eight controller group/evidence copies.',
            traces='24 generations, each at most 1024 rounds; 16 KiB/round plus actual prompt and 2 MiB/request; six receipt/atomic copies.',
            logs='32 MiB/group for worker logs and metadata; post-request accounting stops further generation if this allowance is exceeded.',
            retrieval='Additional full native tree plus controller primary evidence; no compression assumption.',
            reserve='8 GiB per distinct destination filesystem; sum both write budgets on shared devices.',
            exclusions='No checkpoints, optimizer states, feature tensors, new models, or runtime-cache builds admitted.'))
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


def _existing_artifact_credit(plan, native_root, controller_root):
    """Credit only verified artifacts from this fresh, bound attempt.

    Previously written bytes already reduced disk free space. Charging them as
    future writes again can reject later groups at an originally valid capacity
    boundary. Retrieval staging remains entirely future work and is never
    credited. Completed native groups and exact controller evidence are scoped
    by the saved plan/manifest and source identity; unrelated files earn nothing.
    """
    native_root, controller_root = Path(native_root).resolve(), Path(controller_root).resolve()
    credit = dict(native=0, controller=0, files=[])
    plan_path = controller_root / 'PLAN.json'
    if not plan_path.is_file():
        return credit
    allocation = json.loads(plan_path.read_text())
    require(allocation.get('storage_budget') == plan and allocation.get('group_order') == plan['group_order'],
            'Existing controller plan belongs to another attempt')
    manifest_path = controller_root / 'REQUEST_MANIFEST.json'
    require(manifest_path.is_file() and digest(json.loads(manifest_path.read_text())) == allocation.get('manifest_sha256'),
            'Existing request manifest differs from bound current attempt')
    source_path = controller_root / 'SOURCE_MANIFEST.json'
    require(source_path.is_file(), 'Existing attempt source identity missing')
    source = json.loads(source_path.read_text())
    require(source.get('approved_code_sha256') == digest(source.get('related_instance_source_sha256')),
            'Existing attempt source digest differs')
    seen = set()
    def add(path, label):
        require(not path.is_symlink() and path.is_file(), 'Current-attempt credit requires an ordinary file')
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            return
        seen.add(identity)
        # A sparse file must not claim unallocated logical bytes as spent space.
        amount = min(stat.st_size, getattr(stat, 'st_blocks', (stat.st_size + 511) // 512) * 512)
        credit[label] += amount
        credit['files'].append(dict(path=str(path), bytes=amount, destination=label))
    for path in (plan_path, manifest_path, source_path):
        add(path, 'controller')
    runtime_path = controller_root / 'RUNTIME_CONFIG.json'
    if runtime_path.is_file():
        config = json.loads(runtime_path.read_text())
        require(config.get('storage_budget') == plan and Path(config['output_root']).resolve() == native_root and
                Path(config['controller_root']).resolve() == controller_root and
                digest(runtime_payload(config)) == plan['runtime_payload_sha256'] and
                config.get('related_instance_source_sha256') == source['related_instance_source_sha256'],
                'Existing runtime configuration belongs to another attempt')
        add(runtime_path, 'controller')
    previous = controller_root / 'STORAGE_PREFLIGHT.json'
    if previous.is_file():
        saved = json.loads(previous.read_text())
        require(saved.get('storage_plan_sha256') == plan['plan_sha256'], 'Existing storage receipt belongs to another plan')
        add(previous, 'controller')
    progress = controller_root / 'PROGRESS.json'
    if progress.is_file():
        saved = json.loads(progress.read_text())
        selected = saved.get('selected_groups', [])
        require([r.get('group_id') for r in selected] == plan['group_order'] and
                {r['group_id']: r.get('group_payload_sha256') for r in selected} == plan['group_payload_sha256'],
                'Existing progress belongs to different sealed groups')
        add(progress, 'controller')
        events = controller_root / 'EVENTS.jsonl'
        if events.is_file():
            add(events, 'controller')
    # Only completed group trees are credited. The native tree can be nested
    # under the controller root; controller uses an explicit file allowlist,
    # and inode de-duplication prevents double counting in either case.
    for gid in plan['group_order']:
        directory = native_root / gid
        finish, config_path = directory / 'NATIVE_FINISH.json', directory / 'NATIVE_CONFIG.json'
        if not directory.exists():
            continue
        require(not directory.is_symlink() and finish.is_file() and config_path.is_file(),
                'Existing native group is incomplete or not owned by this attempt')
        saved, config = json.loads(finish.read_text()), json.loads(config_path.read_text())
        require(saved.get('status') == 'passed' and saved.get('checkpoint_weights_unchanged') is True and
                saved.get('training_quiescent') is True and saved.get('group_id') == gid and
                config.get('episode_id') == gid and config.get('storage_budget') == plan and
                config.get('related_instance_source_sha256') == source['related_instance_source_sha256'] and
                config.get('original_group', {}).get('group_payload_sha256') == plan['group_payload_sha256'][gid],
                'Existing native group lacks valid current-attempt completion evidence')
        for path in sorted(directory.rglob('*')):
            require(not path.is_symlink(), 'Native artifact symlinks are not creditable')
            if path.is_file():
                require(path.suffix not in ('.safetensors', '.pt', '.pth', '.bin'), 'No model/tensor exports in artifact budget')
                add(path, 'native')
    require(credit['native'] <= plan['native_write_bytes'] and
            credit['controller'] <= plan['controller_primary_write_bytes'],
            'Existing attempt artifacts already exceed planned primary writes')
    return credit


def preflight_paths(plan, native_root, controller_root, reserve_bytes=MIN_RESERVE_BYTES):
    require(plan.get('schema') == SCHEMA and plan.get('gpu_started') is False and
            plan.get('plan_sha256') == _seal(plan), 'Storage plan is missing or changed')
    require(type(reserve_bytes) is int and reserve_bytes >= MIN_RESERVE_BYTES,
            'Preserve at least 8 GiB on every storage filesystem')
    credit = _existing_artifact_credit(plan, native_root, controller_root)
    probes = dict(native=_probe(native_root), controller=_probe(controller_root))
    devices = {}
    for label, probe in probes.items():
        require(type(probe['filesystem_device']) is int and type(probe['free_bytes']) is int and
                probe['free_bytes'] >= 0, 'Invalid filesystem/free-space observation')
        row = devices.setdefault(probe['filesystem_device'], dict(filesystem_device=probe['filesystem_device'],
            destinations=[], write_bytes=0, free_bytes=probe['free_bytes'], reserve_bytes=reserve_bytes))
        row['destinations'].append(label)
        row['write_bytes'] += plan[label + '_write_bytes'] - credit[label]
        row['free_bytes'] = min(row['free_bytes'], probe['free_bytes'])
    rows = list(devices.values())
    for row in rows:
        row['required_bytes'] = row['write_bytes'] + reserve_bytes
        row['shortfall_bytes'] = max(0, row['required_bytes'] - row['free_bytes'])
        row['status'] = 'failed' if row['shortfall_bytes'] else 'passed'
    failed = any(r['status'] == 'failed' for r in rows)
    receipt = dict(status='failed' if failed else 'passed', gpu_started=False,
        storage_plan_sha256=plan['plan_sha256'], destinations=probes, filesystems=rows,
        same_filesystem=len(rows) == 1, reserve_bytes=reserve_bytes, existing_artifact_credit=credit,
        remaining_native_write_bytes=plan['native_write_bytes'] - credit['native'],
        remaining_controller_write_bytes=plan['controller_write_bytes'] - credit['controller'],
        remaining_total_write_bytes=plan['total_write_bytes'] - credit['native'] - credit['controller'],
        native_write_bytes=plan['native_write_bytes'], controller_write_bytes=plan['controller_write_bytes'],
        total_write_bytes=plan['total_write_bytes'], total_required_bytes=sum(r['required_bytes'] for r in rows),
        checkpoint_copies=0, optimizer_state_exports=0,
        scope='Read-only actual-path admission before generation; no cleanup, directories or model work.')
    if failed:
        error = ValueError('Related-instance artifacts, retrieval staging and reserve exceed free space: ' +
            '; '.join(','.join(r['destinations']) + ' shortfall ' + str(r['shortfall_bytes']) + ' bytes'
                for r in rows if r['shortfall_bytes']))
        error.storage_preflight = receipt
        raise error
    return receipt
