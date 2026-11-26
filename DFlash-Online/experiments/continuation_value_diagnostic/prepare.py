"""Authenticate retained endpoints and seal outcome-independent diagnostic anchors."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from experiments.ok_coding.storage import identity
from experiments.frontier_preservation_pilot.point1_prepare import (
    ARCHIVE_MANIFEST_IDENTITY, PARENT_BINDING, PRODUCTION_PLAN_IDENTITY,
    RECOVERY_IDENTITY, EXPORT_IDENTITY, read_bound_json, gpu_inventory, host_memory,
)
from experiments.frontier_preservation_pilot.point3_prepare import (
    CONTROL_RECOVERY, CONTROL_EXPORT, FEATURE_IDENTITIES,
)
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from online_feedback.storage import write_json

BASE_COMMIT = '281516426411ac567129dae7af15ea443a3d0c94'
SELECTION_PREFIX = 'continuation_value_diagnostic_v1|20261126|'
LIMITS = dict(occupied_card_s=3600, elapsed_s=5400, allocated_bytes=2_000_000_000,
              reserve_bytes=400_000_000_000, retry_response_forwards=8)


def select_anchors(record_id, presentation):
    """Hash-rank distinct valid first-pass anchors; preserve native ascending order."""
    candidates = set()
    for sampled in presentation['sampled_anchors']:
        positions, keep = sampled['positions'], sampled['keep_mask']
        if len(positions) != 1 or len(keep) != 1 or len(positions[0]) != len(keep[0]):
            raise ValueError('Expected one native response and aligned keep mask')
        candidates.update(int(a) for a, valid in zip(positions[0], keep[0]) if valid)
    ranked = sorted(candidates, key=lambda anchor: (
        hashlib.sha256(f'{SELECTION_PREFIX}{record_id}|{anchor}'.encode('utf-8')).digest(), anchor))
    return sorted(ranked[:64]), len(candidates)


def prepare(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def bound(path, expected):
        path = str(path)
        actual = identity(path)
        if actual != expected:
            raise ValueError('Retained identity changed: ' + path)
        bindings[path] = actual
        return Path(path)

    def read(path, expected):
        return json.loads(bound(path, expected).read_text())

    archive = read(args.archive, ARCHIVE_MANIFEST_IDENTITY)
    if archive['execution_commit'] != '225802657d630c0985abd68d6e7453f3e47c5b74':
        raise ValueError('Archive is not the retained native CONTROL transition')
    entries = {row['path']: dict(bytes=row['bytes'], sha256=row['sha256']) for row in archive['files']}
    production_path = Path(args.production_plan)
    production = read(production_path, PRODUCTION_PLAN_IDENTITY)
    old_root = Path(production['root'])
    parent = json.loads(Path(production['parent_plan']).read_text())
    if parent['binding'] != PARENT_BINDING:
        raise ValueError('Wrong PAIR1 parent plan')
    # PLAN is itself pinned and binds the config/model files. Bind those exact bytes.
    for path, expected in production['sealed_files'].items():
        bound(path, expected)
    for model in production['model_identities']:
        bound(model['path'], model['expected_identity'])
    bindings[str(Path(production['parent_plan']))] = identity(production['parent_plan'])
    status = read(old_root/'arms/CONTROL/STATUS.json', entries['arms/CONTROL/STATUS.json'])
    event_path = old_root/'arms/CONTROL/trainer/events/event_002/RESULT.json'
    # This existing archive entry authenticates the complete original anchor trajectory.
    event = read(event_path, entries['arms/CONTROL/trainer/events/event_002/RESULT.json'])
    if (status['status'] != 'completed' or
            [status['global_step'], status['presentations'], status['completed_event']] != [256,512,2]
            or event != status['training'] or len(event['records']) != 256):
        raise ValueError('CONTROL state and archived event receipt disagree')
    bound(production['parent_recovery'], RECOVERY_IDENTITY)
    parent_export = bound(Path(production['parent_export'])/'model.safetensors', EXPORT_IDENTITY)
    bound(status['recovery'], CONTROL_RECOVERY)
    child_export = bound(Path(status['checkpoint'])/'model.safetensors', CONTROL_EXPORT)
    features = read(old_root/'shared/FEATURE_IDENTITIES.json', FEATURE_IDENTITIES)
    if len(features) != 128 or [r['source_index'] for r in features] != list(range(128,256)):
        raise ValueError('The fixed 128-response cohort changed')
    first_pass = {r['record_id']:r for r in event['records'] if r['pass'] == 0}
    if len(first_pass) != 128 or len([r for r in event['records'] if r['pass'] == 0]) != 128:
        raise ValueError('Need exactly one retained first-pass presentation per response')
    rows = []
    for ordinal, row in enumerate(features):
        bound(row['path'], row['identity'])
        anchors, available = select_anchors(row['record_id'], first_pass[row['record_id']])
        rows.append(dict(ordinal=ordinal, source_index=row['source_index'], record_id=row['record_id'],
                         feature_path=row['path'], feature_identity=row['identity'], anchors=anchors,
                         available_first_pass_anchors=available, physical_shard=ordinal % len(args.devices)))
    witness_binding = read(old_root/'witness/BINDINGS.json', entries['witness/BINDINGS.json'])
    legacy = {name:read(old_root/f'witness/{filename}.json', entries[f'witness/{filename}.json'])
              for name, filename in [('parent','parent'), ('child','CONTROL')]}
    parent_features_path = Path(parent['root'])/'arms/PAIR/FEATURE_IDENTITIES.json'
    parent_features = json.loads(parent_features_path.read_text())
    for i, row in enumerate(witness_binding['sets']['parent_training']):
        if row['feature_path'] != parent_features[i]['path'] or row['feature_identity'] != parent_features[i]['identity']:
            raise ValueError('Legacy parent feature provenance differs from witness binding')
        bound(row['feature_path'], row['feature_identity'])
    bindings[str(parent_features_path)] = identity(parent_features_path)
    for endpoint, old in legacy.items():
        if old['binding_identity'] != entries['witness/BINDINGS.json']:
            raise ValueError('Legacy endpoint changed witness binding')
        write_json(root/'legacy'/f'{endpoint}.json', old)
    write_json(root/'legacy/BINDINGS.json', witness_binding)
    source_identity = identity(args.source_manifest)
    source = json.loads(Path(args.source_manifest).read_text())
    if source['commit'] != args.commit:
        raise ValueError('Staged source commit mismatch')
    resources = dict(gpus=gpu_inventory(), host=host_memory(), free_bytes=shutil.disk_usage(root).free,
                     local_charged_bytes=args.local_bytes, remote_peak_write_bytes=1_000_000_000,
                     total_projected_study_bytes=args.local_bytes+1_000_000_000)
    if resources['gpus']['processes'] or resources['free_bytes']-1_000_000_000 < LIMITS['reserve_bytes']:
        raise ValueError('GPU occupation or storage reserve prevents admission')
    if resources['total_projected_study_bytes'] > LIMITS['allocated_bytes']:
        raise ValueError('All-location study storage exceeds cap')
    known_devices = {row['uuid'] for row in resources['gpus']['cards']}
    if not 1 <= len(args.devices) <= 2 or len(set(args.devices)) != len(args.devices) or not set(args.devices) <= known_devices:
        raise ValueError('Select one or two distinct available physical GPUs')
    from .statistics import numerical_contract
    contract = dict(schema='continuation_value_diagnostic_v1', root=str(root), created_epoch=time.time(),
        base_commit=BASE_COMMIT, execution_commit=args.commit, source_root=str(Path(__file__).resolve().parents[2]),
        source_manifest=str(Path(args.source_manifest)), source_identity=source_identity,
        original_config=production['original_config'], target=production['target'], draft=production['draft'],
        training_python=production['training_python'], devices=args.devices,
        endpoints=dict(parent=dict(export=str(parent_export.parent), identity=EXPORT_IDENTITY, counters=[128,256,1]),
                       child=dict(export=str(child_export.parent), identity=CONTROL_EXPORT, counters=[256,512,2])),
        features=rows, legacy_binding=witness_binding, identities=bindings,
        selection=dict(prefix=SELECTION_PREFIX, maximum=64, first_pass=0,
                       ranking='SHA256 UTF-8 then numeric anchor; selected coordinates sorted ascending'),
        statistics=numerical_contract(), limits=LIMITS, resource_admission=resources,
        compiler_baseline=compiler_snapshot(), seed=17, block_size=16, projection_chunk_blocks=8,
        mode='eval/no_grad', model_precision='BF16', margin_arithmetic='FP32', offline_reduction='FP64',
        primary_cohort='retained event-2 TRAIN buffer; not held out',
        qualification=dict(response_forwards_per_card=8, endpoint_forwards=4,
                           legacy_responses_per_endpoint=2, primary_first_response_repetitions=2),
        budget=dict(primary_response_forwards=sum(bool(r['anchors']) for r in rows)*2,
                    qualification_response_forwards=8*len(args.devices), maximum_retry_response_forwards=8,
                    primary_paired_blocks=sum(len(r['anchors']) for r in rows),
                    backward=0, adam=0, optimizer_creation=0, teacher=0, generation=0),
        local_charged_bytes=args.local_bytes)
    write_json(root/'CONTRACT.json', contract)
    write_json(root/'IDENTITIES.json', bindings)
    print(json.dumps(dict(status='sealed',responses=len(rows),paired_blocks=contract['budget']['primary_paired_blocks'],
                          free_bytes=resources['free_bytes'],root=str(root))))
    return contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--production-plan',required=True)
    p.add_argument('--archive',required=True)
    p.add_argument('--source-manifest',required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--local-bytes',type=int,required=True)
    p.add_argument('--devices',nargs='+',required=True)
    prepare(p.parse_args())


if __name__=='__main__':
    main()
