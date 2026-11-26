"""Authenticate the fixed warm transition and admit the complete reuse pilot."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

from experiments.frontier_preservation_pilot.point1_prepare import (
    ARCHIVE_MANIFEST_IDENTITY, PRODUCTION_PLAN_IDENTITY, RECOVERY_IDENTITY,
    EXPORT_IDENTITY, gpu_inventory, host_memory,
)
from experiments.frontier_preservation_pilot.point3_prepare import CONTROL_RECOVERY, CONTROL_EXPORT, FEATURE_IDENTITIES
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .configuration import SCHEMA, PARENT_BINDING, digest, build_config

BASE_COMMIT = 'cb8244f79d4d36630316899051994ad901ca0874'
DIAGNOSTIC_CONTRACT = dict(bytes=269270, sha256='a10be1fb2176eac8d785a4792f573f39503610c0649c3d33714f1658085aeff4')
DIAGNOSTIC_MANIFEST = dict(bytes=49428, sha256='760ca8ce26c7132d9abeab71e16700d429353ecd890da13de3262754e404e683')
LIMITS = dict(occupied_card_s=14400, elapsed_s=14400, allocated_bytes=24_000_000_000,
              reserve_bytes=400_000_000_000, qualification_adam_steps=6, training_adam_steps=128,
              scored_generations=768, warmup_generations=24, readout_response_forwards=137)


def prepare(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    sealed = root/'sealed'; sealed.mkdir()
    bindings = {}
    def bound(path, expected):
        path = str(path)
        if identity(path) != expected:
            raise ValueError('Required retained artifact changed: ' + path)
        bindings[path] = expected
        return Path(path)
    def read(path, expected):
        return json.loads(bound(path, expected).read_text())

    archive = read(args.archive, ARCHIVE_MANIFEST_IDENTITY)
    if archive['execution_commit'] != '225802657d630c0985abd68d6e7453f3e47c5b74':
        raise ValueError('Wrong retained BASE execution archive')
    entries = {r['path']: dict(bytes=r['bytes'], sha256=r['sha256']) for r in archive['files']}
    production = read(args.production_plan, PRODUCTION_PLAN_IDENTITY)
    prior_root = Path(production['root'])
    for path, expected in production['sealed_files'].items(): bound(path, expected)
    for model in production['model_identities']: bound(model['path'], model['expected_identity'])
    parent = json.loads(Path(production['parent_plan']).read_text())
    if parent['binding'] != PARENT_BINDING:
        raise ValueError('Authenticated parent is not the original PAIR1 plan')
    bound(production['parent_recovery'], RECOVERY_IDENTITY)
    bound(Path(production['parent_export'])/'model.safetensors', EXPORT_IDENTITY)
    base_status_path = prior_root/'arms/CONTROL/STATUS.json'
    base = read(base_status_path, entries['arms/CONTROL/STATUS.json'])
    event_path = prior_root/'arms/CONTROL/trainer/events/event_002/RESULT.json'
    event = read(event_path, entries['arms/CONTROL/trainer/events/event_002/RESULT.json'])
    if (base['status'] != 'completed' or [base['global_step'], base['presentations'], base['completed_event']] != [256,512,2]
            or event != base['training'] or len(event['records']) != 256):
        raise ValueError('Retained BASE is not the complete matching event')
    bound(base['recovery'], CONTROL_RECOVERY)
    bound(Path(base['checkpoint'])/'model.safetensors', CONTROL_EXPORT)
    feature_manifest = prior_root/'shared/FEATURE_IDENTITIES.json'
    features = read(feature_manifest, FEATURE_IDENTITIES)
    if len(features) != 128 or [r['source_index'] for r in features] != list(range(128,256)):
        raise ValueError('Fixed original event-two buffer changed')
    for row in features: bound(row['path'], row['identity'])
    # PR64 is retained evidence, not a new training data selector.
    diagnostic_root = Path(args.diagnostic_root)
    diagnostic = read(diagnostic_root/'CONTRACT.json', DIAGNOSTIC_CONTRACT)
    diagnostic_manifest = read(diagnostic_root/'RETRIEVAL_MANIFEST.json', DIAGNOSTIC_MANIFEST)
    diagnostic_files = {}
    for item in diagnostic_manifest['files']:
        relative = item['path']
        if relative in ('CONTRACT.json','STATUS.json','legacy/parent.json','legacy/child.json') or relative.startswith('primary/'):
            path = diagnostic_root/relative
            expected = dict(bytes=item['bytes'],sha256=item['sha256'])
            bound(path, expected)
            diagnostic_files[str(path)] = expected
    if json.loads((diagnostic_root/'STATUS.json').read_text())['status'] != 'completed':
        raise ValueError('Retained PR64 readouts are incomplete')
    for rows in diagnostic['legacy_binding']['sets'].values():
        for row in rows: bound(row['feature_path'], row['feature_identity'])
    source = json.loads(Path(args.source_manifest).read_text())
    if source['commit'] != args.commit or source.get('clean') is not True:
        raise ValueError('Require a clean published source identity before model work')
    source_identity = identity(args.source_manifest)
    resources = dict(gpus=gpu_inventory(), host=host_memory(), free_bytes=shutil.disk_usage(root).free,
                     remote_remaining_peak_bytes=22_000_000_000, local_bytes=args.local_bytes)
    resources['all_locations_peak_bytes'] = resources['remote_remaining_peak_bytes'] + args.local_bytes
    idle = not resources['gpus']['processes']
    enough_disk = resources['free_bytes'] - resources['remote_remaining_peak_bytes'] >= LIMITS['reserve_bytes']
    enough_total = resources['all_locations_peak_bytes'] <= LIMITS['allocated_bytes']
    enough_host = resources['host']['available_bytes'] >= 60_000_000_000
    cards = {r['uuid']:r for r in resources['gpus']['cards']}
    enough_gpu = all(g in cards and cards[g]['total_bytes']-cards[g]['used_bytes'] >= 60_000_000_000
                     for g in (args.training_gpu,args.serving_gpu))
    resources['checks'] = dict(idle=idle,disk=enough_disk,total=enough_total,host=enough_host,gpu=enough_gpu)
    resources['admitted'] = all(resources['checks'].values())
    write_json(root/'RESOURCE_ADMISSION.json', resources)
    if not resources['admitted']:
        raise ValueError('Complete pilot does not fit the unchanged resource contract')
    parent_text = Path(parent['configs']['PAIR']).read_text()
    configs = {}
    for arm in ('BASE','TAIL_DISABLED','TAIL'):
        path = sealed/(arm+'.yaml')
        path.write_text(build_config(parent_text, arm, root/'arms'/arm/'trainer'))
        bindings[str(path)] = identity(path)
        configs[arm] = str(path)
    inherited = ('original_config','manifest','canonicals','warmup_canonicals','canonical_provenance',
                 'model_identities','target','draft','worker_python','training_python','serving','devices',
                 'parent_plan','parent_recovery','parent_export','parent_export_identity')
    plan = {key:deepcopy(production[key]) for key in inherited}
    plan.update(schema=SCHEMA, root=str(root), base_commit=BASE_COMMIT, execution_commit=args.commit,
        source_root=str(Path(__file__).resolve().parents[2]), source_manifest=str(args.source_manifest),
        source_identity=source_identity, configs=configs, production_admitted=True,
        resource_admission=resources, limits=LIMITS, reserve_bytes=LIMITS['reserve_bytes'],
        local_charged_bytes=args.local_bytes, local_allocated_bytes=args.local_bytes,
        prior_charged_roots=[], compiler_baseline=compiler_snapshot(),
        base_status=str(base_status_path), base_status_identity=bindings[str(base_status_path)],
        base_export_identity=CONTROL_EXPORT, base_recovery_identity=CONTROL_RECOVERY,
        reuse_BASE=True, feature_manifest=str(feature_manifest), retained_event=str(event_path),
        shared_root=str(prior_root/'shared'), retained_diagnostic_root=str(diagnostic_root),
        retained_diagnostic_contract_identity=DIAGNOSTIC_CONTRACT, retained_diagnostic_files=diagnostic_files,
        qualification_result=str(root/'qualification/RESULT.json'),
        training_gpu=args.training_gpu, serving_gpu=args.serving_gpu, sealed_files=bindings,
        forecast=dict(native_seconds=3000, contingency_seconds=4200,
                      source='Historical BASE564.545s, prior three-arm serving1485s; additional qualification/readout/loading allowance'),
        budget=dict(training_presentations=256, training_adam_steps=128, qualification_adam_max=6,
                    qualification_planned=3, generation_total=792, scored=768, warmups=24,
                    TAIL_readout_forwards=137, new_feedback=0, new_teacher_preparations=0),
        bootstrap_seed=20261126, created_epoch=time.time())
    plan['devices']['A'] = args.serving_gpu
    plan['devices']['B'] = args.training_gpu
    plan['binding'] = digest(plan)
    write_json(sealed/'PLAN.json', plan)
    write_json(root/'IDENTITIES.json', bindings)
    print(json.dumps(dict(status='admitted',plan=str(sealed/'PLAN.json'),binding=plan['binding'],resources=resources)))
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','production-plan','archive','diagnostic-root','source-manifest','commit','training-gpu','serving-gpu'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--local-bytes',type=int,required=True)
    prepare(parser.parse_args())


if __name__=='__main__':
    main()
