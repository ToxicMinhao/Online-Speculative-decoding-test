"""Prepare unchanged C from authenticated PR51 assets and new input-only seals.

No model, optimizer or feedback tensor is copied. Missing historical common-E5
content authentication blocks package creation; a fresh hash cannot replace it.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import tarfile

from experiments.bounded_prefix_allocation.prepare import identity, _digest, _write
from experiments.mathqa_fixed_lr_transfer.prepare import DEVICES
from .data import validate_manifests, evaluation_schedule
from .runtime import ALLOCATION_BYTES, RESERVE_BYTES, LIMITS

TASK_NAME = 'dflash_unchanged_c_mathqa_transfer_20261117'
PR51_TASK = 'dflash_mathqa_fixed_lr_transfer_20261116'
PR51_COMMIT = '15284b323a307d01118e2630e665ed0191ba7c33'
PR51_MANIFEST_SHA256 = 'c2f361a232d06d4ecbd62623e6573ea705b6e621058259c76e191e3975cff205'
INPUTS = dict(data_panel='PANEL.json', warmups='WARMUPS.json', adaptation='TRAINING.json',
    event5_inputs='EVENT5_INPUTS.json', event6_inputs='EVENT6_INPUTS.json',
    evaluation_schedule='EVALUATION_SCHEDULE.json', data_seal='DATA_SEAL.json',
    source_provenance='SOURCE_PROVENANCE.json', exposure_inventory='EXPOSURE_SUPPLEMENT.json')
MUTABLE_PLAN_FIELDS = {'binding', 'compiler_baseline_allocated_bytes', 'estimated_seconds'}


class ReadinessBlocked(ValueError):
    """A required retained identity is missing; no model work may start."""


def blocked_readiness(destination, reason):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='BLOCKED_RETAINED_ASSET_AUTHENTICATION', reason=reason,
        historical_feature_identity_required=True, fresh_hashes_are_not_historical_proof=True,
        model_work=0, gpu_qualified=False)
    _write(destination / 'READINESS.json', receipt)
    raise ReadinessBlocked(reason)


def read(path):
    return json.loads(Path(path).read_text())


def plan_binding(plan):
    return _digest({key: value for key, value in plan.items() if key not in MUTABLE_PLAN_FIELDS})


def storage_plan(prompt_lengths, *, weight_bytes, recovery_bytes, hidden_size, captured_layers):
    """Full capped own E6 pool, duplicate retention, two C and two scratch states."""
    lengths = list(prompt_lengths)
    if (len(lengths) != 128 or any(type(n) is not int or not 0 < n <= 2048 for n in lengths)
            or any(type(n) is not int or n <= 0 for n in (weight_bytes, recovery_bytes, hidden_size, captured_layers))):
        raise ValueError('Require measured tensor geometry and all 128 E6 prompts')
    row_bytes = hidden_size * (captured_layers + 1) * 2 + 12
    features = sum(n + 2048 for n in lengths) * row_bytes
    state_bytes = recovery_bytes + 16 * 2**20
    boundary = state_bytes + weight_bytes
    components = dict(prepared_feature_bytes=features,
        retained_duplicate_capture_bytes=features + 128 * row_bytes,
        temporary_feature_bytes=2 * (max(lengths) + 2049) * row_bytes,
        checkpoint_bytes=2 * boundary, qualification_payload_bytes=2 * boundary,
        atomic_publication_bytes=boundary, compiler_bytes=4 * 2**30,
        compact_bytes=2_000_000_000, source_and_workspace_bytes=2 * 2**30)
    components['serialization_margin_bytes'] = math.ceil(sum(components.values()) * .05)
    peak = sum(components.values())
    return dict(status='passed' if peak <= ALLOCATION_BYTES else 'blocked', peak_new_bytes=peak,
        allocation_cap_bytes=ALLOCATION_BYTES, reserve_bytes=RESERVE_BYTES,
        required_initial_free_bytes=peak + RESERVE_BYTES, row_bytes=row_bytes,
        retained_boundaries=['C5', 'C6'], feature_pools=1, common_event5_copies=0,
        publication_concurrency=1, existing_dependencies_copied=False,
        duplicate_retirement_required_for_admission=False, **components)


def authenticated_retained_assets(workspace):
    """Read historical compact identities without claiming missing packet hashes."""
    previous = Path(workspace) / 'research' / PR51_TASK
    delivery = previous / 'delivery/execution_001'
    manifest_path = delivery / 'RETRIEVAL_MANIFEST.json'
    if identity(manifest_path)['sha256'] != PR51_MANIFEST_SHA256:
        raise ValueError('Historical PR51 transport manifest differs')
    manifest = read(manifest_path)
    consumed = {}
    def checked(relative):
        path = delivery / relative
        if identity(path) != manifest[relative]:
            raise ValueError('Historical compact asset changed: ' + relative)
        consumed[relative] = manifest[relative]
        return read(path)
    old = checked('sealed/PLAN.json')
    if old['execution_commit'] != PR51_COMMIT:
        raise ValueError('Wrong retained PR51 execution source')
    endpoints = {}
    pools = {}
    expected = {p: v for p, v in old['input_identities'].items() if not p.startswith(old['root'] + '/')}
    for task in ('common5', 'N_collect6', 'S_collect6'):
        relative = f'tasks/{task}/FEATURES.json'
        value = checked(relative)
        if (value['status'] != 'completed' or value['binding'] != old['binding']
                or value['execution_commit'] != PR51_COMMIT or value['generated_responses'] != 128
                or value['teacher_preparations'] != 128 or len(value['feature_paths']) != 128):
            raise ValueError('Incomplete retained feedback pool: ' + task)
        result_relative = f'tasks/{task}/RESULT.json'
        if checked(result_relative) != value:
            raise ValueError('Pool manifest and result differ: ' + task)
        pools[task] = dict(manifest=old['root'] + '/' + relative,
            manifest_identity=manifest[relative], result=old['root'] + '/' + result_relative,
            result_identity=manifest[result_relative], binding=old['binding'], execution_commit=PR51_COMMIT,
            own_checkpoint=value['own_checkpoint'], own_weights_identity=value['own_weights_identity'],
            feature_paths=value['feature_paths'])
    for arm in ('N', 'S'):
        for event in (5, 6):
            task = f'{arm}{event}'
            relative = f'tasks/{task}/RESULT.json'
            result = checked(relative)
            event_relative = f'tasks/{task}/trainer/events/event_{event:03d}/RESULT.json'
            event_receipt = checked(event_relative)
            endpoint = result['endpoint']
            if (result['status'] != 'completed' or result['arm'] != arm or result['event'] != event
                    or result['binding'] != old['binding'] or result['execution_commit'] != PR51_COMMIT
                    or endpoint['global_step'] != event * 128 or endpoint['presentations'] != event * 256
                    or result['boundary']['status'] != 'passed' or result['boundary']['export_exact'] is not True):
                raise ValueError('Retained control is not a complete checked boundary: ' + task)
            pool = pools['common5' if event == 5 else arm + '_collect6']
            if (event_receipt['status'] != 'completed' or event_receipt['response_presentations'] != 256
                    or event_receipt['optimizer_steps'] != 128 or len(event_receipt['records']) != 256
                    or [r['path'] for r in event_receipt['features']] != pool['feature_paths']):
                raise ValueError('Control did not use its declared original feedback pool: ' + task)
            parent = old['origin_state'] if event == 5 else endpoints[arm + '5']['endpoint']
            if result['parent'] != parent:
                raise ValueError('Control continuation parent differs: ' + task)
            config_relative = f'tasks/{task}/trainer/events/event_{event:03d}/export/config.json'
            checked(config_relative)
            endpoints[task] = dict(endpoint=deepcopy(endpoint), result=old['root'] + '/' + relative,
                result_path=old['root'] + '/' + relative, result_identity=manifest[relative],
                event_receipt_path=old['root'] + '/' + event_relative,
                event_receipt_identity=manifest[event_relative], source_binding=old['binding'])
            expected[endpoint['state']] = endpoint['state_identity']
            expected[endpoint['export'] + '/model.safetensors'] = endpoint['weights_identity']
            expected[endpoint['export'] + '/config.json'] = manifest[config_relative]
    for task, pool in pools.items():
        producer = old if task == 'common5' else endpoints[task[0] + '5']['endpoint']
        checkpoint = producer['origin_export'] if task == 'common5' else producer['export']
        weights = producer['origin_weights_identity'] if task == 'common5' else producer['weights_identity']
        if pool['own_checkpoint'] != checkpoint or pool['own_weights_identity'] != weights:
            raise ValueError('Own feedback producer differs: ' + task)
    packet_metadata = {}
    for feature in pools['common5']['feature_paths']:
        relative = Path(feature).relative_to(old['root'])
        name = relative.stem
        sidecar_relative = str(relative.with_suffix('.json'))
        equivalence_relative = f'tasks/common5/preparation_retirement/{name}.equivalence.json'
        record_relative = f'tasks/common5/records/{name}.json'
        sidecar, equivalence, record = (checked(path) for path in
                                       (sidecar_relative, equivalence_relative, record_relative))
        if (sidecar['record_id'] != name or sidecar['feature_path'] != feature
                or sidecar['input_ids'] != record['prompt_ids'] + record['response_ids']
                or sidecar['response_only_loss'] is not True
                or equivalence['prepared_packet'] != feature or equivalence['exact_decoded_rows'] is not True
                or record['checkpoint_id'] != old['origin_export']):
            raise ValueError('Retained common-E5 token/producer/sidecar binding differs: ' + feature)
        packet_metadata[feature] = {kind: dict(path=old['root'] + '/' + path, identity=manifest[path])
            for kind, path in (('sidecar', sidecar_relative), ('equivalence', equivalence_relative),
                               ('record', record_relative))}
    pools['common5']['packet_metadata'] = packet_metadata
    for relative, file_identity in consumed.items():
        expected[old['root'] + '/' + relative] = file_identity
    common = pools['common5']
    return dict(status='COMPACT_PROVENANCE_VERIFIED_LIVE_AUTHENTICATION_REQUIRED',
        plan=old, delivery=str(delivery), delivery_manifest_identity=identity(manifest_path),
        common_e5=common, retained_endpoints=endpoints,
        retained_pools={k: v for k, v in pools.items() if k != 'common5'},
        expected_existing_files=expected, compact_files=consumed,
        historical_feature_content_hashes_present=False)


def require_asset_authentication(assets, receipt):
    """Fresh pins alone never satisfy the handoff's historical feature proof."""
    if (receipt.get('status') != 'passed'
            or receipt.get('historical_feature_identity_verified') is not True):
        raise ValueError('Historical common-E5 feature content identity is unverified')
    files = receipt['files']
    for path, expected in assets['expected_existing_files'].items():
        if files.get(path) != expected:
            raise ValueError('Missing or changed live retained asset identity: ' + path)
    for path in assets['common_e5']['feature_paths']:
        value = files.get(path)
        if (not isinstance(value, dict) or set(value) != {'bytes', 'sha256'}
                or type(value['bytes']) is not int or value['bytes'] <= 0
                or not isinstance(value['sha256'], str) or len(value['sha256']) != 64):
            raise ValueError('Missing authenticated common-E5 packet: ' + path)
    return files


def load_inputs(plan, *, package_root=None):
    values = {}
    for key in INPUTS:
        remote = Path(plan[key])
        path = remote if package_root is None else Path(package_root) / remote.relative_to(plan['root'])
        if identity(path) != plan['input_identities'].get(str(remote)):
            raise ValueError('Unbound or changed packaged input: ' + key)
        values[key] = read(path)
    validate_manifests(values['data_panel'], values['warmups'], values['adaptation'])
    provenance = values['source_provenance']
    exposure = values['exposure_inventory']
    if (provenance['status'] != 'passed' or provenance['ancestry_count'] != 768
            or provenance['selected_ancestry_content_overlap'] != 0
            or provenance['selected_PR51_heldout_overlap'] != 0
            or exposure['status'] != 'passed' or exposure['missing_dependencies']):
        raise ValueError('Exposure or complete ancestry provenance is unresolved')
    seal = values['data_seal']
    if seal['status'] != 'passed' or seal['first_model_work_before_seal'] is not False or not seal['input_only_eligibility']:
        raise ValueError('Input-only seal is incomplete')
    for name, expected in seal['files'].items():
        if plan['input_identities'].get(plan['root'] + '/sealed/' + name) != expected:
            raise ValueError('Original data seal identity differs: ' + name)
    train = values['adaptation']['train_requests']
    for event, offset in ((5, 0), (6, 128)):
        part = values[f'event{event}_inputs']
        if (part['event'] != event or part['start_ordinal'] != 512 + offset
                or part['stop_ordinal'] != 640 + offset or part['requests'] != train[offset:offset + 128]):
            raise ValueError('Original ordered continuation slice differs')
    schedule = values['evaluation_schedule']
    if (schedule['activations'] != evaluation_schedule(values['data_panel']['requests'])
            or schedule['scored_executions'] != 2304 or schedule['warmup_executions'] != 72):
        raise ValueError('Fixed three-arm schedule differs')
    return values


def input_readiness(plan, *, package_root=None):
    load_inputs(plan, package_root=package_root)
    return dict(status='passed', scored_inputs=256, warmup_inputs=2, adaptation_inputs=256,
        new_canonical_generations=258, before_worker_construction=True, model_work=0)


def reference_readiness(plan, *, package_root=None):
    for key in ('references', 'warmup_references'):
        remote = Path(plan[key])
        path = remote if package_root is None else Path(package_root) / remote.relative_to(plan['root'])
        if path.exists():
            raise ValueError('New-panel canonical generation has already been attempted')
    return input_readiness(plan, package_root=package_root)


def build_archive(plan, package_root, archive_path):
    ready = reference_readiness(plan, package_root=package_root)
    if read(Path(package_root) / 'sealed/PLAN.json') != plan:
        raise ValueError('Archive plan differs from inspected plan')
    with tarfile.open(archive_path, 'x:gz') as archive:
        for path in sorted(Path(package_root).rglob('*')):
            if path.is_symlink():
                raise ValueError('Input package must contain regular files only')
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(package_root)), recursive=False)
    return dict(status='passed', archive_identity=identity(archive_path), readiness=ready)


def unpack_archive(archive_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, 'r:gz') as archive:
        if any(not (m.isfile() or m.isdir()) for m in archive.getmembers()):
            raise ValueError('Input package contains a link or special file')
        archive.extractall(destination, filter='data')
    plan = read(destination / 'sealed/PLAN.json')
    return dict(plan=plan, readiness=input_readiness(plan, package_root=destination))


def build(workspace, destination, *, remote_root, source_root, execution_commit, asset_authentication=None):
    import yaml
    from .state import validate_config_text
    workspace, destination = Path(workspace), Path(destination)
    task = workspace / 'research' / TASK_NAME
    assets = authenticated_retained_assets(workspace)
    authentication = Path(asset_authentication) if asset_authentication is not None else task / 'ASSET_AUTHENTICATION.json'
    if not authentication.is_file():
        blocked_readiness(destination, 'Missing live and historical common-E5 asset authentication: ' + str(authentication))
    try:
        authenticated = require_asset_authentication(assets, read(authentication))
    except (ValueError, KeyError) as error:
        blocked_readiness(destination, str(error))
    old = assets['plan']; delivery = Path(assets['delivery'])
    data = task / 'data_readiness/sealed'; data_seal = read(data / 'DATA_SEAL.json')
    for name, expected in data_seal['files'].items():
        if identity(data / name) != expected:
            raise ValueError('Selected input changed after sealing: ' + name)
    destination.mkdir(parents=True, exist_ok=False); sealed = destination / 'sealed'; sealed.mkdir()
    inherited = ('target', 'draft', 'training_python', 'worker_python', 'origin_state', 'origin_export',
        'origin_state_identity', 'origin_weights_identity', 'origin_binding', 'devices', 'qualified_compilers',
        'host_resource_policy', 'runtime_contract', 'compiler_root', 'compiler_growth_reserved_bytes',
        'checkpoint_boundary_allowance_s', 'final_activation_allowance_s')
    plan = {key: deepcopy(old[key]) for key in inherited}
    plan.update(schema='unchanged_c_mathqa_transfer_v1', root=remote_root, source_root=source_root,
        execution_commit=execution_commit, source_manifest=remote_root + '/SOURCE.json',
        methods=['N','C','S'], events=[5,6], configs={}, limits=dict(LIMITS),
        input_identities=deepcopy(assets['expected_existing_files']), automatic_retry=False,
        common_e5=deepcopy(assets['common_e5']), retained_endpoints=deepcopy(assets['retained_endpoints']),
        retained_pools=deepcopy(assets['retained_pools']), pr51_binding=old['binding'],
        pr51_execution_commit=PR51_COMMIT, compiler_baseline_allocated_bytes=0,
        allocation_roots=[remote_root,source_root], reference_reuse=False, native_control_reuse=True,
        new_reference_generations=258,
        expected_work=dict(adam=268,response_forward=544,response_backward=544,training_initialization=4,
            feedback_generation=128,teacher_preparation=128,canonical_generation=258,
            qualification_generation=8,warmup_generation=72,scored_generation=2304),
        expected_training_initializations=dict(qualification=2,production=2),
        estimated_seconds=dict(training_A=180.,training_B=180.,serving_A=45.,serving_B=45.,
            references=420.,C5=600.,C_collect6=250.,C6=600.,final=2400.),
        estimate_basis='Initial conservative PR51-based values; measured qualification updates admission')
    plan['common_e5']['feature_identities'] = {path: authenticated[path] for path in plan['common_e5']['feature_paths']}
    plan['input_identities'].update(plan['common_e5']['feature_identities'])
    def copy(source, name, key=None):
        shutil.copyfile(source, sealed / name)
        remote = remote_root + '/sealed/' + name
        if key:
            plan[key] = remote
        plan['input_identities'][remote] = identity(sealed / name)
        return remote
    for key, name in INPUTS.items():
        copy(data / name, name, key)
    for name, key in (('ORIGINAL_CONFIG.yaml','origin_config'),('MODEL_PIN.json','model_pin'),
            ('RUNTIME_PIN.json','serving_pin'),('TRAINING_PIN.json','training_pin'),
            ('ORIGIN_STATUS.json','origin_status'),('ORIGIN_DRAFT_CONFIG.json','origin_draft_config'),
            ('ORIGIN_TRAINING.json','expected_ancestry'),('ORIGIN_REMOTE_CHECK.json','origin_remote_check'),
            ('HISTORICAL_PLAN.json','historical_plan'),('HISTORICAL_SEAL.json','historical_seal')):
        source = delivery / 'sealed' / name
        manifest = read(delivery / 'RETRIEVAL_MANIFEST.json')
        if identity(source) != manifest['sealed/' + name]:
            raise ValueError('Original configuration or provenance changed: ' + name)
        copy(source, name, key)
    copy(delivery / 'sealed/PLAN.json', 'PR51_PLAN.json', 'pr51_plan')
    copy(delivery / 'RETRIEVAL_MANIFEST.json', 'PR51_RETRIEVAL_MANIFEST.json', 'pr51_manifest')
    remote_auth = copy(authentication, 'ASSET_AUTHENTICATION.json')
    plan['asset_authentication'] = dict(path=remote_auth, identity=identity(authentication))
    original_text = (sealed / 'ORIGINAL_CONFIG.yaml').read_text()
    config = yaml.safe_load(original_text)
    config.update(run_id='unchanged-c-mathqa-transfer-C', output_dir=remote_root + '/training/C')
    config['training']['learning_rate'] = 3e-4
    text = yaml.safe_dump(config, sort_keys=False)
    validate_config_text(text, 'C', original_text)
    (sealed / 'config_C.yaml').write_text(text)
    plan['configs']['C'] = remote_root + '/sealed/config_C.yaml'
    plan['input_identities'][plan['configs']['C']] = identity(sealed / 'config_C.yaml')
    plan['references'] = remote_root + '/references/CANONICALS.json'
    plan['warmup_references'] = remote_root + '/references/WARMUP_CANONICALS.json'
    geometry = read(sealed / 'ORIGIN_DRAFT_CONFIG.json')
    lengths = [len(r['prompt_ids']) for r in read(sealed / 'EVENT6_INPUTS.json')['requests']]
    plan['storage'] = storage_plan(lengths, weight_bytes=old['origin_weights_identity']['bytes'],
        recovery_bytes=old['origin_state_identity']['bytes'], hidden_size=geometry['hidden_size'],
        captured_layers=len(geometry['dflash_config']['target_layer_ids']))
    plan['binding'] = plan_binding(plan)
    _write(sealed / 'PLAN.json', plan)
    _write(sealed / 'SEAL.json', dict(binding=plan['binding'], execution_commit=execution_commit,
        input_identities=plan['input_identities']))
    _write(destination / 'INPUT_READINESS.json', reference_readiness(plan, package_root=destination))
    _write(destination / 'READINESS.json', dict(status='CPU_PREPARED_REMOTE_PREFLIGHT_PENDING',
        execution_commit=execution_commit, binding=plan['binding'], origin_event=4,
        native_control_reuse=True, scored_requests=256,warmup_requests=2,canonical_generations=258,
        storage=plan['storage'], model_work=0, gpu_qualified=False))
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace');parser.add_argument('destination')
    for name in ('remote-root','source-root','execution-commit'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--asset-authentication')
    args=parser.parse_args()
    try:
        plan=build(args.workspace,args.destination,remote_root=args.remote_root,source_root=args.source_root,
            execution_commit=args.execution_commit,asset_authentication=args.asset_authentication)
    except ReadinessBlocked as error:
        print(json.dumps(dict(status='blocked',reason=str(error),model_work=0)))
        raise SystemExit(2)
    print(json.dumps(dict(status='prepared',binding=plan['binding'],storage=plan['storage'])))


if __name__ == '__main__':
    main()
