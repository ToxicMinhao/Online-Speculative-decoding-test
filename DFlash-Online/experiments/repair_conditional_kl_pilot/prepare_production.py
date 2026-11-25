"""Authenticate PR58 reuse and seal a fresh, possibly resource-blocked production plan.

This module never constructs a trainer, starts a GPU clock, or generates data.
The original qualification plan and preparation command remain immutable.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.kplus_full_gsm8k.storage import compiler_snapshot
from experiments.opencode_native_comparison.contract import validate_manifest
from experiments.opencode_native_comparison.evaluation import check_reference
from experiments.osd_components.prepare import verify_native_runtime
from .configuration import build_config, validate_config_text
from .contract import ARMS, BASE_COMMIT, LIMITS, SERVING, digest, validate_inputs
from .resources import read, verify_inputs, boundary

QUALIFIED_COMMIT = '79b8c076610e9b4d310169b3891cb76ac6d0f9bb'
REVIEWED_COMMIT = 'e38c942b5f607bb0b816cb05f3535a78c48110ec'
QUALIFIED_SOURCE_IDENTITY = dict(
    bytes=932583,
    sha256='49c001d90156cb835b7ac6d94b042c555accb1156a219b4d65ae004094b9169c',
)
PUBLISHED_ARTIFACTS = (
    Path(__file__).resolve().parents[2]
    / 'reports/dflash_repair_conditional_kl_20261124/ARTIFACTS.json'
)
PUBLISHED_ARTIFACTS_IDENTITY = dict(
    bytes=25405,
    sha256='7cfc21b688b38d0354fe0dbe37269c60f96bfad4f70e3491a5dfcdaea18106e2',
)
PILOT = 'experiments/repair_conditional_kl_pilot/'
# No objective, training wrapper, configuration, sampler or native dependency is
# permitted here. Publication additions are separately labelled non-runtime.
PLUMBING = frozenset(
    PILOT + name
    for name in (
        'campaign.py',
        'resources.py',
        'witness.py',
        'production.py',
        'production_storage.py',
        'prepare_production.py',
        'test_production.py',
        'test_production_storage.py',
        'test_prepare_production.py',
        'test_witness.py',
    )
)
PUBLICATION_PREFIXES = (
    'reports/dflash_repair_conditional_kl_20261124/',
    'reports/dflash_pair_one_event_20261124/',
)


def _same(path, expected):
    if identity(path) != expected:
        raise ValueError('Authenticated artifact changed: ' + str(path))


def _relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or str(path) != value:
        raise ValueError('Unsafe source-manifest path: ' + value)
    return path


def verify_source_reuse(old_manifest, new_manifest, source_root, execution_commit):
    """Compare every staged dependency, allowing only the reviewed plumbing."""
    if (
        old_manifest['commit'] != QUALIFIED_COMMIT
        or new_manifest['commit'] != execution_commit
    ):
        raise ValueError('Qualified or new source commit differs')
    root = Path(source_root)
    changed, publications = [], []
    for relative in sorted(set(old_manifest['files']) | set(new_manifest['files'])):
        path = root / _relative(relative)
        expected = new_manifest['files'].get(relative)
        if expected is None:
            raise ValueError('A qualified source file was removed: ' + relative)
        if expected.get('kind') == 'symlink':
            if not path.is_symlink() or os.readlink(path) != expected['target']:
                raise ValueError('Staged source link changed: ' + relative)
        else:
            if path.is_symlink():
                raise ValueError('Unexpected source symlink: ' + relative)
            _same(path, expected)
        if old_manifest['files'].get(relative) == expected:
            continue
        if relative in PLUMBING:
            changed.append(relative)
        elif relative == PILOT + 'README.md' or relative.startswith(
            PUBLICATION_PREFIXES
        ):
            publications.append(relative)
        else:
            raise ValueError(
                'Qualification-invalidating source difference: ' + relative
            )
    for path in root.rglob('*.py'):
        relative = str(path.relative_to(root))
        if relative not in new_manifest['files']:
            raise ValueError('Unmanifested executable source: ' + relative)
    return dict(
        status='passed',
        qualified_execution_commit=QUALIFIED_COMMIT,
        execution_commit=execution_commit,
        scientific_dependencies_unchanged=True,
        reviewed_plumbing_changes=changed,
        nonruntime_publication_changes=publications,
        staged_source_files=len(new_manifest['files']),
    )


def authenticate_qualification(root, *, source_manifest=None):
    """Read and bind actual passed evidence; reuse executes zero qualification work."""
    root = Path(root)
    # These identities are from the reviewed e38c942 publication, not hashes
    # first accepted from the current mutable run directory.
    _same(PUBLISHED_ARTIFACTS, PUBLISHED_ARTIFACTS_IDENTITY)
    published = read(PUBLISHED_ARTIFACTS)
    if published['execution_commit'] != QUALIFIED_COMMIT:
        raise ValueError('Published qualification execution identity differs')
    evidence = {str(PUBLISHED_ARTIFACTS): PUBLISHED_ARTIFACTS_IDENTITY}
    for relative, expected in published['compact_retrieval'].items():
        path = root / _relative(relative)
        _same(path, expected)
        evidence[str(path)] = expected

    def retained(relative):
        path = root / relative
        value = read(path)
        evidence[str(path)] = identity(path)
        return value

    plan = retained('sealed/PLAN.json')
    if (
        plan['binding'] != digest({k: v for k, v in plan.items() if k != 'binding'})
        or plan['execution_commit'] != QUALIFIED_COMMIT
        or plan['base_commit'] != BASE_COMMIT
        or plan['execution_mode'] != 'qualification_only'
        or plan['production_admitted']
    ):
        raise ValueError('Historical qualification plan identity is invalid')
    old_source = Path(source_manifest or plan['source_manifest'])
    _same(old_source, QUALIFIED_SOURCE_IDENTITY)
    for path, expected in plan['sealed_files'].items():
        actual = old_source if path == plan['source_manifest'] else Path(path)
        _same(actual, expected)
        evidence[str(actual)] = expected
    if plan['source_manifest'] not in plan['sealed_files']:
        raise ValueError('Historical source manifest is not authenticated')
    summary, launch = retained('QUALIFICATION.json'), retained('LAUNCH.json')
    preflight, audit = retained('CPU_PREFLIGHT.json'), retained('FINAL_CPU_AUDIT.json')
    if (
        summary.get('status') != 'passed'
        or summary.get('execution_commit') != QUALIFIED_COMMIT
        or summary.get('arms') != list(ARMS)
        or summary.get('optimizer_steps') != 2
        or summary.get('production_started') is not False
        or launch.get('status') != 'completed'
        or launch.get('execution_commit') != QUALIFIED_COMMIT
        or launch.get('automatic_retry') is not False
        or preflight.get('status') != 'passed'
        or preflight.get('model_work') != 0
        or audit.get('status') != 'passed'
        or audit.get('execution_commit') != QUALIFIED_COMMIT
        or audit.get('cuda_initialized') is not False
        or not audit.get('exact_initialization_comparison', {}).get(
            'exact_weights_empty_Adam_RNG'
        )
        or len(audit.get('raw_native_records', [])) != 8
        or not all(row.get('valid') for row in audit['raw_native_records'])
    ):
        raise ValueError('Historical combined native qualification is not passed')
    expected_work = dict(
        feedback_generation=2,
        teacher_preparation=2,
        training_initialization=1,
        response_forward=2,
        response_backward=2,
        adam=1,
        evaluation_generation=2,
    )
    for arm in ARMS:
        training = retained(f'qualification/{arm}/TRAINING_QUALIFICATION.json')
        status = retained(f'qualification_inputs/{arm}/STATUS.json')
        work = retained(f'qualification_inputs/{arm}/WORK.json')
        restoration = retained(f'qualification/{arm}/startup/EXACT_RESTORATION.json')
        if (
            training.get('status') != 'passed'
            or training.get('arm') != arm
            or training.get('execution_commit') != QUALIFIED_COMMIT
            or training.get('binding') != plan['binding']
            or training.get('global_step') != 1
            or training.get('presentations') != 2
            or training.get('exact_export_and_state_reload') is not True
            or status.get('status') != 'passed'
            or status.get('phase') != 'completed'
            or status.get('execution_commit') != QUALIFIED_COMMIT
            or status.get('canonical_and_repeat_checks') is not True
            or work.get('pending')
            or work.get('attempted') != expected_work
            or work.get('completed') != expected_work
            or restoration.get('exact_equal') is not True
        ):
            raise ValueError('Incomplete historical arm qualification: ' + arm)
        for name in (
            'EXPORT_RELOAD_0.json',
            'FROZEN_RELOAD.json',
            'EXPORT_RELOAD_1.json',
        ):
            reload = retained(f'qualification_inputs/{arm}/{name}')
            if (
                reload.get('errors') != []
                or reload.get('weights', {}).get('all_equal') is not True
            ):
                raise ValueError(
                    'Historical native reload did not pass: ' + arm + '/' + name
                )
    receipt = dict(
        status='reused_passed',
        qualified_execution_commit=QUALIFIED_COMMIT,
        reviewed_publication_commit=REVIEWED_COMMIT,
        qualified_plan_binding=plan['binding'],
        qualification_root=str(root),
        evidence=evidence,
        optimizer_steps_executed=0,
        model_forwards_executed=0,
        generations_executed=0,
        fresh_production_origin_required=True,
        qualification_states_are_not_origins=True,
        historical_cost=launch['clock'],
    )
    return plan, read(old_source), receipt


def authenticate_inputs(source_run, qualified):
    """Reuse exactly the already authenticated ordered subset and frozen-F canonicals."""
    old = Path(source_run)
    prior = read(old / 'sealed/PLAN_BUDGET_RESUME_001.json')
    if prior['execution_commit'] != BASE_COMMIT:
        raise ValueError('Require the executed PR56 recipe')
    paths = {
        name: old / 'sealed' / name
        for name in (
            'INPUTS.json',
            'ORIGINAL_CONFIG.yaml',
            'MODEL_PIN.json',
            'RUNTIME_PIN.json',
            'TRAINING_PIN.json',
        )
    }
    for path in paths.values():
        _same(path, prior['input_identities'][str(path)])
    source_inputs = read(paths['INPUTS.json'])
    validate_manifest(source_inputs)
    selected = {
        key: source_inputs[key][:count]
        for key, count in (
            ('train_requests', 128),
            ('evaluation_requests', 128),
            ('qualification_requests', 2),
            ('warmup_requests', 2),
        )
    }
    validate_inputs(selected)
    if selected != read(qualified['manifest']):
        raise ValueError('The original PR58 input subset changed')
    canonical_path = old / 'precheck/CANONICALS.json'
    warmup_path = old / 'qualification_policy_001/WARMUP_CANONICALS.json'
    for path in (
        paths['INPUTS.json'],
        paths['ORIGINAL_CONFIG.yaml'],
        canonical_path,
        warmup_path,
    ):
        _same(path, qualified['source_identities'][str(path)])
    references = read(canonical_path)
    complete = read(old / 'precheck/AR_F_PRECHECK.json')
    if complete['status'] != 'passed' or complete['canonical_digest'] != digest(
        references
    ):
        raise ValueError('Frozen-F canonical completion authentication failed')
    canonicals = {
        r['request_id']: references[r['request_id']]['F']
        for r in selected['evaluation_requests']
    }
    all_warmup = read(warmup_path)
    warmups = {
        r['request_id']: all_warmup[r['request_id']]['F']
        for r in selected['warmup_requests']
    }
    for key, table in (
        ('evaluation_requests', canonicals),
        ('warmup_requests', warmups),
    ):
        for request in selected[key]:
            check_reference(
                request,
                table[request['request_id']],
                prior['canonical_provenance'],
                method='F',
            )
    old_seal = Path(qualified['root']) / 'sealed'
    if canonicals != read(old_seal / 'CANONICALS.json') or warmups != read(
        old_seal / 'WARMUP_CANONICALS.json'
    ):
        raise ValueError('Retained canonical policy changed since PR58')
    for name in (
        'target',
        'draft',
        'worker_python',
        'training_python',
        'devices',
        'canonical_provenance',
    ):
        if prior[name] != qualified[name]:
            raise ValueError('Qualified model/runtime identity changed: ' + name)
    checked = [
        entry
        for entry in prior['authenticated_files']
        if entry['role'] in ('released_draft', 'released_target')
    ]
    if checked != qualified['model_identities'] or not checked:
        raise ValueError('Qualified released-model artifact set changed')
    for entry in checked:
        _same(entry['path'], entry['expected_identity'])
    for name in (
        'MODEL_PIN.json',
        'RUNTIME_PIN.json',
        'TRAINING_PIN.json',
        'ORIGINAL_CONFIG.yaml',
    ):
        if paths[name].read_bytes() != (old_seal / name).read_bytes():
            raise ValueError('Qualified scientific pin changed: ' + name)
    return dict(
        prior=prior,
        paths=paths,
        selected=selected,
        canonicals=canonicals,
        warmups=warmups,
        model_identities=checked,
        source_identities={
            str(path): identity(path)
            for path in (
                *paths.values(),
                canonical_path,
                warmup_path,
                old / 'precheck/AR_F_PRECHECK.json',
            )
        },
    )


def authenticate_runtime(
    worker_python, runtime_pin, training_pin, *, training_python, source_root
):
    import torch
    import transformers
    import specforge

    source_root = Path(source_root).resolve()
    if (
        Path(sys.executable).resolve() != Path(training_python).resolve()
        or not Path(__file__).resolve().is_relative_to(source_root)
        or not Path(specforge.__file__)
        .resolve()
        .is_relative_to(source_root / 'training')
    ):
        raise ValueError(
            'Run CPU preparation with the sealed training interpreter and source paths'
        )
    environment = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
    )
    if torch.cuda.is_initialized() or environment != training_pin['environment']:
        raise ValueError(
            'Training environment differs or CPU preflight initialized CUDA'
        )
    native = verify_native_runtime(worker_python, runtime_pin)
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only authentication initialized CUDA')
    return dict(
        training_environment=environment,
        native_runtime=native,
        cuda_initialized=False,
        training_python=str(Path(sys.executable).resolve()),
        specforge_source=str(Path(specforge.__file__).resolve()),
    )


def prepare(
    source_run,
    qualification_run,
    root,
    execution_commit,
    source_root,
    *,
    feature_strategy='local_single',
    feature_roots=None,
    extra_reserve_bytes=2_000_000_000,
    qualified_source_manifest=None,
    transfer_probe=None,
):
    from .production_storage import prepare_storage, filesystem

    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    if root.exists():
        raise FileExistsError('New production preparation requires an unused run root')
    qualified, old_source, reuse = authenticate_qualification(
        qualification_run, source_manifest=qualified_source_manifest
    )
    source_manifest = source_root.parent / 'SOURCE.json'
    source_check = verify_source_reuse(
        old_source, read(source_manifest), source_root, execution_commit
    )
    authenticated = authenticate_inputs(source_run, qualified)
    prior, inputs, paths = (
        authenticated['prior'],
        authenticated['selected'],
        authenticated['paths'],
    )
    runtime = authenticate_runtime(
        prior['worker_python'],
        read(paths['RUNTIME_PIN.json']),
        read(paths['TRAINING_PIN.json']),
        training_python=prior['training_python'],
        source_root=source_root,
    )
    root.mkdir(parents=True, exist_ok=False)
    seal = root / 'sealed'
    seal.mkdir()
    write_json(seal / 'INPUTS.json', inputs)
    write_json(seal / 'CANONICALS.json', authenticated['canonicals'])
    write_json(seal / 'WARMUP_CANONICALS.json', authenticated['warmups'])
    for name in (
        'MODEL_PIN.json',
        'RUNTIME_PIN.json',
        'TRAINING_PIN.json',
        'ORIGINAL_CONFIG.yaml',
    ):
        shutil.copy2(paths[name], seal / name)
    configs = {}
    for arm in ARMS:
        config = build_config(
            paths['ORIGINAL_CONFIG.yaml'].read_text(),
            arm=arm,
            output_dir=root / 'arms' / arm / 'trainer',
            maximum_events=1,
            run_id='pair-one-event-' + arm.lower(),
        )
        validate_config_text(
            config, paths['ORIGINAL_CONFIG.yaml'].read_text(), maximum_events=1
        )
        destination = seal / (arm + '.yaml')
        destination.write_text(config)
        configs[arm] = str(destination)
    feature_roots = feature_roots or {arm: str(root / 'arms' / arm) for arm in ARMS}
    if set(feature_roots) != set(ARMS):
        raise ValueError('Both own feature roots must be specified')
    feature_roots = {
        arm: str(Path(path).resolve()) for arm, path in feature_roots.items()
    }
    if len(set(feature_roots.values())) != 2:
        raise ValueError('The two own-feedback roots must be distinct')
    storage = prepare_storage(
        inputs['train_requests'],
        root=root,
        source_root=source_root,
        feature_roots=feature_roots,
        strategy=feature_strategy,
        extra_reserve_bytes=extra_reserve_bytes,
    )
    compiler_baseline = compiler_snapshot()
    main = next(group for group in storage['groups'] if group['name'] == 'main')
    for path in compiler_baseline['directories'].values():
        actual = filesystem(path)
        if (
            actual['capacity_key'] != main['filesystem']['capacity_key']
            or actual['device'] != main['filesystem']['device']
            or actual['readonly']
            or not actual['durable']
        ):
            storage['blockers'].append(
                'Compiler cache is not on admitted main storage: ' + path
            )
            storage['admitted'] = False
    reuse['source_authentication'] = source_check
    reuse['runtime_authentication'] = runtime
    reuse_path = seal / 'QUALIFICATION_REUSE.json'
    write_json(reuse_path, reuse)
    blockers = deepcopy(storage['blockers'])
    forecast = dict(
        retained_source='PR56 C event1 update589.38s + feedback154.95s per arm; first128 F maximum139.822s/pass',
        scored_passes=6,
        expected_elapsed_s=3600,
        conservative_elapsed_s=7200,
        conservative_card_s=12600,
        report_reserved_s=600,
        includes='both own events, startup, three witness readouts, graph/loading/export, 768 scores, 24 warmups and reporting',
        admitted_feature_transfer_seconds=0,
        transfer_strategy='none for local_single/local_split',
    )
    if transfer_probe is not None:
        probe = read(transfer_probe)
        write_json(seal / 'REJECTED_TRANSFER_PROBE.json', probe)
        forecast['rejected_transfer_probe'] = str(seal / 'REJECTED_TRANSFER_PROBE.json')
    plan = dict(
        schema='repair_pair_one_event_production_v1',
        root=str(root),
        source_root=str(source_root),
        source_manifest=str(source_manifest),
        base_commit=BASE_COMMIT,
        reviewed_commit=REVIEWED_COMMIT,
        execution_commit=execution_commit,
        maximum_events=1,
        production_admitted=storage['admitted'],
        execution_mode='one_event_production',
        arms=list(ARMS),
        serving=SERVING,
        limits=LIMITS,
        configs=configs,
        original_config=str(seal / 'ORIGINAL_CONFIG.yaml'),
        manifest=str(seal / 'INPUTS.json'),
        target=prior['target'],
        draft=prior['draft'],
        worker_python=prior['worker_python'],
        training_python=prior['training_python'],
        devices=prior['devices'],
        canonical_provenance=prior['canonical_provenance'],
        storage=storage,
        feature_strategy=feature_strategy,
        feature_roots=feature_roots,
        qualification_reuse=str(reuse_path),
        compiler_baseline=compiler_baseline,
        allocation_roots=[str(root), str(source_root.parent), *feature_roots.values()],
        forecast=forecast,
        production_blockers=blockers,
        source_identities=authenticated['source_identities'],
        model_identities=authenticated['model_identities'],
        canonical_policy='Every speculative output matches existing frozen-F canonical',
        source_subset='First128train and first128evaluation in retained ordered PR56 manifest',
        auxiliary_coefficient='1-mu, no tuned coefficient',
        bootstrap_seed=20261124,
        expected_production_work=dict(
            feedback_generation=256,
            teacher_preparation=256,
            response_forward=512,
            response_backward=512,
            adam=256,
            scored_generation=768,
            warmup_generation=24,
        ),
        separate_work=dict(
            startup_response_forward=4,
            startup_response_backward=4,
            startup_adam=0,
            witness_readouts=3,
            maximum_witness_draft_forwards=24,
            witness_teacher_forwards=0,
            new_qualification_optimizer_steps=0,
        ),
    )
    current_admission = None
    if storage['admitted']:
        try:
            current_admission = boundary(plan)
        except (RuntimeError, ValueError, TimeoutError) as error:
            blocker = 'CPU resource boundary rejected production: ' + repr(error)
            storage['admitted'] = plan['production_admitted'] = False
            storage['blockers'].append(blocker)
            blockers.append(blocker)
    plan['sealed_files'] = {
        str(path): identity(path) for path in seal.iterdir() if path.is_file()
    }
    plan['sealed_files'][str(source_manifest)] = identity(source_manifest)
    plan['sealed_files'].update(reuse['evidence'])
    plan['sealed_files'].update(authenticated['source_identities'])
    plan['binding'] = digest(plan)
    write_json(seal / 'PLAN.json', plan)
    source_verification = verify_inputs(plan)
    result = dict(
        status='passed' if storage['admitted'] else 'blocked',
        authentication_status='passed',
        execution_commit=execution_commit,
        source_verification=source_verification,
        qualification_status='reused_passed',
        new_qualification_optimizer_steps=0,
        input_validation=validate_inputs(inputs),
        model_artifacts_checked=len(authenticated['model_identities']),
        runtime=runtime,
        storage=storage,
        admission=current_admission,
        blockers=blockers,
        model_work=0,
        clock_started=False,
    )
    write_json(root / 'CPU_PREFLIGHT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        'source-run',
        'qualification-run',
        'root',
        'execution-commit',
        'source-root',
    ):
        parser.add_argument('--' + key, required=True)
    parser.add_argument(
        '--feature-strategy',
        choices=('local_single', 'local_split'),
        default='local_single',
    )
    parser.add_argument('--feature-root-base')
    parser.add_argument('--feature-root-pair')
    parser.add_argument('--extra-reserve-bytes', type=int, default=2_000_000_000)
    parser.add_argument('--qualified-source-manifest')
    parser.add_argument('--transfer-probe')
    args = vars(parser.parse_args())
    base, pair = args.pop('feature_root_base'), args.pop('feature_root_pair')
    if bool(base) != bool(pair):
        parser.error('Specify both feature roots together')
    if base:
        args['feature_roots'] = dict(BASE=base, PAIR=pair)
    print(json.dumps(prepare(**args), indent=2))
