"""CPU-only seal for the qualified, bounded PR59-parent event-two continuation.

No model is constructed and no phase clock is created. Live idle-card/storage
admission remains the coordinator's responsibility, before collection and again
after the actual shared feature pool has been measured.
"""

import argparse
from copy import deepcopy
import difflib
import json
import math
from pathlib import Path
import shutil
import time

import yaml

from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.contract import digest, validate_manifest
from experiments.opencode_native_comparison.evaluation import check_reference
from experiments.repair_conditional_kl_pilot.evaluation import frozen_reference
from experiments.repair_conditional_kl_pilot.resources import read, verify_inputs
from online_feedback.storage import write_json

from .allocation_contract import numerical_contract
from .allocation_prepare import PARENT_EXPORT_IDENTITY
from .fork import ARMS
from .native import select_parent
from .prepare import INPUT_IDENTITY
from .qualify import STOP_GRACE_S, verify_source

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PIN_NAMES = (
    'MODEL_PIN.json',
    'RUNTIME_PIN.json',
    'TRAINING_PIN.json',
    'ORIGINAL_CONFIG.yaml',
)


def same(path, expected, label):
    if identity(path) != expected:
        raise ValueError(label + ' changed: ' + str(path))


def source_continuity(source, integrated, *, source_root=None, integrated_root=None):
    """Keep integrated science exact; admit only the reviewed three reporting lines."""
    reporting_path = 'experiments/frontier_preservation_pilot/allocation_training.py'
    reporting_changes = []
    for relative, expected in integrated['files'].items():
        if source['files'].get(relative) != expected:
            if (
                relative != reporting_path
                or source_root is None
                or integrated_root is None
            ):
                raise ValueError(
                    'Production changed an integrated source file: ' + relative
                )
            before_path, after_path = (
                Path(integrated_root) / relative,
                Path(source_root) / relative,
            )
            same(before_path, expected, 'Integrated production wrapper')
            same(after_path, source['files'][relative], 'Current production wrapper')
            before, after = before_path.read_text(), after_path.read_text()
            replacements = (
                (
                    "                fast = scorer.fast_scores(preview['w'])\n",
                    "                fast = scorer.fast_scores(preview['w'])\n                torch.cuda.synchronize()\n",
                ),
                (
                    "                    pass_index=pass_index)\n                diagnostics.update(total_window_s=time.perf_counter()-started,\n",
                    "                    pass_index=pass_index)\n                work.value['native_adam_steps'] = 1\n                work.save()\n                diagnostics.update(total_window_s=time.perf_counter()-started,\n",
                ),
            )
            approved = before
            for old, new in replacements:
                if approved.count(old) != 1:
                    raise ValueError(
                        'Reviewed production reporting patch no longer applies exactly'
                    )
                approved = approved.replace(old, new)
            if after != approved:
                raise ValueError(
                    'Production wrapper differs beyond the reviewed reporting patch'
                )
            reporting_changes.append(
                dict(
                    path=relative,
                    integrated_identity=expected,
                    production_identity=source['files'][relative],
                    scope='GPU timing synchronization and durable successful-step accounting only; ForkTrainer not executed by integration',
                    diff=''.join(
                        difflib.unified_diff(
                            before.splitlines(True),
                            after.splitlines(True),
                            fromfile='integrated/' + relative,
                            tofile='production/' + relative,
                        )
                    ),
                )
            )
    return dict(
        status='passed',
        unchanged_integrated_files=len(integrated['files']) - len(reporting_changes),
        integrated_commit=integrated['commit'],
        production_commit=source['commit'],
        reviewed_reporting_changes=reporting_changes,
    )


def clone_parent_config(text, arm, output):
    if arm not in ARMS:
        raise ValueError('Require an explicitly named PAIR-enabled fork')
    original = yaml.safe_load(text)
    current = deepcopy(original)
    current['run_id'] = 'frontier-event-two-' + arm.lower()
    current['output_dir'] = str(output)
    comparable = deepcopy(current)
    for key in ('run_id', 'output_dir'):
        if key in original:
            comparable[key] = original[key]
        else:
            comparable.pop(key)
    if comparable != original:
        raise ValueError('Fork configuration changed native scientific settings')
    return yaml.safe_dump(current, sort_keys=False)


def forecasts(requests, complete_candidate_window_s):
    if (
        len(requests) != 128
        or not math.isfinite(complete_candidate_window_s)
        or complete_candidate_window_s <= 0
    ):
        raise ValueError(
            'Require the full shared buffer and measured complete candidate-window time'
        )
    lengths = [len(request['prompt_ids']) for request in requests]
    rows = sum(length + 2048 for length in lengths) + max(lengths) + 2049
    collection_bytes = (49164 * rows * 105 + 99) // 100 + 2_000_000_000
    continuation_seconds = 1.5 * (128 * complete_candidate_window_s + 554 + 914 + 600)
    return dict(
        collection_write_bound_bytes=collection_bytes,
        continuation_write_bound_bytes=20_000_000_000,
        collection_wall_bound_s=900.0,
        continuation_wall_bound_s=continuation_seconds,
        complete_candidate_window_s=complete_candidate_window_s,
        candidate_windows=128,
        parent_control_training_reference_s=554.0,
        parent_evaluation_reference_s=914.0,
        load_witness_persistence_allowance_s=600.0,
        continuation_contingency_multiplier=1.5,
        collection_bound=dict(
            row_bytes=49164,
            token_rows_including_one_packet_overlap=rows,
            tensor_margin=0.05,
            source_log_compiler_allowance_bytes=2_000_000_000,
        ),
        continuation_storage_scope='Two final recoveries/exports plus 3.2GB overlap/compiler/log allowance; actual recovery size gate remains active',
        interpretation='Forecast from one qualified window and retained PR59 workload; source response lengths may differ',
    )


def completed_process(root):
    process = read(Path(root) / 'QUALIFICATION_PROCESS.json')
    if (
        process.get('exit_code') != 0
        or process.get('stopped_reason') is not None
        or process.get('final_storage_error') is not None
    ):
        raise ValueError('Qualification supervisor did not finish cleanly')
    for key in ('occupied_card_s', 'cumulative_occupied_card_s'):
        if not math.isfinite(process[key]) or process[key] <= 0:
            raise ValueError('Qualification occupied-card cost is incomplete')
    return process


def authenticate_qualifications(parent_plan, numerical_root, integration_root):
    numerical_root, integration_root = Path(numerical_root), Path(integration_root)
    numerical_plan, integration_plan = read(numerical_root / 'PLAN.json'), read(
        integration_root / 'PLAN.json'
    )
    numerical_path, integration_path = (
        numerical_root / 'native/RESULT.json',
        integration_root / 'native/RESULT.json',
    )
    numerical, integration = read(numerical_path), read(integration_path)
    if (
        numerical.get('status') != 'numerically_qualified'
        or numerical.get('passed') is not True
        or numerical.get('native_adam_steps') != 0
        or numerical.get('restoration', {}).get('exact_resume') is not True
    ):
        raise ValueError(
            'Complete restored zero-Adam numerical qualification is required'
        )
    if (
        integration.get('status') != 'passed'
        or integration.get('passed') is not True
        or integration.get('production_qualified') is not True
        or integration.get('candidate_active') is not True
        or integration.get('native_adam_steps') != 4
        or integration.get('restoration', {}).get('exact_resume') is not True
    ):
        raise ValueError(
            'Restored four-step native integration and attributable activity are required'
        )
    if (
        numerical_plan['parent_plan'] != str(parent_plan)
        or integration_plan['parent_plan'] != str(parent_plan)
        or integration_plan['numerical_result'] != str(numerical_path)
    ):
        raise ValueError(
            'Qualifications do not share the exact parent and numerical audit'
        )
    same(
        numerical_path,
        integration_plan['numerical_result_identity'],
        'Numerical admission',
    )
    if integration['numerical_audit_identity'] != identity(numerical_path):
        raise ValueError('Native integration result does not bind this numerical audit')
    contract_path = numerical_plan['numerical_contract']
    same(
        contract_path,
        numerical_plan['numerical_contract_identity'],
        'Numerical contract',
    )
    if read(contract_path) != numerical_contract() or numerical[
        'numerical_contract_identity'
    ] != identity(contract_path):
        raise ValueError('The prospective allocation criteria changed')
    numerical_process, integration_process = completed_process(
        numerical_root
    ), completed_process(integration_root)
    if (
        integration_plan['prior_occupied_card_s']
        < numerical_process['cumulative_occupied_card_s']
        or integration_process['prior_occupied_card_s']
        != integration_plan['prior_occupied_card_s']
        or not math.isclose(
            integration_process['cumulative_occupied_card_s'],
            integration_process['prior_occupied_card_s']
            + integration_process['occupied_card_s'],
            rel_tol=0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError('Historical-plus-new occupied-card accounting is incomplete')
    clock_path = Path(integration_plan['phase_clock'])
    clock = read(clock_path)
    if (
        clock != read(numerical_root / 'CLOCK.json')
        or clock['elapsed_limit_s'] != 14400
        or clock['occupied_card_limit_s'] != 28800
    ):
        raise ValueError('The allocation phase clock or limits changed')
    evidence = {
        str(path): identity(path)
        for path in (
            numerical_root / 'PLAN.json',
            numerical_root / 'QUALIFICATION_PROCESS.json',
            numerical_path,
            integration_root / 'PLAN.json',
            integration_root / 'QUALIFICATION_PROCESS.json',
            integration_path,
            clock_path,
            Path(contract_path),
        )
    }
    for path, expected in numerical_plan['prior_record_identities'].items():
        same(path, expected, 'Retained historical failure evidence')
        evidence[path] = expected
    for name in ('ALLOCATION_REFERENCE_LEDGER.json', 'ALLOCATION_STEP_LEDGER.json'):
        path = numerical_root.parent / name
        ledger = read(path)
        if name == 'ALLOCATION_REFERENCE_LEDGER.json':
            counts = [
                len(attempt['records']) for attempt in ledger['attempts'].values()
            ]
            valid = (
                len(counts) <= 2 and max(counts, default=0) <= 60 and sum(counts) <= 120
            )
        else:
            valid = len(ledger['steps']) <= 8 and len(ledger['fixed_windows']) <= 2
        if not valid:
            raise ValueError('A cumulative qualification allowance was exceeded')
        evidence[str(path)] = identity(path)
    return dict(
        numerical_plan=numerical_plan,
        integration_plan=integration_plan,
        numerical_path=str(numerical_path),
        integration_path=str(integration_path),
        integration=integration,
        clock_path=str(clock_path),
        clock=clock,
        prior_card_s=integration_process['cumulative_occupied_card_s'],
        evidence=evidence,
    )


def prepare(
    parent_plan,
    numerical_root,
    integration_root,
    root,
    source_manifest,
    execution_commit,
    local_allocated_bytes,
):
    root = Path(root)
    if root.exists():
        raise FileExistsError(
            'Production preparation needs a new root; old plans stay immutable'
        )
    if type(local_allocated_bytes) is not int or local_allocated_bytes < 0:
        raise ValueError('Local retained storage must be counted explicitly')
    parent, recovery, _ = select_parent(parent_plan)
    qualified = authenticate_qualifications(
        parent_plan, numerical_root, integration_root
    )
    numerical_plan, integration_plan = (
        qualified['numerical_plan'],
        qualified['integration_plan'],
    )
    source_manifest = Path(source_manifest)
    source = read(source_manifest)
    if source['commit'] != execution_commit:
        raise ValueError('Production source manifest commit differs')
    same(
        integration_plan['source_manifest'],
        integration_plan['source_identity'],
        'Integrated source manifest',
    )
    continuity = source_continuity(
        source,
        read(integration_plan['source_manifest']),
        source_root=SOURCE_ROOT,
        integrated_root=integration_plan['source_root'],
    )
    verify_source(
        dict(
            source_root=str(SOURCE_ROOT),
            source_manifest=str(source_manifest),
            source_identity=identity(source_manifest),
            execution_commit=execution_commit,
        )
    )
    manifest_path = Path(numerical_plan['full_input_manifest'])
    same(manifest_path, INPUT_IDENTITY, 'Full PR56 input manifest')
    manifest = read(manifest_path)
    validate_manifest(manifest)
    training = manifest['train_requests'][128:256]
    if [request['request_index'] for request in training] != list(range(128, 256)):
        raise ValueError('Training source indices must remain128..255')
    parent_inputs = read(parent['manifest'])
    if (
        parent_inputs['evaluation_requests'] != manifest['evaluation_requests'][:128]
        or parent_inputs['warmup_requests'] != manifest['warmup_requests']
    ):
        raise ValueError('The inherited held-out panel or warmups changed')
    parent_seal = Path(parent['root']) / 'sealed'
    inherited_files = {}
    for name in (
        *PIN_NAMES,
        'PAIR.yaml',
        'INPUTS.json',
        'CANONICALS.json',
        'WARMUP_CANONICALS.json',
    ):
        path = parent_seal / name
        same(path, parent['sealed_files'][str(path)], 'Inherited parent input/pin')
        inherited_files[str(path)] = identity(path)
    for name, requests in (
        ('CANONICALS.json', manifest['evaluation_requests'][:128]),
        ('WARMUP_CANONICALS.json', manifest['warmup_requests']),
    ):
        references = read(parent_seal / name)
        if set(references) != {request['request_id'] for request in requests}:
            raise ValueError(
                'Frozen-F canonical coverage differs from the authorized panel'
            )
        for request in requests:
            check_reference(
                request,
                frozen_reference(references[request['request_id']]),
                parent['canonical_provenance'],
                method='F',
            )
    parent_export_file = Path(numerical_plan['parent_export'])
    same(parent_export_file, PARENT_EXPORT_IDENTITY, 'Exact PAIR1 parent export')
    forecast = forecasts(
        training, qualified['integration']['complete_candidate_window_s']
    )
    now = time.time()
    remaining = min(
        14400 - (now - qualified['clock']['first_gpu_action_epoch']),
        28800 - qualified['prior_card_s'],
    )
    required_seconds = (
        forecast['collection_wall_bound_s'] + forecast['continuation_wall_bound_s']
    )
    time_admitted = required_seconds + STOP_GRACE_S < remaining
    root.mkdir(parents=True, exist_ok=False)
    seal = root / 'sealed'
    seal.mkdir()
    configs = {}
    for arm in ARMS:
        destination = seal / (arm + '.yaml')
        destination.write_text(
            clone_parent_config(
                Path(parent['configs']['PAIR']).read_text(),
                arm,
                root / 'arms' / arm / 'trainer',
            )
        )
        configs[arm] = str(destination)
    for name in (*PIN_NAMES, 'CANONICALS.json', 'WARMUP_CANONICALS.json'):
        shutil.copy2(parent_seal / name, seal / name)
    write_json(seal / 'SOURCE_CONTINUITY.json', continuity)
    write_json(
        seal / 'INPUT_SELECTION.json',
        dict(
            training=[
                dict(
                    local_ordinal=i,
                    source_index=row['request_index'],
                    request_id=row['request_id'],
                )
                for i, row in enumerate(training)
            ],
            evaluation_request_ids=[
                row['request_id'] for row in manifest['evaluation_requests'][:128]
            ],
            warmup_request_ids=[
                row['request_id'] for row in manifest['warmup_requests']
            ],
            full_manifest_identity=INPUT_IDENTITY,
            parent_plan_identity=identity(parent_plan),
        ),
    )
    plan = dict(
        schema='frontier_allocation_production_v1',
        root=str(root),
        production_admitted=time_admitted,
        resource_admission='pending_staged_live_coordinator_checks',
        arms=list(ARMS),
        event_id='event_002',
        bootstrap_seed=20261125,
        binding_policy='new fork metadata; immutable original PAIR1 history and native recipe',
        source_root=str(SOURCE_ROOT),
        source_manifest=str(source_manifest),
        source_identity=identity(source_manifest),
        execution_commit=execution_commit,
        parent_plan=str(parent_plan),
        parent_recovery=str(recovery),
        parent_export=str(parent_export_file.parent),
        parent_export_identity=PARENT_EXPORT_IDENTITY,
        numerical_result=qualified['numerical_path'],
        numerical_result_identity=identity(qualified['numerical_path']),
        integration_result=qualified['integration_path'],
        integration_result_identity=identity(qualified['integration_path']),
        phase_clock=qualified['clock_path'],
        phase_clock_identity=identity(qualified['clock_path']),
        prior_occupied_card_s=qualified['prior_card_s'],
        configs=configs,
        original_config=str(seal / 'ORIGINAL_CONFIG.yaml'),
        manifest=str(manifest_path),
        canonicals=str(seal / 'CANONICALS.json'),
        warmup_canonicals=str(seal / 'WARMUP_CANONICALS.json'),
        canonical_provenance=parent['canonical_provenance'],
        model_identities=parent['model_identities'],
        target=parent['target'],
        draft=parent['draft'],
        worker_python=parent['worker_python'],
        training_python=parent['training_python'],
        devices=parent['devices'],
        serving=parent['serving'],
        compiler_baseline=numerical_plan['compiler_baseline'],
        local_allocated_bytes=local_allocated_bytes,
        prior_charged_roots=list(
            dict.fromkeys(
                [
                    *numerical_plan['prior_charged_roots'],
                    str(numerical_root),
                    str(integration_root),
                    str(Path(numerical_plan['source_manifest']).parent),
                    str(Path(integration_plan['source_manifest']).parent),
                ]
            )
        ),
        limits=dict(
            elapsed_s=14400,
            occupied_card_s=28800,
            additional_allocated_bytes=40_000_000_000,
        ),
        reserve_bytes=400_000_000_000,
        forecast=forecast,
        **{
            key: forecast[key]
            for key in (
                'collection_write_bound_bytes',
                'continuation_write_bound_bytes',
                'collection_wall_bound_s',
                'continuation_wall_bound_s',
            )
        },
        expected_production_work=dict(
            feedback_generation=128,
            teacher_preparation=128,
            response_forward=512,
            response_backward=512,
            adam=256,
            scored_generation=768,
            warmup_generation=24,
        ),
        historical_hidden_state_gate='failed_retained_diagnostic_not_production_veto',
    )
    plan['sealed_files'] = {str(path): identity(path) for path in seal.iterdir()}
    plan['sealed_files'].update(qualified['evidence'])
    plan['sealed_files'].update(inherited_files)
    plan['sealed_files'].update(
        {
            str(source_manifest): identity(source_manifest),
            str(parent_plan): identity(parent_plan),
            str(manifest_path): INPUT_IDENTITY,
        }
    )
    plan['binding'] = digest(plan)
    write_json(seal / 'PLAN.json', plan)
    verification = verify_inputs(plan)
    result = dict(
        status='prepared' if time_admitted else 'time_forecast_blocked',
        plan=str(seal / 'PLAN.json'),
        execution_commit=execution_commit,
        source_verification=verification,
        inherited_pins_authenticated=list(PIN_NAMES),
        required_forecast_s=required_seconds,
        remaining_time_budget_s=remaining,
        production_admitted=time_admitted,
        live_resource_admission=False,
        model_work=0,
        new_optimizer_steps=0,
        clock_started=False,
    )
    write_json(root / 'CPU_PREFLIGHT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        'parent-plan',
        'numerical-root',
        'integration-root',
        'root',
        'source-manifest',
        'execution-commit',
    ):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--local-allocated-bytes', type=int, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
