"""Point3's fixed contract and whole-campaign resource admission (CPU only)."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.contract import digest
from online_feedback.storage import write_json
from .point1_prepare import (
    ARCHIVE_MANIFEST_IDENTITY, EXPORT_IDENTITY, PARENT_BINDING,
    PRODUCTION_PLAN_IDENTITY, RECOVERY_IDENTITY, WINDOW_IDENTITY,
    authenticate_witness, read_bound_json,
)

SCHEMA = 'point3_cumulative_max_v1'
BASE_COMMIT = 'a1558c0c78c543fb257d1e21889b333be1675eec'
CASES = ('C', 'F_C_SUM', 'F_C_MAX')
ARMS = CASES[1:]
HISTORICAL_CARD_S = 5790.93129579185
LIMITS = dict(elapsed_s=21600, occupied_card_s=14400,
    allocated_bytes=25_000_000_000, reserve_bytes=400_000_000_000,
    qualification_adam_calls=6, qualification_attempts=2,
    oracle_backwards_total=60, event_adam_calls_per_arm=128,
    attempted_adam_calls=262, generation_calls=1188)
CONTROL_RECOVERY = dict(bytes=6301108315,
    sha256='3f3e5075926e3f08c06958c7ab77eb0eb9bf9af2cd5b8ea2cb77aeab70750ddd')
CONTROL_EXPORT = dict(bytes=2097259104,
    sha256='b675b559c0fe2f7db4251a585a53740c7ea66868137cdf3be432c5cb6e634cc6')
FEATURE_IDENTITIES = dict(bytes=3161988,
    sha256='0f4de1defebe878ea3135a07e4570dbe364c8d7ed1538a3d289d9e1c04b07eee')


def prospective_contract():
    """No model/RNG action: declare the full experiment, including failed work."""
    return dict(schema=SCHEMA, base_commit=BASE_COMMIT, cases=list(CASES),
        arms=list(ARMS), limits=dict(LIMITS), prior_occupied_card_s=HISTORICAL_CARD_S,
        scientific_change='SUM -> cumulative MAX outer assessment only',
        reference_gradient='actual normalized native C preclip gradient; no reference Adam step',
        allocator='unchanged beta-centered, beta-budget, cap 0.25',
        assessment_tie_rule='earliest margin minimum, strict-less replacement',
        live_precision='native BF16', shadow_precision='FP32',
        diagnostic_product_reduction='FP64',
        parent_counters=[128, 256, 1], qualification_counters=[129, 258, 1],
        final_counters=[256, 512, 2], event_id='event_002',
        parent_binding=PARENT_BINDING, original_window_identity=WINDOW_IDENTITY,
        parent_recovery_identity=RECOVERY_IDENTITY, parent_export_identity=EXPORT_IDENTITY,
        full_events=2, new_feedback_generations=0, planned_qualification_adam_calls=3,
        planned_oracle_backwards=28, expected_oracle_pool_lengths=[13, 15],
        committed_presentations=512, committed_adam_calls=256,
        evaluation=dict(methods=list(CASES), groups=2, requests_per_group=64,
            repeats=3, rotation='(repeat + group) % 3', activations=18,
            warmups_per_activation=2, scored_generations=1152, warmup_generations=36,
            total_generations=1188, bootstrap_draws=10000, bootstrap_seed=20261125),
        freeze_rule='complete/valid distinct candidate: latency<C and calls<=C; lowest latency, then calls, then SUM; otherwise C',
        inherited_deletion_allowed=False, full_event_restart_allowed=False,
        point2b_executed=False, status='prospective')


def forecast():
    """Historical costs are planning evidence, not a promise for new gradients."""
    training = 2 * 2439.323 * 1.35
    components = dict(qualification_s=900., training_s=training,
        serving_s=2100., readouts_loading_persistence_s=900.)
    raw = sum(components.values())
    return dict(components=components, raw_s=raw, contingency_factor=1.25,
        forecast_s=raw * 1.25, reference='PR60 candidate 2439.323s; new C-reference route unmeasured',
        transfer_s=0., storage=dict(recovery_bytes_each=6_400_000_000,
            export_bytes_each=2_100_000_000, endpoints_bytes=17_000_000_000,
            remote_source_logs_cache_allowance_bytes=3_000_000_000,
            remote_peak_bytes=20_000_000_000, local_allowance_bytes=600_000_000,
            all_locations_peak_bytes=20_600_000_000,
            replacement_overlap='new outputs only; pending recovery rename is not a second copy'))


def resource_admission(*, filesystems, all_locations_peak_bytes,
                       host_available_bytes, gpu_free_bytes, gpu_busy,
                       forecast_s, occupied_forecast_s):
    """Each physical filesystem gets one peak; a symlink cannot add capacity."""
    if not filesystems or len({row['device'] for row in filesystems}) != len(filesystems):
        raise ValueError('List each physical filesystem once')
    original = [row for row in filesystems if row.get('role') == 'a800_original']
    if len(original) != 1 or original[0]['reserve_bytes'] != LIMITS['reserve_bytes']:
        raise ValueError('The original A800 filesystem must retain the 400GB reserve')
    if all_locations_peak_bytes < sum(row['remaining_peak_bytes'] for row in filesystems):
        raise ValueError('All-location peak omits planned filesystem writes')
    checked = []
    for row in filesystems:
        if min(row['free_bytes'], row['remaining_peak_bytes'], row['reserve_bytes']) < 0:
            raise ValueError('Negative resource budget')
        checked.append(dict(row, projected_free_bytes=row['free_bytes'] - row['remaining_peak_bytes'],
            deficit_bytes=max(0, row['reserve_bytes'] + row['remaining_peak_bytes'] - row['free_bytes']),
            passed=row['free_bytes'] - row['remaining_peak_bytes'] >= row['reserve_bytes']))
    checks = dict(filesystems=all(row['passed'] for row in checked),
        storage_cap=0 <= all_locations_peak_bytes <= LIMITS['allocated_bytes'],
        host_memory=host_available_bytes >= 55_000_000_000 + 32_000_000_000,
        gpu_memory=gpu_free_bytes >= 60_000_000_000, gpu_idle=not gpu_busy,
        elapsed=0 < forecast_s <= LIMITS['elapsed_s'],
        card_time=0 < occupied_forecast_s <= LIMITS['occupied_card_s'])
    return dict(admitted=all(checks.values()), checks=checks, filesystems=checked,
        all_locations_peak_bytes=all_locations_peak_bytes, forecast_s=forecast_s,
        occupied_forecast_s=occupied_forecast_s, host_available_bytes=host_available_bytes,
        gpu_free_bytes=gpu_free_bytes, gpu_busy=gpu_busy)


def authenticated_inputs(production_plan, point2a_root, archive_manifest):
    """Resolve all private paths from pinned receipts, including the retained C arm."""
    source = Path(__file__).resolve().parents[2]
    public = json.loads((source / 'reports/dflash_point2a_c_reference_20261125/diagnostic_001/IDENTITIES.json').read_text())
    receipts = public['raw_input_receipts']
    previous = read_bound_json(Path(point2a_root) / 'CONTRACT.json', receipts['CONTRACT.json'])
    production = read_bound_json(production_plan, PRODUCTION_PLAN_IDENTITY)
    for path, expected in production['sealed_files'].items():
        if identity(path) != expected:
            raise ValueError('Retained production input changed: ' + path)
    archive = read_bound_json(archive_manifest, ARCHIVE_MANIFEST_IDENTITY)
    entries = {row['path']: dict(bytes=row['bytes'], sha256=row['sha256']) for row in archive['files']}
    old_root = Path(production['root'])
    shared = read_bound_json(old_root/'shared/STATUS.json', entries['shared/STATUS.json'])
    features = read_bound_json(old_root/'shared/FEATURE_IDENTITIES.json', FEATURE_IDENTITIES)
    control_path = old_root/'arms/CONTROL/STATUS.json'
    control = read_bound_json(control_path, entries['arms/CONTROL/STATUS.json'])
    if (shared['status'] != 'completed' or shared['completed'] != 128
            or shared['source_indices'] != list(range(128, 256))
            or len(features) != 128 or [row['source_index'] for row in features] != list(range(128, 256))
            or control['status'] != 'completed'
            or [control['global_step'], control['presentations'], control['completed_event']] != [256,512,2]):
        raise ValueError('Retained shared pool or C endpoint is incomplete')
    for path, expected in (
        (previous['parent_recovery'], RECOVERY_IDENTITY),
        (Path(production['parent_export'])/'model.safetensors', EXPORT_IDENTITY),
        (control['recovery'], CONTROL_RECOVERY),
        (Path(control['checkpoint'])/'model.safetensors', CONTROL_EXPORT),
    ):
        if identity(path) != expected:
            raise ValueError('Protected tensor changed: ' + str(path))
    for row in features:
        if identity(row['path']) != row['identity']:
            raise ValueError('Shared prepared feedback changed: ' + row['path'])
    window = read_bound_json(previous['original_window'], WINDOW_IDENTITY)
    witness = authenticate_witness(old_root, archive_manifest, previous['parent_features'], window)
    bindings = read_bound_json(witness['binding_path'], witness['binding_identity'])
    for row in bindings['sets']['parent_training']:
        if identity(row['feature_path']) != row['feature_identity']:
            raise ValueError('Parent-training witness changed')
    controls = {}
    for current, old in (('C','C'),('F_C_SUM','F_C')):
        relative = f'attempt_001/native/{old}.json'
        path = Path(point2a_root)/relative
        read_bound_json(path, receipts[relative])
        controls[current] = dict(path=str(path), identity=receipts[relative])
    return dict(production=production, previous=previous, historical_controls=controls,
        shared_status=shared, features=features, control=control,
        retained_C_status=str(control_path), retained_C_status_identity=entries['arms/CONTROL/STATUS.json'],
        witness=witness, witness_bindings=bindings)


def prepare(*, production_plan, point2a_root, archive_manifest, root,
            source_manifest, execution_commit, admission_path):
    admission = json.loads(Path(admission_path).read_text())
    recomputed = resource_admission(**{key: admission[key] for key in (
        'filesystems','all_locations_peak_bytes','host_available_bytes',
        'gpu_free_bytes','gpu_busy','forecast_s','occupied_forecast_s')})
    if admission['checks'] != recomputed['checks'] or admission['admitted'] != recomputed['admitted']:
        raise ValueError('Resource admission summary does not match its inputs')
    if not admission['admitted']:
        raise RuntimeError('Whole Point3 campaign must be admitted before native preparation')
    from .qualify import verify_source
    import yaml
    inputs = authenticated_inputs(production_plan, point2a_root, archive_manifest)
    production, previous = inputs['production'], inputs['previous']
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    plan = prospective_contract()
    plan.update({key: deepcopy(previous[key]) for key in (
        'parent_plan','parent_plan_identity','parent_recovery','parent_export','parent_features',
        'original_probe','original_window','devices','training_python','compiler_baseline')})
    plan.update({key: deepcopy(production[key]) for key in (
        'manifest','canonicals','warmup_canonicals','canonical_provenance','target','draft',
        'worker_python','serving','model_identities','original_config')})
    configs = {}
    for arm in ARMS:
        config = yaml.safe_load(Path(production['configs']['CONTROL']).read_text())
        config.update(run_id='point3-'+arm.lower(), output_dir=str(root/'arms'/arm/'trainer'))
        path = root/'configs'/(arm+'.yaml')
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[arm] = str(path)
    plan.update(root=str(root), source_root=str(Path(__file__).resolve().parents[2]),
        execution_commit=execution_commit, source_manifest=str(source_manifest),
        source_identity=identity(source_manifest), production_plan=str(production_plan),
        production_plan_identity=PRODUCTION_PLAN_IDENTITY, point2a_root=str(point2a_root),
        historical_controls=inputs['historical_controls'], witness=inputs['witness'],
        configs=configs, shared_root=str(Path(production['root'])/'shared'),
        features=inputs['features'], retained_C_status=inputs['retained_C_status'],
        retained_C_status_identity=inputs['retained_C_status_identity'],
        retained_C_schedule=inputs['control']['training'], retained_C_checkpoint=inputs['control']['checkpoint'],
        resource_admission=admission, forecast=forecast(), created_epoch=time.time(),
        status='admitted', production_admitted=True, bootstrap_seed=20261125)
    plan['binding'] = digest(plan)
    verify_source(plan)
    write_json(root/'CONTRACT.json', plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('production-plan','point2a-root','archive-manifest','root',
                 'source-manifest','execution-commit','admission-path'):
        parser.add_argument('--'+name, required=True)
    prepared = prepare(**vars(parser.parse_args()))
    print(json.dumps(dict(status=prepared['status'], root=prepared['root'])))
