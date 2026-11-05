"""Add OSD to the sealed screen: reuse feedback, train O, and freshly compare F/O.

The original campaign is read-only. Its C arm supplies sampling/RNG controls,
not timing measurements. The first complete group qualifies this exact run;
null or harmful results do not change the remaining groups.
"""
from pathlib import Path
import argparse
import json
import shutil
import time

import torch
import yaml
from accelerate.utils import set_seed

from online_feedback.native import NativeDFlashBackend
from online_feedback.qualify import assert_state_equal
from online_feedback.storage import write_json

from .checks import (
    BLOCK, CAP, EXPECTED_LRS, EXPORT, FEATURE_ALLOWANCE, FULL_STATE, RESERVE,
    check_frozen, check_pair, compare_learning, frozen_versions,
    validate_learning, validate_plan,
)
from .report import _validate
from .run import generate, run_arm


def check_frozen_reproduction(actual, retained):
    check_pair(actual, retained)
    if actual['metadata']['verification_calls'] != retained['metadata']['verification_calls']:
        raise ValueError(f'{actual["record_id"]}: fresh F verification calls differ from retained F')


def check_feature(path, canonical):
    """Bind the reused teacher features to the original response, including its mask."""
    feature = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    expected_ids = torch.tensor([canonical['prompt_ids'] + canonical['response_ids']])
    assert feature['record_id'] == canonical['record_id']
    assert torch.equal(feature['input_ids'], expected_ids)
    mask = feature['loss_mask']
    expected_mask = torch.zeros_like(mask)
    expected_mask[:, len(canonical['prompt_ids']):] = 1
    assert torch.equal(mask, expected_mask), 'Reused feature must supervise only its response'
    assert feature['target_last_hidden_states'].shape[:2] == expected_ids.shape


def load_source_group(source, group):
    directory = source / group['name']
    canonical, features = {}, []
    for request in group['requests']:
        record = json.loads((directory / 'F/records' / f'{request["request_id"]}.json').read_text())
        errors, _ = _validate(record, request, CAP)
        if errors:
            raise ValueError(f'Invalid retained F {request["request_id"]}: {errors}')
        canonical[request['request_id']] = record
        if request['role'] == 'A':
            path = directory / 'F/features' / f'{request["request_id"]}.pt'
            check_feature(path, record)
            features.append(path)
    control = json.loads((directory / 'C/TRAINING.json').read_text())
    ids = [request['request_id'] for request in group['requests'][:8]]
    validate_learning(control, ids, control['actual_learning_rates'])
    checks = json.loads((directory / 'C/CHECKS.json').read_text())
    assert checks['initial_released_equal'] and checks['fresh_optimizer']
    assert checks['publication_equal'] and checks['outputs_equal']
    for name in ('initial_rng.pt', 'final_rng.pt'):
        if not (directory / 'C' / name).is_file():
            raise FileNotFoundError(directory / 'C' / name)
    return canonical, features, control


def match_training(control, source_directory, receipt, directory):
    compare_learning(control, receipt)
    for name in ('initial_rng.pt', 'final_rng.pt'):
        retained = torch.load(source_directory / 'C' / name, map_location='cpu', weights_only=False)
        actual = torch.load(directory / name, map_location='cpu', weights_only=False)
        assert_state_equal(retained, actual, 'historical_C_' + name)
    write_json(directory.parent / 'TRAINING_MATCH.json', {
        'status': 'passed', 'reference_arm': str(source_directory / 'C'),
        'matched_anchors': True, 'matched_exposure': True, 'matched_learning_rates': True,
        'matched_initial_rng': True, 'matched_final_rng': True,
        'supervised_positions': receipt['exposure']['total_supervised_positions'],
    })


def admit_followup_storage(root):
    # Source features already exist. Retain three exports and one rolling Adam state.
    required = FULL_STATE + 3 * EXPORT + FEATURE_ALLOWANCE + RESERVE
    free = shutil.disk_usage(root).free
    write_json(root / 'storage/initial.json', {
        'free_bytes': free, 'required_additional_bytes': required, 'reserve_bytes': RESERVE,
        'rolling_full_state_bytes': FULL_STATE, 'retained_export_bytes': 3 * EXPORT,
        'metadata_and_staging_allowance': FEATURE_ALLOWANCE, 'source_features_reused': True,
    })
    if free < required:
        raise ValueError(f'Storage admission failed: {free} < {required}')


def training_counts(root, groups):
    """Keep partial completed work visible if training or a later check fails."""
    steps, presentations = 0, 0
    for group in groups:
        directory = root / group['name'] / 'O/trainer/events/update'
        for name in ('RESULT.json', 'FAILED.json'):
            path = directory / name
            if path.exists():
                receipt = json.loads(path.read_text())
                counters = receipt.get('counters', receipt)
                steps += counters['global_step']
                presentations += counters['presentations']
                break
    return steps, presentations


def run(args):
    root, source = Path(args.output).resolve(), Path(args.source).resolve()
    if root == source or source in root.parents:
        raise ValueError('Follow-up output must be outside the immutable source campaign')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'STATUS.json').exists():
        raise FileExistsError('Existing follow-up status; no automatic replay/retraining')
    manifest = json.loads(Path(args.manifest).read_text())
    retained_manifest = json.loads((source / 'manifest.json').read_text())
    validate_plan(manifest, yaml.safe_load(Path(args.config).read_text()), args.target, args.draft)
    groups = manifest['groups']
    for collection in (groups, retained_manifest['groups']):
        for group in collection:
            for index, request in enumerate(group['requests'], start=1):
                request['index'] = index
    assert groups == retained_manifest['groups'], 'Use the unchanged sealed request set'
    write_json(root / 'manifest.json', manifest)
    admit_followup_storage(root)
    sources = {group['name']: load_source_group(source, group) for group in groups}
    write_json(root / 'SOURCE_INVENTORY.json', {
        'source_campaign': str(source), 'groups': {
            name: {'feedback_features': [str(path) for path in item[1]],
                   'canonical_records': len(item[0]),
                   'source_supervised_positions': item[2]['exposure']['total_supervised_positions']}
            for name, item in sources.items()},
        'exact_feedback_token_and_mask_checks': True,
    })
    write_json(root / 'PLAN.json', {
        'source_campaign': str(source), 'target': args.target, 'draft': args.draft,
        'config_template': args.config, 'groups': [group['name'] for group in groups],
        'budget': {'generations': 96, 'optimizer_steps': 24, 'response_presentations': 48},
        'first_group_qualification_included': True, 'objective': 'teacher_forward_kl',
        'temperature': 1.0, 'hard_label_mixture': 0.0, 'block_size': BLOCK, 'output_cap': CAP,
        'context_limit': 2048, 'prompt_cap': 1024, 'seed': 17, 'greedy': True, 'concurrency': 1,
        'loss_reduction': 'response_mean_before_accumulation',
        'expected_used_learning_rates': EXPECTED_LRS,
        'fresh_timing_arms': ['F', 'O'], 'timing_repeats': 1,
        'compact_verification_trace_enabled': True, 'feedback_capture_in_timed_generation': False,
        'cache_policy': 'fresh_target_and_draft_per_request', 'extra_generation_warmups': 0,
        'feedback': 'reuse_exact_original_F_A_features; no_new_teacher_preparation',
        'retention': 'keep_exports; retire_completed_full_state_only_after_valid_native_replays',
    })
    status = {'status': 'running', 'completed_groups': [], 'generations_completed': 0,
              'new_generations_completed': 0, 'training_events_completed': 0,
              'started_unix': time.time()}
    write_json(root / 'STATUS.json', status)

    def completed_generation():
        status['generations_completed'] += 1
        status['new_generations_completed'] += 1
        write_json(root / 'STATUS.json', status)

    def completed_training():
        status['training_events_completed'] += 1
        write_json(root / 'STATUS.json', status)

    from .osd_report import build_report
    started = time.perf_counter()
    try:
        set_seed(17)
        backend = NativeDFlashBackend(args.target, args.draft, context_limit=2048, block_size=BLOCK)
        status['native_loading_s'] = backend.loading_s
        target_versions = frozen_versions(backend.target)
        released = {name: tensor.detach().cpu().clone() for name, tensor in backend.draft.state_dict().items()}
        for group in groups:
            name = group['name']
            group_root = root / name
            group_root.mkdir()
            canonical, features, control = sources[name]
            status.update(current_group=name, current_stage='frozen_generation')
            write_json(root / 'STATUS.json', status)
            if backend.checkpoint_id != str(Path(args.draft)):
                write_json(group_root / 'F_RESET.json', backend.publish(args.draft))
            reproduced = []
            for request in group['requests']:
                record = generate(backend, request, group_root / 'F', arm='F', group=name,
                                  canonical=canonical[request['request_id']],
                                  completed_generation=completed_generation)
                check_frozen_reproduction(record, canonical[request['request_id']])
                reproduced.append({'request_id': request['request_id'], 'status': 'passed',
                                   'verification_calls': record['metadata']['verification_calls']})
                write_json(group_root / 'F_REPRODUCTION.json', {'records': reproduced})
            status['current_stage'] = 'O_training_and_evaluation'
            write_json(root / 'STATUS.json', status)
            _, checks = run_arm(
                backend, released, group, 'O', features, canonical, root, args.config,
                completed_training, completed_generation,
                training_check=lambda receipt, directory: match_training(control, source / name, receipt, directory),
            )
            check_frozen(backend.target, target_versions)
            write_json(group_root / 'QUALIFICATION.json', {
                'status': 'passed', 'frozen_reproduction': True, 'target_frozen': True,
                'matched_anchors': True, 'matched_exposure': True,
                'matched_initial_rng': True, 'matched_final_rng': True,
                'arms': {'O': checks}, 'counts': {'generations': 32, 'presentations': 16, 'steps': 8},
            })
            status['completed_groups'].append(name)
            write_json(root / 'STATUS.json', status)
            build_report(root)
        assert status['generations_completed'] == 96 and status['training_events_completed'] == 3
        assert training_counts(root, groups) == (24, 48)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        steps, presentations = training_counts(root, groups)
        status.update(optimizer_steps_completed=steps, response_presentations_completed=presentations,
                      wall_s=time.perf_counter() - started, finished_unix=time.time())
        write_json(root / 'STATUS.json', status)
        build_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'manifest', 'target', 'draft', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
