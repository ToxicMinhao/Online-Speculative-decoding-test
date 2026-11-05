"""One bounded F/C/P screen on a sealed manifest; no retries or retuning.

This is experiment orchestration around the existing online-feedback components.
F supplies the immutable feedback pool. C/P each train independently once, then
replay both A and B. Compact verification traces remain identically enabled;
training-feature capture is absent from every timed generation.
"""
from pathlib import Path
import argparse
import gc
import json
import time

import torch
import yaml
from accelerate.utils import set_seed

from online_feedback.native import NativeDFlashBackend
from online_feedback.training import SpecForgeEventTrainer
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json, save_torch

from .checks import (
    BLOCK, CAP, EXPECTED_LRS, FEATURE_ALLOWANCE, ReferenceCheck,
    admit_storage, check_frozen, check_pair, compare_learning, exposure,
    frozen_versions, retired_state_metadata, validate_learning, validate_plan,
)
from .report import build_report


def generate(backend, request, destination, *, arm, group, completed_generation, canonical=None):
    # Exact rendered IDs are checked on the native tokenizer before any generation.
    assert backend.render(request['messages']) == request['prompt_ids']
    assert len(request['prompt_ids']) <= 1024
    purpose = 'serving' if arm == 'F' and request['role'] == 'A' else 'evaluation'
    record = backend.generate(
        request['prompt_ids'], max_new_tokens=CAP, request_id=request['request_id'],
        session_id=group, request_index=request['index'], purpose=purpose,
        save_to=destination / 'records' / f'{request["request_id"]}.json',
    )
    # A completed execution still consumes the budget when output validation fails.
    completed_generation()
    if canonical is not None:
        check_pair(record, canonical)
    return record

def run_arm(backend, released_state, group, arm, feature_paths, canonical, root, base_config,
            completed_training, completed_generation, *, training_check=None):
    directory = root / group['name'] / arm
    directory.mkdir(parents=True)
    admit_storage(root, f'{group["name"]}_{arm}')
    config = yaml.safe_load(Path(base_config).read_text())
    config['training']['max_steps'] = 8
    config['run_id'] = f'projected-target-{group["name"]}-{arm}'
    config['output_dir'] = str(directory / 'trainer')
    config['data']['hidden_states_path'] = str(Path(feature_paths[0]).parent)
    config_path = directory / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    objective = {'C': 'hard_ce', 'P': 'projected', 'O': 'osd'}[arm]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = SpecForgeEventTrainer(config_path, directory / 'trainer', objective=objective)
    loading_s = time.perf_counter() - started
    checks = {'initial_released_equal': False, 'publication_equal': False, 'outputs_equal': False}
    diagnostic_seconds = 0.0
    try:
        diagnostic_start = time.perf_counter()
        assert_state_equal(trainer.bundle.draft_model.state_dict(), released_state, 'released_initialization')
        checks['initial_released_equal'] = True
        optimizer = trainer.backend.optimizer
        assert not optimizer.optimizer.state
        assert trainer.global_step == 0 and trainer.presentations == 0
        checks['fresh_optimizer'] = True
        frozen_tables = [trainer.bundle.model.lm_head, trainer.bundle.model.embed_tokens]
        table_versions = [frozen_versions(module) for module in frozen_tables]
        trainable_ids = {id(p) for p in optimizer.model_params}
        assert all(id(p) not in trainable_ids for module in frozen_tables for p in module.parameters())
        checks['target_excluded_from_optimizer'] = True
        before_rng = capture_rng()
        save_torch(directory / 'initial_rng.pt', before_rng)
        diagnostic_seconds += time.perf_counter() - diagnostic_start
        model = trainer.bundle.model
        reference_check = ReferenceCheck(released_state, trainable_ids, checks) if arm == 'P' else None
        receipt = trainer.train_event('update', feature_paths, passes=2,
                                      reference_check=reference_check)
        if reference_check is not None:
            diagnostic_seconds += reference_check.elapsed_s
        lrs = [row['learning_rate'] for row in receipt['records'] if row['optimizer_stepped']]
        completed_training()
        receipt['actual_learning_rates'] = lrs
        write_json(directory / 'TRAINING.json', receipt)
        request_ids = [request['request_id'] for request in group['requests'] if request['role'] == 'A']
        validate_learning(receipt, request_ids, lrs)
        assert trainer.core.accumulation_remainder == 0 and model.reference is None
        receipt['exposure'] = exposure(receipt['records'], canonical)
        receipt['model_loading_s'] = loading_s
        receipt['diagnostic_validation_s'] = diagnostic_seconds
        receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        write_json(directory / 'TRAINING.json', receipt)
        save_torch(directory / 'final_rng.pt', capture_rng())
        if training_check is not None:
            training_check(receipt, directory)
        diagnostic_start = time.perf_counter()
        for module, versions in zip(frozen_tables, table_versions):
            check_frozen(module, versions)
        activation = backend.publish(receipt['export_path'])
        write_json(directory / 'ACTIVATION.json', activation)
        assert_state_equal(backend.draft.state_dict(), trainer.bundle.draft_model.state_dict(), 'published_weights')
        checks['publication_equal'] = True
        diagnostic_seconds += time.perf_counter() - diagnostic_start - activation['activation_s']
        checks['publication_validation_s'] = diagnostic_seconds - receipt['diagnostic_validation_s']
        checks['frozen_target_guard'] = 'requires_grad/optimizer_exclusion/tensor_identity_and_version'
    finally:
        trainer.close()
    del trainer, model, optimizer, frozen_tables, module, reference_check
    gc.collect()
    torch.cuda.empty_cache()
    write_json(directory / 'CHECKS.json', checks)
    for request in group['requests']:
        generate(backend, request, directory, arm=arm, group=group['name'],
                 canonical=canonical[request['request_id']], completed_generation=completed_generation)
    checks['outputs_equal'] = True
    write_json(directory / 'CHECKS.json', checks)
    # Native output comparisons finish before losing exact optimizer continuation.
    retired_state_metadata(directory / 'trainer', receipt, checks)
    return receipt, checks

def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'STATUS.json').exists():
        raise FileExistsError('Existing campaign status; no automatic replay/retraining')
    manifest = json.loads(Path(args.manifest).read_text())
    validate_plan(manifest, yaml.safe_load(Path(args.config).read_text()), args.target, args.draft)
    groups = manifest['groups']
    assert len(groups) == 3 and sum(len(group['requests']) for group in groups) == 48
    for group in groups:
        assert [request['role'] for request in group['requests']] == ['A'] * 8 + ['B'] * 8
        for index, request in enumerate(group['requests'], start=1):
            request['index'] = index
    write_json(root / 'manifest.json', manifest)
    admit_storage(root, 'initial', initial=True)
    plan = {
        'budget': {'generations': 144, 'response_presentations': 96, 'optimizer_steps': 48},
        'first_group_qualification_included': True, 'groups': [group['name'] for group in groups],
        'target': args.target, 'draft': args.draft, 'config_template': args.config,
        'block_size': BLOCK, 'output_cap': CAP, 'context_limit': 2048, 'prompt_cap': 1024,
        'greedy': True, 'seed': 17, 'margin_ratio': 1.1, 'concurrency': 1,
        'expected_used_learning_rates': EXPECTED_LRS, 'effective_warmup_steps': 0,
        'optimizer': 'AdamW with CPU FP32 masters/moments',
        'adam_betas': [0.9, 0.999], 'adam_epsilon': 1e-8, 'weight_decay': 0.0,
        'gradient_norm_clip': 1.0, 'per_group_feature_storage_allowance': FEATURE_ALLOWANCE,
        'loss_reduction': 'response_mean_before_accumulation',
        'feedback_capture_in_timed_generation': False, 'compact_verification_trace_enabled': True,
        'extra_generation_warmups': 0, 'cache_policy': 'fresh_target_and_draft_per_request',
        'retention': 'keep_all_exports_and_compact_evidence; retire_completed_full_states_only_after_valid_replay',
        'reused_frozen_first_group': None,
    }
    write_json(root / 'PLAN.json', plan)
    status = {'status': 'running', 'completed_groups': [], 'generations_completed': 0,
              'generations_reused': 0, 'new_generations_completed': 0,
              'training_events_completed': 0, 'started_unix': time.time(),
              'previous_attempt': None}
    write_json(root / 'STATUS.json', status)

    def completed_generation():
        status['generations_completed'] += 1
        status['new_generations_completed'] += 1
        write_json(root / 'STATUS.json', status)

    def completed_training():
        status['training_events_completed'] += 1
        write_json(root / 'STATUS.json', status)

    started = time.perf_counter()
    try:
        set_seed(17)
        backend = NativeDFlashBackend(args.target, args.draft, context_limit=2048, block_size=BLOCK)
        status['native_loading_s'] = backend.loading_s
        target_versions = frozen_versions(backend.target)
        released_state = {name: tensor.detach().cpu().clone() for name, tensor in backend.draft.state_dict().items()}
        for group in groups:
            group_root = root / group['name']
            group_root.mkdir()
            status.update(current_group=group['name'], current_stage='frozen_generation')
            write_json(root / 'STATUS.json', status)
            if backend.checkpoint_id != str(Path(args.draft)):
                write_json(group_root / 'F_RESET.json', backend.publish(args.draft))
            canonical = {}
            for request in group['requests']:
                record = generate(backend, request, group_root / 'F', arm='F', group=group['name'],
                                  completed_generation=completed_generation)
                canonical[request['request_id']] = record
            preparation, feature_paths = [], []
            for request in group['requests'][:8]:
                path = group_root / 'F/features' / f'{request["request_id"]}.pt'
                prepared = backend.prepare_feedback(canonical[request['request_id']], path)
                preparation.append(prepared)
                write_json(group_root / 'F/PREPARATION.json', {'records': preparation})
                if prepared['status'] != 'prepared':
                    raise ValueError('Ineligible actual response; preserve case and stop without resampling')
                feature_paths.append(path)
            arms, checks = {}, {}
            for arm in ('C', 'P'):
                status['current_stage'] = f'{arm}_training_and_evaluation'
                write_json(root / 'STATUS.json', status)
                arms[arm], checks[arm] = run_arm(backend, released_state, group, arm, feature_paths,
                                                canonical, root, args.config,
                                                completed_training, completed_generation)
                write_json(root / 'STATUS.json', status)
            compare_learning(arms['C'], arms['P'])
            for filename in ('initial_rng.pt', 'final_rng.pt'):
                left = torch.load(group_root / 'C' / filename, weights_only=False)
                right = torch.load(group_root / 'P' / filename, weights_only=False)
                assert_state_equal(left, right, 'matched_' + filename)
            check_frozen(backend.target, target_versions)
            write_json(group_root / 'QUALIFICATION.json', {
                'status': 'passed', 'matched_anchors': True, 'matched_exposure': True,
                'matched_initial_rng': True, 'matched_final_rng': True, 'arms': checks,
                'counts': {'generations': 48, 'presentations': 32, 'steps': 16},
                'frozen_target_guard': 'version-based, optimizer-disjoint; not bitwise whole-target scan',
            })
            status['completed_groups'].append(group['name'])
            write_json(root / 'STATUS.json', status)
            build_report(root)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status.update(wall_s=time.perf_counter() - started, finished_unix=time.time())
        write_json(root / 'STATUS.json', status)
        build_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'target', 'draft', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
