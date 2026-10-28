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
import math
import shutil
import time

import torch
import yaml
from accelerate.utils import set_seed

from online_feedback.native import NativeDFlashBackend, validate_generation
from online_feedback.training import SpecForgeEventTrainer
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json, save_torch

RESERVE = 8 * 2**30
FULL_STATE = 14_682_000_000
EXPORT = 2_101_000_000
# Eight <=1536-token BF16 records: five 4096-wide target feature layers plus
# final hidden states require <0.61 GB; 1 GiB also covers atomic staging/metadata.
FEATURE_ALLOWANCE = 2**30
CAP = 512
BLOCK = 16
# Qualified upstream nested scheduler on Torch 2.13: its second initialization
# get_lr call multiplies the initial LR by 2/(1+cos(pi/8)). Preserve that behavior.
EXPECTED_LRS = [6e-4 * (1 + math.cos(math.pi * step / 8)) / (1 + math.cos(math.pi / 8))
                for step in range(8)]


def validate_plan(manifest, config, target, draft):
    """Refuse provisional inputs or a silently different training recipe."""
    assert manifest['status'] == 'sealed_before_generation'
    assert manifest['selection_seed'] == 20261022
    groups = manifest['groups']
    assert [group['name'] for group in groups] == ['gsm8k', 'codealpaca', 'sharegpt']
    requests = [request for group in groups for request in group['requests']]
    assert len(requests) == 48
    assert len({request['request_id'] for request in requests}) == 48
    assert len({tuple(request['prompt_ids']) for request in requests}) == 48
    for group in groups:
        assert [request['role'] for request in group['requests']] == ['A'] * 8 + ['B'] * 8
    for request in requests:
        assert 0 < len(request['prompt_ids']) <= 1024
    model, training = config['model'], config['training']
    assert model['target_model_path'] == target
    assert model['draft_checkpoint_path'] == model['draft_model_config'] == draft
    assert model['torch_dtype'] == 'bfloat16' and model['use_liger_kernel'] is False
    expected = {'seed': 17, 'strategy': 'dflash', 'fsdp_sharding': 'NO_SHARD',
                'optimizer_cpu_offload': True, 'batch_size': 1, 'accumulation_steps': 2,
                'learning_rate': 0.0006, 'lr_scheduler': 'cosine', 'warmup_ratio': 0.04,
                'max_grad_norm': 1.0, 'attention_backend': 'eager', 'num_anchors': 64,
                'objective_chunk_blocks': 8, 'loss_type': 'dflash', 'lk_loss_type': None,
                'loss_decay_gamma': 7.0}
    assert all(training[key] == value for key, value in expected.items())
    assert training.get('total_steps') in (None, 8)
    assert config['data']['max_length'] == 2048


def check_pair(actual, canonical):
    for name in ('prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens'):
        if actual[name] != canonical[name]:
            raise ValueError(f'{actual["record_id"]}: native {name} differs from F')


def frozen_versions(module):
    """Cheap mutation guard; this is not a bitwise whole-target comparison."""
    return {name: (id(tensor), tensor._version, tensor.requires_grad)
            for name, tensor in module.named_parameters()}


def check_frozen(module, before):
    assert frozen_versions(module) == before, 'Frozen parameter identity/version/gradient flag changed'
    assert all(not tensor.requires_grad and tensor.grad is None for tensor in module.parameters())


def exposure(records, canonical):
    """Count each presentation independently, avoiding accumulated metric overlap."""
    rows = []
    for row in records:
        response = canonical[row['record_id']]
        prompt_length = len(response['prompt_ids'])
        sequence_length = prompt_length + len(response['response_ids'])
        positions, weight = 0, 0.0
        assert len(row['sampled_anchors']) == 1
        sampled = row['sampled_anchors'][0]
        for anchor, keep in zip(sampled['positions'][0], sampled['keep_mask'][0]):
            if keep:
                for offset in range(1, BLOCK):
                    if prompt_length <= anchor + offset < sequence_length:
                        positions += 1
                        weight += math.exp(-(offset - 1) / 7.0)
        if positions == 0:
            raise ValueError('Zero-supervision presentation; no resampling allowed')
        rows.append({'record_id': row['record_id'], 'pass': row['pass'],
                     'positions': positions, 'weight': weight})
    return {'rows': rows, 'total_supervised_positions': sum(row['positions'] for row in rows),
            'total_weight': sum(row['weight'] for row in rows)}


def validate_learning(receipt, request_ids, learning_rates):
    expected_order = [(request_id, pass_index) for pass_index in range(2) for request_id in request_ids]
    actual_order = [(row['record_id'], row['pass']) for row in receipt['records']]
    assert actual_order == expected_order
    assert receipt['optimizer_steps'] == 8 and receipt['counters']['presentations'] == 16
    assert [row['optimizer_stepped'] for row in receipt['records']] == [False, True] * 8
    assert len(learning_rates) == 8
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
               for a, b in zip(learning_rates, EXPECTED_LRS))


def compare_learning(control, projected):
    assert [row['sampled_anchors'] for row in control['records']] == [
        row['sampled_anchors'] for row in projected['records']]
    assert control['exposure'] == projected['exposure']
    assert control['actual_learning_rates'] == projected['actual_learning_rates']


def reusable_frozen_records(previous_root, manifest, released_path):
    """Resume the stopped feedback preparation without regenerating its 16 answers."""
    previous_root = Path(previous_root)
    previous = json.loads((previous_root / 'STATUS.json').read_text())
    assert previous['status'] == 'failed' and not previous['completed_groups']
    assert previous['generations_completed'] == 16 and previous['training_events_completed'] == 0
    assert json.loads((previous_root / 'manifest.json').read_text()) == manifest
    assert not list(previous_root.glob('*/C')) and not list(previous_root.glob('*/P'))
    group = manifest['groups'][0]
    records = {}
    for request in group['requests']:
        path = previous_root / group['name'] / 'F/records' / (request['request_id'] + '.json')
        record = json.loads(path.read_text())
        assert record['record_id'] == request['request_id']
        assert record['checkpoint_id'] == released_path
        assert record['prompt_ids'] == request['prompt_ids']
        assert record['max_new_tokens'] == CAP
        assert record['purpose'] == ('serving' if request['role'] == 'A' else 'evaluation')
        response, finish = validate_generation(record['prompt_ids'], record['prompt_ids'] + record['response_ids'],
            record['metadata']['trace'], max_new_tokens=CAP, stop_tokens=record['stop_token_ids'])
        assert response == record['response_ids'] and finish == record['finish_reason']
        records[request['request_id']] = record
    return records, previous


def admit_storage(root, name, *, initial=False):
    # Completed arm states are retired only after valid paired replay; exports stay.
    # Initial admission also budgets all six retained exports and three feature pools.
    required = (FULL_STATE + 6 * EXPORT + 3 * FEATURE_ALLOWANCE + RESERVE
                if initial else FULL_STATE + EXPORT + FEATURE_ALLOWANCE + RESERVE)
    free = shutil.disk_usage(root).free
    write_json(root / 'storage' / f'{name}.json', {
        'free_bytes': free, 'required_additional_bytes': required,
        'reserve_bytes': RESERVE, 'initial_campaign_admission': initial,
    })
    if free < required:
        raise ValueError(f'Storage admission failed: {free} < {required}')


def retired_state_metadata(trainer_root, training, checks):
    """Preserve metadata/RNG and verified export; retire completed optimizer state."""
    assert checks['publication_equal'] and checks['outputs_equal']
    state_path = trainer_root / 'training_state.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
    assert state['global_step'] == 8 and state['presentations'] == 16
    compact = {key: value for key, value in state.items() if key not in {'draft_state_dict', 'backend'}}
    compact['retained_export'] = training['export_path']
    compact['retired_capability'] = 'Exact optimizer/master continuation; all trained BF16 weights retained in verified export.'
    destination = trainer_root / 'completed_state_metadata.pt'
    save_torch(destination, compact)
    reloaded = torch.load(destination, map_location='cpu', weights_only=False)
    assert_state_equal(reloaded, compact, 'compact_state')
    size = state_path.stat().st_size
    del state, compact, reloaded
    gc.collect()
    receipt = {'path': str(state_path), 'bytes': size, 'retained_export': training['export_path'],
               'retained_metadata': str(destination), 'checks': checks,
               'lost_capability': 'Exact Adam/master continuation; not archived.',
               'free_before': shutil.disk_usage(trainer_root).free}
    write_json(trainer_root / 'RETIREMENT_PREPARED.json', receipt)
    state_path.unlink()
    receipt.update(deleted=True, free_after=shutil.disk_usage(trainer_root).free)
    write_json(trainer_root / 'RETIREMENT.json', receipt)


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
            completed_training, completed_generation):
    directory = root / group['name'] / arm
    directory.mkdir(parents=True)
    admit_storage(root, f'{group["name"]}_{arm}')
    config = yaml.safe_load(Path(base_config).read_text())
    config['training']['max_steps'] = 8
    config['run_id'] = f'projected-target-{group["name"]}-{arm}'
    config['output_dir'] = str(directory / 'trainer')
    config['data']['hidden_states_path'] = str(root / group['name'] / 'F' / 'features')
    config_path = directory / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    objective = {'C': 'hard_ce', 'P': 'projected'}[arm]
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
        lrs = []
        ordinary_step = optimizer.step

        def recorded_step(*args, **kwargs):
            lrs.append(optimizer.optimizer.param_groups[0]['lr'])
            return ordinary_step(*args, **kwargs)

        optimizer.step = recorded_step
        model = trainer.bundle.model
        begin_event, end_event = model.begin_event, model.end_event

        def checked_begin():
            nonlocal diagnostic_seconds
            begin_event()
            if arm == 'P':
                start = time.perf_counter()
                reference = model.reference.model.draft_model
                assert_state_equal(reference.state_dict(), released_state, 'reference_initial')
                assert all(not p.requires_grad and id(p) not in trainable_ids for p in reference.parameters())
                checks['reference_initial_equal'] = True
                diagnostic_seconds += time.perf_counter() - start

        def checked_end():
            nonlocal diagnostic_seconds
            if arm == 'P' and model.reference is not None:
                start = time.perf_counter()
                assert_state_equal(model.reference.model.draft_model.state_dict(), released_state, 'reference_final')
                checks['reference_unchanged'] = True
                diagnostic_seconds += time.perf_counter() - start
            return end_event()

        model.begin_event, model.end_event = checked_begin, checked_end
        receipt = trainer.train_event('update', feature_paths, passes=2)
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
    del trainer, model, optimizer, begin_event, end_event, ordinary_step
    del recorded_step, checked_begin, checked_end, frozen_tables, module
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
    reused, previous_status = ({}, None)
    if args.reuse_frozen_first_group:
        reused, previous_status = reusable_frozen_records(args.reuse_frozen_first_group, manifest, args.draft)
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
        'reused_frozen_first_group': args.reuse_frozen_first_group,
    }
    write_json(root / 'PLAN.json', plan)
    status = {'status': 'running', 'completed_groups': [], 'generations_completed': 0,
              'generations_reused': 0, 'new_generations_completed': 0,
              'training_events_completed': 0, 'started_unix': time.time(),
              'previous_attempt': previous_status}
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
                if request['request_id'] in reused:
                    record = reused[request['request_id']]
                    assert backend.render(request['messages']) == record['prompt_ids']
                    write_json(group_root / 'F/records' / (request['request_id'] + '.json'), record)
                    status['generations_completed'] += 1
                    status['generations_reused'] += 1
                    write_json(root / 'STATUS.json', status)
                else:
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
            from report import build_report
            build_report(root)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status.update(wall_s=time.perf_counter() - started, finished_unix=time.time())
        write_json(root / 'STATUS.json', status)
        from report import build_report
        build_report(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'config', 'target', 'draft', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--reuse-frozen-first-group', default=None,
                        help='Explicit recovery from failed first-group feedback preparation; zero prior training only.')
    run(parser.parse_args())
