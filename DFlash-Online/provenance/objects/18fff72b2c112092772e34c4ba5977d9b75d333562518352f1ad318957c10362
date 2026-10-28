"""Protocol checks and storage policy for this fixed screen, outside the reading path."""
from pathlib import Path
import gc
import math
import shutil
import time

import torch

from online_feedback.qualify import assert_state_equal
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


class ReferenceCheck:
    """Qualification-only observation; the shared trainer owns reference lifetime."""

    def __init__(self, released_state, trainable_ids, checks):
        self.released_state = released_state
        self.trainable_ids = trainable_ids
        self.checks = checks
        self.elapsed_s = 0.0

    def __call__(self, model, phase):
        started = time.perf_counter()
        reference = model.reference.model.draft_model
        assert_state_equal(reference.state_dict(), self.released_state, "reference_" + phase)
        assert all(not p.requires_grad and id(p) not in self.trainable_ids
                   for p in reference.parameters())
        name = "reference_initial_equal" if phase == "before" else "reference_unchanged"
        self.checks[name] = True
        self.elapsed_s += time.perf_counter() - started
