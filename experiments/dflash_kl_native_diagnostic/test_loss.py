"""Small CPU tensors only: not native GPU/model qualification."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
from torch import nn
from torch.nn import functional as F
from . import loss


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def record(index=0, length=13):
    return dict(group_id='group', request_id=f'heldout_{index}', record_id=f'group:heldout_{index}',
        role='evaluation', input_ids=[1, 3, 5, 7, 2, 4, 6],
        output_ids=[(index + 2*i) % 19 for i in range(length)], output_cap=1024)


def models(dtype=torch.float32):
    torch.manual_seed(41)
    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)
            self.mix = nn.Linear(4, 4)
            self.dropout = nn.Dropout(.7)
            self.calls = []
        def forward(self, *, position_ids, noise_embedding, target_hidden, attention_mask):
            assert not torch.is_grad_enabled()
            assert all(not p.requires_grad for p in self.parameters())
            self.calls.append(dict(position_ids=position_ids.clone(), target_hidden=target_hidden.clone(),
                noise_embedding=noise_embedding.clone(), attention_mask=attention_mask.clone()))
            kv = torch.cat((self.fc(target_hidden), noise_embedding), dim=1).unsqueeze(1)
            hidden = F.scaled_dot_product_attention(noise_embedding.unsqueeze(1), kv, kv,
                attn_mask=attention_mask).squeeze(1)
            return self.mix(self.dropout(hidden + noise_embedding)), None
    draft = Draft().to(dtype)
    draft.dropout.eval()  # mixed mode must be restored exactly
    embed = nn.Embedding(20, 4).to(dtype).requires_grad_(False).eval()
    head = nn.Linear(4, 20, bias=False).to(dtype).requires_grad_(False).eval()
    calls = []
    def teacher(ids):
        assert not torch.is_grad_enabled()
        calls.append(ids.clone())
        base = embed(ids)
        causal = base.cumsum(1)/torch.arange(1, ids.shape[1]+1, device=ids.device,
            dtype=dtype).view(1, -1, 1)
        return torch.cat((base, causal), dim=-1), causal
    return draft, embed, head, teacher, calls


def evaluate(parts, rows, checkpoint='F'):
    return loss.evaluate_responses(*parts[:4], rows, checkpoint_id=checkpoint,
        checkpoint_identity_sha256=('a' if checkpoint == 'F' else 'b')*64,
        mask_token_id=19, attention_backend='sdpa', projection_chunk=7)


def reference(parts, row, seed):
    draft, embed, head, teacher, _ = parts
    ids = torch.tensor([row['input_ids'] + row['output_ids']])
    modules = [(m, m.training) for root in parts[:3] for m in root.modules()]
    flags = [(p, p.requires_grad) for root in parts[:3] for p in root.parameters()]
    try:
        for root in parts[:3]:
            root.eval().requires_grad_(False)
        with torch.no_grad():
            hidden, target = teacher(ids)
            ids = torch.cat((ids, ids.new_full((1, 9), 19)), 1)
            hidden = torch.cat((hidden, hidden.new_zeros((1, 9, 8))), 1)
            target = torch.cat((target, target.new_zeros((1, 9, 4))), 1)
            mask = torch.zeros_like(ids, dtype=torch.float32)
            mask[:, len(row['input_ids']):len(row['input_ids'])+len(row['output_ids'])] = 1
            return loss.author.author_response_loss(draft, embed, head, ids, hidden, target, mask,
                mask_token_id=19, generator=torch.Generator().manual_seed(seed), attention_backend='sdpa',
                projection_chunk=7, checkpoint_projection=False)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, mode in modules:
            m.training = mode


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_read_only_loss_exactly_matches_qualified_objective(dtype):
    parts = models(dtype)
    row = record()
    actual = evaluate(parts, [row])['response_losses'][0]
    expected, metadata = reference(parts, row, actual['anchor_seed'])
    assert actual['loss'] == float(expected)
    assert all(actual[key] == value for key, value in metadata.items())
    assert actual['role'] == 'evaluation' and actual['loss_improvement_eligible']
    assert actual['eligible_positions'] > actual['unique_supervised_positions']
    json.dumps(actual, allow_nan=False)


def test_exact_teacher_conditioning_full_context_and_response_only_mask():
    parts = models()
    row = record()
    result = evaluate(parts, [row])['response_losses'][0]
    expected = row['input_ids'] + row['output_ids']
    assert parts[4][0].tolist() == [expected]  # teacher never sees synthetic tail
    call = parts[0].calls[0]
    assert call['target_hidden'].shape[1] == len(expected) + 9
    assert not call['target_hidden'][:, -9:].any()
    assert result['response_loss_mask'] == [0]*7 + [1]*13 + [0]*9
    assert min(result['unique_supervised_position_ids']) >= 7
    assert max(result['unique_supervised_position_ids']) < len(expected)
    for labels, teacher, weights in zip(result['label_positions'], result['teacher_positions'], result['position_weights']):
        assert all(t == y-1 for y, t, w in zip(labels, teacher, weights) if w > 0)
        assert weights[0] == 0
    for anchor_index, anchor in enumerate(result['anchor_positions']):
        mask = call['attention_mask'][0, 0, anchor_index*10]
        assert mask[:anchor].all() and not mask[anchor:len(expected)+9].any()
        start = len(expected)+9+anchor_index*10
        assert mask[start:start+10].all()


def test_checkpoint_independent_anchors_and_rng_modes_weights_preserved():
    parts = models()
    row = record()
    before = loss._state_hash(loss._rng_state())
    modes = [m.training for root in parts[:3] for m in root.modules()]
    flags = [p.requires_grad for root in parts[:3] for p in root.parameters()]
    states = [copy.deepcopy(root.state_dict()) for root in parts[:3]]
    with patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer forbidden')), \
         patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward forbidden')):
        frozen = evaluate(parts, [row])
        updated = evaluate(parts, [row], 'S8')
    assert frozen['response_losses'][0]['anchor_seed'] == updated['response_losses'][0]['anchor_seed']
    assert frozen['response_losses'][0]['sampling_and_weights_sha256'] == updated['response_losses'][0]['sampling_and_weights_sha256']
    assert frozen['response_losses'][0]['anchor_rng_after_sha256'] == updated['response_losses'][0]['anchor_rng_after_sha256']
    assert frozen['response_losses'][0]['loss'] == updated['response_losses'][0]['loss']
    assert before == loss._state_hash(loss._rng_state())
    assert modes == [m.training for root in parts[:3] for m in root.modules()]
    assert flags == [p.requires_grad for root in parts[:3] for p in root.parameters()]
    for original, root in zip(states, parts[:3]):
        assert all(torch.equal(value, root.state_dict()[name]) for name, value in original.items())
    assert all(p.grad is None for root in parts[:3] for p in root.parameters())
    assert frozen['optimizer_steps'] == 0 and not frozen['optimizer_created']


def test_sampling_independent_of_iteration_order_and_global_randomness():
    parts = models()
    rows = [record(0), record(1)]
    first = evaluate(parts, rows)['response_losses']
    torch.rand(200)
    second = evaluate(parts, rows[::-1], 'S8')['response_losses'][::-1]
    for a, b in zip(first, second):
        assert a['anchor_seed'] == b['anchor_seed']
        assert a['sampling_and_weights_sha256'] == b['sampling_and_weights_sha256']
        assert a['loss'] == b['loss']
    assert first[0]['anchor_seed'] != first[1]['anchor_seed']


def test_teacher_randomness_isolated_and_restored_on_exception():
    parts = list(models())
    modes = [m.training for root in parts[:3] for m in root.modules()]
    flags = [p.requires_grad for root in parts[:3] for p in root.parameters()]
    before = loss._state_hash(loss._rng_state())
    def failing_teacher(ids):
        torch.rand(17)
        loss.random.random()
        raise RuntimeError('controlled teacher error')
    parts[3] = failing_teacher
    with pytest.raises(RuntimeError, match='controlled teacher error') as failure:
        evaluate(parts, [record()])
    assert failure.value.diagnostic_partial['global_rng_restored']
    assert failure.value.diagnostic_partial['status'] == 'failed'
    assert failure.value.diagnostic_partial['optimizer_steps'] == 0
    assert before == loss._state_hash(loss._rng_state())
    assert modes == [m.training for root in parts[:3] for m in root.modules()]
    assert flags == [p.requires_grad for root in parts[:3] for p in root.parameters()]


@pytest.mark.parametrize('length', [0, 1])
def test_short_response_excluded_without_teacher_or_sampler(length):
    parts = models()
    with patch.object(loss.author, 'sample_anchor_positions', side_effect=AssertionError('no invented anchor')):
        result = evaluate(parts, [record(length=length)])
    row = result['response_losses'][0]
    assert row['status'] == 'excluded' and row['loss'] is None
    assert not row['loss_improvement_eligible'] and row['zero_supervision']
    assert row['anchor_sampling_attempts'] == 0
    assert row['anchor_positions'] == [] and row['normalization_denominator'] is None
    assert row['anchor_rng_before_sha256'] == row['anchor_rng_after_sha256']
    assert result['loss_mean'] is None and result['excluded_response_count'] == 1
    assert not parts[4] and not parts[0].calls


def test_real_two_token_zero_supervision_not_resampled_or_in_mean():
    parts = models()
    # Select a deterministic identity whose unchanged author sampler takes the
    # terminal anchor. This is fixture construction, never production resampling.
    chosen = None
    for index in range(30):
        candidate = record(index, length=2)
        audit = loss.audit_record(candidate)
        mask = torch.tensor([audit['response_loss_mask']], dtype=torch.float32)
        anchors, _ = loss.author.sample_anchor_positions(mask.shape[1], mask,
            generator=torch.Generator().manual_seed(loss.anchor_seed(audit)))
        if anchors[0, 0].item() == audit['response_end_exclusive']-1:
            chosen = candidate
            break
    assert chosen is not None
    original = loss.author.sample_anchor_positions
    with patch.object(loss.author, 'sample_anchor_positions', wraps=original) as sampler:
        result = evaluate(parts, [chosen, record(99)])
    assert sampler.call_count == 2
    row = result['response_losses'][0]
    assert row['zero_supervision'] and row['status'] == 'excluded' and row['loss'] == 0
    assert row['excluded_reason'] == 'sample_has_zero_supervision'
    assert row['sampled_blocks'] == 1 and row['eligible_positions'] == 0
    assert row['normalization_denominator'] > 0
    assert result['loss_mean'] == result['response_losses'][1]['loss']


def test_historical_training_guards_unchanged():
    assert loss.author.OUTPUT_CAP == 256
    assert loss.author.RECIPE['passes'] == 2
    with pytest.raises(ValueError, match='Invalid response turn'):
        loss.author.audit_response(record())
    for role in ('feedback', 'source', None):
        row = record()
        row['role'] = role
        with pytest.raises(ValueError, match='evaluation records only'):
            evaluate(models(), [row])


def test_validation_and_nonfinite_stop_without_valid_result():
    row = record()
    row['output_ids_sha256'] = '0'*64
    with pytest.raises(ValueError, match='hash differs'):
        evaluate(models(), [row])
    with pytest.raises(ValueError, match='Duplicate'):
        evaluate(models(), [record(), record()])
    parts = list(models())
    original = parts[3]
    def nonfinite_teacher(ids):
        hidden, teacher = original(ids)
        teacher[0, 0, 0] = float('nan')
        return hidden, teacher
    parts[3] = nonfinite_teacher
    with pytest.raises(ValueError, match='nonfinite') as failure:
        evaluate(parts, [record()])
    assert failure.value.diagnostic_partial['response_losses'] == []


def test_tensor_mutation_is_rejected():
    parts = list(models())
    original = parts[3]
    def bad_teacher(ids):
        parts[0].mix.bias.add_(1)
        return original(ids)
    parts[3] = bad_teacher
    with pytest.raises(ValueError, match='mutated model'):
        evaluate(parts, [record()])


def test_module_import_does_not_import_torch_or_construct_training_state():
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc, sys
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch imported at metadata load')
sys.meta_path.insert(0, RejectTorch())
from experiments.dflash_kl_native_diagnostic import loss
assert loss.anchor_seed(loss.audit_record(dict(group_id='g',request_id='r',role='evaluation',input_ids=[1],output_ids=[2,3]))) >= 0
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_real_loss_rows_satisfy_paired_controller_metadata_contract():
    from .experiment import SAMPLING_FIELDS, validate_loss
    parts = models()
    for source in (record(), record(length=1)):
        row = evaluate(parts, [source])['response_losses'][0]
        assert row['sampling_and_weights_sha256'] == loss.digest({key: row[key] for key in SAMPLING_FIELDS})
        assert row['diagnostic_model_mode'] == 'eval_no_grad' and row['training_rng_preserved']
        assert row['excluded_from_loss_improvement'] == row['zero_supervision']
        validate_loss(row, source, 'a'*64)
