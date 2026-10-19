"""CPU service integration with actual sampler, compression and spectral writes.

The tiny synthetic head/backbone never loads a model or qualifies GPU serving.
It exercises the complete prepare/feedback state boundary and saved artifacts.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.dflash_correction_bank.artifacts import atomic_json, binding, load_snapshot
from .core import Factors, fuse
from .feedback import FullHiddenObjective
from .state import _hash
from . import training

torch.set_num_threads(1)


def service_fixture(tmp_path, monkeypatch):
    torch.manual_seed(71)
    width, vocab = 64, 32

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = torch.nn.Parameter(torch.ones(1))

        def forward(self, *, position_ids, noise_embedding, target_hidden, attention_mask):
            self.observed_mask = attention_mask.clone()
            self.observed_positions = position_ids.clone()
            self.observed_target = target_hidden.clone()
            return noise_embedding, None

    service = training.BankTrainingService.__new__(training.BankTrainingService)
    service.config = {'episode_id': 'toy', 'target': {'identity_sha256': 'a' * 64},
                      'bank_qualification': True}
    service.bank_draft = Draft().eval().requires_grad_(False)
    service.bank_versions = tuple((n, id(p), p._version) for n, p in service.bank_draft.named_parameters())
    embed = torch.nn.Embedding(vocab, width).bfloat16().eval().requires_grad_(False)
    head = torch.nn.Linear(width, vocab, bias=False).bfloat16().eval().requires_grad_(False)
    service.components = SimpleNamespace(embed_tokens=embed, lm_head=head)
    service.bank_generator = torch.Generator().manual_seed(17)
    service.bank_directory = tmp_path / 'bank'
    service.bank_directory.mkdir()
    service.arrays = service.bank_directory / 'arrays'
    service.bank = None
    service.snapshot_reference = None
    service.last_turn = 0
    service.qualification_replay = {}
    service._memory_objectives = {}
    service.check_frozen = lambda: None
    teacher_inputs = []

    def teacher(ids):
        teacher_inputs.append(ids.detach().clone())
        h = embed(ids)
        return torch.cat((h, h), dim=-1), h

    service.teacher_features = teacher
    real_sampler = training.sampled_features
    monkeypatch.setattr(training, 'sampled_features', lambda *args, **kwargs:
                        real_sampler(*args, **kwargs, attention_backend='sdpa', mask_token_id=31))
    service.prepare({})
    return service, teacher_inputs


def cost_profile():
    return {'bins': {'8192': {'cycle_s': {'0': 1., '1': 1.1, '2': 1.2},
                             'base_s': 1., 'observations': 0}},
            'head_increment_s': {'0': 0., '1': .1, '2': .2}}


def event(service, turn, *, final=False, qualification=True, output_length=18,
          corrupt_proposal=False,block_start=0):
    response = dict(record_id=f'toy:BANK:t{turn}', episode_id='toy', turn=turn, target_id='T0',
                    target_identity_sha256='a' * 64, capture_checkpoint_sha256='b' * 64,
                    input_ids=[3] * (200 + turn), output_ids=list(range(3, 3 + output_length)),
                    input_token_count=200 + turn, output_cap=1024)
    descriptor = np.zeros(64)
    descriptor[0] = 1
    read = service.bank.read(descriptor, len(response['input_ids']))
    saved_read = dict(coefficients=read.alpha.tolist(), descriptor=descriptor.tolist(),
                      forecast=read.forecast, diagnostics=read.diagnostics,
                      snapshot_sha256=service.snapshot_reference['sha256'])
    rows, blocks = [], []
    if output_length > 1:
        generator = torch.Generator().manual_seed(100 + turn)
        h = torch.randn(9, 64, generator=generator).bfloat16().float().numpy()
        positions=list(range(block_start+1,block_start+10))
        ids = [response['output_ids'][pos] for pos in positions if pos<output_length]
        eligible = np.array([position < output_length for position in positions])
        candidate_obj = FullHiddenObjective(h, np.zeros_like(h), np.full(9, 1 / 9),
                                           service.components.lm_head.weight)
        # Beyond-end rows are still retained; only eligible rows may enter memory.
        canonical = ids + [0] * (9 - len(ids))
        proposed = candidate_obj.native_evaluate(read.effective, canonical)['predictions'].tolist()
        if corrupt_proposal:
            proposed[0] = (proposed[0] + 1) % 32
        rows.append(h)
        blocks.append(dict(block_start=block_start, hidden_index=0, eligible_mask=eligible.tolist(),
                           output_positions=positions, proposal_token_ids=proposed,
                           canonical_token_ids=[response['output_ids'][p] if p < output_length else None
                                                for p in positions]))
    path = service.bank_directory / f'capture_{turn}.npz'
    np.savez(path, frozen_h=np.stack(rows) if rows else np.empty((0, 9, 64), np.float32))
    document = dict(input_ids_sha256=training.per_turn.digest(response['input_ids']),
                    output_ids_sha256=training.per_turn.digest(response['output_ids']),
                    tensors=binding(path), blocks=blocks, costs={'transfer_s': 0.})
    capture = atomic_json(path.with_suffix('.json'), document)
    return dict(turn=turn, final=final, qualification=qualification, response=response,
                served_read=saved_read, before_snapshot=service.snapshot_reference,
                native_head_capture=capture, cost_profile=cost_profile(), cycle_observations=[])


def test_three_response_feedback_executes_real_masks_compression_and_native_validation(tmp_path, monkeypatch):
    service, teacher_inputs = service_fixture(tmp_path, monkeypatch)
    order = []
    original_add = service.bank.add_transfer
    original_write = service.bank.write

    def add(*args, **kwargs):
        result = original_add(*args, **kwargs)
        order.append(('insert', len(service.bank.memory), len(service.bank.protection)))
        return result

    def write(*args, **kwargs):
        assert order[-1][0] == 'insert'
        assert order[-1][1] == len(service.bank.memory)
        order.append(('write', len(service.bank.memory), len(service.bank.protection)))
        return original_write(*args, **kwargs)

    service.bank.add_transfer, service.bank.write = add, write
    for turn in (1, 2, 3):
        body = event(service, turn, final=turn == 3)
        before = service.bank.identity()
        result = service.feedback(body)
        assert result['qualification_checks']['compressed_direct']['passed']
        assert result['qualification_checks']['direct_historical']['passed']
        assert result['qualification_checks']['compiled_fusion']['passed']
        assert result['qualification_checks']['masks_and_conditioning']
        assert result['qualification_checks']['native_precision_reported']
        assert len(result['qualification_checks']['native_precision']['cases']) == 5
        native_fusion=result['qualification_checks']['sampled_native_fusion']
        assert native_fusion['passed'] and native_fusion['available']
        assert native_fusion['compiled_rank']==64 and native_fusion['device']=='cpu'
        assert result['source_fit']['native_position_presentations'] == result['accounting']['eligible_positions']
        assert set(result['source_fit']['native_correct_positions']) == {'frozen', 'served', 'fitted'}
        assert result['costs']['feedback_compression_s'] >= 0
        assert result['costs']['source_fit_diagnostic_only_s'] >= 0
        assert result['new_memory_blocks'] == (0 if turn == 3 else 1)
        assert result['scheduled_steps'] == (0 if turn == 3 else 2)
        assert 0 <= result['accepted_steps'] <= result['scheduled_steps']
        saved = load_snapshot(result['feedback'])
        assert all(row['passed'] for row in saved['canonical_checks'])
        assert saved['teacher_compression']['author_native_probability_max_abs_difference'] >= 0
        assert saved['memory'][0]['teacher_compression']['stored_moment_dtype'] == 'float32'
        assert all(pos >= len(body['response']['input_ids']) for pos in result['accounting']['unique_supervised_position_ids'])
        assert not set(result['accounting']['excluded_memory_absolute_positions']) & \
            set(result['accounting']['unique_supervised_position_ids'])
        assert teacher_inputs[-1].tolist()[0] == body['response']['input_ids'] + body['response']['output_ids']
        assert result['qualification_replay']
        if turn == 3:
            assert service.bank.identity() == before
    assert [kind for kind, _, _ in order] == ['insert', 'write', 'insert', 'write']
    assert len(service.bank.memory) == 2


@pytest.mark.parametrize('length', [0, 1])
def test_short_response_noop_does_not_add_memory_or_allocate_and_final_does_not_initialize_costs(tmp_path, monkeypatch, length):
    service, teacher_inputs = service_fixture(tmp_path, monkeypatch)
    body = event(service, 1, final=True, output_length=length)
    before = service.bank.identity()
    result = service.feedback(body)
    assert service.bank.identity() == before
    assert not service.bank.cost_profile
    assert not teacher_inputs
    assert result['scheduled_steps'] == result['accepted_steps'] == 0
    assert result['no_new_routing_evidence']
    assert result['source_fit']['native_position_presentations'] == 0
    assert result['accounting']['no_op_reason'] == 'fewer_than_two_eligible_response_anchors'


def test_nonfinal_short_response_retains_scheduled_noop_without_sampler_or_writes(tmp_path, monkeypatch):
    service, teacher_inputs = service_fixture(tmp_path, monkeypatch)
    body = event(service, 1, output_length=1)
    rng = service.bank_generator.get_state().clone()
    result = service.feedback(body)
    assert result['scheduled_steps'] == 2 and result['accepted_steps'] == 0
    assert result['skip_reason'] == 'fewer_than_two_eligible_response_anchors'
    assert result['blocked_writes'] == 1 and not any(service.bank.core.allocated)
    assert not teacher_inputs and torch.equal(rng, service.bank_generator.get_state())


def test_mismatched_native_head_capture_is_preserved_and_cannot_pass_qualification(tmp_path, monkeypatch):
    service, _ = service_fixture(tmp_path, monkeypatch)
    body = event(service, 1, corrupt_proposal=True)
    with pytest.raises(ValueError, match='reconstruct actual served proposals'):
        service.feedback(body)
    assert service.last_turn == 0 and not any(service.bank.core.allocated)
    evidence = load_snapshot(binding(service.bank_directory / 'FAILED_NATIVE_01.json'))
    assert not evidence['checks'][0]['passed']


def test_wrong_feedback_ownership_and_noncausal_action_are_rejected_before_teacher(tmp_path, monkeypatch):
    service, teacher_inputs = service_fixture(tmp_path, monkeypatch)
    body = event(service, 1)
    bad = copy.deepcopy(body)
    bad['response']['episode_id'] = 'another'
    with pytest.raises(ValueError, match='ownership'):
        service.feedback(bad)
    bad = copy.deepcopy(body)
    bad['served_read']['coefficients'] = [1., 0., 0., 0.]
    with pytest.raises(ValueError, match='saved causal read'):
        service.feedback(bad)
    assert not teacher_inputs


def test_native_block_loader_preserves_absolute_response_binding_and_filters_tail(tmp_path, monkeypatch):
    service, _ = service_fixture(tmp_path, monkeypatch)
    body = event(service, 1, output_length=4)
    blocks, _ = training.load_native_blocks(body['native_head_capture'], body['response'])
    assert len(blocks) == 1
    assert blocks[0]['output_positions'] == [1, 2, 3]
    assert blocks[0]['h'].shape == (3, 64)
    assert blocks[0]['full_native_h'].shape == (9,64)
    assert blocks[0]['native_row_indices']==[0,1,2]
    assert len(blocks[0]['served_proposals']) == 3
    bad = copy.deepcopy(body['response'])
    bad['input_ids'][0] = 4
    with pytest.raises(ValueError, match='another canonical response'):
        training.load_native_blocks(body['native_head_capture'], bad)


def test_partial_memory_block_preserves_full_native_head_geometry_through_feedback(tmp_path,monkeypatch):
    service,_=service_fixture(tmp_path,monkeypatch)
    body=event(service,1,block_start=9)
    result=service.feedback(body)
    assert result['new_memory_blocks']==1
    block=service.bank.memory[0]
    assert block['h'].shape==(8,64) and block['native_h'].shape==(9,64)
    assert block['native_row_indices'].tolist()==list(range(8))
    obj=service._memory_objectives[block['block_id']]
    assert obj.native_evaluate(Factors.zero(64),block['token_ids'])['native_head_row_counts']==[9]
