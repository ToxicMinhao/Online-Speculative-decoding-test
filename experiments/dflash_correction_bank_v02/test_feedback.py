"""Synthetic CPU feedback checks; these do not qualify GPU native integration."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .core import Factors
from .feedback import (FullHiddenObjective, compress_teacher_moments,
                       reserve_memory_blocks, sampled_features, author)

torch.set_num_threads(1)


def fixture(seed=17, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    E = torch.randn(23, 7, generator=generator).to(dtype)
    h = torch.randn(8, 7, generator=generator).to(dtype)
    th = torch.randn(8, 7, generator=generator).to(dtype)
    weights = np.arange(1., 9.)
    weights /= weights.sum()
    objective = FullHiddenObjective.from_teacher(h, th, weights, E, row_chunk=3, vocab_chunk=5)
    rng = np.random.default_rng(seed)
    factors = Factors(rng.normal(size=(7, 2)) * .05, rng.normal(size=(2, 7)))
    return objective, factors


def test_full_hidden_teacher_compression_matches_native_logits_and_discloses_rounding():
    objective, _ = fixture()
    teacher = torch.tensor(objective.teacher_h).to(objective.E_native.dtype)
    expected = torch.nn.functional.linear(teacher, objective.E_native).double().softmax(-1) @ objective.E_native.double()
    np.testing.assert_array_equal(objective.mu, expected.float().numpy())
    assert objective.mu.dtype == np.float32 and objective.features.dtype == np.float32
    assert not hasattr(objective, 'E')  # No persistent full precision duplicate.
    assert objective.compression['full_high_precision_vocabulary_copy'] is False
    assert objective.compression['moment_storage_max_abs_error'] > 0
    assert objective.compression['author_native_probability_max_abs_difference'] > 0
    assert objective.compression['author_native_probability_mass_max_error'] > 0


def test_compressed_difference_gradient_and_covariance_use_unclipped_full_features():
    objective, factors = fixture()
    loss, gradient = objective.loss_gradient(factors)
    direct, dg = objective.direct_loss_gradient(factors)
    assert abs(loss - direct) < 2e-7
    np.testing.assert_allclose(gradient.dense(), dg, atol=2e-7, rtol=2e-6)
    rng = np.random.default_rng(5)
    W = factors.dense()
    step = rng.normal(size=W.shape)
    finite = (objective.loss(W + 1e-6 * step) - objective.loss(W - 1e-6 * step)) / 2e-6
    assert finite == pytest.approx(np.sum(gradient.dense() * step), abs=2e-8)
    h = objective.features.astype(np.float64)
    np.testing.assert_allclose(objective.covariance(), (h.T * objective.weights) @ h, atol=1e-12)
    assert np.max(np.linalg.norm(h, axis=1)) > 1
    assert objective.loss(Factors.zero(7)) == 0


def test_native_precision_checks_survival_without_turning_it_into_success_gate():
    objective, factors = fixture()
    tiny = Factors(factors.B * 1e-10, factors.A)
    zero = objective.precision_diagnostic(Factors.zero(7))
    vanished = objective.precision_diagnostic(tiny)
    fitted = objective.precision_diagnostic(factors)
    assert zero['changed_positions'] == 0 and zero['native_loss_change'] == 0
    assert vanished['nonzero_precast_coordinates'] > 0 and vanished['changed_positions'] == 0
    assert fitted['changed_positions'] > 0
    assert fitted['native_to_reference_logit_delta_norm_ratio'] > 0
    assert not fitted['performance_gate']


def test_full_vocabulary_mathematical_margins_and_native_prefix_are_separate():
    E = torch.tensor([[1., 0.], [.5, .3], [-1., 0.], [.9, 2.]], dtype=torch.bfloat16)
    h = torch.tensor([[1., 0.], [-1., 0.], [1., 1.]], dtype=torch.bfloat16)
    obj = FullHiddenObjective(h, np.zeros((3, 2)), np.full(3, 1 / 3), E, vocab_chunk=2)
    ids = [0, 2, 1]
    W = Factors(np.array([[0.], [1.]]), np.array([[.25, 0.]]))
    x = h.double().numpy() + h.double().numpy() @ W.dense().T
    logits = x @ E.double().numpy().T
    target = logits[np.arange(3), ids].copy()
    logits[np.arange(3), ids] = -np.inf
    np.testing.assert_allclose(obj.mathematical_margins(W, ids), target - logits.max(-1))
    native = obj.native_evaluate(Factors.zero(2), ids, [(0, 2), (2, 3)])
    assert native['predictions'].tolist() == [0, 2, 3]
    assert native['prefix_lengths'] == [2, 0] and not native['seed_included']
    assert obj.native_evaluate(None, ids, [(0, 2)], eos_token_ids=[0])['prefix_lengths'] == [1]


def test_memory_reservation_is_deterministic_nonoverlapping_and_retains_training_positions():
    blocks = [dict(block_start=start, output_positions=list(range(start + 1, start + 10)),
                   h=np.ones((9, 3)), observed_gain=100 - start) for start in (0, 4, 9)]
    first = reserve_memory_blocks('stream', 1, blocks, list(range(1, 20)))
    second = reserve_memory_blocks('stream', 1, list(reversed(blocks)), list(range(1, 20)))
    assert first['excluded_output_positions'] == second['excluded_output_positions']
    selected_sets = [set(x['output_positions']) for x in first['selected']]
    assert len(selected_sets) <= 2
    if len(selected_sets) == 2:
        assert not selected_sets[0] & selected_sets[1]
    assert first['training_positions_after'] > 0
    no_room = reserve_memory_blocks('stream', 1, [blocks[0]], [1, 2])
    assert no_room['selected'] == [] and no_room['no_new_routing_evidence']
    assert no_room['training_positions_after'] == 2


def test_memory_reservation_does_not_fill_slots_with_empty_or_invalid_blocks():
    empty = reserve_memory_blocks('stream', 2, [dict(block_start=0, output_positions=[])], [1])
    assert empty['new_routing_memory_blocks'] == 0
    with pytest.raises(ValueError, match='exclude the seed'):
        reserve_memory_blocks('stream', 1, [dict(block_start=0, output_positions=[0, 1])], [1, 2])
    with pytest.raises(ValueError, match='distinct'):
        reserve_memory_blocks('stream', 1, [dict(block_start=0, output_positions=[1])] * 2, [1, 2])


def fake_source(output_length=18):
    torch.manual_seed(2)

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 4, bias=False)

        def forward(self, *, position_ids, noise_embedding, target_hidden, attention_mask):
            self.last_mask = attention_mask.detach().clone()
            return noise_embedding, None

    draft = Draft().eval().requires_grad_(False)
    embed = torch.nn.Embedding(32, 4).eval().requires_grad_(False)
    response = dict(record_id='test:BANK:t2', episode_id='test', turn=2, target_id='T0',
                    target_identity_sha256='a' * 64, capture_checkpoint_sha256='b' * 64,
                    input_ids=[3] * 200, output_ids=list(range(3, 3 + output_length)),
                    input_token_count=200, output_cap=1024)

    def teacher(ids):
        x = embed(ids)
        return torch.cat((x, x), -1), x

    return draft, embed, teacher, response


def test_native_memory_exclusion_masks_every_overlap_without_changing_author_anchors():
    draft, embed, teacher, response = fake_source()
    native_h = np.arange(36, dtype=np.float32).reshape(9, 4)
    blocks = [dict(block_start=0, output_positions=list(range(1, 10)), h=native_h)]
    generator = torch.Generator().manual_seed(17)
    features = sampled_features(draft, embed, teacher, response, generator=generator, native_blocks=blocks,
                                mask_token_id=31, attention_backend='sdpa')
    metadata = features['metadata']
    assert metadata['memory_blocks_reserved'] == 1
    assert not set(range(201, 210)) & set(metadata['unique_supervised_position_ids'])
    labels = np.asarray(metadata['label_positions'])
    weights = np.asarray(metadata['position_weights'])
    assert np.count_nonzero(weights[np.isin(labels, range(201, 210))]) == 0
    assert metadata['eligible_positions'] < metadata['original_sampled_eligible_positions']
    independent = torch.Generator().manual_seed(17)
    mask = torch.tensor([[0] * 200 + [1] * 18 + [0] * 9], dtype=torch.float32)
    anchors, _ = author.sample_anchor_positions(227, mask, generator=independent)
    assert anchors[0].tolist() == metadata['anchor_positions']
    assert np.array_equal(features['rng_after'], independent.get_state().numpy())
    assert features['omega'].sum() == pytest.approx(1)
    memory = features['memory_blocks'][0]
    np.testing.assert_array_equal(memory['h'], native_h)
    assert memory['token_ids'] == response['output_ids'][1:10]
    assert memory['metadata']['teacher_positions'] == list(range(200, 209))
    expected_teacher = embed(torch.tensor(response['input_ids'] + response['output_ids']))
    np.testing.assert_array_equal(memory['teacher_h'], expected_teacher[200:209].numpy())
    for block, anchor in enumerate(metadata['anchor_positions']):
        row = draft.last_mask[0, 0, block * 10]
        assert bool(row[:anchor].all()) and not bool(row[anchor:227].any())


@pytest.mark.parametrize('length', [0, 1])
def test_short_response_is_explicit_bank_noop_without_consuming_rng_or_teacher_work(length):
    draft, embed, _, response = fake_source(length)
    generator = torch.Generator().manual_seed(17)
    before = generator.get_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError('No teacher or source sampler for short response')

    features = sampled_features(draft, embed, forbidden, response, generator=generator)
    assert features['metadata']['zero_supervision']
    assert features['metadata']['no_op_reason'] == 'fewer_than_two_eligible_response_anchors'
    assert features['reservation']['no_new_routing_evidence']
    assert torch.equal(before, generator.get_state())
    assert features['h'].shape == (0, 4)


def test_empty_objective_and_portable_round_trip():
    E = torch.eye(3)
    empty = FullHiddenObjective(np.empty((0, 3)), np.empty((0, 3)), np.empty(0), E)
    loss, gradient = empty.loss_gradient(Factors.zero(3))
    assert loss == 0 and np.count_nonzero(gradient.dense()) == 0
    objective, factors = fixture()
    data = objective.portable()
    restored = FullHiddenObjective(data['h'], data['mu'], data['omega'], objective.E_native,
                                  teacher_h=data['teacher_h'], row_chunk=data['row_chunk'],
                                  vocab_chunk=data['vocab_chunk'])
    assert restored.loss(factors) == objective.loss(factors)
    assert restored.native_evaluate(factors, list(range(8)))['predictions'].tolist() == \
        objective.native_evaluate(factors, list(range(8)))['predictions'].tolist()


def test_teacher_compression_is_vocabulary_chunk_independent_with_native_head_unchanged():
    objective, _ = fixture()
    first, _ = compress_teacher_moments(objective.teacher_h, objective.E_native, vocab_chunk=3)
    second, _ = compress_teacher_moments(objective.teacher_h, objective.E_native, vocab_chunk=23)
    np.testing.assert_array_equal(first, second)


def test_short_memory_prefix_keeps_nine_native_head_rows_before_selecting_eligible_positions(monkeypatch):
    generator=torch.Generator().manual_seed(81)
    E=torch.randn(19,5,generator=generator).bfloat16()
    full=torch.randn(9,5,generator=generator).bfloat16().float().numpy()
    obj=FullHiddenObjective(full[:3],np.zeros((3,5)),np.full(3,1/3),E,
                            native_h=full,native_row_indices=[0,1,2])
    original=torch.nn.functional.linear
    expected=original(torch.from_numpy(full).bfloat16(),E)[:3].argmax(-1).tolist()
    observed=[]
    def record(features,weight,*args,**kwargs):
        observed.append(features.shape[0])
        return original(features,weight,*args,**kwargs)
    monkeypatch.setattr(torch.nn.functional,'linear',record)
    result=obj.native_evaluate(Factors.zero(5),expected)
    assert observed==[9] and result['native_head_row_counts']==[9]
    assert result['predictions'].tolist()==expected and result['prefix_lengths']==[3]
    portable=obj.portable()
    assert portable['native_h'].shape==(9,5) and portable['h'].shape==(3,5)
    restored=FullHiddenObjective(portable['h'],portable['mu'],portable['omega'],E,
        native_h=portable['native_h'],native_row_indices=portable['native_row_indices'])
    assert restored.native_evaluate(None,expected)['prefix_lengths']==[3]
    with pytest.raises(ValueError,match='all nine'):
        FullHiddenObjective(full[:3],np.zeros((3,5)),np.full(3,1/3),E,
                            native_h=full[:3],native_row_indices=[0,1,2])
