"""CPU tests of observed native rounding, with no learning or model execution."""
import json

import numpy as np
import pytest
import torch

from .feedback import FrozenResponseObjective
from .precision import qualify_native_precision


def _objective(*, empty=False, common_logit_shift=False):
    generator = torch.Generator().manual_seed(17)
    E = (torch.randn(32, 8, generator=generator) / 4).to(torch.bfloat16)
    if common_logit_shift:
        E = torch.full((32, 8), .125, dtype=torch.bfloat16)
    n = 0 if empty else 5
    h = torch.ones(n, 8, dtype=torch.bfloat16)
    teacher_h = h.clone()
    if n:
        teacher_h[:, 1] = 1.5
    weights = np.full(n, 1/n) if n else np.empty(0)
    U = np.eye(8, dtype=np.float64)[:, :2]
    V = U.copy()
    return FrozenResponseObjective(h, teacher_h, weights, E, U, V, projection_chunk=2)


def test_zero_tiny_and_surviving_corrections_are_observations():
    objective = _objective()
    matrices = {'zero': np.zeros((2, 2)),
                'served_tiny': np.eye(2)*1e-6,
                'controlled_nonzero': np.eye(2)*.25}
    before_h = objective.h.clone()
    before_teacher = objective.teacher_h.clone()
    before_calls = list(objective.calls)
    result = qualify_native_precision(objective, matrices, chunk_size=2)
    rows = {row['label']: row for row in result['cases']}
    assert result['status'] == 'measured' and result['performance_gate'] is False
    assert result['dtype']['native_hidden_and_projection'] == 'torch.bfloat16'
    assert result['sampled_positions'] == 5
    zero = rows['zero']
    assert zero['native_loss_change'] == 0 and zero['reference_loss_change'] == 0
    assert zero['fraction_changed_positions'] == 0
    assert zero['fraction_changed_coordinates'] == 0
    assert zero['centered_logit_native_to_reference_norm_ratio'] is None
    assert zero['feature_native_to_precast_norm_ratio'] is None
    tiny = rows['served_tiny']
    assert tiny['weighted_centered_reference_logit_delta_norm'] > 0
    assert tiny['weighted_feature_precast_delta_norm'] > 0
    assert tiny['weighted_feature_native_delta_norm'] == 0
    assert tiny['weighted_centered_native_logit_delta_norm'] == 0
    assert tiny['native_loss_change'] == 0
    assert tiny['fraction_precast_nonzero_coordinates_surviving'] == 0
    assert tiny['centered_logit_native_to_reference_norm_ratio'] == 0
    assert tiny['feature_rounding_relative_error'] == pytest.approx(1.)
    assert tiny['nonzero_reference_but_zero_native_logit_change'] is True
    large = rows['controlled_nonzero']
    assert large['correction_survived_feature_rounding'] is True
    assert large['fraction_changed_positions'] == 1.
    assert large['fraction_changed_coordinates'] == .25
    assert large['fraction_precast_nonzero_coordinates_surviving'] == 1.
    assert large['weighted_centered_native_logit_delta_norm'] > 0
    assert large['centered_logit_native_to_reference_norm_ratio'] > 0
    # Diagnostic evaluation does not train, change features, or add observations
    # to the mathematical objective's learning-call ledger.
    assert torch.equal(objective.h, before_h)
    assert torch.equal(objective.teacher_h, before_teacher)
    assert objective.calls == before_calls
    assert all(len(row['rows']) == 5 for row in result['cases'])
    json.dumps(result, allow_nan=False)
    # Compare to the independent retained objective after checking no mutation.
    for label, C in matrices.items():
        loss, _ = objective.loss_gradient(C)
        assert rows[label]['reference_loss_change'] == pytest.approx(loss, abs=1e-12)
        assert rows[label]['reference_absolute_kl'] - rows[label]['reference_baseline_kl'] == pytest.approx(loss, abs=1e-12)
        assert rows[label]['native_absolute_kl'] - rows[label]['native_baseline_kl'] == pytest.approx(rows[label]['native_loss_change'], abs=1e-12)


def test_empty_sample_is_explicit_with_null_ratios():
    result = qualify_native_precision(_objective(empty=True), {'fitted': np.eye(2)})
    row = result['cases'][0]
    assert result['sampled_positions'] == 0 and row['no_eligible_anchors']
    assert row['rows'] == []
    assert row['fraction_changed_positions'] is None
    assert row['fraction_changed_coordinates'] is None
    assert row['fraction_precast_nonzero_coordinates_surviving'] is None
    assert row['feature_native_to_reference_norm_ratio'] is None
    assert row['centered_logit_native_to_reference_norm_ratio'] is None
    assert row['centered_logit_relative_discrepancy'] is None
    assert row['correction_survived_feature_rounding'] is None
    assert row['native_loss_change'] == 0 and row['reference_loss_change'] == 0
    json.dumps(result, allow_nan=False)


def test_centering_removes_softmax_invariant_logit_shift():
    result = qualify_native_precision(_objective(common_logit_shift=True), {'controlled': np.eye(2)*.25})
    row = result['cases'][0]
    assert row['fraction_changed_positions'] == 1.
    # All vocabulary rows are identical: the residual changes logits only by a
    # common offset, so its softmax-relevant change is exactly zero.
    assert row['weighted_centered_reference_logit_delta_norm'] == 0.
    assert row['weighted_centered_native_logit_delta_norm'] == 0.
    assert row['centered_logit_native_to_reference_norm_ratio'] is None
    assert abs(row['reference_loss_change']) < 1e-14
    assert abs(row['native_loss_change']) < 1e-14


def test_precision_qualification_keeps_bounded_chunks_and_rejects_invalid_cases():
    objective = _objective()
    with pytest.raises(ValueError, match='bounded chunks'):
        qualify_native_precision(objective, {'case': np.eye(2)}, chunk_size=33)
    with pytest.raises(ValueError, match='Invalid correction'):
        qualify_native_precision(objective, {'case': np.ones((3, 3))})
    with pytest.raises(ValueError, match='Invalid correction'):
        qualify_native_precision(objective, {'case': np.full((2, 2), np.nan)})
    empty_cases = qualify_native_precision(objective, {})
    assert empty_cases['cases'] == []
    json.dumps(empty_cases, allow_nan=False)
