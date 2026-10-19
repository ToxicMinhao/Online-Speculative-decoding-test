"""Synthetic CPU checks; these tests do not inspect campaign outcomes."""
import inspect

import numpy as np
import pytest

from experiments.dflash_correction_bank.projections import FixedProjections
from experiments.dflash_osd_attribution.bound import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    clipped_feature,
    pairwise_capacity,
)


def simple(kappa=1.0, **kwargs):
    return pairwise_capacity(
        np.array([-1.0, 0.0]), np.array([1.0, 0.0]), np.zeros(2),
        np.eye(2), np.eye(2), kappa, **kwargs
    )


def test_feature_map_matches_original_clipping_without_mutation():
    original = FixedProjections(np.eye(2), np.eye(2), np.eye(2), np.eye(2), 2.0)
    for hidden in (np.array([-3.0, 4.0]), np.array([0.3, 0.4]), np.zeros(2)):
        saved = hidden.copy()
        np.testing.assert_array_equal(clipped_feature(hidden, original.V_in), original.phi(hidden))
        np.testing.assert_array_equal(hidden, saved)
    np.testing.assert_array_equal(clipped_feature([-3.0, 4.0], np.eye(2)), [-0.6, 0.8])


def test_clipping_applies_to_capacity():
    row = pairwise_capacity([-3.0, 4.0], [1.0, 0.0], [0.0, 0.0], np.eye(2), np.eye(2), 2)
    assert row["phi_raw_norm"] == 5
    assert row["phi_norm"] == 1
    assert row["phi_clipped"]
    assert row["deficit"] == 3
    assert row["bound"] == 0.5


def test_radius_is_combined_radius_not_module_count():
    row = simple(2)
    assert row["effective_radius"] == 0.5
    assert row["bound"] == 0.5
    assert row["classification"] == "excluded"
    assert row["radius_applies_to"] == "combined_effective_matrix_including_fusion"
    # Four selected components do not enlarge ||C_eff||_F <= r.
    assert "module_count" not in inspect.signature(pairwise_capacity).parameters
    components = np.repeat(np.array([[[0.125, 0], [0, 0]]]), 4, axis=0)
    assert np.linalg.norm(components.sum(axis=0)) == row["effective_radius"]


def test_bound_attained_by_rank_one_unconstrained_pairwise_witness():
    rng = np.random.default_rng(5)
    hidden = rng.normal(size=6)
    target = rng.normal(size=6)
    competitor = rng.normal(size=6)
    basis = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    projection = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    row = pairwise_capacity(hidden, target, competitor, basis, projection, 3.0)
    phi = clipped_feature(hidden, projection)
    q = basis.T @ (target - competitor)
    witness = row["effective_radius"] * np.outer(q / np.linalg.norm(q), phi / np.linalg.norm(phi))
    pairwise_gain = (target - competitor) @ basis @ witness @ phi
    assert np.linalg.norm(witness) == pytest.approx(row["effective_radius"])
    assert pairwise_gain == pytest.approx(row["bound"], abs=1e-14)
    for _ in range(10):
        correction = rng.normal(size=(3, 2))
        correction *= row["effective_radius"] / np.linalg.norm(correction)
        assert (target - competitor) @ basis @ correction @ phi <= row["bound"] + 1e-14


@pytest.mark.parametrize("kappa,expected", [(2, "excluded"), (1, "borderline"), (0.5, "not_excluded")])
def test_classification_uses_declared_separation_tolerance(kappa, expected):
    row = simple(kappa)
    assert row["classification"] == expected
    assert row["deficit"] == 1
    assert row["tolerance"] == ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(1, row["bound"])
    assert row["arithmetic_dtype"] == "float64"


@pytest.mark.parametrize("gap,expected", [(5e-8, "borderline"), (-5e-8, "borderline"),
                                         (1e-5, "excluded"), (-1e-5, "not_excluded")])
def test_small_numerical_differences_are_inconclusive(gap, expected):
    assert simple(1.0 / (1 - gap))["classification"] == expected


def test_native_bf16_margin_does_not_enter_mathematical_screen():
    lower = simple(2, native_bf16_margin=-1000.0)
    higher = simple(2, native_bf16_margin=1000.0)
    assert lower["native_bf16_canonical_margin"] == -1000
    assert higher["native_bf16_canonical_margin"] == 1000
    for key in ("classification", "deficit", "bound", "tolerance"):
        assert lower[key] == higher[key]
    assert lower["native_margin_used_for_classification"] is False


@pytest.mark.parametrize("kwargs,reason", [
    ({"valid_pair": False}, "invalid_pair"),
    ({"frozen_error": False}, "frozen_prediction_not_an_error"),
    ({"osd_repaired": False}, "not_repaired_by_osd"),
])
def test_ineligible_decisions_do_not_contribute_to_exclusion_fraction(kwargs, reason):
    row = simple(2, **kwargs)
    assert not row["eligible"]
    assert row["classification"] == "not_evaluated"
    assert row["ineligibility_reasons"] == [reason]


def test_zero_projected_difference_and_zero_feature_are_supported():
    no_projection = pairwise_capacity([-1.0, 0], [1.0, 0], [0.0, 0],
                                     np.zeros((2, 1)), np.eye(2), 2)
    no_feature = pairwise_capacity([-1.0, 0], [1.0, 0], [0.0, 0],
                                  np.eye(2), np.zeros((2, 1)), 2)
    for row in (no_projection, no_feature):
        assert row["bound"] == 0
        assert row["classification"] == "excluded"


@pytest.mark.parametrize("kappa", [0, -1, float("nan"), float("inf"), True, [1, 2], 1j])
def test_invalid_radius_artifacts_are_rejected(kappa):
    with pytest.raises(ValueError):
        simple(kappa)


@pytest.mark.parametrize("field,value", [
    ("h", [[1.0, 2.0]]), ("h", [np.nan, 0]), ("h", [1j, 0]),
    ("E_y", [1.0]), ("E_j", [np.inf, 0]),
    ("U", np.ones((3, 2))), ("U", np.empty((2, 0))),
    ("V_in", np.ones((3, 2))), ("V_in", np.array([[1, 0], [0, np.nan]])),
])
def test_invalid_arrays_are_rejected(field, value):
    arguments = dict(h=[-1.0, 0], E_y=[1.0, 0], E_j=[0.0, 0],
                     U=np.eye(2), V_in=np.eye(2), kappa=1.0)
    arguments[field] = value
    with pytest.raises(ValueError):
        pairwise_capacity(**arguments)


def test_input_float32_values_are_promoted_before_subtraction_and_products():
    hidden = np.array([-1.1234567, 0.7654321], dtype=np.float32)
    target = np.array([1234.5678, 1234.5678], dtype=np.float32)
    competitor = np.array([1234.5680, 1234.5676], dtype=np.float32)
    row = pairwise_capacity(hidden, target, competitor, np.eye(2), np.eye(2), 2)
    expected = (competitor.astype(np.float64) - target.astype(np.float64)) @ hidden.astype(np.float64)
    assert row["deficit"] == expected


def test_invalid_native_metadata_and_boolean_flags_are_rejected():
    with pytest.raises(ValueError):
        simple(native_bf16_margin=float("nan"))
    with pytest.raises(ValueError):
        simple(valid_pair="false")
