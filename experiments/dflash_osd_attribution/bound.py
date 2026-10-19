"""Pairwise capacity of the *combined* correction-bank v0.1 matrix.

This is an attribution diagnostic, not a correction solver. For the original
mathematical head ``h + U @ C_eff @ phi(h)`` and ``||C_eff||_F <= 1/kappa``,
Cauchy--Schwarz gives the largest possible pairwise improvement against a fixed
competitor. Passing this screen does not imply a feasible shared correction,
preserved historical gains, or a change in native BF16 drafting.

The tolerance below is declared in source before inspecting campaign outcomes.
It is a conservative separation convention for float64 diagnostics, not a
certificate for every floating-point implementation. Native BF16 margins are
recorded separately and never enter this mathematical comparison.
"""
from __future__ import annotations

from typing import Any

import numpy as np


ABSOLUTE_TOLERANCE = 1e-8
RELATIVE_TOLERANCE = 1e-7
TOLERANCE_FORMULA = "1e-8 + 1e-7 * max(1, abs(deficit), abs(bound))"
MATHEMATICAL_SCOPE = (
    "float64 evaluation of the fixed mathematical head; effective combined "
    "matrix Frobenius radius 1/kappa, including permitted fusion; excludes no "
    "native BF16 implementation by itself"
)


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    initial = np.asarray(value)
    if np.iscomplexobj(initial):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a float64-compatible array") from error
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def clipped_feature(h: Any, V_in: Any) -> np.ndarray:
    """Apply the original feature map, including its unit-norm clipping.

    The vectors and fixed projection are converted to float64 before either
    projection or clipping. The input artifacts are never changed.
    """
    hidden = _array(h, "h", 1)
    projection = _array(V_in, "V_in", 2)
    if projection.shape[0] != hidden.size:
        raise ValueError("V_in input dimension must equal the hidden dimension")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        raw = hidden @ projection
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm):
            raise ValueError("Projected feature norm is not finite in float64")
        return raw / max(1.0, norm)


def pairwise_capacity(
    h: Any,
    E_y: Any,
    E_j: Any,
    U: Any,
    V_in: Any,
    kappa: float,
    *,
    valid_pair: bool = True,
    frozen_error: bool = True,
    osd_repaired: bool = True,
    native_bf16_margin: float | None = None,
) -> dict[str, Any]:
    """Screen one OSD-repaired frozen decision against the bank's capacity.

    ``h`` is the *frozen* pre-vocabulary-projection feature. ``E_y`` is the
    canonical target-token row and ``E_j`` the row for the frozen native winning
    competitor, both from the same fixed vocabulary projection. Caller-owned
    token IDs and artifact identities belong in the surrounding evidence row.

    ``native_bf16_margin`` means canonical logit minus its strongest competitor
    in native BF16 inference. It is optional reporting metadata, not ``-d``:
    the native strongest competitor and the mathematical pair can differ.

    Classification is attempted only for a valid pair, a frozen error, and an
    OSD repair. Otherwise the quantities are retained but classification is
    ``not_evaluated``. An eligible case is excluded only when ``d - B > tol``;
    ``abs(d - B) <= tol`` is borderline; all other cases are not excluded.
    """
    hidden = _array(h, "h", 1)
    target = _array(E_y, "E_y", 1)
    competitor = _array(E_j, "E_j", 1)
    basis = _array(U, "U", 2)
    feature_projection = _array(V_in, "V_in", 2)
    if target.shape != hidden.shape or competitor.shape != hidden.shape:
        raise ValueError("Vocabulary rows must match the hidden dimension")
    if basis.shape[0] != hidden.size or feature_projection.shape[0] != hidden.size:
        raise ValueError("Projection input dimensions must match the hidden dimension")
    if isinstance(kappa, (bool, np.bool_)) or not np.isscalar(kappa):
        raise ValueError("kappa must be a finite positive scalar")
    try:
        kappa_value = float(kappa)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("kappa must be a finite positive scalar") from error
    if not np.isfinite(kappa_value) or kappa_value <= 0:
        raise ValueError("kappa must be a finite positive scalar")
    for name, value in (("valid_pair", valid_pair), ("frozen_error", frozen_error),
                        ("osd_repaired", osd_repaired)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    if native_bf16_margin is not None:
        native_bf16_margin = float(native_bf16_margin)
        if not np.isfinite(native_bf16_margin):
            raise ValueError("Native BF16 margin must be finite when present")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        radius = 1.0 / kappa_value
        phi = clipped_feature(hidden, feature_projection)
        raw_norm = float(np.linalg.norm(hidden @ feature_projection))
        phi_norm = float(np.linalg.norm(phi))
        difference = target - competitor
        projected_difference = basis.T @ difference
        projected_difference_norm = float(np.linalg.norm(projected_difference))
        deficit = float((competitor - target) @ hidden)
        bound = float(radius * projected_difference_norm * phi_norm)
        gap = deficit - bound
        tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(
            1.0, abs(deficit), abs(bound)
        )
    if not all(np.isfinite(value) for value in (
        radius, raw_norm, phi_norm, projected_difference_norm, deficit, bound,
        gap, tolerance,
    )):
        raise ValueError("Capacity calculation produced a nonfinite float64 quantity")

    reasons = []
    if not valid_pair:
        reasons.append("invalid_pair")
    if not frozen_error:
        reasons.append("frozen_prediction_not_an_error")
    if not osd_repaired:
        reasons.append("not_repaired_by_osd")
    if reasons:
        classification = "not_evaluated"
    elif abs(gap) <= tolerance:
        classification = "borderline"
    elif gap > tolerance:
        classification = "excluded"
    else:
        classification = "not_excluded"
    return {
        "classification": classification,
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "deficit": deficit,
        "bound": bound,
        "deficit_minus_bound": gap,
        "tolerance": tolerance,
        "tolerance_formula": TOLERANCE_FORMULA,
        "arithmetic_dtype": "float64",
        "effective_radius": radius,
        "kappa": kappa_value,
        "radius_applies_to": "combined_effective_matrix_including_fusion",
        "phi_raw_norm": raw_norm,
        "phi_norm": phi_norm,
        "phi_clipped": raw_norm > 1.0,
        "projected_token_difference_norm": projected_difference_norm,
        "native_bf16_canonical_margin": native_bf16_margin,
        "native_margin_used_for_classification": False,
        "interpretation_scope": MATHEMATICAL_SCOPE,
    }
