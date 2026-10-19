"""Qualification-only measurement of the mathematical/native precision boundary.

This does not update the bank, select anchors, or certify native retention. It
measures the exact serving helper (FP32 correction, native hidden-state cast,
then the native vocabulary projection) against the fixed-feature FP64 objective
on the already sampled response positions. Full vocabulary rows are transient.
"""
from __future__ import annotations

import math
import time
from typing import Mapping

import numpy as np

from .native_hooks import corrected_features


_SCOPE = (
    "The protected-loss inequalities apply to the fixed-feature FP64 affine-head "
    "objective. This observation measures the separate quantized native path; "
    "it does not extend those guarantees to native logits, greedy acceptance, "
    "or serving latency. Neither correction survival nor a gain is a pass gate."
)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _norm(square: float) -> float:
    return math.sqrt(max(0., square))


def qualify_native_precision(objective, matrices: Mapping[str, np.ndarray], *,
                             chunk_size: int = 32) -> dict:
    """Measure caller-labelled served/fitted/controlled corrections without learning.

    ``objective`` is the existing ``FrozenResponseObjective``. Each matrix is
    evaluated on exactly its retained h/teacher_h/omega, with the same teacher
    probabilities as its direct objective. A caller may provide zero or more
    matrices; labels are provenance and do not alter the calculation.

    Weighted norms mean sqrt(sum_a omega[a] * sum_coordinates delta[a]**2).
    Logit deltas are centered across vocabulary per position before comparing
    magnitudes, because adding one constant to every logit is softmax-invariant.
    Loss changes use each execution path's own zero-correction baseline.
    """
    import torch
    import torch.nn.functional as F

    if type(chunk_size) is not int or not 1 <= chunk_size <= 32:
        raise ValueError("Precision qualification requires bounded chunks of 1--32 positions")
    if not isinstance(matrices, Mapping):
        raise TypeError("Matrices must be a caller-labelled mapping")
    rank_o, rank_i = objective.U.shape[1], objective.V_in.shape[1]
    validated = {}
    for label, matrix in matrices.items():
        if not isinstance(label, str) or not label:
            raise ValueError("Precision case labels must be nonempty strings")
        matrix = np.array(matrix, dtype=np.float64, copy=True)
        if matrix.shape != (rank_o, rank_i) or not np.all(np.isfinite(matrix)):
            raise ValueError("Invalid correction matrix for precision qualification")
        validated[label] = matrix
    n = len(objective.h)
    if len(objective.teacher_h) != n or len(objective.omega) != n:
        raise ValueError("Qualification must use aligned existing sampled positions")
    vocab, width = tuple(objective.E_native.shape)
    if n and (not np.all(np.isfinite(objective.omega)) or
              not np.isclose(np.sum(objective.omega), 1., atol=1e-12)):
        raise ValueError("Sampled response weights must be normalized")
    accumulators = {}
    for label, matrix in validated.items():
        accumulators[label] = dict(
            label=label, correction_frobenius_norm=float(np.linalg.norm(matrix)),
            zero_correction=not bool(np.any(matrix)), anchor_count=n,
            sampled_weight_sum=float(np.sum(objective.omega)),
            changed_positions=0, changed_coordinates=0,
            nonzero_precast_coordinates=0, surviving_precast_coordinates=0,
            weighted_changed_positions=0., weighted_changed_coordinate_fraction=0.,
            reference_loss_change=0., native_loss_change=0.,
            reference_absolute_kl=0., reference_baseline_kl=0.,
            native_absolute_kl=0., native_baseline_kl=0.,
            feature_reference_square=0., feature_precast_square=0.,
            feature_native_square=0., feature_rounding_error_square=0.,
            logit_reference_square=0., logit_native_square=0.,
            logit_discrepancy_square=0., max_centered_logit_discrepancy=0.,
            rows=[])
    objective.sync()
    started = time.perf_counter()
    with torch.no_grad():
        U32 = objective.U.to(dtype=torch.float32)
        V32 = objective.V_in.to(dtype=torch.float32)
        candidates = {label: (
            torch.as_tensor(matrix, dtype=torch.float64, device=objective.device),
            torch.as_tensor(matrix, dtype=torch.float32, device=objective.device))
            for label, matrix in validated.items()}
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            hidden64 = objective.h[start:stop].to(objective.device)
            hidden_native = hidden64.to(dtype=objective.E_native.dtype)
            teacher_h = objective.teacher_h[start:stop].to(objective.device)
            omega = torch.as_tensor(objective.omega[start:stop], dtype=torch.float64,
                                    device=objective.device)
            phi64 = torch.as_tensor(objective.phi[start:stop], dtype=torch.float64,
                                   device=objective.device)
            teacher_logits = objective._teacher_logits(teacher_h).double()
            teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
            teacher = torch.softmax(teacher_logits, dim=-1)
            native_baseline = F.linear(hidden_native, objective.E_native).double()
            reference_baseline = hidden64 @ objective.E.T
            reference_logq0 = torch.log_softmax(reference_baseline, dim=-1)
            native_logq0 = torch.log_softmax(native_baseline, dim=-1)
            baseline_reference_kl = (teacher * (teacher_logp - reference_logq0)).sum(dim=-1)
            baseline_native_kl = (teacher * (teacher_logp - native_logq0)).sum(dim=-1)
            phi32 = hidden_native.float() @ V32
            phi32 = phi32 / phi32.norm(dim=-1, keepdim=True).clamp_min(1.)
            for label, (C64, C32) in candidates.items():
                output = corrected_features(hidden_native, C32, U32, V32)
                native_logits = F.linear(output, objective.E_native).double()
                reference_delta = (phi64 @ C64.T) @ objective.P.T
                reference_logits = reference_baseline + reference_delta
                native_delta = native_logits - native_baseline
                reference_centered = reference_delta - reference_delta.mean(dim=-1, keepdim=True)
                native_centered = native_delta - native_delta.mean(dim=-1, keepdim=True)
                logit_error = native_centered - reference_centered
                feature_precast = (phi32 @ C32.T) @ U32.T
                feature_reference = (phi64 @ C64.T) @ objective.U.T
                feature_native = output.double() - hidden_native.double()
                feature_rounding_error = feature_native - feature_precast.double()
                changed = output != hidden_native
                changed_positions = changed.any(dim=-1)
                nonzero_precast = feature_precast != 0
                retained_coordinates = changed & nonzero_precast
                reference_logq = torch.log_softmax(reference_logits, dim=-1)
                native_logq = torch.log_softmax(native_logits, dim=-1)
                reference_loss_change = (teacher * (reference_logq0 - reference_logq)).sum(dim=-1)
                native_loss_change = (teacher * (native_logq0 - native_logq)).sum(dim=-1)
                reference_kl = (teacher * (teacher_logp - reference_logq)).sum(dim=-1)
                native_kl = (teacher * (teacher_logp - native_logq)).sum(dim=-1)
                per_position = {
                    "reference_loss_change": reference_loss_change,
                    "native_loss_change": native_loss_change,
                    "reference_absolute_kl": reference_kl,
                    "reference_baseline_kl": baseline_reference_kl,
                    "native_absolute_kl": native_kl,
                    "native_baseline_kl": baseline_native_kl,
                    "feature_reference_square": feature_reference.square().sum(dim=-1),
                    "feature_precast_square": feature_precast.double().square().sum(dim=-1),
                    "feature_native_square": feature_native.square().sum(dim=-1),
                    "feature_rounding_error_square": feature_rounding_error.square().sum(dim=-1),
                    "logit_reference_square": reference_centered.square().sum(dim=-1),
                    "logit_native_square": native_centered.square().sum(dim=-1),
                    "logit_discrepancy_square": logit_error.square().sum(dim=-1),
                }
                accumulator = accumulators[label]
                for key, value in per_position.items():
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("Nonfinite precision diagnostic: " + key)
                    accumulator[key] += float((omega * value).sum())
                accumulator["changed_positions"] += int(changed_positions.sum())
                accumulator["changed_coordinates"] += int(changed.sum())
                accumulator["nonzero_precast_coordinates"] += int(nonzero_precast.sum())
                accumulator["surviving_precast_coordinates"] += int(retained_coordinates.sum())
                accumulator["weighted_changed_positions"] += float((omega * changed_positions).sum())
                accumulator["weighted_changed_coordinate_fraction"] += float((omega * changed.double().mean(dim=-1)).sum())
                accumulator["max_centered_logit_discrepancy"] = max(
                    accumulator["max_centered_logit_discrepancy"], float(logit_error.abs().max()))
                # Small per-anchor evidence only; never retain vocabulary logits.
                small = {key: value.detach().cpu().tolist() for key, value in per_position.items()}
                changed_count = changed.sum(dim=-1).cpu().tolist()
                weights = omega.cpu().tolist()
                for i in range(stop-start):
                    accumulator["rows"].append(dict(
                        sampled_position_index=start+i, weight=weights[i],
                        changed_feature_coordinates=changed_count[i],
                        **{key: values[i] for key, values in small.items()}))
    objective.sync()
    wall_s = time.perf_counter() - started
    cases = []
    for accumulator in accumulators.values():
        feature_reference_norm = _norm(accumulator.pop("feature_reference_square"))
        feature_precast_norm = _norm(accumulator.pop("feature_precast_square"))
        feature_native_norm = _norm(accumulator.pop("feature_native_square"))
        rounding_error_norm = _norm(accumulator.pop("feature_rounding_error_square"))
        reference_norm = _norm(accumulator.pop("logit_reference_square"))
        native_norm = _norm(accumulator.pop("logit_native_square"))
        error_norm = _norm(accumulator.pop("logit_discrepancy_square"))
        accumulator.update(
            no_eligible_anchors=n == 0,
            fraction_changed_positions=_ratio(accumulator["changed_positions"], n),
            fraction_changed_coordinates=_ratio(accumulator["changed_coordinates"], n*width),
            fraction_precast_nonzero_coordinates_surviving=_ratio(
                accumulator["surviving_precast_coordinates"], accumulator["nonzero_precast_coordinates"]),
            weighted_feature_reference_delta_norm=feature_reference_norm,
            weighted_feature_precast_delta_norm=feature_precast_norm,
            weighted_feature_native_delta_norm=feature_native_norm,
            feature_native_to_reference_norm_ratio=_ratio(feature_native_norm, feature_reference_norm),
            feature_native_to_precast_norm_ratio=_ratio(feature_native_norm, feature_precast_norm),
            feature_rounding_relative_error=_ratio(rounding_error_norm, feature_precast_norm),
            weighted_centered_reference_logit_delta_norm=reference_norm,
            weighted_centered_native_logit_delta_norm=native_norm,
            centered_logit_native_to_reference_norm_ratio=_ratio(native_norm, reference_norm),
            weighted_centered_logit_discrepancy_norm=error_norm,
            centered_logit_relative_discrepancy=_ratio(error_norm, reference_norm),
            native_minus_reference_loss_change=(accumulator["native_loss_change"]-
                                               accumulator["reference_loss_change"]),
            correction_survived_feature_rounding=bool(accumulator["changed_positions"]) if n else None,
            nonzero_reference_but_zero_native_logit_change=bool(reference_norm > 0 and native_norm == 0) if n else None)
        cases.append(accumulator)
    return dict(schema="correction_bank_native_precision_v1", status="measured",
        qualification_only=True, observational_only=True, performance_gate=False,
        retention_scope=_SCOPE, wall_s=wall_s, sampled_positions=n,
        hidden_width=width, vocabulary_size=vocab, chunk_size=chunk_size,
        dtype=dict(native_hidden_and_projection=str(objective.E_native.dtype),
                   correction="torch.float32", reference="torch.float64",
                   teacher="native projection logits, float64 softmax; same for both losses"),
        norm_definition="sqrt(sum_position omega * sum_coordinate centered_delta**2); logit centering is per vocabulary row",
        baseline_definition="Each loss change is relative to its own uncorrected path, using identical teacher probabilities",
        cases=cases)
