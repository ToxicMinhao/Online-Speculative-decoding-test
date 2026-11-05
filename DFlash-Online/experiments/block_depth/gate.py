"""Fixed CPU calibration for the frozen K+ block-depth gate."""

import hashlib
import math

import numpy as np

from .data import STUDY_ID


FEATURES = 97
RIDGE_PENALTY = 1e-3
SCALE_FLOOR = 1e-6


def select_gate_starts(request_id, observed_rounds):
    """Hash-select at most 64 native starts, returned in replay order."""
    if observed_rounds < 0:
        raise ValueError("Native round count cannot be negative")
    ordered = sorted(range(observed_rounds), key=lambda index: hashlib.sha256(
        f"{STUDY_ID}|GATE|{request_id}|{index}".encode()).digest())
    return sorted(ordered[:64])


def progress_for_proposals(proposals, canonical_suffix_after_seed, stop_token_ids):
    """Native accepted prefix plus repair/bonus, excluding the known seed.

    The saved canonical suffix ends at EOS or the output cap. Matching EOS
    commits that token without a further bonus. A cap-ending suffix censors
    any additional proposal or bonus beyond the saved boundary.
    """
    if len(proposals) != 15:
        raise ValueError("A block must contain all 15 unknown proposals")
    suffix, stops = list(canonical_suffix_after_seed), set(stop_token_ids)
    if any(token in stops for token in suffix[:-1]):
        raise ValueError("Canonical continuation extends beyond EOS")
    accepted, matched_eos = 0, False
    for proposal, target in zip(proposals, suffix):
        if proposal != target:
            break
        accepted += 1
        if target in stops:
            matched_eos = True
            break
    progress = min(len(suffix), accepted + (not matched_eos))
    return dict(accepted_draft_tokens=accepted, committed_progress=progress,
                matched_eos=matched_eos,
                censored_at_cap=bool(suffix and suffix[-1] not in stops and progress == len(suffix)))


def pooled_features(rank_features, committed_prefix_length):
    """CPU reference for the three proposal groups and log-prefix feature."""
    features = np.asarray(rank_features, dtype=np.float64)
    if features.shape != (15, 32) or committed_prefix_length < 0:
        raise ValueError("Gate features require 15 by 32 proposal features and a valid prefix")
    return np.concatenate([features[:3].mean(0), features[3:7].mean(0),
                           features[7:15].mean(0), [math.log1p(committed_prefix_length)]])


def weighted_ridge(features, targets, weights):
    """FP64 solve of mean weighted square error plus 1e-3 ||beta||².

    All centering/scaling is computed from these GATE rows only. The intercept
    is not penalized. Returned coefficients operate directly on native features.
    """
    x, y, w = (np.asarray(value, dtype=np.float64) for value in (features, targets, weights))
    if x.ndim != 2 or x.shape[1] != FEATURES or y.shape != (len(x),) or w.shape != y.shape:
        raise ValueError("Invalid gate design dimensions")
    if not len(x) or not all(np.isfinite(v).all() for v in (x, y, w)) or np.any(w <= 0):
        raise ValueError("Gate calibration requires finite rows with positive weights")
    normalized = w / w.sum()
    center = normalized @ x
    centered = x - center
    scale = np.maximum(np.sqrt(normalized @ (centered ** 2)), SCALE_FLOOR)
    standardized = centered / scale
    intercept = float(normalized @ y)
    gram = standardized.T @ (normalized[:, None] * standardized)
    rhs = standardized.T @ (normalized * (y - intercept))
    coefficient = np.linalg.solve(gram + RIDGE_PENALTY * np.eye(FEATURES), rhs)
    folded = coefficient / scale
    folded_intercept = float(intercept - center @ folded)
    prediction = x @ folded + folded_intercept
    return dict(
        feature_mean=center.tolist(), feature_scale=scale.tolist(),
        standardized_weight=coefficient.tolist(), standardized_intercept=intercept,
        folded_weight=folded.tolist(), folded_intercept=folded_intercept,
        weighted_mse=float(normalized @ ((prediction - y) ** 2)),
        penalized_objective=float(normalized @ ((prediction - y) ** 2) + RIDGE_PENALTY * (coefficient @ coefficient)),
        weight_sum=float(w.sum()), rows=len(x), precision="numpy float64",
        solver="numpy.linalg.solve; centered weighted Gram plus lambda I",
        native_coefficient_precision="float32; folded FP64 solution cast once by native loader",
        ridge_penalty=RIDGE_PENALTY, scale_floor=SCALE_FLOOR,
        scaling="GATE-only weighted population mean/std with observed_rounds/sampled_rounds",
        objective="sum(w*(prediction-y)^2)/sum(w) + 1e-3*sum(standardized_weight^2)",
    )


def fit_gate(rows, behavior_records, continuation_cost_ms):
    """Validate the fixed 128-request calibration and fit its signed target."""
    if len(behavior_records) != 128 or not math.isfinite(continuation_cost_ms):
        raise ValueError("Calibration requires all 128 behaviors and a finite priced continuation cost")
    by_request = {r["request_id"]: r for r in behavior_records}
    if len(by_request) != 128:
        raise ValueError("Duplicate calibration behavior")
    groups = {key: [] for key in by_request}
    for row in rows:
        groups[row["request_id"]].append(row)
    for ordinal, behavior in enumerate(behavior_records):
        expected = "FIVE" if ordinal % 2 == 0 else "THREE"
        if behavior["behavior"] != expected:
            raise ValueError("GATE behavior must alternate by sealed ordinal")
        samples = groups[behavior["request_id"]]
        n = behavior["observed_rounds"]
        indices = select_gate_starts(behavior["request_id"], n)
        if sorted(r["round_index"] for r in samples) != indices:
            raise ValueError("Calibration starts differ from the sealed hash selection")
        for row in samples:
            if row["observed_rounds"] != n or row["sampled_rounds"] != len(indices):
                raise ValueError("Calibration expansion weight does not match native rounds")
            if any(type(row[key]) is not int or not 0 <= row[key] <= 16 for key in ("p3", "p5")):
                raise ValueError("Progress labels must be native whole-token counts")
    weights = [r["observed_rounds"] / r["sampled_rounds"] for r in rows]
    fit = weighted_ridge([r["features"] for r in rows], [r["p5"] - r["p3"] for r in rows], weights)
    five = [r for r in behavior_records if r["behavior"] == "FIVE"]
    time = math.fsum(r["decode_round_wall_s"] for r in five)
    progress = sum(r["committed_progress"] for r in five)
    if not math.isfinite(time) or time <= 0 or progress <= 0:
        raise ValueError("Native FIVE decode cost/progress cannot define kappa")
    fit.update(kappa_ms_per_token=1000 * time / progress, continuation_cost_ms=float(continuation_cost_ms),
               kappa_decode_round_wall_s=time, kappa_committed_progress=progress,
               kappa_behavior_requests=len(five), requests=len(behavior_records),
               target="signed native committed progress P5 - P3; prefill excluded",
               target_min=min(r["p5"] - r["p3"] for r in rows),
               target_max=max(r["p5"] - r["p3"] for r in rows),
               negative_targets=sum(r["p5"] < r["p3"] for r in rows),
               decision="continue to FIVE iff kappa_ms_per_token * prediction >= continuation_cost_ms")
    return fit
