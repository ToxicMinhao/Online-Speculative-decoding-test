"""Response-grouped out-of-fold reachability prediction on retained CPU records."""

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


SEED = 20261127
FOLDS = 4
PENALTY = 1e-3
BOOTSTRAP_DRAWS = 2000
PROBABILITY_CLIP = 1e-12
STRUCT_FEATURES = (
    "e", "r", "n", "distance", "error_count", "u_old", "w",
    "log1p_first_error_deficit", "log1p_prefix_margin", "empty_prefix",
)
SEVERITY_FEATURES = STRUCT_FEATURES + (
    "log1p_later_mean_deficit", "log1p_later_max_deficit",
)
COMPARISONS = (("CONST", "STRUCT"), ("STRUCT", "SEVERITY"), ("CONST", "SEVERITY"))


def _features(rows, severity=False):
    features = [[
        row["e"], row["r"], row["n"], row["distance"], row["error_count"],
        row["u_old"], row["w"], np.log1p(row["first_error_deficit"]),
        np.log1p(row["prefix_margin"]), float(row["empty_prefix"]),
    ] for row in rows]
    if severity:
        for values, row in zip(features, rows):
            values.extend((np.log1p(row["later_mean_deficit"]),
                           np.log1p(row["later_max_deficit"])))
    return np.asarray(features, dtype=np.float64)


def _sample_weights(rows):
    """Each eligible response has equal mass, then each block, then follower."""
    block_counts = Counter((row["record_id"], row["anchor"]) for row in rows)
    blocks_per_response = Counter(record_id for record_id, _ in block_counts)
    response_count = len(blocks_per_response)
    return np.asarray([
        1.0 / (response_count * blocks_per_response[row["record_id"]]
               * block_counts[row["record_id"], row["anchor"]])
        for row in rows
    ], dtype=np.float64)


def _logistic_objective(parameters, design, target, weights):
    """Weighted mean BCE plus L2 on standardized coefficients, never intercept."""
    intercept, coefficients = parameters[0], parameters[1:]
    logits = intercept + design @ coefficients
    value = np.dot(weights, np.logaddexp(0, logits) - target * logits)
    value += PENALTY * np.dot(coefficients, coefficients) / 2
    residual = weights * (expit(logits) - target)
    gradient = np.concatenate(([residual.sum()], design.T @ residual + PENALTY * coefficients))
    return float(value), gradient


def _fit_logistic(train, target, weights, test):
    weights = weights / weights.sum()
    if np.unique(target).size < 2:
        return None, {"status": "unavailable", "reason": "training fold has one target class"}
    mean = weights @ train
    variance = weights @ ((train - mean) ** 2)
    scale = np.sqrt(variance)
    # A constant column remains constant even if its weighted mean rounds.
    scale[np.ptp(train, axis=0) == 0] = 1.0
    standardized = (train - mean) / scale
    prevalence = float(weights @ target)
    initial = np.zeros(train.shape[1] + 1)
    initial[0] = np.log(prevalence / (1 - prevalence))
    fit = minimize(_logistic_objective, initial, args=(standardized, target, weights),
                   jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-9, "ftol": 1e-12})
    information = {
        "status": "fitted" if fit.success else "unavailable",
        "message": str(fit.message), "iterations": int(fit.nit),
        "mean": mean.tolist(), "scale": scale.tolist(),
        "intercept": float(fit.x[0]), "coefficients": fit.x[1:].tolist(),
        "objective": float(fit.fun), "gradient_max_abs": float(np.abs(fit.jac).max()),
        "training_prevalence": prevalence,
    }
    if not (np.isfinite(fit.x).all() and np.isfinite(fit.fun) and np.isfinite(fit.jac).all()):
        information.update(status="unavailable", reason="nonfinite optimizer result")
    if information["status"] != "fitted":
        return None, information
    return expit(fit.x[0] + ((test - mean) / scale) @ fit.x[1:]), information


def _losses(probability, target):
    clipped = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return {
        "log_loss": -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped)),
        "brier": (probability - target) ** 2,
    }


def _weighted_metrics(probability, target, weights, mask):
    eligible = mask & np.isfinite(probability)
    result = {"rows": int(eligible.sum()), "unavailable_rows": int((mask & ~eligible).sum())}
    if not eligible.any():
        return dict(result, log_loss=None, brier=None)
    losses = _losses(probability[eligible], target[eligible])
    result.update({name: float(np.average(value, weights=weights[eligible]))
                   for name, value in losses.items()})
    return result


def _bootstrap_comparison(differences, rows, weights, draws):
    """Bootstrap fixed OOF response means; no model refits or replacement draws."""
    response_ids = sorted({row["record_id"] for row in rows})
    by_response = defaultdict(list)
    for index, row in enumerate(rows):
        by_response[row["record_id"]].append(index)
    means = np.asarray([
        np.average(differences[by_response[record_id]], weights=weights[by_response[record_id]])
        for record_id in response_ids
    ])
    estimates = means[draws].mean(axis=1)
    finite = np.isfinite(estimates)
    interval = np.quantile(estimates[finite], [0.025, 0.975]).tolist() if finite.any() else None
    return {
        "draws": [float(value) if np.isfinite(value) else None for value in estimates],
        "ci95": interval,
        "valid_draws": int(finite.sum()),
        "unidentifiable_draw_indices": np.flatnonzero(~finite).tolist(),
        "response_means": [float(value) if np.isfinite(value) else None for value in means],
    }


def run_prediction(follower_rows):
    """Fit the fixed three-model family; return public OOF and fit evidence.

    Input rows are eligible correct followers beyond the next parent mismatch.
    n denotes available followers (m-e); error_count counts errors through j.
    record_id is used only for grouped splitting and never returned publicly.
    """
    rows = sorted(follower_rows, key=lambda row: (row["record_id"], row["anchor"], row["position"]))
    result = {
        "settings": {
            "seed": SEED, "rng": "PCG64", "folds": FOLDS, "l2_penalty": PENALTY,
            "intercept_penalized": False, "standardization": "training-fold weighted mean and variance",
            "weights": "equal eligible responses, equal eligible blocks per response, equal followers per block",
            "probability_clip_for_log_loss": PROBABILITY_CLIP,
            "optimizer": "L-BFGS-B", "maxiter": 2000, "gtol": 1e-9, "ftol": 1e-12,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_scope": "fixed OOF loss differences, whole eligible responses; conditional on transition and fitted folds",
        },
        "folds": [], "models": [], "predictions": [], "metrics": [],
        "comparisons": [], "bootstrap": {},
    }
    if not rows:
        return dict(result, status="unavailable", reason="no eligible correct followers beyond the next mismatch")
    keys = [(row["record_id"], row["anchor"], row["position"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate response/anchor/follower rows would change their weighting")
    response_ordinals = {}
    for row in rows:
        record_id, ordinal = row["record_id"], int(row["response_ordinal"])
        if record_id in response_ordinals and response_ordinals[record_id] != ordinal:
            raise ValueError("One response identity has inconsistent public ordinals")
        response_ordinals[record_id] = ordinal
    if len(set(response_ordinals.values())) != len(response_ordinals):
        raise ValueError("Different response identities share a public ordinal")
    identifiers = sorted(response_ordinals)
    shuffled = np.random.Generator(np.random.PCG64(SEED)).permutation(identifiers).tolist()
    fold_by_response = {record_id: index % FOLDS for index, record_id in enumerate(shuffled)}
    fold_index = np.asarray([fold_by_response[row["record_id"]] for row in rows])
    target = np.asarray([float(row["target"]) for row in rows])
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Reachability targets must be binary")
    weights = _sample_weights(rows)
    designs = {"STRUCT": _features(rows), "SEVERITY": _features(rows, severity=True)}
    if not all(np.isfinite(design).all() for design in designs.values()):
        raise ValueError("Predictor inputs must be finite")
    probabilities = {name: np.full(len(rows), np.nan) for name in ("CONST", "STRUCT", "SEVERITY")}

    for fold in range(FOLDS):
        train, test = fold_index != fold, fold_index == fold
        train_responses = sorted({row["response_ordinal"] for row, use in zip(rows, train) if use})
        test_responses = sorted({row["response_ordinal"] for row, use in zip(rows, test) if use})
        result["folds"].append({"fold": fold, "train_response_ordinals": train_responses,
                                "test_response_ordinals": test_responses})
        common = {"fold": fold, "training_rows": int(train.sum()), "test_rows": int(test.sum())}
        if not train.any() or not test.any():
            for name in probabilities:
                result["models"].append(dict(common, model=name, status="unavailable",
                                              reason="empty training or test fold"))
            continue
        prevalence = float(np.average(target[train], weights=weights[train]))
        probabilities["CONST"][test] = prevalence
        result["models"].append(dict(common, model="CONST", status="fitted", probability=prevalence))
        for name, design in designs.items():
            predicted, model = _fit_logistic(design[train], target[train], weights[train], design[test])
            result["models"].append(dict(common, model=name,
                feature_names=list(STRUCT_FEATURES if name == "STRUCT" else SEVERITY_FEATURES), **model))
            if predicted is not None:
                probabilities[name][test] = predicted

    for index, row in enumerate(rows):
        public_row = {name: value for name, value in row.items() if name != "record_id"}
        public_row.update(fold=int(fold_index[index]), sample_weight=float(weights[index]))
        for name, probability in probabilities.items():
            public_row[f"rho_{name}"] = float(probability[index]) if np.isfinite(probability[index]) else None
        result["predictions"].append(public_row)
    all_rows = np.ones(len(rows), dtype=bool)
    for name, probability in probabilities.items():
        result["metrics"].append({"model": name, "fold": None,
            **_weighted_metrics(probability, target, weights, all_rows)})
        for fold in range(FOLDS):
            result["metrics"].append({"model": name, "fold": fold,
                **_weighted_metrics(probability, target, weights, fold_index == fold)})

    draws = np.random.Generator(np.random.PCG64(SEED)).integers(
        0, len(identifiers), size=(BOOTSTRAP_DRAWS, len(identifiers)))
    for simpler, richer in COMPARISONS:
        paired = np.isfinite(probabilities[simpler]) & np.isfinite(probabilities[richer])
        simpler_loss = _losses(probabilities[simpler], target)
        richer_loss = _losses(probabilities[richer], target)
        differences = {metric: simpler_loss[metric] - richer_loss[metric] for metric in simpler_loss}
        comparison = {"simpler": simpler, "richer": richer, "paired_rows": int(paired.sum()),
                      "unavailable_rows": int((~paired).sum()), "foldwise": []}
        for metric, difference in differences.items():
            comparison[f"{metric}_improvement"] = float(np.average(difference[paired], weights=weights[paired])) if paired.any() else None
        for fold in range(FOLDS):
            use = paired & (fold_index == fold)
            comparison["foldwise"].append({"fold": fold, "paired_rows": int(use.sum()), **{
                f"{metric}_improvement": float(np.average(difference[use], weights=weights[use])) if use.any() else None
                for metric, difference in differences.items()
            }})
        name = f"{richer}_over_{simpler}"
        result["bootstrap"][name] = {
            "response_ordinals": [response_ordinals[record_id] for record_id in identifiers],
            **{metric: _bootstrap_comparison(difference, rows, weights, draws)
               for metric, difference in differences.items()},
        }
        result["comparisons"].append(comparison)
    result["status"] = "complete" if all(np.isfinite(p).all() for p in probabilities.values()) else "incomplete_folds"
    result["eligible_rows"] = len(rows)
    result["eligible_responses"] = len(identifiers)
    result["eligible_blocks"] = len({(row["record_id"], row["anchor"]) for row in rows})
    return result
