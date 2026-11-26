"""Prespecified response-cluster continuation-value association analysis.

All model choices use parent predictors. Endpoint outcomes never choose columns,
rank cutoffs, response weights, cluster membership, or bootstrap draws.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np

RANK_RCOND = 1e-10
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261126
DIRECT_CHECK_DRAWS = 20
FIT_ATOL, FIT_RTOL = 1e-8, 1e-7
MAX_WHITENING_CONDITION = 1e6
MIN_GRAM_EIGEN_RATIO = 1e-8
OUTCOMES = ("Y", "useful_repair", "prefix_damaged")
CONTINUOUS = ("x_error", "x_error_squared", "x_prefix", "x_prefix_squared",
              "h1", "h1_squared", "r_h1_over14", "r_h1_squared_over14")
INDICATORS = ("error_tie", "prefix_tie", "empty_prefix")


def numerical_contract():
    return dict(
        schema="continuation_value_statistics_v1", arithmetic="FP64",
        rank_rule="singular_value > 1e-10 * largest_singular_value; weighted within-response design",
        rank_rcond=RANK_RCOND, theta_identification="rank(full) == rank(baseline) + 1",
        scaling="parent-only response-equal beta-weighted mean/SD, frozen for all resamples; zero SD uses scale 1",
        fixed_effects="response intercepts removed by weighted centering; exact joint categorical (e,r,n)",
        bootstrap=dict(draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, generator="PCG64",
                       unit="whole response with multiplicity", percentiles="linear 2.5/97.5",
                       cluster_order="sorted original UTF-8 record_id; anchors ascending within response",
                       direct_validation_draws=DIRECT_CHECK_DRAWS,
                       keep_invalid_draws=True, minimum_identifiable_fraction=.95),
        support=dict(minimum_responses=30, D_range_strictly_greater=1e-12,
                     residual_u_energy_strictly_greater_weight_factor=1e-24,
                     minimum_residual_df=30, minimum_positive_valid_LOO_fraction=.9),
        acceleration=dict(method="global SVD whitening, response cross-products, reconstructed-design SVD",
                          maximum_global_whitening_condition=MAX_WHITENING_CONDITION,
                          minimum_positive_Gram_eigen_ratio=MIN_GRAM_EIGEN_RATIO,
                          discarded_factor_bound_fraction_of_rank_cutoff=.01,
                          fallback="direct weighted-design SVD with the identical rank rule",
                          fit_check_absolute=FIT_ATOL, fit_check_relative=FIT_RTOL),
        outcome="Y", secondary_outcomes=list(OUTCOMES[1:]),
        confidence_intervals="primary full-model Y only; secondary fits descriptive",
        influence_policy="report all LOO failures and energy concentration; no invented dominance cutoff",
        cpu_execution="launch NumPy analysis with OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1",
    )


@dataclass
class Design:
    rows: list
    response_ids: list
    clusters: np.ndarray
    weights: np.ndarray
    matrix: np.ndarray
    outcomes: np.ndarray
    columns: list
    scaling: dict
    categories: list

    @property
    def u_scale(self):
        return self.scaling["u"]["scale"]


def make_design(rows, *, include_immediate=True):
    """Freeze parent-only scaling; center separately within each response."""
    selected = sorted((row for row in rows if row["high"]),
                      key=lambda row: (row["record_id"], row["anchor"]))
    if not selected:
        raise ValueError("The origin-HIGH population is empty")
    identities = [(row["record_id"], row["anchor"]) for row in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate response/anchor occurrences")
    response_ids = sorted({row["record_id"] for row in selected})
    response_index = {name: index for index, name in enumerate(response_ids)}
    clusters = np.array([response_index[row["record_id"]] for row in selected])
    beta = np.array([row["beta"] for row in selected], dtype=np.float64)
    if not np.isfinite(beta).all() or np.any(beta <= 0):
        raise ValueError("HIGH analysis weights must be finite and positive")
    weights = beta / np.bincount(clusters, weights=beta)[clusters]
    categories = sorted({(row["e"], row["r"], row["n"]) for row in selected})
    category_index = {key: index for index, key in enumerate(categories)}
    categorical = np.zeros((len(selected), len(categories)), dtype=np.float64)
    for index, row in enumerate(selected):
        if not (row["e"] >= 1 and 1 <= row["r"] <= row["n"] <= 15 - row["e"]):
            raise ValueError("Origin-HIGH geometry violates block-16 support")
        categorical[index, category_index[row["e"], row["r"], row["n"]]] = 1
    error = np.log1p([row["d"] for row in selected])
    prefix = np.log1p([max(0., row["minimum_original_prefix_margin"]) for row in selected])
    h1 = np.array([row["h1"] for row in selected], dtype=np.float64)
    run = np.array([row["r"] for row in selected], dtype=np.float64)
    raw = dict(zip(CONTINUOUS, (error, error ** 2, prefix, prefix ** 2,
                               h1, h1 ** 2, run * h1 / 14, run * h1 ** 2 / 14)))
    raw["u"] = np.array([row["u"] for row in selected], dtype=np.float64)
    names = list(CONTINUOUS if include_immediate else CONTINUOUS[:4])
    scaling = {}
    for name in names + ["u"]:
        mean = float(np.average(raw[name], weights=weights))
        sd = float(np.sqrt(np.average((raw[name] - mean) ** 2, weights=weights)))
        scaling[name] = dict(mean=mean, sd=sd, scale=sd if sd > 0 else 1.)
        raw[name] = (raw[name] - mean) / scaling[name]["scale"]
    indicators = np.array([[row[name] for name in INDICATORS] for row in selected], dtype=np.float64)
    matrix = np.column_stack([categorical, *[raw[name] for name in names], indicators, raw["u"]])
    outcomes = np.array([[row[name] for name in OUTCOMES] for row in selected], dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(outcomes).all():
        raise ValueError("Nonfinite predictor or outcome; no silent row deletion")
    for cluster in range(len(response_ids)):
        chosen = clusters == cluster
        matrix[chosen] -= np.average(matrix[chosen], weights=weights[chosen], axis=0)
        outcomes[chosen] -= np.average(outcomes[chosen], weights=weights[chosen], axis=0)
    matrix *= np.sqrt(weights)[:, None]
    outcomes *= np.sqrt(weights)[:, None]
    columns = [f"joint_{e}_{r}_{n}" for e, r, n in categories] + names + list(INDICATORS) + ["u"]
    return Design(selected, response_ids, clusters, weights, matrix, outcomes, columns, scaling, categories)


def decomposition(matrix):
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    threshold = RANK_RCOND * singular[0] if len(singular) else 0.
    rank = int(np.count_nonzero(singular > threshold))
    return left, singular, right, rank, threshold


def factor_fit(matrix, outcomes, u_scale):
    """Direct weighted-design SVD and its independent FWL scalar check."""
    left, singular, right, rank, cutoff = decomposition(matrix)
    base_left, base_singular, _, base_rank, base_cutoff = decomposition(matrix[:, :-1])
    basis = base_left[:, :base_rank]
    score = matrix[:, -1] - basis @ (basis.T @ matrix[:, -1])
    outcome = outcomes - basis @ (basis.T @ outcomes)
    energy = float(score @ score)
    result = dict(status="unidentifiable", baseline_rank=base_rank, full_design_rank=rank,
                  rank_cutoff=cutoff, baseline_rank_cutoff=base_cutoff,
                  singular_values=singular.tolist(), baseline_singular_values=base_singular.tolist(),
                  theta=None, coefficients={name: None for name in OUTCOMES},
                  residual_u=score * u_scale, residual_u_energy=energy * u_scale ** 2,
                  fwl_max_absolute_discrepancy=None)
    if rank != base_rank + 1 or energy == 0:
        return result
    fwl = (score @ outcome) / energy / u_scale
    full_coefficients = (right[:rank].T @ ((left[:, :rank].T @ outcomes) / singular[:rank, None]))[-1] / u_scale
    discrepancy = float(np.max(np.abs(fwl - full_coefficients)))
    result["fwl_max_absolute_discrepancy"] = discrepancy
    if not np.isfinite(fwl).all() or not np.allclose(fwl, full_coefficients, atol=FIT_ATOL, rtol=FIT_RTOL):
        result["status"] = "invalid_numerical"
        return result
    result.update(status="identified", theta=float(fwl[0]),
                  coefficients={name: float(value) for name, value in zip(OUTCOMES, fwl)})
    return result


def direct_fit(design, multiplicity=None):
    if multiplicity is None:
        multiplicity = np.ones(len(design.response_ids), dtype=np.int64)
    row_multiplicity = np.asarray(multiplicity)[design.clusters]
    kept = row_multiplicity > 0
    factors = np.sqrt(row_multiplicity[kept])[:, None]
    result = factor_fit(design.matrix[kept] * factors, design.outcomes[kept] * factors, design.u_scale)
    result.update(solver="direct_svd", response_FE_rank=int(np.count_nonzero(multiplicity)),
                  rows=int(row_multiplicity.sum()))
    result["residual_df"] = result["rows"] - result["response_FE_rank"] - result["full_design_rank"]
    return result


class ClusterSolver:
    """Whiten before cross-products, then SVD a weighted-design factor.

    Small/ambiguous Gram modes are checked against actual weighted rows. Unsafe
    conditioning always falls back to the identical direct SVD. The Gram cutoff
    is only an acceleration safety test, never the scientific rank convention.
    """
    def __init__(self, design):
        self.design = design
        left, singular, right = np.linalg.svd(design.matrix, full_matrices=False)
        machine_cutoff = np.finfo(np.float64).eps * max(design.matrix.shape) * singular[0]
        keep = singular > machine_cutoff
        self.white = left[:, keep]
        self.mapping = singular[keep, None] * right[keep]
        self.condition = float(singular[0] / singular[keep][-1]) if keep.any() else float("inf")
        self.mapping_norm = float(singular[0])
        self.grams = np.array([self.white[design.clusters == i].T @ self.white[design.clusters == i]
                              for i in range(len(design.response_ids))])
        self.crosses = np.array([self.white[design.clusters == i].T @ design.outcomes[design.clusters == i]
                                for i in range(len(design.response_ids))])
        self.counts = Counter()

    def fallback(self, multiplicity, reason):
        self.counts["direct_" + reason] += 1
        result = direct_fit(self.design, multiplicity)
        result["fallback_reason"] = reason
        return result

    def fit(self, multiplicity):
        if self.condition > MAX_WHITENING_CONDITION:
            return self.fallback(multiplicity, "global_condition")
        gram = np.einsum("r,rij->ij", multiplicity, self.grams, optimize=True)
        cross = np.einsum("r,rij->ij", multiplicity, self.crosses, optimize=True)
        values, vectors = np.linalg.eigh(gram)
        largest = float(values[-1]) if len(values) else 0.
        noise = 128 * np.finfo(np.float64).eps * max(1, len(values)) * largest
        if largest <= 0 or values[0] < -noise:
            return self.fallback(multiplicity, "invalid_gram")
        keep = values > noise
        if values[keep][0] / largest < MIN_GRAM_EIGEN_RATIO:
            return self.fallback(multiplicity, "gram_condition")
        roots = np.sqrt(values[keep])
        factor = (roots[:, None] * vectors[:, keep].T) @ self.mapping
        response = (vectors[:, keep].T @ cross) / roots[:, None]
        if not keep.all():
            # Bound the discarded original-design factor using actual rows.
            row_weights = np.sqrt(np.asarray(multiplicity)[self.design.clusters])
            discarded = (self.white @ vectors[:, ~keep]) * row_weights[:, None]
            bound = np.linalg.norm(discarded) * self.mapping_norm
            reference = np.linalg.norm(factor, ord=2)
            if bound > .01 * RANK_RCOND * reference:
                return self.fallback(multiplicity, "ambiguous_nullspace")
        result = factor_fit(factor, response, self.design.u_scale)
        # Near the scientific cutoff, use original rows instead of trusting a
        # cross-product-derived singular value on either side of the boundary.
        for key, threshold_key in (("singular_values", "rank_cutoff"),
                                   ("baseline_singular_values", "baseline_rank_cutoff")):
            threshold = result[threshold_key]
            if any(.5 * threshold <= value <= 2 * threshold for value in result[key]) and threshold > 0:
                return self.fallback(multiplicity, "rank_boundary")
        if result["status"] == "invalid_numerical":
            return self.fallback(multiplicity, "fwl_check")
        self.counts["accelerated_svd"] += 1
        result.update(solver="whitened_crossproduct_svd",
                      response_FE_rank=int(np.count_nonzero(multiplicity)),
                      rows=int(np.asarray(multiplicity)[self.design.clusters].sum()))
        result["residual_df"] = result["rows"] - result["response_FE_rank"] - result["full_design_rank"]
        return result


def compact_fit(result, residual_u_sd=None):
    output = {key: value for key, value in result.items()
              if key not in ("residual_u", "singular_values", "baseline_singular_values")}
    output["theta_per_0_1_u"] = None if result["theta"] is None else .1 * result["theta"]
    output["residual_u_sd"] = residual_u_sd
    output["theta_per_residual_u_sd"] = (None if result["theta"] is None or residual_u_sd is None
                                         else residual_u_sd * result["theta"])
    return output


def check_fit_equivalence(first, second):
    for name in ("status", "baseline_rank", "full_design_rank"):
        if first[name] != second[name]:
            raise ValueError("Accelerated/direct SVD disagreement: " + name)
    if first["theta"] is not None and not np.allclose(
            list(first["coefficients"].values()), list(second["coefficients"].values()),
            atol=FIT_ATOL, rtol=FIT_RTOL):
        raise ValueError("Accelerated/direct coefficient disagreement")


def numerical_failure(error):
    return dict(status="invalid_numerical", theta=None, coefficients={name: None for name in OUTCOMES},
                baseline_rank=None, full_design_rank=None, error=repr(error))


def bootstrap(design, solver, residual_u_sd, *, draws=BOOTSTRAP_DRAWS):
    clusters = len(design.response_ids)
    samples = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(0, clusters, (draws, clusters))
    coefficients, statuses, baseline_ranks, full_ranks, failures = [], [], [], [], []
    for index, sample in enumerate(samples):
        counts = np.bincount(sample, minlength=clusters)
        try:
            result = solver.fit(counts)
            if index < DIRECT_CHECK_DRAWS:
                check_fit_equivalence(result, direct_fit(design, counts))
        except np.linalg.LinAlgError as error:
            result = numerical_failure(error)
            failures.append(dict(draw=index, error=repr(error)))
        coefficients.append(result["theta"])
        statuses.append(result["status"])
        baseline_ranks.append(result["baseline_rank"])
        full_ranks.append(result["full_design_rank"])
    valid = np.array([value for value in coefficients if value is not None], dtype=np.float64)
    interval = np.percentile(valid, [2.5, 97.5], method="linear").tolist() if len(valid) else None
    scaled = lambda scale: [value * scale for value in interval] if interval else None
    return dict(planned=draws, valid=len(valid), invalid=draws - len(valid), valid_fraction=len(valid) / draws,
                theta_ci95=interval, theta_per_0_1_u_ci95=scaled(.1),
                theta_per_residual_u_sd_ci95=scaled(residual_u_sd), frozen_residual_u_sd=residual_u_sd,
                coefficients=coefficients, statuses=statuses, baseline_ranks=baseline_ranks, full_ranks=full_ranks,
                numerical_failures=failures,
                status_counts=dict(Counter(statuses)), solver_counts=dict(solver.counts),
                direct_checked_draws=min(draws, DIRECT_CHECK_DRAWS),
                seed=BOOTSTRAP_SEED, generator="PCG64", unit="whole response with multiplicity",
                draw_sha256=hashlib.sha256(samples.astype("<i8", copy=False).tobytes()).hexdigest())


def decision(primary, support, bootstrap_result, influence, *, correctness_passed, structural_absence):
    if not correctness_passed:
        return dict(label="BLOCKED", reasons=["provenance_or_correctness_gate"])
    if structural_absence:
        return dict(label="NOT_SUPPORTED_IN_THIS_DIAGNOSTIC", reasons=["D_is_identically_zero_exactly"])
    reasons = []
    if primary["status"] != "identified":
        reasons.append("theta_numerically_unidentifiable_or_invalid")
    if support < 30:
        reasons.append("fewer_than_30_supported_response_clusters")
    if primary["residual_df"] < 30:
        reasons.append("residual_degrees_of_freedom_below_30")
    if bootstrap_result["valid_fraction"] < .95:
        reasons.append("fewer_than_95_percent_identifiable_bootstrap_draws")
    if reasons:
        return dict(label="INCONCLUSIVE", reasons=reasons)
    interval = bootstrap_result["theta_ci95"]
    if interval is None:
        return dict(label="INCONCLUSIVE", reasons=["no_identifiable_bootstrap_interval"])
    if interval[1] <= 0:
        return dict(label="NOT_SUPPORTED_IN_THIS_DIAGNOSTIC", reasons=["identified_interval_entirely_at_or_below_zero"])
    if interval[0] <= 0:
        return dict(label="INCONCLUSIVE", reasons=["primary_interval_includes_zero"])
    valid = [row["theta"] for row in influence if row["theta"] is not None]
    positive_fraction = sum(value > 0 for value in valid) / len(valid) if valid else 0.
    if positive_fraction < .9:
        return dict(label="INCONCLUSIVE", reasons=["fewer_than_90_percent_valid_LOO_coefficients_positive"])
    return dict(label="SUPPORTED_FOR_A_SEPARATE_PILOT", reasons=["all_prespecified_support_and_inference_gates_pass"])


def coverage(rows):
    responses = sorted({row["record_id"] for row in rows})
    pools = []
    for name in responses:
        group = [row for row in rows if row["record_id"] == name]
        high = [row for row in group if row["high"]]
        low = sum(row["e"] >= 1 and not row["high"] for row in group)
        varying = len(high) >= 2 and max(row["u"] for row in high) > min(row["u"] for row in high)
        pools.append(dict(record_id=name, blocks=len(group), high=len(high), low=low,
                          no_error=sum(row["m"] > 0 and row["e"] < 1 for row in group),
                          empty=sum(row["m"] == 0 for row in group),
                          mixed_HIGH_LOW=bool(high and low), at_least_two_HIGH=len(high) >= 2,
                          varying_HIGH_score=varying, eligible=bool(low and varying)))
    return dict(blocks=len(rows), responses=len(responses), HIGH=sum(row["high"] for row in rows),
                LOW=sum(row["e"] >= 1 and not row["high"] for row in rows),
                no_error=sum(row["m"] > 0 and row["e"] < 1 for row in rows),
                empty=sum(row["m"] == 0 for row in rows), pools=pools,
                scope="Per-response selected diagnostic anchors; not historical training allocation pools")


def analyze(rows, *, correctness_passed=True):
    """Run the fixed full-model diagnostic; all outcomes remain in the sample."""
    began = time.perf_counter()
    observed = coverage(rows)
    if not observed["HIGH"]:
        return dict(contract=numerical_contract(), coverage=observed,
                    primary=dict(status="unidentifiable", theta=None, residual_df=0),
                    secondary_without_immediate=None, outcome_decomposition={}, per_response=[],
                    bootstrap=dict(planned=BOOTSTRAP_DRAWS, attempted=0, unattempted=BOOTSTRAP_DRAWS,
                                   valid=0, invalid=0, valid_fraction=0., theta_ci95=None),
                    influence=[], raw_association=None, exact_run_summaries=[],
                    decision=dict(label="INCONCLUSIVE" if correctness_passed else "BLOCKED",
                                  reasons=["empty_origin_HIGH_population"]),
                    timings=dict(total_cpu_wall_s=time.perf_counter() - began))
    design = make_design(rows)
    main = direct_fit(design)
    residual_sd = math.sqrt(main["residual_u_energy"] / design.weights.sum())
    per_response = []
    for cluster, name in enumerate(design.response_ids):
        chosen = design.clusters == cluster
        deficits = np.array([row["D"] for row, keep in zip(design.rows, chosen) if keep], dtype=np.float64)
        if not np.isfinite(deficits).all():
            raise ValueError("Nonfinite parent-only D values")
        spread = float(deficits.max() - deficits.min())
        energy = float(np.sum(main["residual_u"][chosen] ** 2))
        weight = float(design.weights[chosen].sum())
        per_response.append(dict(record_id=name, high=int(chosen.sum()), total_analysis_weight=weight,
                                 D_range=spread, residual_u_energy=energy,
                                 residual_u_energy_share=energy / main["residual_u_energy"] if main["residual_u_energy"] else 0.,
                                 supports_later_information=bool(spread > 1e-12 and energy > 1e-24 * weight)))
    support = sum(row["supports_later_information"] for row in per_response)
    primary = compact_fit(main, residual_sd)
    primary.update(supported_response_clusters=support,
                   maximum_response_energy_share=max(row["residual_u_energy_share"] for row in per_response),
                   columns=design.columns, parent_scaling=design.scaling,
                   categorical_levels=[list(key) for key in design.categories],
                   min_numerical_rank_ratio=RANK_RCOND)
    reduced = direct_fit(make_design(rows, include_immediate=False))
    fit_seconds = time.perf_counter() - began
    solver = ClusterSolver(design)
    resampling_started = time.perf_counter()
    boot = bootstrap(design, solver, residual_sd, draws=BOOTSTRAP_DRAWS)
    boot.update(attempted=BOOTSTRAP_DRAWS, unattempted=0, response_order=design.response_ids)
    resampling_seconds = time.perf_counter() - resampling_started
    influence_started = time.perf_counter()
    influence = []
    for omitted, name in enumerate(design.response_ids):
        multiplicity = np.ones(len(design.response_ids), dtype=np.int64)
        multiplicity[omitted] = 0
        try:
            fitted = solver.fit(multiplicity)
        except np.linalg.LinAlgError as error:
            fitted = numerical_failure(error)
        influence.append(dict(record_id=name, **compact_fit(fitted, residual_sd)))
    valid_loo = [row["theta"] for row in influence if row["theta"] is not None]
    primary["valid_LOO"] = len(valid_loo)
    primary["invalid_LOO"] = len(influence) - len(valid_loo)
    primary["positive_valid_LOO_fraction"] = sum(value > 0 for value in valid_loo) / len(valid_loo) if valid_loo else None
    structural_absence = all(row["D"] == 0 for row in design.rows)
    classification = decision(primary, support, boot, influence,
                              correctness_passed=correctness_passed, structural_absence=structural_absence)
    weights = design.weights
    u = np.array([row["u"] for row in design.rows], dtype=np.float64)
    y = np.array([row["Y"] for row in design.rows], dtype=np.float64)
    centered_u = u - np.average(u, weights=weights)
    centered_y = y - np.average(y, weights=weights)
    variance_u = float(np.average(centered_u ** 2, weights=weights))
    variance_y = float(np.average(centered_y ** 2, weights=weights))
    covariance = float(np.average(centered_u * centered_y, weights=weights))
    raw = dict(weighting="equal response total; beta within origin HIGH",
               slope=covariance / variance_u if variance_u else None,
               correlation=covariance / math.sqrt(variance_u * variance_y) if variance_u and variance_y else None,
               weighted_mean_u=float(np.average(u, weights=weights)), weighted_mean_Y=float(np.average(y, weights=weights)),
               weighted_variance_u=variance_u, D_identically_zero=structural_absence)
    summaries = []
    for category in design.categories:
        positions = [index for index, row in enumerate(design.rows) if (row["e"], row["r"], row["n"]) == category]
        selected = [design.rows[index] for index in positions]
        local_weights = weights[positions]
        summaries.append(dict(e=category[0], r=category[1], n=category[2], occurrences=len(selected),
                              responses=len({row["record_id"] for row in selected}),
                              analysis_weight=float(local_weights.sum()),
                              weighted_mean_Y=float(np.average([row["Y"] for row in selected], weights=local_weights)),
                              weighted_mean_u=float(np.average([row["u"] for row in selected], weights=local_weights)),
                              weighted_mean_D=float(np.average([row["D"] for row in selected], weights=local_weights))))
    outcome_decomposition = {
        name: dict(theta=main["coefficients"][name], theta_per_0_1_u=(None if main["coefficients"][name] is None else .1 * main["coefficients"][name]),
                   weighted_incidence=float(np.average([row[name] for row in design.rows], weights=weights)),
                   status=main["status"], confidence_interval=None,
                   interpretation="Descriptive full-specification linear-probability coefficient; not a selection criterion")
        for name in OUTCOMES[1:]
    }
    return dict(contract=numerical_contract(), coverage=observed, primary=primary,
                secondary_without_immediate=compact_fit(reduced), outcome_decomposition=outcome_decomposition,
                per_response=per_response, bootstrap=boot, influence=influence, decision=classification,
                raw_association=raw, exact_run_summaries=summaries,
                timings=dict(main_and_reduced_fit_s=fit_seconds, bootstrap_s=resampling_seconds,
                             influence_s=time.perf_counter() - influence_started,
                             total_cpu_wall_s=time.perf_counter() - began),
                memory=dict(whitened_cross_products_bytes=solver.grams.nbytes + solver.crosses.nbytes,
                            design_bytes=design.matrix.nbytes + design.outcomes.nbytes),
                interpretation="Conditional association in one retained training-buffer transition; no weight-allocation or serving causal effect")
