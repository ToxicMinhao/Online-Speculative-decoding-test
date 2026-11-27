"""Response-weighted association of added shared credit with native outcomes.

This adapts PR64's fixed-effects design and reuses its efficient numerical
solver. Only parent HIGH blocks enter this regression. The two outcomes share
one design and the same whole-response bootstrap draws.
"""
from collections import Counter
import math
import time

import numpy as np

from experiments.continuation_value_diagnostic import statistics as native


BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20261127
DIRECT_CHECK_DRAWS = 5
OUTCOMES = ("Y_full", "beyond_K")
CONTROLS = (*native.CONTINUOUS, "u_old", "later_error_count")


def make_design(block_rows):
    """Add old score and later-error count before the final delta_u column."""
    rows = sorted((row for row in block_rows if row["high"]),
                  key=lambda row: (row["record_id"], row["anchor"]))
    if not rows:
        raise ValueError("The parent HIGH population is empty")
    identities = [(row["record_id"], row["anchor"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate response/anchor observations")
    response_ids = sorted({row["record_id"] for row in rows})
    response_index = {name: index for index, name in enumerate(response_ids)}
    clusters = np.array([response_index[row["record_id"]] for row in rows])
    beta = np.array([row["beta"] for row in rows], dtype=np.float64)
    if not np.isfinite(beta).all() or np.any(beta <= 0):
        raise ValueError("Parent HIGH weights must be finite and positive")
    weights = beta / np.bincount(clusters, weights=beta)[clusters]

    categories = sorted({(row["e"], row["r"], row["n"]) for row in rows})
    category_index = {value: index for index, value in enumerate(categories)}
    categorical = np.zeros((len(rows), len(categories)), dtype=np.float64)
    for index, row in enumerate(rows):
        if not (row["e"] >= 1 and 1 <= row["r"] <= row["n"] <= 15 - row["e"]):
            raise ValueError("Parent HIGH geometry violates block-16 support")
        categorical[index, category_index[row["e"], row["r"], row["n"]]] = 1

    difficulty = np.log1p([row["d"] for row in rows])
    fragility = np.log1p([max(0., row["minimum_original_prefix_margin"]) for row in rows])
    confidence = np.array([row["h1"] for row in rows], dtype=np.float64)
    run_length = np.array([row["r"] for row in rows], dtype=np.float64)
    raw = dict(zip(native.CONTINUOUS, (
        difficulty, difficulty ** 2, fragility, fragility ** 2,
        confidence, confidence ** 2, run_length * confidence / 14,
        run_length * confidence ** 2 / 14,
    )))
    raw.update({name: np.array([row[name] for row in rows], dtype=np.float64)
                for name in ("u_old", "later_error_count", "delta_u")})
    scaling = {}
    for name in (*CONTROLS, "delta_u"):
        mean = float(np.average(raw[name], weights=weights))
        sd = float(np.sqrt(np.average((raw[name] - mean) ** 2, weights=weights)))
        scaling[name] = dict(mean=mean, sd=sd, scale=sd if sd > 0 else 1.)
        raw[name] = (raw[name] - mean) / scaling[name]["scale"]
    indicators = np.array([[row[name] for name in native.INDICATORS] for row in rows],
                          dtype=np.float64)
    matrix = np.column_stack([
        categorical, *[raw[name] for name in CONTROLS], indicators, raw["delta_u"],
    ])
    outcomes = np.array([[row[name] for name in OUTCOMES] for row in rows], dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(outcomes).all():
        raise ValueError("Nonfinite association data; no silent row deletion")
    for cluster in range(len(response_ids)):
        chosen = clusters == cluster
        matrix[chosen] -= np.average(matrix[chosen], weights=weights[chosen], axis=0)
        outcomes[chosen] -= np.average(outcomes[chosen], weights=weights[chosen], axis=0)
    matrix *= np.sqrt(weights)[:, None]
    outcomes *= np.sqrt(weights)[:, None]
    columns = [f"joint_{e}_{r}_{n}" for e, r, n in categories]
    columns += [*CONTROLS, *native.INDICATORS, "delta_u"]
    # Design's historical API calls the final regressor "u". This is only an
    # internal scale alias; the matrix and every public label use delta_u.
    scaling["u"] = scaling["delta_u"]
    return native.Design(rows, response_ids, clusters, weights, matrix, outcomes,
                         columns, scaling, categories)


def _coefficients(fit):
    # The unchanged solver labels output columns with PR64 names. It receives
    # exactly our two real outcomes; translate those positional labels here.
    return {name: fit["coefficients"][legacy]
            for name, legacy in zip(OUTCOMES, native.OUTCOMES)}


def _fit_record(fit):
    coefficients = _coefficients(fit)
    return dict(
        status=fit["status"], coefficients=coefficients,
        coefficients_per_0_1_delta_u={
            name: None if value is None else .1 * value for name, value in coefficients.items()
        },
        baseline_rank=fit["baseline_rank"], full_design_rank=fit["full_design_rank"],
        response_FE_rank=fit.get("response_FE_rank"), residual_df=fit.get("residual_df"),
        fwl_max_absolute_discrepancy=fit.get("fwl_max_absolute_discrepancy"),
        solver=fit.get("solver"), fallback_reason=fit.get("fallback_reason"),
    )


def _bootstrap(design, solver, *, draws=BOOTSTRAP_DRAWS):
    """Retain every fixed whole-response draw, including failed identification."""
    clusters = len(design.response_ids)
    samples = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, clusters, size=(draws, clusters),
    )
    records = []
    checked = 0
    for index, sample in enumerate(samples):
        multiplicity = np.bincount(sample, minlength=clusters)
        try:
            fit = solver.fit(multiplicity)
            if index < DIRECT_CHECK_DRAWS:
                native.check_fit_equivalence(fit, native.direct_fit(design, multiplicity))
                checked += 1
            record = _fit_record(fit)
        except np.linalg.LinAlgError as error:
            record = dict(
                status="invalid_numerical", error=repr(error),
                coefficients={name: None for name in OUTCOMES},
                coefficients_per_0_1_delta_u={name: None for name in OUTCOMES},
                baseline_rank=None, full_design_rank=None,
            )
        records.append(dict(draw=index, response_multiplicity=multiplicity.tolist(), **record))
    intervals = {}
    for name in OUTCOMES:
        values = [record["coefficients"][name] for record in records
                  if record["status"] == "identified"]
        interval = np.percentile(values, [2.5, 97.5], method="linear").tolist() if values else None
        intervals[name] = dict(
            identified_draws=len(values), coefficient_ci95=interval,
            coefficient_per_0_1_delta_u_ci95=(None if interval is None else [.1 * x for x in interval]),
        )
    return dict(
        planned=draws, attempted=len(records), unattempted=0,
        status_counts=dict(Counter(record["status"] for record in records)),
        intervals=intervals, draw_records=records, direct_checked_draws=checked,
        solver_counts=dict(solver.counts),
        interval_scope="Percentiles of identified draws only; every unidentifiable or failed draw is retained, never replaced.",
    )


def run_association(block_rows):
    """Fit only the fixed HIGH model, with two outcomes and 2,000 shared draws."""
    started = time.perf_counter()
    settings = dict(
        population="parent HIGH only; all native child outcomes retained",
        weighting="equal response totals; beta-proportional within each response",
        response_effects="weighted within-response centering",
        geometry="joint categorical (e,r,n), where n is available followers",
        continuous_controls=list(CONTROLS), indicators=list(native.INDICATORS),
        tested_regressor="delta_u", primary_outcome="Y_full", secondary_outcome="beyond_K",
        scaling="parent-only weighted mean/SD, fixed across draws; zero SD uses scale 1",
        arithmetic="FP64", rank_rcond=native.RANK_RCOND,
        identification="rank(full) == rank(controls) + 1",
        bootstrap=dict(draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, generator="PCG64",
                       unit="whole response with multiplicity", direct_check_draws=DIRECT_CHECK_DRAWS,
                       percentiles="linear 2.5/97.5", replacement_of_failed_draws=False),
        numerical_checks=dict(atol=native.FIT_ATOL, rtol=native.FIT_RTOL,
                              direct_full_fit=True, accelerated_full_fit=True),
    )
    if not any(row["high"] for row in block_rows):
        return dict(settings=settings, model=None, primary=dict(status="unidentifiable", reason="no_HIGH_blocks"),
                    secondary=dict(status="unidentifiable", reason="no_HIGH_blocks"), variability=None,
                    bootstrap=dict(planned=BOOTSTRAP_DRAWS, attempted=0, unattempted=BOOTSTRAP_DRAWS,
                                   status_counts={}, draw_records=[]),
                    timings=dict(total_cpu_wall_s=time.perf_counter() - started))

    design = make_design(block_rows)
    fit = native.direct_fit(design)
    solver = native.ClusterSolver(design)
    native.check_fit_equivalence(fit, solver.fit(np.ones(len(design.response_ids), dtype=np.int64)))
    fit_seconds = time.perf_counter() - started
    bootstrap_started = time.perf_counter()
    bootstrap = _bootstrap(design, solver)
    bootstrap_seconds = time.perf_counter() - bootstrap_started

    raw_delta = np.array([row["delta_u"] for row in design.rows], dtype=np.float64)
    residual_sd = math.sqrt(fit["residual_u_energy"] / design.weights.sum())
    per_response = []
    for cluster, name in enumerate(design.response_ids):
        chosen = design.clusters == cluster
        local_delta = raw_delta[chosen]
        local_weight = design.weights[chosen]
        local_mean = float(np.average(local_delta, weights=local_weight))
        per_response.append(dict(
            response_ordinal=int(design.rows[int(np.flatnonzero(chosen)[0])]["response_ordinal"]),
            high_blocks=int(chosen.sum()), analysis_weight=float(local_weight.sum()),
            delta_u_min=float(local_delta.min()), delta_u_max=float(local_delta.max()),
            weighted_mean_delta_u=local_mean,
            weighted_variance_delta_u=float(np.average((local_delta - local_mean) ** 2, weights=local_weight)),
            adjusted_delta_u_energy=float(np.sum(fit["residual_u"][chosen] ** 2)),
        ))
    estimates = _fit_record(fit)
    results = {}
    for name in OUTCOMES:
        coefficient = estimates["coefficients"][name]
        results[name] = dict(
            status=fit["status"], coefficient=coefficient,
            coefficient_per_0_1_delta_u=estimates["coefficients_per_0_1_delta_u"][name],
            coefficient_per_adjusted_delta_u_sd=None if coefficient is None else coefficient * residual_sd,
            **bootstrap["intervals"][name],
        )
    bootstrap["response_order"] = [row["response_ordinal"] for row in per_response]
    return dict(
        settings=settings,
        model=dict(rows=len(design.rows), responses=len(design.response_ids), columns=design.columns,
                   categorical_levels=[list(value) for value in design.categories],
                   parent_scaling={name: values for name, values in design.scaling.items() if name != "u"},
                   **estimates),
        primary=results["Y_full"], secondary=results["beyond_K"],
        variability=dict(
            raw_delta_u_min=float(raw_delta.min()), raw_delta_u_max=float(raw_delta.max()),
            response_weighted_mean_delta_u=design.scaling["delta_u"]["mean"],
            response_weighted_sd_delta_u=design.scaling["delta_u"]["sd"],
            adjusted_delta_u_sd=residual_sd, adjusted_delta_u_energy=fit["residual_u_energy"],
            responses_with_raw_delta_variation=sum(row["delta_u_max"] > row["delta_u_min"] for row in per_response),
            per_response=per_response,
        ),
        bootstrap=bootstrap,
        memory=dict(design_bytes=design.matrix.nbytes + design.outcomes.nbytes,
                    crossproduct_bytes=solver.grams.nbytes + solver.crosses.nbytes),
        timings=dict(design_and_fit_s=fit_seconds, bootstrap_s=bootstrap_seconds,
                     total_cpu_wall_s=time.perf_counter() - started),
        interpretation="Retrospective adjusted associations under one fixed historical native update; not effects of applying shared-credit allocation.",
    )
