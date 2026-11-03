"""Independent NumPy reconstruction of the fixed Track 2 mathematics.

This file imports no experiment implementation. It reads no labels or outcomes
when reconstructing selection and exposes no configurable scientific settings.
"""
import numpy as np

ARMS = ("K", "U", "R")
PAIRS = (("R", "U"), ("R", "K"), ("U", "K"))
COUNT_KEYS = ("verification_calls", "accepted_draft_tokens", "committed_progress", "output_tokens")
TIME_KEYS = ("request_latency_s", "native_generate_s", "cache_maintenance_s")


def selected_records(event, fresh, memory):
    """Return exact chosen IDs, coverage trajectory and sorted two-pass inputs."""
    fresh = sorted(fresh, key=lambda row: row["global_ordinal"])
    old = sorted(memory, key=lambda row: row["global_ordinal"])
    fresh_ids = [row["record_id"] for row in fresh]
    if event == 1:
        return {"passes": {arm: [fresh_ids, fresh_ids] for arm in ARMS}, "uniform": None, "coverage": None}
    zq = np.array([row["descriptor"] for row in fresh], dtype=np.float64)
    zm = np.array([row["descriptor"] for row in old], dtype=np.float64)
    similarities = np.einsum("qd,md->qm", zq, zm, optimize=False)
    similarities = np.maximum(0.0, np.clip(similarities, -1.0, 1.0))
    fresh_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261102, event, 0])))
    old_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261102, event, 1])))
    fresh_subset = [fresh[int(index)] for index in fresh_rng.choice(128, 96, replace=False)]
    uniform = [int(index) for index in old_rng.choice(128, 32, replace=False)]
    chosen, remaining = [], set(range(128))
    best_by_query = np.zeros(128, dtype=np.float64)
    for _ in range(32):
        scores = {index: float(np.sum(np.maximum(best_by_query, similarities[:, index]), dtype=np.float64) / 128)
                  for index in remaining}
        choice = min(remaining, key=lambda index: (-scores[index], old[index]["global_ordinal"]))
        chosen.append(choice)
        remaining.remove(choice)
        best_by_query = np.maximum(best_by_query, similarities[:, choice])
    result = {"passes": {"K": [fresh_ids, fresh_ids]}}
    for arm, indices, key in (("U", uniform, "uniform"), ("R", chosen, "coverage")):
        second = sorted(fresh_subset + [old[index] for index in indices], key=lambda row: row["global_ordinal"])
        result["passes"][arm] = [fresh_ids, [row["record_id"] for row in second]]
        covered, score, trajectory = np.zeros(128), 0.0, []
        for index in indices:
            covered = np.maximum(covered, similarities[:, index])
            after = float(np.sum(covered, dtype=np.float64) / 128)
            trajectory.append(dict(record_id=old[index]["record_id"], coverage=after, marginal_gain=after - score))
            score = after
        result[key] = dict(record_ids=[old[index]["record_id"] for index in indices],
                           coverage=score, steps=trajectory)
    return result


def admit_fresh(slots, fresh, generator, seen):
    """Algorithm R reference: chronological nth unique record chooses [0,n)."""
    result = list(slots)
    steps = []
    for record in sorted(fresh, key=lambda row: row["global_ordinal"]):
        seen += 1
        choice = seen - 1 if seen <= 128 else int(generator.integers(seen))
        evicted = None
        if seen <= 128:
            result.append(record)
        elif choice < 128:
            evicted = result[choice]["record_id"]
            result[choice] = record
        steps.append(dict(record_id=record["record_id"], global_ordinal=record["global_ordinal"],
            seen_count=seen, draw=choice if seen > 128 else None,
            admitted_slot=choice if choice < 128 else None, evicted_record_id=evicted))
    return result, seen, steps


def endpoint_statistics(paired):
    """Paired 128+128 stratified 10,000-draw request bootstrap; no trajectory CI."""
    generator = np.random.Generator(np.random.PCG64(20261102))
    math_draws, coding_draws = [], []
    for _ in range(10000):
        math_draws.append(generator.integers(0, 128, 128))
        coding_draws.append(generator.integers(0, 128, 128))
    math_draws, coding_draws = np.array(math_draws), np.array(coding_draws)
    sampled = np.concatenate((2 * math_draws, 2 * coding_draws + 1), axis=1)
    result = _panel_summary(paired, sampled)
    result["domains"] = {"math": _panel_summary(paired[::2], math_draws),
                         "coding": _panel_summary(paired[1::2], coding_draws)}
    return result


def _reduction(candidate, reference, indices):
    base = reference.sum()
    denominators = reference[indices].sum(axis=1)
    values = np.full(10000, np.nan)
    np.divide(candidate[indices].sum(axis=1), denominators, out=values, where=denominators > 0)
    values = 100 * (1 - values)
    finite = values[np.isfinite(values)]
    return dict(point_pct=float(100 * (1 - candidate.sum() / base)) if base else None,
                ci95_pct=np.quantile(finite, [.025, .975], method="linear").tolist() if len(finite) else None,
                undefined_bootstrap_draws=int(10000 - len(finite)), candidate_sum=float(candidate.sum()), reference_sum=float(base))


def _panel_summary(paired, indices):
    arrays = {arm: {key: np.array([row[arm][key] for row in paired], dtype=np.float64)
                    for key in COUNT_KEYS + TIME_KEYS} for arm in ARMS}
    totals, effects = {}, {}
    for arm, data in arrays.items():
        values = {key: int(data[key].sum()) for key in COUNT_KEYS}
        values.update({key: float(data[key].sum()) for key in TIME_KEYS})
        calls = values["verification_calls"]
        values.update(accepted_tokens_per_call=values["accepted_draft_tokens"] / calls,
                      committed_progress_per_call=values["committed_progress"] / calls,
                      output_tokens_per_call=values["output_tokens"] / calls,
                      output_tokens_per_s=values["output_tokens"] / values["request_latency_s"])
        totals[arm] = values
    for a, b in PAIRS:
        delta = arrays[b]["verification_calls"] - arrays[a]["verification_calls"]
        effect = dict(call_reduction=_reduction(arrays[a]["verification_calls"], arrays[b]["verification_calls"], indices),
            request_wins=int(np.count_nonzero(delta > 0)), request_regressions=int(np.count_nonzero(delta < 0)),
            request_ties=int(np.count_nonzero(delta == 0)), request_denominator=len(paired),
            gross_calls_saved=int(np.maximum(delta, 0).sum()), gross_calls_added=int(np.maximum(-delta, 0).sum()))
        for key in TIME_KEYS:
            effect[key + "_reduction"] = _reduction(arrays[a][key], arrays[b][key], indices)
        ratios = arrays[b]["request_latency_s"][indices].sum(axis=1) / arrays[a]["request_latency_s"][indices].sum(axis=1)
        effect["serving_throughput_ratio"] = dict(point=float(arrays[b]["request_latency_s"].sum() / arrays[a]["request_latency_s"].sum()),
            ci95=np.quantile(ratios, [.025, .975], method="linear").tolist())
        effects[a + "_vs_" + b] = effect
    return dict(request_count=len(paired), method_totals=totals, comparisons=effects)
