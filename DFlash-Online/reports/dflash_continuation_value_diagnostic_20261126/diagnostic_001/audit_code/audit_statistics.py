"""Independent, CPU-only retained-result audit; imports no experimental helpers."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


RCOND = 1e-10
OUTCOMES = ('Y', 'useful_repair', 'prefix_damaged')


def identity(path):
    content = Path(path).read_bytes()
    return dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def construct(rows, immediate=True):
    """Independent raw design, frozen standardization, and response demeaning."""
    rows = sorted((row for row in rows if row['high']),
                  key=lambda row: (row['record_id'], row['anchor']))
    names = sorted({row['record_id'] for row in rows})
    categories = sorted({(row['e'], row['r'], row['n']) for row in rows})
    group = np.array([names.index(row['record_id']) for row in rows])
    weights = np.array([row['beta'] for row in rows], dtype=np.float64)
    for index in range(len(names)):
        selected = group == index
        weights[selected] /= weights[selected].sum()
    raw = []
    for row in rows:
        error = math.log1p(row['d'])
        prefix = math.log1p(max(0, row['minimum_original_prefix_margin']))
        h, r = row['h1'], row['r']
        values = [error, error * error, prefix, prefix * prefix]
        if immediate:
            values += [h, h * h, r * h / 14, r * h * h / 14]
        values += [row['u']]
        raw.append(values)
    continuous = np.array(raw, dtype=np.float64)
    means = (continuous * weights[:, None]).sum(0) / weights.sum()
    scales = np.sqrt(((continuous - means) ** 2 * weights[:, None]).sum(0) / weights.sum())
    scales[scales == 0] = 1
    continuous = (continuous - means) / scales
    categorical = np.array([[int((row['e'], row['r'], row['n']) == category)
                             for category in categories] for row in rows], dtype=np.float64)
    flags = np.array([[row['error_tie'], row['prefix_tie'], row['empty_prefix']]
                      for row in rows], dtype=np.float64)
    uncentered = np.column_stack((categorical, continuous[:, :-1], flags, continuous[:, -1]))
    outcome = np.array([[row[name] for name in OUTCOMES] for row in rows], dtype=np.float64)
    matrix, centered_y = uncentered.copy(), outcome.copy()
    for index in range(len(names)):
        selected = group == index
        denominator = weights[selected].sum()
        matrix[selected] -= (matrix[selected] * weights[selected, None]).sum(0) / denominator
        centered_y[selected] -= (centered_y[selected] * weights[selected, None]).sum(0) / denominator
    return dict(rows=rows, names=names, groups=group, weights=weights,
                matrix=matrix * np.sqrt(weights)[:, None],
                outcomes=centered_y * np.sqrt(weights)[:, None],
                uncentered=uncentered, raw_outcomes=outcome,
                u_scale=float(scales[-1]), categories=categories)


def basis(matrix):
    u, values, v = np.linalg.svd(matrix, full_matrices=False)
    rank = int((values > RCOND * values[0]).sum()) if len(values) else 0
    return u[:, :rank], rank


def fit(design, multiplicity=None):
    counts = np.ones(len(design['names']), dtype=np.int64) if multiplicity is None else multiplicity
    repeats = np.asarray(counts)[design['groups']]
    selected = repeats > 0
    x = design['matrix'][selected] * np.sqrt(repeats[selected, None])
    y = design['outcomes'][selected] * np.sqrt(repeats[selected, None])
    baseline, nuisance_rank = basis(x[:, :-1])
    _, full_rank = basis(x)
    residual_x = x[:, -1] - baseline @ (baseline.T @ x[:, -1])
    residual_y = y - baseline @ (baseline.T @ y)
    energy = float(residual_x @ residual_x)
    identified = full_rank == nuisance_rank + 1 and energy > 0
    coefficient = (residual_x @ residual_y / energy / design['u_scale']) if identified else None
    return dict(identified=identified, baseline_rank=nuisance_rank, full_design_rank=full_rank,
                coefficients=coefficient, theta=None if coefficient is None else float(coefficient[0]),
                residual_u=residual_x * design['u_scale'],
                residual_df=int(repeats.sum()) - int(np.count_nonzero(counts)) - full_rank)


def explicit_response_fit(design):
    intercepts = np.eye(len(design['names']))[design['groups']]
    matrix = np.column_stack((intercepts, design['uncentered']))
    matrix *= np.sqrt(design['weights'])[:, None]
    outcomes = design['raw_outcomes'] * np.sqrt(design['weights'])[:, None]
    coef = np.linalg.lstsq(matrix, outcomes, rcond=RCOND)[0][-1] / design['u_scale']
    return coef


class Audit:
    def __init__(self):
        self.checks = 0

    def equal(self, actual, expected, message):
        self.checks += 1
        if actual != expected:
            raise AssertionError(f'{message}: {actual!r} != {expected!r}')

    def close(self, actual, expected, message, atol=1e-8, rtol=1e-7):
        self.checks += 1
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            raise AssertionError(message)

    def fit_matches(self, independent, retained, message):
        self.equal(independent['baseline_rank'], retained['baseline_rank'], message + ' nuisance rank')
        self.equal(independent['full_design_rank'], retained['full_design_rank'], message + ' full rank')
        self.equal(independent['identified'], retained['status'] == 'identified', message + ' identification')
        if independent['identified']:
            self.close(independent['theta'], retained['theta'], message + ' theta')


def audit(root):
    start = time.perf_counter()
    check = Audit()
    rows = json.loads((root / 'BLOCK_READOUTS.json').read_text())
    result = json.loads((root / 'ANALYSIS_PRIVATE.json').read_text())
    contract = json.loads((root / 'CONTRACT.json').read_text())
    numerical = result['contract']
    check.equal(numerical['rank_rcond'], RCOND, 'sealed rank rule')
    check.equal(numerical['bootstrap']['seed'], 20261126, 'bootstrap seed')
    check.equal(numerical['bootstrap']['draws'], 10000, 'bootstrap planned draws')
    check.equal(numerical, contract['statistics'], 'analysis follows sealed numerical policy')
    design = construct(rows)
    independent = fit(design)
    check.fit_matches(independent, result['primary'], 'main fit')
    check.equal(independent['residual_df'], result['primary']['residual_df'], 'main residual degrees of freedom')
    explicit = explicit_response_fit(design)
    if independent['identified']:
        check.close(explicit, independent['coefficients'], 'explicit response-FE versus centered WLS')
    reduced = fit(construct(rows, immediate=False))
    check.fit_matches(reduced, result['secondary_without_immediate'], 'reduced immediate-confidence control')
    for index, name in enumerate(OUTCOMES[1:], 1):
        old = result['outcome_decomposition'][name]
        check.close(old['weighted_incidence'], np.average(
            [row[name] for row in design['rows']], weights=design['weights']), 'outcome incidence')
        if independent['identified']:
            check.close(independent['coefficients'][index], old['theta'], 'secondary full-model coefficient')

    retained_responses = {row['record_id']: row for row in result['per_response']}
    energies, support = [], 0
    for index, name in enumerate(design['names']):
        selected = design['groups'] == index
        group = [row for row, keep in zip(design['rows'], selected) if keep]
        deficits = []
        for row in group:
            minima = row['running_minima_fp64'][:row['r']]
            deficit = math.fsum(row['h1'] - value for value in minima) / 14
            check.close(deficit, row['D'], 'later-bottleneck identity', atol=1e-15, rtol=1e-14)
            check.close(math.fsum(row['running_minima_fp64']) / 14, row['u'], 'area score identity', atol=1e-15, rtol=1e-14)
            deficits.append(deficit)
        spread = max(deficits) - min(deficits)
        energy = float((independent['residual_u'][selected] ** 2).sum())
        weight = float(design['weights'][selected].sum())
        supported = spread > 1e-12 and energy > 1e-24 * weight
        support += supported
        energies.append(energy)
        old = retained_responses[name]
        check.close(weight, 1, 'equal response total weight', atol=1e-14, rtol=1e-14)
        check.close(spread, old['D_range'], 'response D range', atol=1e-15, rtol=1e-14)
        check.close(energy, old['residual_u_energy'], 'response residual-u energy', atol=1e-20, rtol=1e-7)
        check.equal(bool(supported), old['supports_later_information'], 'later-information support')
    check.equal(support, result['primary']['supported_response_clusters'], 'supported response count')
    sd = math.sqrt(math.fsum(energies) / design['weights'].sum())
    check.close(sd, result['primary']['residual_u_sd'], 'frozen residual-u effect scale')
    check.close(max(energies) / math.fsum(energies) if math.fsum(energies) else 0,
                result['primary']['maximum_response_energy_share'], 'maximum response energy share')

    bootstrap = result['bootstrap']
    check.equal(bootstrap['planned'], 10000, 'fixed bootstrap budget')
    check.equal(bootstrap['attempted'], 10000, 'all bootstrap draws attempted')
    check.equal(bootstrap['unattempted'], 0, 'no unattempted bootstrap draws')
    sample = np.random.Generator(np.random.PCG64(20261126)).integers(
        0, len(design['names']), (10000, len(design['names'])))
    digest = hashlib.sha256(sample.astype('<i8', copy=False).tobytes()).hexdigest()
    check.equal(digest, bootstrap['draw_sha256'], 'all fixed whole-response bootstrap draws')
    check.equal(bootstrap['response_order'], design['names'], 'bootstrap response order')
    check.equal(len(bootstrap['coefficients']), 10000, 'one coefficient/invalid receipt per planned draw')
    valid = []
    counts = {}
    for value, status in zip(bootstrap['coefficients'], bootstrap['statuses'], strict=True):
        counts[status] = counts.get(status, 0) + 1
        check.equal(value is not None, status == 'identified', 'draw status and coefficient eligibility')
        if value is not None:
            check.equal(math.isfinite(value), True, 'finite identified bootstrap coefficient')
            valid.append(value)
    check.equal(counts, bootstrap['status_counts'], 'all bootstrap status counts')
    check.equal(len(valid), bootstrap['valid'], 'identifiable bootstrap draws')
    check.equal(10000 - len(valid), bootstrap['invalid'], 'invalid bootstrap draws retained')
    check.close(len(valid) / 10000, bootstrap['valid_fraction'], 'valid draw fraction')
    interval = np.percentile(valid, [2.5, 97.5], method='linear') if valid else None
    if interval is not None:
        check.close(interval, bootstrap['theta_ci95'], 'retained-coefficient percentile interval')
        check.close(interval * .1, bootstrap['theta_per_0_1_u_ci95'], 'per-0.1 effect interval')
        check.close(interval * sd, bootstrap['theta_per_residual_u_sd_ci95'], 'fixed-scale effect interval')
    for index in range(20):
        multiplicity = np.bincount(sample[index], minlength=len(design['names']))
        fitted = fit(design, multiplicity)
        check.equal(fitted['baseline_rank'], bootstrap['baseline_ranks'][index], 'bootstrap direct nuisance rank')
        check.equal(fitted['full_design_rank'], bootstrap['full_ranks'][index], 'bootstrap direct full rank')
        check.equal(fitted['identified'], bootstrap['statuses'][index] == 'identified', 'bootstrap direct identification')
        if fitted['identified']:
            check.close(fitted['theta'], bootstrap['coefficients'][index], 'bootstrap direct coefficient')
    old_loo = {row['record_id']: row for row in result['influence']}
    loo = []
    for omitted, name in enumerate(design['names']):
        multiplicity = np.ones(len(design['names']), dtype=np.int64)
        multiplicity[omitted] = 0
        fitted = fit(design, multiplicity)
        check.fit_matches(fitted, old_loo[name], 'leave-one-response-out')
        loo.append(fitted['theta'])
    positive = sum(value is not None and value > 0 for value in loo)
    identified = sum(value is not None for value in loo)
    positive_fraction = positive / identified if identified else 0
    check.close(positive_fraction,
                result['primary']['positive_valid_LOO_fraction'] or 0, 'positive valid LOO fraction')
    check.equal(identified, result['primary']['valid_LOO'], 'valid LOO count')
    check.equal(len(loo) - identified, result['primary']['invalid_LOO'], 'invalid LOO count')
    if independent['identified']:
        check.close(.1 * independent['theta'], result['primary']['theta_per_0_1_u'], 'main per-0.1 effect')
        check.close(sd * independent['theta'], result['primary']['theta_per_residual_u_sd'], 'main fixed-scale effect')
    structural_absence = all(row['D'] == 0 for row in design['rows'])
    if structural_absence:
        decision = 'NOT_SUPPORTED_IN_THIS_DIAGNOSTIC'
    elif (not independent['identified'] or support < 30 or independent['residual_df'] < 30
          or len(valid) < 9500 or interval is None):
        decision = 'INCONCLUSIVE'
    elif interval[1] <= 0:
        decision = 'NOT_SUPPORTED_IN_THIS_DIAGNOSTIC'
    elif interval[0] <= 0 or positive_fraction < .9:
        decision = 'INCONCLUSIVE'
    else:
        decision = 'SUPPORTED_FOR_A_SEPARATE_PILOT'
    check.equal(decision, result['decision']['label'], 'prespecified decision rule')
    return dict(schema='continuation_value_independent_statistics_audit_v1', status='passed',
                verification_status='ANALYZED: independent retained-result reconstruction, no model rerun',
                checks=check.checks, execution_commit=contract['execution_commit'],
                rows=len(rows), high_rows=len(design['rows']), response_clusters=len(design['names']),
                main_theta=independent['theta'], explicit_response_FE_coefficients=explicit.tolist(),
                residual_df=independent['residual_df'], supported_response_clusters=support,
                residual_u_sd=sd, bootstrap_draws_authenticated=10000,
                direct_bootstrap_refits=20, direct_LOO_refits=len(loo),
                independently_recomputed_decision=decision,
                positive_valid_LOO_fraction=positive_fraction,
                theta_ci95=None if interval is None else interval.tolist(),
                root_identities={name:identity(root/name) for name in
                                 ('BLOCK_READOUTS.json','ANALYSIS_PRIVATE.json','CONTRACT.json')},
                cpu_wall_s=time.perf_counter()-start, gpu_work=0,
                limitations=['Same assistant team; independently written audit, not blinded external review',
                             'NumPy FP64 SVD solver family is shared; construction and reductions are independent',
                             'Only first20 bootstrap draws are refitted; all10000 draw identities and saved-coefficient intervals are checked',
                             'Native primitives and file provenance are checked by the companion audit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except Exception as error:
        result = dict(status='failed', error=repr(error), gpu_work=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key:result[key] for key in ('status','checks','cpu_wall_s')}, indent=2))


if __name__ == '__main__':
    main()
