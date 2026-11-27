"""CPU FP64 replay of repair coefficients on each response's diagnostic pool."""

from collections import defaultdict
import math

import numpy as np


SCORES = ('u_old', 'u_shared', 'u_const', 'u_adapt')


def allocate(rows, score='u_old', gamma=7.0, eta=0.5, tail=True):
    """Keep the HIGH gate; normalize by every selected prefix/error occurrence."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['response_ordinal']].append(row)
    coefficients, pools = [], []
    for ordinal, group in sorted(grouped.items()):
        denominator = math.fsum(
            math.exp(-(position - 1) / gamma)
            for row in group
            for position in range(1, (row['e'] if row['has_error'] else row['m']) + 1)
        ) + 1e-6
        repairs = [row for row in group if row['has_error']]
        beta = np.array([math.exp(-(row['e'] - 1) / gamma) for row in repairs])
        high = np.array([row['high'] for row in repairs], dtype=bool)
        scores = np.array([row[score] if row['high'] else 0.0 for row in repairs], dtype=float)
        mass, high_mass = float(beta.sum()), float(beta[high].sum())
        fraction = high_mass / mass if mass else 0.0
        increase = eta * (1 - fraction) / (1 + fraction)
        native = beta * np.where(high, 1 + increase, 1 - eta * fraction / (1 + fraction))
        mean = float(np.dot(beta[high], scores[high]) / high_mass) if high_mass else 0.0
        # eta sensitivity calls tail=False; the TAIL recipe itself stays eta=.5.
        kappa = min(increase, eta - increase)
        active = bool(tail and 1 < high.sum() < len(repairs)
                      and np.ptp(scores[high]) > 0)
        applied = native.copy()
        if active:
            applied[high] += kappa * beta[high] * (scores[high] - mean)
        tolerance = 1e-12 + 1e-10 * beta
        if (not np.isfinite(applied).all()
                or not np.array_equal(applied[~high], native[~high])
                or abs(float(applied[high].sum() - native[high].sum())) > float(tolerance[high].sum())
                or np.any(applied[high] < beta[high] - tolerance[high])
                or np.any(applied[high] > (1 + eta) * beta[high] + tolerance[high])):
            raise ValueError('Repair allocation violated its mass, support or bounds')
        for row, weight, base, value in zip(repairs, beta, native, applied):
            coefficients.append(dict(response_ordinal=ordinal, anchor=row['anchor'],
                high=row['high'], e=row['e'], score=score, score_value=row[score],
                gamma=gamma, eta=eta, tail=tail, beta=float(weight),
                native_coefficient=float(base), coefficient=float(value),
                denominator=denominator, effective_coefficient=float(value / denominator)))
        pools.append(dict(response_ordinal=ordinal, score=score, gamma=gamma, eta=eta,
            tail=tail, blocks=len(group), repairs=len(repairs), high=int(high.sum()),
            denominator=denominator, repair_mass=mass, high_beta_mass=high_mass,
            native_high_mass=float(native[high].sum()), active=active,
            weighted_score_mean=mean, kappa=kappa,
            moved_from_native=float(np.abs(applied - native).sum() / 2),
            high_mass_error=float(applied[high].sum() - native[high].sum())))
    return coefficients, pools


def attach_oof_scores(rows, follower_rows, predictions):
    lookup = {(row['response_ordinal'], row['anchor'], row['position']): row
              for row in predictions}
    followers = defaultdict(list)
    for follower in follower_rows:
        followers[(follower['response_ordinal'], follower['anchor'])].append(follower)
    updated = []
    for row in rows:
        current = dict(row)
        for model, key in (('CONST', 'u_const'), ('SEVERITY', 'u_adapt')):
            if not row['high']:
                current[key] = None  # OOF models cover only HIGH blocks; LOW weights never change.
                continue
            additions = []
            for follower in followers[(row['response_ordinal'], row['anchor'])]:
                prediction = lookup[(row['response_ordinal'], row['anchor'], follower['position'])]
                probability = prediction['rho_' + model]
                if probability is None:
                    raise ValueError('Unavailable OOF probability; candidate allocation must be deferred')
                additions.append(follower['credit'] * probability)
            current[key] = row['u_old'] + math.fsum(additions)
            if not row['u_old'] - 1e-12 <= current[key] <= row['u_shared'] + 1e-12:
                raise ValueError('OOF candidate score left the old/shared bounds')
        updated.append(current)
    return updated


def compare_allocations(rows, scores=SCORES):
    """Report actual centered movement and weighted increment energy removed."""
    tables, pool_tables = {}, {}
    for score in scores:
        tables[score], pool_tables[score] = allocate(rows, score)
    old = {(row['response_ordinal'], row['anchor']): row for row in tables['u_old']}
    by_response = defaultdict(list)
    for row in rows:
        by_response[row['response_ordinal']].append(row)
    summary, changes = [], []
    for score in scores:
        for pool in pool_tables[score]:
            ordinal = pool['response_ordinal']
            current = [row for row in tables[score] if row['response_ordinal'] == ordinal]
            differences, normalized = [], []
            for row in current:
                previous = old[(ordinal, row['anchor'])]
                change = row['coefficient'] - previous['coefficient']
                effective = row['effective_coefficient'] - previous['effective_coefficient']
                differences.append(change)
                normalized.append(effective)
                changes.append(dict(row, coefficient_change_vs_old=change,
                                    effective_change_vs_old=effective))
            high = [row for row in by_response[ordinal] if row['high']]
            weighted_sum = math.fsum(row['beta'] * (row[score] - row['u_old']) for row in high)
            energy = math.fsum(row['beta'] * (row[score] - row['u_old']) ** 2 for row in high)
            mean_removed = weighted_sum / pool['high_beta_mass'] if high else 0.0
            centered_energy = math.fsum(row['beta'] * (row[score] - row['u_old'] - mean_removed) ** 2
                                       for row in high)
            moved = math.fsum(abs(value) for value in differences) / 2
            summary.append(dict(pool, changed_repairs=sum(value != 0 for value in differences),
                moved_vs_old=moved,
                moved_fraction_of_all_repair_mass=moved / pool['repair_mass'] if pool['repair_mass'] else None,
                moved_fraction_of_native_high_mass=moved / pool['native_high_mass'] if pool['native_high_mass'] else None,
                effective_half_l1_vs_old=math.fsum(abs(value) for value in normalized) / 2,
                added_high_score_mean_removed=mean_removed,
                increment_energy=energy, centered_increment_energy=centered_energy,
                increment_energy_fraction_removed=1 - centered_energy / energy if energy else None))
    return changes, summary


def sensitivity(rows, scores=SCORES):
    """One-at-a-time gamma changes; eta changes stop before TAIL allocation."""
    output, summaries = [], []
    settings = [('gamma', value, score, value, .5, True) for value in (5., 7., 10.) for score in scores]
    settings += [('eta', value, 'u_old', 7., value, False) for value in (.25, .5, .75)]
    for parameter, value, score, gamma, eta, tail in settings:
        candidate, pools = allocate(rows, score, gamma, eta, tail)
        baseline, _ = allocate(rows, score, 7., .5, tail)
        base = {(row['response_ordinal'], row['anchor']): row for row in baseline}
        for row in candidate:
            before = base[(row['response_ordinal'], row['anchor'])]
            output.append(dict(row, parameter=parameter, value=value,
                coefficient_change=row['coefficient'] - before['coefficient'],
                effective_change=row['effective_coefficient'] - before['effective_coefficient']))
        for pool in pools:
            selected = [row for row in output if row['parameter'] == parameter and row['value'] == value
                        and row['score'] == score and row['response_ordinal'] == pool['response_ordinal']]
            summaries.append(dict(pool, parameter=parameter, value=value,
                coefficient_half_l1=math.fsum(abs(row['coefficient_change']) for row in selected) / 2,
                effective_half_l1=math.fsum(abs(row['effective_change']) for row in selected) / 2,
                max_absolute_effective_change=max((abs(row['effective_change']) for row in selected), default=0.)))
    return output, summaries
