"""Tiny ideal-arithmetic checks. No model imports, optimizer or native BF16 claim."""
import json
import math
import time


EPSILON = 1e-6


def probability(logits):
    values = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(values) for x in values]


def kl(target, logits):
    q = probability(logits)
    return sum(p * (math.log(p) - math.log(qi))
               for p, qi in zip(target, q) if p)


def check():
    started = time.perf_counter()
    target = [.6, .3, .1]
    logits = [[.2, -.1, .4], [.1, .7, -.3], [-.3, .4, .8], [.9, .1, -.1]]
    weights = [1., .5, .25, .125]
    selected = [True, True, True, False]
    losses = [kl(target, z) for z in logits]
    d = sum(weights) + EPSILON
    z = sum(w for w, use in zip(weights, selected) if use) + EPSILON
    numerator = sum(w * loss for w, loss, use in zip(weights, losses, selected) if use)
    support, k = numerator / d, numerator / z
    assert math.isclose(k, d / z * support, abs_tol=1e-14)

    # Position zero is a prefix here; positions 1/2 represent two error groups.
    # These are abstract sampled occurrences, not slots within a single block.
    w1, w2 = weights[1:3]
    n1, n2 = w1 * losses[1], w2 * losses[2]
    c1 = 1 - .5 * w2 / (w1 + 2 * w2)
    c2 = 1 + .5 * w1 / (w1 + 2 * w2)
    mixed = numerator + .5 * (w1 * n2 - w2 * n1) / (w1 + 2 * w2)
    explicit = weights[0] * losses[0] + c1 * n1 + c2 * n2
    assert math.isclose(mixed, explicit, abs_tol=1e-14)
    assert math.isclose(c1 * w1 + c2 * w2, w1 + w2, abs_tol=1e-14)

    derivative_checks = []
    for p in (target, [.6001, .3001, .1001]):
        current = logits[1]
        q = probability(current)
        analytic = [qi * sum(p) - pi for qi, pi in zip(q, p)]
        finite = []
        for j in range(len(current)):
            plus, minus = current.copy(), current.copy()
            plus[j] += 1e-6
            minus[j] -= 1e-6
            finite.append((kl(p, plus) - kl(p, minus)) / 2e-6)
        error = max(abs(a-b) for a, b in zip(analytic, finite))
        assert error < 1e-8
        derivative_checks.append(dict(target_mass=sum(p), max_finite_difference_error=error))

    # Equal response averaging does not make unequal response factors global.
    support_gradient = [.5, .5]
    k_gradient = [1., 2.]  # response factors 2 and 4 applied before averaging
    determinant = support_gradient[0] * k_gradient[1] - support_gradient[1] * k_gradient[0]
    assert determinant != 0

    # Conserved scalar mass can redirect Jacobian-projected parameter gradients.
    old_gradient = [w1, 2*w2]
    new_gradient = [c1*w1, 2*c2*w2]
    assert old_gradient[0]*new_gradient[1] != old_gradient[1]*new_gradient[0]
    assert sum(x*x for x in old_gradient) != sum(x*x for x in new_gradient)

    assert 0 / EPSILON == 0
    return dict(status='passed', arithmetic='Python float64 ideal algebra, fixed masks',
                normalization=dict(D=d, Z=z, scale=d/z, support=support, K=k),
                redistribution=dict(c1=c1, c2=c2, original_mass=w1+w2,
                                    new_mass=c1*w1+c2*w2),
                derivative_checks=derivative_checks,
                unequal_response_factors_change_direction=True,
                mass_conservation_does_not_guarantee_direction_or_norm=True,
                empty_loss=0,
                caveat='Algebra examples do not test native BF16 gradients or serving efficacy.',
                model_loads=0, model_forwards=0, optimizer_applications=0,
                wall_s=time.perf_counter()-started)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
