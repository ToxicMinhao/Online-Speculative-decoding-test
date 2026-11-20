"""Independent list/loop oracle: no Torch or implementation imports.

The oracle receives native row losses as numbers, so it independently checks
support, scores, coefficients and reduction, not the upstream KL operation.
"""


def response_oracle(probabilities, proposals, labels, native_beta, row_losses,
                    eos_token_ids=()):
    valid_rows, support_rows, score_rows = [], [], []
    for q, proposed, canonical, beta in zip(probabilities, proposals, labels, native_beta):
        valid = [False] * len(canonical)
        support = [False] * len(canonical)
        alive = canonical[0] not in eos_token_ids
        seen_error = False
        products = [0.0] * len(canonical)
        product = 1.0
        for j in range(1, len(canonical)):
            if not alive or beta[j] <= 0:
                alive = False
                continue
            valid[j] = True
            if not seen_error:
                support[j] = True
            seen_error = seen_error or proposed[j] != canonical[j]
            product *= 0.5 + 0.5 * q[j]
            products[j] = product
            if canonical[j] in eos_token_ids:
                alive = False
        scores = [sum(products[j:]) if valid[j] else 0.0 for j in range(len(canonical))]
        valid_rows.append(valid)
        support_rows.append(support)
        score_rows.append(scores)
    B = sum(beta[j] for beta, support in zip(native_beta, support_rows)
            for j in range(len(beta)) if support[j])
    R = sum(score[j] for score, support in zip(score_rows, support_rows)
            for j in range(len(score)) if support[j])
    coefficients = [[B * score[j] / R if support[j] else 0.0
                     for j in range(len(score))]
                    for score, support in zip(score_rows, support_rows)]
    numerator = sum(a * loss for coeff, losses in zip(coefficients, row_losses)
                    for a, loss in zip(coeff, losses))
    return dict(valid=valid_rows, support=support_rows, scores=score_rows,
                B=B, R=R, coefficients=coefficients, loss=numerator / (B + 1e-6))
