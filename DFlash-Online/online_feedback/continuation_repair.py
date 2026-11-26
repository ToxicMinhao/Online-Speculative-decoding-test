"""Redistribute native HIGH repair mass using current continuation supports.

Only three additive moments cross vocabulary chunks. Detached compact repair
rows support exact neutral branches and coefficient diagnostics; they are never
collected during checkpoint recomputation and never receive gradients.
"""

from contextlib import contextmanager

import torch

from .first_error import valid_prefix_mask
from .kplus import first_error_groups


def continuation_repair_contract(enabled=True):
    return dict(
        name='continuation_repair_high_pool_v1', enabled=bool(enabled),
        base='native_C_Kplus_PAIR', eta=0.5, follower_capacity=14,
        support='relu(tanh((FP32(native_canonical)-FP32(native_strongest_wrong))/2))',
        score='sum_running_minimum_follower_support/14',
        allocation='a0 + kappa*beta*(u-beta_weighted_HIGH_mean) on HIGH only',
        kappa='min(c,0.5-c), c=0.5*(1-f)/(1+f), f=B_HIGH/B_ALL',
        pool='all current repair positions from all sampled blocks of one response',
        correction='kappa*(S1-(U/B_HIGH)*S0), full FKL+PAIR repair objective',
        allocation_precision='float32', coefficient_atol=2e-7, coefficient_beta_rtol=5e-6,
        additional_forwards=0, additional_head_projections=0,
    )


@torch.no_grad()
def continuation_scores(logits, proposals, labels, weights, selection, eos_token_ids=()):
    """Return one score per block, using the current first error and native ties."""
    if logits.dtype != torch.bfloat16 or labels.shape[-1] != 16:
        raise ValueError('Continuation repair requires native BF16 sixteen-slot blocks')
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    low, high = first_error_groups(proposals, labels, weights, selection, eos_token_ids)
    errors = low | high
    slots = torch.arange(16, device=labels.device)
    error = torch.where(errors, slots, 16).amin(-1)
    followers = valid & (slots > error.unsqueeze(-1))
    selected = logits.detach()[followers]
    canonical = labels[followers]
    wrong = selected.clone()
    wrong.scatter_(-1, canonical.unsqueeze(-1), -torch.inf)
    rivals = wrong.argmax(-1)
    native_y = selected.gather(-1, canonical.unsqueeze(-1)).squeeze(-1).float()
    native_rival = selected.gather(-1, rivals.unsqueeze(-1)).squeeze(-1).float()
    margins = native_y - native_rival
    if not bool(torch.isfinite(margins).all()):
        raise ValueError('Nonfinite continuation margin')
    support = torch.zeros_like(weights, dtype=torch.float32)
    support[followers] = torch.tanh(margins / 2).clamp_min(0)
    # Pre-error slots are multiplicative identities for the running minimum;
    # invalid suffix slots contribute nothing and cannot reopen continuation.
    running = torch.where(followers, support, 1.0).cummin(-1).values
    score = (running * followers).sum(-1) / 14
    if not bool(torch.isfinite(score).all()) or bool(((score < 0) | (score > 1)).any()):
        raise ValueError('Continuation score left its fixed [0,1] support')
    return score.detach(), low, high


@torch.no_grad()
def high_pool_coefficients(beta, high, scores, native_metrics, weighted_score):
    """Literal complete-response coefficients and prospective FP32 invariants."""
    beta, high, scores = beta.detach().float(), high.detach(), scores.detach().float()
    weight_low = native_metrics['kplus_W1'].detach()
    weight_high = native_metrics['kplus_W2'].detach()
    coefficient_low = native_metrics['kplus_coefficient1'].detach()
    coefficient_high = native_metrics['kplus_coefficient2'].detach()
    native = beta * torch.where(high, coefficient_high, coefficient_low)
    zero = weight_high.new_zeros(())
    count = int(high.sum())
    neutral = (not len(beta) or not count or count == len(beta) or count == 1
               or bool((scores[high] == scores[high][0]).all()))
    kappa, mean = zero, zero
    candidate = native
    if count:
        mean = weighted_score.detach() / weight_high
    if count and count < len(beta):
        fraction = weight_high / (weight_low + weight_high)
        c = 0.5 * (1 - fraction) / (1 + fraction)
        kappa = torch.minimum(c, 0.5 - c)
    if not neutral:
        candidate = native.clone()
        candidate[high] = native[high] + kappa * beta[high] * (scores[high] - mean)
    tolerance = 2e-7 + 5e-6 * beta.abs()
    difference = candidate - native
    mass_error = candidate[high].sum() - native[high].sum()
    if (not bool(torch.isfinite(candidate).all())
            or not torch.equal(candidate[~high], native[~high])
            or bool((candidate[high] < beta[high] - tolerance[high]).any())
            or bool((candidate[high] > 1.5 * beta[high] + tolerance[high]).any())
            or bool(mass_error.abs() > tolerance[high].sum())):
        raise ValueError('Continuation coefficients violate finite/budget/LOW/bound contract')
    return candidate, native, dict(
        active=not neutral, kappa=kappa, mean=mean, high_count=zero + count,
        low_count=zero + int((~high).sum()),
        changed_high_positions=((difference != 0) & high).sum().float(),
        moved_weight=difference.abs().sum() * 0.5,
        high_mass_before=native[high].sum(), high_mass_after=candidate[high].sum(),
        total_mass_error=candidate.sum() - native.sum(), high_mass_error=mass_error,
        score_min=scores[high].amin() if count else zero,
        score_max=scores[high].amax() if count else zero,
    )


class ContinuationRepair:
    def __init__(self):
        self._collecting = False
        self._rows = []
        self._prefix_rows = []
        self.last_summary = None
        self.coefficient_probe = None

    @contextmanager
    def collect_response(self):
        """Collect only original-forward metadata, then close before backward."""
        self._rows = []
        self._prefix_rows = []
        self.last_summary = None
        self._collecting = True
        try:
            yield
        except BaseException:
            self._rows = []
            self._prefix_rows = []
            raise
        finally:
            self._collecting = False

    def observe_prefix_coefficients(self, payload):
        """Qualification only: retain the existing native C output, not a recomputation."""
        if self._collecting and self.coefficient_probe is not None:
            self._prefix_rows.append((payload['candidate'].detach().reshape(-1, 16),
                                      payload['prefix'].detach().reshape(-1, 16)))

    def chunk_terms(self, logits, token_loss, proposals, labels, weights, selection,
                    pair_rows, eos_token_ids=()):
        score, low, high = continuation_scores(
            logits, proposals, labels, weights, selection, eos_token_ids)
        errors, pair_extra = pair_rows
        repair_score = score.unsqueeze(-1).expand_as(weights)[errors]
        beta = weights.detach()[errors].float()
        is_high = high[errors]
        weighted_score = beta * is_high * repair_score
        if self._collecting:
            self._rows.append((beta.detach(), is_high.detach(), repair_score.detach()))
        return (weighted_score.sum().detach(),
                (weighted_score * token_loss[errors]).sum(),
                (weighted_score * pair_extra).sum())

    def reduce_response(self, native_numerator, tail_terms, native_terms, pair_terms,
                        native_metrics, preservation_terms=()):
        """Apply one factorized correction to the assembled C+K+PAIR numerator."""
        weighted_score, weighted_fkl, weighted_pair = tail_terms
        if self._rows:
            beta, high, scores = (torch.cat(values) for values in zip(*self._rows))
        else:
            beta = weighted_score.new_empty(0)
            high = torch.empty(0, dtype=torch.bool, device=beta.device)
            scores = beta
        self._rows = []
        candidate, native, pool = high_pool_coefficients(
            beta, high, scores, native_metrics, weighted_score)
        zero = weighted_score.new_zeros(())
        correction = correction_fkl = correction_pair = zero
        numerator = native_numerator
        if pool['active']:
            kappa, mean = pool['kappa'].detach(), pool['mean'].detach()
            high_fkl, high_pair = native_terms[8], pair_terms[1]
            # Scalar cancellation must not suppress a nonzero gradient change.
            correction = kappa * ((weighted_fkl + weighted_pair) - mean * (high_fkl + high_pair))
            numerator = native_numerator + correction
            correction_fkl = kappa * (weighted_fkl.detach() - mean * high_fkl.detach())
            correction_pair = kappa * (weighted_pair.detach() - mean * high_pair.detach())
        summary = {'continuation_' + name: value.detach()
                   for name, value in pool.items() if name != 'active'}
        summary.update(
            continuation_active_response=zero + float(pool['active']),
            continuation_fk_correction=correction_fkl.detach(),
            continuation_pair_correction=correction_pair.detach(),
            continuation_total_correction=correction.detach(),
        )
        self.last_summary = summary
        prefix_rows, self._prefix_rows = self._prefix_rows, []
        if self.coefficient_probe is not None:
            preservation, prefix = (torch.cat(values) for values in zip(*prefix_rows))
            self.coefficient_probe(dict(beta=beta, high=high, scores=scores,
                                        native=native, candidate=candidate, summary=summary,
                                        preservation_coefficients=preservation,
                                        preservation_mask=prefix,
                                        preservation_terms=tuple(term.detach() for term in preservation_terms),
                                        native_terms=tuple(term.detach() for term in native_terms),
                                        pair_terms=tuple(term.detach() for term in pair_terms)))
        return numerator, summary


def install_continuation_repair(model, *, enabled=True):
    """Install only after the wrapper has restored its exact native parent."""
    if type(enabled) is not bool:
        raise ValueError('Continuation enable flag must be boolean')
    if enabled:
        from .repair_conditional_kl import validate_repair_conditional_kl
        validate_repair_conditional_kl(model)
        if not getattr(model, 'repair_conditional_kl', False) or model.block_size != 16:
            raise ValueError('Continuation repair requires native block16 C+K+PAIR')
    model.continuation_repair = ContinuationRepair() if enabled else None
    model.prefix_allocation.coefficient_probe = (
        model.continuation_repair.observe_prefix_coefficients if enabled else None)
    return continuation_repair_contract(enabled)
