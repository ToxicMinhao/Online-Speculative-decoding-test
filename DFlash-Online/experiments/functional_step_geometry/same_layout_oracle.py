"""Independent formula oracle at the production BF16 head partition.

The caller supplies one live complete-block pre-state graph and detached native
hidden states. No production prefix, weighting, sensitivity or proxy helper is
called here. Head slices retain the same eight-block shape, including the final
partial slice. This oracle does not revise the earlier cross-layout results.
"""
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint

from online_feedback.rng import preserve_rng


CHUNK_BLOCKS = 8
MARGIN_FLOOR = .05


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite independent same-layout '+name)


@torch.no_grad()
def reference_weights(proposals, labels, validity, canonical_probability,
                      rival_probability, eos_token_ids=()):
    """Direct block recurrence, followed by one response-wide normalization."""
    _finite(canonical_probability, 'canonical probability')
    _finite(rival_probability, 'rival probability')
    terminal = torch.zeros_like(labels, dtype=torch.bool)
    for token in eos_token_ids:
        terminal |= labels == token
    continuing = ~terminal[..., 0]
    matching = torch.ones_like(continuing)
    protected = torch.zeros_like(labels, dtype=torch.bool)
    for position in range(1, labels.shape[-1]):
        continuing = continuing & (validity[..., position] > 0)
        matching = matching & (proposals[..., position] == labels[..., position])
        protected[..., position] = continuing & matching
        continuing = continuing & ~terminal[..., position]
    lengths = protected.sum(-1, keepdim=True)
    position = torch.arange(labels.shape[-1], device=labels.device)
    triangular = torch.where(protected, lengths-position+1, 0)
    margin = canonical_probability.double()-rival_probability.double()
    raw = triangular.double()/torch.maximum(margin, margin.new_tensor(MARGIN_FLOOR)).square()
    denominator = raw.sum(dtype=torch.float64)
    _finite(denominator, 'response denominator')
    weights = (raw/denominator).float() if float(denominator) else torch.zeros_like(raw, dtype=torch.float32)
    return dict(prefix_mask=protected, margins=margin, raw_weights=raw,
                weights=weights.detach(), denominator=float(denominator))


@torch.no_grad()
def sensitivity_terms(reference_logits, native_logits, weights):
    """Evaluate the centered categorical covariance formula directly.

    The one-half response factor does not enter s or c; it belongs in the later
    column normalization sqrt(2*c), which this reference does not implement.
    """
    _finite(reference_logits, 'pre-state logits')
    _finite(native_logits, 'native logits')
    q = torch.softmax(reference_logits.detach().float(), dim=-1)
    displacement = native_logits.detach().float()-reference_logits.detach().float()
    mean = torch.sum(q*displacement, dim=-1, keepdim=True)
    centered = displacement-mean
    weighted_probability = weights.detach().float().unsqueeze(-1)*q
    sensitivity = weighted_probability*centered
    energy = torch.sum(weighted_probability.double()*centered.double().square(), dtype=torch.float64)
    # Independent actual-risk arithmetic is separate from FP32 geometry q.
    log_reference = torch.log_softmax(reference_logits.detach().double(), dim=-1)
    log_native = torch.log_softmax(native_logits.detach().double(), dim=-1)
    risk_weights = weights.detach().double().unsqueeze(-1)*log_reference.exp()
    risk = torch.sum(risk_weights*(log_reference-log_native), dtype=torch.float64)
    absolute = torch.sum(risk_weights*(log_reference.abs()+log_native.abs()), dtype=torch.float64)
    for name, value in [('sensitivity', sensitivity), ('energy', energy), ('risk', risk), ('absolute risk', absolute)]:
        _finite(value, name)
    summary = dict(reference_logit_sum=float(reference_logits.double().sum()),
        native_logit_sum=float(native_logits.double().sum()),
        probability_sum=float(q.double().sum()), displacement_norm=float(displacement.double().norm()),
        mean_norm=float(mean.double().norm()), sensitivity_norm=float(sensitivity.double().norm()),
        sensitivity_sum=float(sensitivity.double().sum()), energy=float(energy))
    return sensitivity.detach(), energy, risk, absolute, summary


@dataclass
class SameLayoutReference:
    proposals: torch.Tensor
    prefix_mask: torch.Tensor
    margins: torch.Tensor
    weights: torch.Tensor
    c: float
    gradients: tuple
    native_risk_terms: tuple
    statistics: dict


def same_layout_reference(head, pre_hidden, native_hidden, labels, validity_weights,
                          active_parameters, *, eos_token_ids=(), chunk_blocks=CHUNK_BLOCKS,
                          work_callback=None):
    """Recompute references and one native VJP without production formula helpers.

    The caller owns model mode/caches and the original complete-block graph.
    This function preserves RNG and existing .grad buffers. The callback uses
    the existing ('additional_backwards', status) convention for its one VJP;
    physical head calls are counted locally and by the caller's module hooks.
    Full-vocabulary activations are checkpointed, not retained over the response.
    """
    if chunk_blocks != CHUNK_BLOCKS:
        raise ValueError('Independent native reference is pinned to eight-block head slices')
    if (pre_hidden.ndim != 4 or pre_hidden.shape != native_hidden.shape
            or pre_hidden.shape[0] != 1 or pre_hidden.shape[1] == 0
            or labels.shape != pre_hidden.shape[:-1] or validity_weights.shape != labels.shape):
        raise ValueError('Independent reference requires aligned complete response blocks')
    parameters = tuple(active_parameters)
    labels = labels.to(pre_hidden.device)
    validity = validity_weights.to(pre_hidden.device)
    regions = [(start, min(start+CHUNK_BLOCKS, pre_hidden.shape[1]))
               for start in range(0, pre_hidden.shape[1], CHUNK_BLOCKS)]
    counts = dict(reference_head=0, proxy_forward_head=0, proxy_recompute_head=0,
                  proxy_calls=0, backward_attempted=0, backward_completed=0)
    summaries = {}

    def project(hidden, phase):
        # Checkpoint recomputation may stop inside a head invocation after its
        # required saved tensors exist. Retain attempts as well as completions.
        attempt_key = phase+'_attempted'
        counts[attempt_key] = counts.get(attempt_key, 0)+1
        result = head(hidden)
        counts[phase] += 1
        _finite(result.detach(), 'head logits')
        return result

    @contextmanager
    def backward_work():
        counts['backward_attempted'] += 1
        if work_callback is not None:
            work_callback('additional_backwards', 'attempted')
        try:
            yield
        except BaseException:
            if work_callback is not None:
                work_callback('additional_backwards', 'failed')
            raise
        else:
            counts['backward_completed'] += 1
            if work_callback is not None:
                work_callback('additional_backwards', 'completed')

    with preserve_rng():
        predictions, canonical, rivals = [], [], []
        with torch.no_grad():
            for start, stop in regions:
                logits = project(pre_hidden[:, start:stop].detach(), 'reference_head')
                probability = logits.float().softmax(-1)
                local_labels = labels[:, start:stop, :, None]
                canonical.append(probability.gather(-1, local_labels).squeeze(-1))
                probability.scatter_(-1, local_labels, -torch.inf)
                rivals.append(probability.max(-1).values)
                predictions.append(logits.argmax(-1))
            proposals = torch.cat(predictions, dim=1)
            fixed = reference_weights(proposals, labels, validity, torch.cat(canonical, 1),
                                      torch.cat(rivals, 1), eos_token_ids)
        weights = fixed['weights']
        gradients = tuple(None for _ in parameters)
        total_proxy = None
        total_energy = pre_hidden.new_zeros((), dtype=torch.float64)
        total_risk = total_energy.clone()
        total_absolute = total_energy.clone()
        if bool(fixed['prefix_mask'].any()):
            def proxy(current, native, local_weights, block_start):
                counts['proxy_calls'] += 1
                start = int(block_start)
                phase = 'proxy_forward_head' if start not in summaries else 'proxy_recompute_head'
                logits = project(current, phase)
                with torch.no_grad():
                    endpoint_logits = project(native, phase)
                    sensitivity, energy, risk, absolute, summary = sensitivity_terms(
                        logits, endpoint_logits, local_weights)
                    if start not in summaries:
                        summaries[start] = dict(start=start, blocks=current.shape[1], **summary)
                value = torch.sum(logits.float()*sensitivity, dtype=torch.float32)
                return value, energy, risk, absolute

            with torch.enable_grad():
                for start, stop in regions:
                    terms = checkpoint(proxy, pre_hidden[:, start:stop],
                        native_hidden[:, start:stop].detach(), weights[:, start:stop], start,
                        use_reentrant=False)
                    total_proxy = terms[0] if total_proxy is None else total_proxy+terms[0]
                    total_energy = total_energy+terms[1]
                    total_risk = total_risk+terms[2]
                    total_absolute = total_absolute+terms[3]
                if float(total_energy) > 0:
                    with backward_work():
                        values = torch.autograd.grad(total_proxy, parameters, allow_unused=True)
                    gradients = tuple(None if value is None else value.detach() for value in values)
                    for value in gradients:
                        if value is not None:
                            _finite(value, 'native parameter VJP')
    statistics = dict(response_weight_denominator=fixed['denominator'],
        protected_occurrences=int(fixed['prefix_mask'].sum()),
        weight_sum=float(weights.double().sum()), chunks=[stop-start for start, stop in regions],
        chunk_statistics=[summaries[start] for start in sorted(summaries)], work=counts,
        proxy=None if total_proxy is None else float(total_proxy.detach()),
        formula_source='independent recurrence and centered categorical covariance; native BF16 head layout')
    return SameLayoutReference(proposals, fixed['prefix_mask'], fixed['margins'], weights,
        float(total_energy), gradients, (float(total_risk), float(total_absolute)), statistics)
