"""Inactive CPU reference for one-response direct support-plus-score allocation.

No installer or production import path activates this module. Callers supply
the already computed native row losses and log probabilities; there are no
model forwards, vocabulary projections, optimizer steps or mutable run state.
"""

from dataclasses import dataclass

import torch

from online_feedback.first_error import first_error_mask, valid_prefix_mask


SMOOTHING = 0.5
NORMALIZATION_EPSILON = 1e-6


def operation_contract():
    return {
        "name": "inactive_cpu_direct_support_score_v1",
        "status": "matched_adapted_reference_not_official_reproduction",
        "input": "native forward-KL rows and native BF16 log_softmax; one response",
        "support": "current matching prefix plus first error; native validity/EOS",
        "smoothing": SMOOTHING,
        "probability": "exp(FP32 detached canonical gather from native log_softmax)",
        "score": "reverse_sum(valid * cumulative_product(where(valid, 0.5+0.5*q, 1)))",
        "pool": "all support positions in all blocks/chunks of one response",
        "B": "sum native beta over support",
        "R": "sum detached score r over support",
        "coefficient": "B*r/R on support; zero elsewhere; no extra beta factor",
        "loss": "sum(coefficient*native_forward_KL_row)/(B+1e-6)",
        "empty_support": "graph-connected zero supervision; no division by zero R",
        "chunk_boundary": "between complete blocks only; never between block slots",
        "runtime": "CPU only; inactive; no production installer",
    }


@dataclass(frozen=True)
class AllocationRows:
    valid: torch.Tensor
    support: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True)
class ChunkTerms:
    beta_mass: torch.Tensor
    score_mass: torch.Tensor
    score_loss_sum: torch.Tensor


@dataclass(frozen=True)
class ResponseReduction:
    loss: torch.Tensor
    beta_mass: torch.Tensor
    score_mass: torch.Tensor
    coefficient_scale: torch.Tensor


@torch.no_grad()
def allocation_rows(log_probability, proposals, labels, native_beta, eos_token_ids=()):
    """Build detached support and full-valid-block scores for complete blocks.

    Arrays are [block, slot], with [block, slot, vocabulary] log probabilities.
    A leading response/batch dimension is deliberately rejected: pooling two
    responses here would silently change the specified normalization.
    """
    tensors = (log_probability, proposals, labels, native_beta)
    if any(x.device.type != "cpu" for x in tensors):
        raise ValueError("This inactive reference accepts CPU tensors only")
    if (labels.ndim != 2 or proposals.shape != labels.shape
            or native_beta.shape != labels.shape
            or log_probability.ndim != 3
            or log_probability.shape[:-1] != labels.shape):
        raise ValueError("Pass complete [block, slot] arrays for exactly one response")
    valid = valid_prefix_mask(labels, native_beta, eos_token_ids)
    support = first_error_mask(proposals, labels, native_beta, eos_token_ids)
    q = log_probability.detach().gather(-1, labels.unsqueeze(-1)).squeeze(-1).float().exp()
    factors = torch.where(valid, SMOOTHING + (1 - SMOOTHING) * q, 1.0)
    # Multiplying by valid BEFORE the reverse sum prevents padding from
    # extending the score horizon. Invalid factors alone are insufficient.
    products = factors.cumprod(-1) * valid
    score = products.flip(-1).cumsum(-1).flip(-1)
    return AllocationRows(valid, support, torch.where(valid, score, 0.0))


def chunk_terms(log_probability, token_loss, proposals, labels, native_beta,
                eos_token_ids=()):
    """Return three additive scalars; only score_loss_sum retains a gradient."""
    if token_loss.device.type != "cpu" or token_loss.shape != labels.shape:
        raise ValueError("Pass native CPU token-loss rows matching labels")
    rows = allocation_rows(log_probability, proposals, labels, native_beta, eos_token_ids)
    selected_score = torch.where(rows.support, rows.score, 0.0)
    beta_mass = (native_beta.detach().float() * rows.support).sum()
    return ChunkTerms(beta_mass, selected_score.sum(), (selected_score * token_loss).sum())


def reduce_response(chunks):
    """Reduce all chunks, allocate one pool, and normalize once per response.

    An all-invalid response still supplies its native graph-connected chunk(s).
    No epsilon is inserted into positive R; smoothing guarantees R > 0 when
    support is nonempty. B+1e-6 is the native loss denominator.
    """
    chunks = tuple(chunks)
    if not chunks:
        raise ValueError("Supply at least one native chunk, including for empty support")
    beta_mass = torch.stack([x.beta_mass for x in chunks]).sum().detach()
    score_mass = torch.stack([x.score_mass for x in chunks]).sum().detach()
    score_loss_sum = torch.stack([x.score_loss_sum for x in chunks]).sum()
    scale = beta_mass / score_mass if bool(score_mass > 0) else beta_mass.new_zeros(())
    numerator = scale * score_loss_sum
    loss = numerator / (beta_mass + NORMALIZATION_EPSILON)
    return ResponseReduction(loss, beta_mass, score_mass, scale)


@torch.no_grad()
def direct_coefficients(rows, response):
    """Inspect coefficients using the already reduced response-wide pool."""
    return torch.where(rows.support, rows.score * response.coefficient_scale, 0.0)
