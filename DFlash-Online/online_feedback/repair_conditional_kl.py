"""First-error conditional KL on the unchanged native BF16 C+K+ objective.

Only the selected error rows are promoted to FP32. The two additive error
numerators cross head chunks, then reuse K+'s detached response multipliers.
The original BF16 forward KL and bounded-prefix correction remain separate.
"""

import torch
import torch.nn.functional as F

from .kplus import first_error_groups


SUMMARY_FIELDS = (
    "repair_rows", "active_rows", "mu_sum", "complement_mass_sum",
    "conditional_kl_sum", "teacher_y_preferred_rows", "teacher_tie_rows",
    "teacher_r_preferred_rows", "log_mu_roundoff_clamps",
)
LOG_MU_ROUNDOFF_ULPS = 8


def repair_conditional_kl_contract(enabled=True):
    if type(enabled) is not bool:
        raise ValueError("Repair conditional KL enable flag must be boolean")
    return {
        "name": "repair_conditional_kl_v1", "enabled": enabled,
        "base": "native_bf16_kplus_bounded_prefix_allocation",
        "support": "current first valid mismatch only",
        "pair": "aligned canonical token and native detached draft argmax",
        "auxiliary": "(1-mu)*KL(target_pair||draft_pair)",
        "coefficient": "same final detached K+ repair coefficient as native FKL",
        "pooling": "all sampled blocks of one response",
        "normalization": "native selected beta sum + 1e-6 once per response",
        "temperature": 1, "auxiliary_precision": "float32",
        "log_mu_roundoff_tolerance": "8*float32_eps*max(1,abs(pair_lse),abs(full_lse))",
        "additional_model_forwards": 0, "additional_vocabulary_projections": 0,
    }


def validate_repair_conditional_kl(model):
    """Reject unsupported combinations, including hooks added after installation."""
    if (model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16
            or getattr(model, "first_error_redistribution", None) is None
            or getattr(model, "prefix_allocation", None) is None):
        raise ValueError("Repair conditional KL requires installed native BF16 C+K+")
    conflicts = (
        "point_loss_extension", "role_loss_extension", "prefix_guard", "native_prefix_roles",
        "recent_repair_retention", "survival_head_extension", "adversarial_block",
        "successor_shuffle", "block_observer",
    )
    if any(getattr(model, name, None) is not None for name in conflicts):
        raise ValueError("Repair conditional KL cannot combine other objective hooks")


def install_repair_conditional_kl(model, *, enabled=True):
    contract = repair_conditional_kl_contract(enabled)
    if enabled:
        validate_repair_conditional_kl(model)
    model.repair_conditional_kl = enabled
    return contract


def conditional_pair_values(draft_pair, teacher_pair, teacher_rows):
    """FP32 pair KL and complement, without a full-vocabulary FP32 softmax.

    log(mu) can exceed zero only by subtraction/reduction roundoff. Permit
    eight FP32 epsilons scaled by the two log partitions; count every clamp.
    Any larger excess or nonfinite input/intermediate fails visibly.
    """
    draft_pair = draft_pair.float()
    teacher_pair = teacher_pair.detach().float()
    teacher_rows = teacher_rows.detach().float()
    if not all(bool(torch.isfinite(value).all()) for value in (
            draft_pair, teacher_pair, teacher_rows)):
        raise ValueError("Repair conditional KL received nonfinite selected logits")
    with torch.no_grad():
        log_p = F.log_softmax(teacher_pair, dim=-1)
        pair_lse = torch.logsumexp(teacher_pair, dim=-1)
        full_lse = torch.logsumexp(teacher_rows, dim=-1)
        log_mu = pair_lse - full_lse
        tolerance = LOG_MU_ROUNDOFF_ULPS * torch.finfo(torch.float32).eps * torch.maximum(
            torch.ones_like(full_lse), torch.maximum(pair_lse.abs(), full_lse.abs()))
        if not bool(torch.isfinite(log_mu).all()) or bool((log_mu > tolerance).any()):
            raise ValueError("Repair conditional KL pair mass exceeds one beyond FP32 roundoff")
        roundoff_clamps = (log_mu > 0).sum().float()
        log_mu = log_mu.clamp_max(0)
        complement = -torch.expm1(log_mu)
    log_q = F.log_softmax(draft_pair, dim=-1)
    pair_kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    extra = complement * pair_kl
    if not bool(torch.isfinite(pair_kl).all()) or not bool(torch.isfinite(extra).all()):
        raise ValueError("Repair conditional KL produced a nonfinite auxiliary")
    return extra, log_mu.exp(), complement, pair_kl, roundoff_clamps


def repair_pair_chunk_terms(logits, teacher_logits, proposals, labels, weights,
                            selection, eos_token_ids=()):
    """Additive priority-1/2 auxiliary numerators plus detached diagnostics."""
    if teacher_logits is None:
        raise ValueError("Repair conditional KL requires native frozen teacher logits")
    group1, group2 = first_error_groups(
        proposals, labels, weights, selection, eos_token_ids)
    errors = group1 | group2
    if not bool(errors.any()):
        zero = logits.new_zeros((), dtype=torch.float32)
        return zero, zero, zero.new_zeros(len(SUMMARY_FIELDS))
    # Select BEFORE the full-vocabulary FP32 cast: at most one row per block.
    teacher_rows = teacher_logits.detach()[errors].float()
    pair = torch.stack((labels[errors], proposals.detach()[errors]), dim=-1)
    teacher_pair = teacher_rows.gather(-1, pair)
    draft_pair = logits[errors].gather(-1, pair).float()
    extra, mu, complement, pair_kl, clamps = conditional_pair_values(
        draft_pair, teacher_pair, teacher_rows)
    weighted = weights.detach()[errors] * extra
    priority2 = group2[errors]
    numerator1 = weighted[~priority2].sum()
    numerator2 = weighted[priority2].sum()
    with torch.no_grad():
        preference = teacher_pair[:, 0] - teacher_pair[:, 1]
        summary = torch.stack((
            errors.sum().float(), (extra > 0).sum().float(), mu.sum(), complement.sum(),
            pair_kl.sum(), (preference > 0).sum().float(), (preference == 0).sum().float(),
            (preference < 0).sum().float(), clamps,
        ))
    return numerator1, numerator2, summary


def repair_pair_reduce_response(terms, native_metrics):
    """Use the actual native response-wide multipliers, never raw beta alone."""
    numerator1, numerator2, summary = terms
    auxiliary = (native_metrics["kplus_coefficient1"].detach() * numerator1
                 + native_metrics["kplus_coefficient2"].detach() * numerator2)
    metrics = {"repair_pair_" + name: value.detach()
               for name, value in zip(SUMMARY_FIELDS, summary)}
    metrics["repair_pair_auxiliary_numerator"] = auxiliary.detach()
    return auxiliary, metrics
