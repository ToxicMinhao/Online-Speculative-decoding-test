"""Paper-defined AdaFlash mixture loss on ordinary OSD response positions.

The experiment keeps the native BF16 vocabulary projection and probability
arithmetic. Position weights and the response reduction remain FP32. This is
entry-wise clipping before vocabulary reduction, not the released trainer's
post-normalization clipping variant.
"""

import math

import torch
import torch.nn.functional as F


CE_FRACTION = 0.8
RKL_FRACTION = 0.2
ENTRY_CAP = 0.01
DIAGNOSTIC_NAMES = (
    "adaflash_ce_numerator",
    "adaflash_rkl_numerator",
    "adaflash_clipped_entries",
    "adaflash_negative_entries",
    "adaflash_eligible_entries",
    "adaflash_clipped_positions",
    "adaflash_available_positions",
    "adaflash_teacher_argmax_canonical_mismatches",
)
# Declared before running the fixtures. BF16 compares two explicitly equivalent
# arithmetic paths; the CPU tests also check an analytic FP64 softmax gradient.
NUMERICAL_TOLERANCES = {
    "bfloat16": {"rtol": 0.0, "atol": 0.0},
    "float64": {"rtol": 1e-12, "atol": 1e-12},
}


def adaflash_opd_contract():
    """Fixed loss definition for checkpoint and experiment provenance."""
    return {
        "name": "adaflash_paper_opd_v1",
        "main_loss": "0.8_teacher_argmax_ce_plus_0.2_entry_clipped_reverse_kl",
        "temperature": 1.0,
        "ce_fraction": CE_FRACTION,
        "reverse_kl_fraction": RKL_FRACTION,
        "entry_cap": ENTRY_CAP,
        "clipping_scope": "each_vocabulary_entry_before_vocabulary_sum",
        "negative_entries": "unchanged",
        "cap_equality_derivative": "torch_clamp_max_native",
        "teacher": "frozen_target_predecessor_hidden",
        "ce_labels": "current_frozen_teacher_projection_argmax",
        "draft_projection_precision": "bfloat16",
        "teacher_projection_precision": "bfloat16",
        "probability_arithmetic": "bfloat16_stable_log_softmax_then_exp",
        "entry_arithmetic_and_vocabulary_sum": "bfloat16",
        "position_weight_and_response_reduction_precision": "float32",
        "selection": "all_native_osd_valid_response_proposals_excluding_seed",
        "position_decay_and_response_denominator": "unchanged_native_osd",
        "denominator_epsilon": 1e-6,
        "reference_model": None,
        "adaptive_length_head": False,
        "public_trainer_variant_reproduced": False,
    }


def paper_position_loss(draft_logits, teacher_logits):
    """Return mixture/CE/RKL rows, raw entries and detached teacher labels.

Do not form log(p) from softmax(p): a tiny probability may round to zero in
BF16 even though log_softmax remains finite. A clipped entry may be negative
after the vocabulary sum; no positivity clamp belongs to this objective.
"""
    log_q = F.log_softmax(draft_logits, dim=-1)
    with torch.no_grad():
        log_p = F.log_softmax(teacher_logits.detach(), dim=-1)
        target_ids = teacher_logits.detach().argmax(dim=-1)
    q = log_q.exp()
    entries = q * (log_q - log_p)
    reverse_kl = entries.clamp_max(ENTRY_CAP).sum(dim=-1)
    cross_entropy = -log_q.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    mixture = CE_FRACTION * cross_entropy + RKL_FRACTION * reverse_kl
    return mixture, cross_entropy, reverse_kl, entries, target_ids


def chunk_terms(draft_logits, teacher_logits, labels, weights):
    """Additive response terms; checkpoint recomputation mutates no counters."""
    mixture, ce, rkl, entries, teacher_ids = paper_position_loss(draft_logits, teacher_logits)
    loss_sum = (mixture * weights).sum()
    with torch.no_grad():
        eligible = weights > 0
        clipped = (entries > ENTRY_CAP) & eligible.unsqueeze(-1)
        negative = (entries < 0) & eligible.unsqueeze(-1)
        positions = eligible.sum().float()
        correct = ((draft_logits.argmax(-1) == labels) & eligible).sum().float()
        diagnostics = torch.stack((
            (ce * weights).sum(),
            (rkl * weights).sum(),
            clipped.sum().float(),
            negative.sum().float(),
            positions * draft_logits.shape[-1],
            clipped.any(-1).sum().float(),
            positions,
            ((teacher_ids != labels) & eligible).sum().float(),
        ))
    return loss_sum, weights.sum(), correct, positions, diagnostics


def response_metrics(terms, denominator):
    """Emit statistics once after projection chunks have been combined."""
    diagnostics = dict(zip(DIAGNOSTIC_NAMES, terms[4]))
    diagnostics["adaflash_response_count"] = denominator.new_ones(())
    diagnostics["adaflash_loss_numerator"] = terms[0].detach()
    diagnostics["adaflash_loss_denominator"] = denominator.detach()
    ratios = {
        "adaflash_teacher_argmax_ce": (diagnostics["adaflash_ce_numerator"], denominator.detach()),
        "adaflash_entry_clipped_reverse_kl": (diagnostics["adaflash_rkl_numerator"], denominator.detach()),
        "adaflash_entry_clipping_rate": (
            diagnostics["adaflash_clipped_entries"], diagnostics["adaflash_eligible_entries"],
        ),
        "adaflash_position_clipping_rate": (
            diagnostics["adaflash_clipped_positions"], diagnostics["adaflash_available_positions"],
        ),
    }
    return diagnostics, ratios


class PaperAdaFlashOPD:
    """An explicit point-loss extension; no model, sampler or optimizer state."""

    chunk_terms = staticmethod(chunk_terms)
    response_metrics = staticmethod(response_metrics)


def install_adaflash_opd(model):
    """Install only on the unchanged all-position native BF16 OSD path."""
    if model.objective != "osd" or model.loss_dtype != torch.bfloat16:
        raise ValueError("AdaFlash paper OPD requires ordinary all-position BF16 OSD")
    if any(getattr(model, name, None) is not None for name in (
            "prefix_guard", "first_error_redistribution", "point_loss_extension")):
        raise ValueError("AdaFlash paper OPD cannot be combined with another objective extension")
    model.point_loss_extension = PaperAdaFlashOPD()
    return adaflash_opd_contract()


def _direct_reference(draft_logits, teacher_logits, weights):
    """Independent qualification expression with where, not clamp_max."""
    log_q = torch.log_softmax(draft_logits, dim=-1)
    log_p = torch.log_softmax(teacher_logits.detach(), dim=-1)
    raw = log_q.exp() * (log_q - log_p)
    capped = torch.where(raw > ENTRY_CAP, torch.full_like(raw, ENTRY_CAP), raw)
    target = teacher_logits.detach().argmax(-1)
    ce = -torch.take_along_dim(log_q, target.unsqueeze(-1), dim=-1).squeeze(-1)
    loss_rows = CE_FRACTION * ce + RKL_FRACTION * capped.sum(-1)
    return (loss_rows * weights).sum() / (weights.sum() + 1e-6)


def numerical_qualification(device="cpu"):
    """Tiny direct loss/gradient checks; zero model forwards or optimizer steps.

    Uses only constant inputs, so no training RNG is consumed. The caller writes
    the returned receipt and stops if status is failed. Tolerances are fixed.
    """
    fixtures = {
        "mixed_clipped_unclipped_negative": (
            [[0.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.4, 0.0]],
            [[2.0, -2.0, 0.0, 0.0], [0.0, -0.1, 0.3, 0.1]],
        ),
        "unclipped_equal": (
            [[0.0, 0.2, -0.1, 0.3], [0.3, -0.1, 0.2, 0.0]],
            [[0.0, 0.2, -0.1, 0.3], [0.3, -0.1, 0.2, 0.0]],
        ),
        "extreme_finite_logits": (
            [[1000.0, -1000.0, 0.0, -500.0], [-1000.0, 1000.0, 0.0, -500.0]],
            [[-1000.0, 1000.0, 0.0, -500.0], [1000.0, -1000.0, 0.0, -500.0]],
        ),
    }
    receipt = {
        "status": "passed", "device": str(device), "contract": adaflash_opd_contract(),
        "tolerances": NUMERICAL_TOLERANCES, "cases": [],
        "work": {"synthetic_loss_forwards": 0, "synthetic_loss_backwards": 0,
                 "model_forwards": 0, "optimizer_steps": 0, "rng_draws": 0},
    }

    def scalar(value):
        number = float(value.detach())
        return number if math.isfinite(number) else str(number)

    for precision, tolerance in NUMERICAL_TOLERANCES.items():
        dtype = getattr(torch, precision)
        for name, (draft_values, teacher_values) in fixtures.items():
            draft = torch.tensor(draft_values, dtype=dtype, device=device, requires_grad=True)
            reference_draft = draft.detach().clone().requires_grad_()
            teacher = torch.tensor(teacher_values, dtype=dtype, device=device)
            weights = torch.tensor([1.0, 0.37], dtype=torch.float32, device=device)
            actual_rows, ce, rkl, entries, _ = paper_position_loss(draft, teacher)
            actual = (actual_rows * weights).sum() / (weights.sum() + 1e-6)
            expected = _direct_reference(reference_draft, teacher, weights)
            receipt["work"]["synthetic_loss_forwards"] += 2
            actual_gradient = torch.autograd.grad(actual, draft)[0]
            expected_gradient = torch.autograd.grad(expected, reference_draft)[0]
            receipt["work"]["synthetic_loss_backwards"] += 2
            finite = all(bool(torch.isfinite(value).all()) for value in (
                actual, expected, actual_gradient, expected_gradient))
            value_matches = bool(torch.isclose(actual, expected, **tolerance))
            gradient_matches = bool(torch.isclose(actual_gradient, expected_gradient, **tolerance).all())
            with torch.no_grad():
                wrong_rows = CE_FRACTION * ce + RKL_FRACTION * entries.sum(-1).clamp_max(ENTRY_CAP)
                wrong_loss = (wrong_rows * weights).sum() / (weights.sum() + 1e-6)
                sum_clip_difference = float((actual.detach() - wrong_loss).abs())
            distinguishes_sum_clipping = (
                name != "mixed_clipped_unclipped_negative" or sum_clip_difference > 0
            )
            row = {
                "fixture": name, "precision": precision,
                "actual_loss": scalar(actual), "reference_loss": scalar(expected),
                "max_gradient_error": scalar((actual_gradient - expected_gradient).abs().max()),
                "clipped_entries": int((entries > ENTRY_CAP).sum()),
                "negative_entries": int((entries < 0).sum()),
                "minimum_clipped_rkl": scalar(rkl.min()),
                "absolute_difference_from_wrong_sum_clipping": scalar((actual.detach() - wrong_loss).abs()),
                "finite": finite, "value_matches": value_matches,
                "gradient_matches": gradient_matches,
                "distinguishes_sum_clipping": distinguishes_sum_clipping,
            }
            receipt["cases"].append(row)
            if not (finite and value_matches and gradient_matches and distinguishes_sum_clipping):
                receipt["status"] = "failed"
                row["draft_gradient"] = [scalar(x) for x in actual_gradient.flatten()]
                row["reference_gradient"] = [scalar(x) for x in expected_gradient.flatten()]
                return receipt
    return receipt
