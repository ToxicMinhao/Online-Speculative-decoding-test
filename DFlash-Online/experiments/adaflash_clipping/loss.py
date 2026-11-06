"""Compare two clipping rules while sharing the qualified probability path.

This is a common-arithmetic diagnostic, not a reproduction of the official
AdaFlash kernels. Both branches use the existing BF16 log-softmax/exp and CE.
Only the position-clipped branch uses FP32 for normalization and rescaling;
its effective RKL row is cast back before the shared BF16 mixture operation.
"""

import torch

from online_feedback.adaflash_opd import (
    CE_FRACTION, RKL_FRACTION, ENTRY_CAP, DIAGNOSTIC_NAMES,
    paper_position_loss,
)


MODES = ("paper_entry", "repo_position")
COMPARISON_NAMES = (
    "raw_reverse_kl_numerator",
    "paper_entry_clipped_count",
    "paper_entry_clipped_positions",
    "repo_position_clipped_count",
    "eligible_entries",
    "available_positions",
)


def clipping_contract(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown clipping rule: {mode}")
    return {
        "mode": mode,
        "ce_fraction": CE_FRACTION,
        "reverse_kl_fraction": RKL_FRACTION,
        "cap": ENTRY_CAP,
        "probability_path": "shared_native_bf16_log_softmax_exp",
        "position_cap_normalization": "full_response_fp32_denominator",
        "position_cap_rescale": "fp32_then_cast_to_probability_dtype",
        "mixture": "shared_probability_dtype_before_fp32_weighted_sum",
        "response_denominator": "sum_all_sampled_block_weights_plus_1e-6",
        "official_kernel_reproduction": False,
    }


def position_clipped_rows(raw_reverse_kl, denominator):
    """Return numerator-equivalent rows and normalized rows before clipping.

The outer objective divides the complete response numerator by D. Multiplying
the clipped normalized rows by D here preserves that existing final reduction.
FP64 is supported only for the mathematical CPU oracle; production uses FP32.
"""
    decision_dtype = (
        torch.float64 if raw_reverse_kl.dtype == torch.float64 else torch.float32
    )
    raw = raw_reverse_kl.to(decision_dtype)
    full_denominator = denominator.to(decision_dtype)
    normalized = raw / full_denominator
    effective = normalized.clamp_max(ENTRY_CAP) * full_denominator
    return effective.to(raw_reverse_kl.dtype), normalized


class ClippingComparisonLoss:
    """Response-bound point loss for the existing OSD objective extension.

Call prepare_response once, with every sampled block's weights, before the
projection chunks. Finish that response's backward before preparing the next
response. Checkpoint recomputation then reads the same detached denominator;
chunk execution never changes state or appends diagnostics.
"""

    def __init__(self, mode, *, chunk_blocks=8):
        self.contract = clipping_contract(mode)
        self.mode = mode
        self.chunk_blocks = chunk_blocks
        self.denominator = None

    def prepare_response(self, weights):
        # Clone rather than retaining a view into a reusable mask buffer.
        # Match the existing outer reduction's FP32 addition order too. This
        # still computes one denominator across the entire response.
        chunks = weights.detach().split(self.chunk_blocks or weights.shape[1], dim=1)
        total = chunks[0].sum()
        for chunk in chunks[1:]:
            total = total + chunk.sum()
        self.denominator = total.add(1e-6).clone()
        return self.denominator

    def chunk_terms(self, draft_logits, teacher_logits, labels, weights):
        if self.denominator is None:
            raise RuntimeError("Prepare the complete response before projection chunks")
        paper_mixture, ce, paper_rkl, entries, teacher_ids = paper_position_loss(
            draft_logits, teacher_logits,
        )
        raw_rkl = entries.sum(dim=-1)
        position_rkl, normalized_rkl = position_clipped_rows(raw_rkl, self.denominator)
        if self.mode == "paper_entry":
            # Reuse the old mixture tensor: no arithmetic change to A58.
            mixture, rkl = paper_mixture, paper_rkl
        else:
            rkl = position_rkl
            mixture = CE_FRACTION * ce + RKL_FRACTION * rkl
        loss_sum = (mixture * weights).sum()
        with torch.no_grad():
            eligible = weights > 0
            entry_clips = (entries > ENTRY_CAP) & eligible.unsqueeze(-1)
            position_clips = (normalized_rkl > ENTRY_CAP) & eligible
            negative = (entries < 0) & eligible.unsqueeze(-1)
            positions = eligible.sum().float()
            correct = ((draft_logits.argmax(-1) == labels) & eligible).sum().float()
            # The original diagnostic slots remain additive and in the same
            # order. For repo_position, RKL is the effective numerator row.
            diagnostics = torch.stack((
                (ce * weights).sum(), (rkl * weights).sum(),
                entry_clips.sum().float(), negative.sum().float(),
                positions * draft_logits.shape[-1],
                entry_clips.any(-1).sum().float(), positions,
                ((teacher_ids != labels) & eligible).sum().float(),
            ))
            comparison = torch.stack((
                (raw_rkl * weights).sum(), entry_clips.sum().float(),
                entry_clips.any(-1).sum().float(), position_clips.sum().float(),
                positions * draft_logits.shape[-1], positions,
            ))
        return loss_sum, weights.sum(), correct, positions, diagnostics, comparison

    def response_metrics(self, terms, denominator):
        diagnostics = dict(zip(DIAGNOSTIC_NAMES, terms[4]))
        comparison = dict(zip(COMPARISON_NAMES, terms[5]))
        diagnostics.update({f"clipping_{key}": value for key, value in comparison.items()})
        diagnostics.update(
            adaflash_response_count=denominator.new_ones(()),
            adaflash_loss_numerator=terms[0].detach(),
            adaflash_loss_denominator=denominator.detach(),
            clipping_full_response_denominator=self.denominator,
        )
        ratios = {
            "adaflash_teacher_argmax_ce": (
                diagnostics["adaflash_ce_numerator"], denominator.detach(),
            ),
            "clipping_effective_reverse_kl_loss": (
                diagnostics["adaflash_rkl_numerator"], denominator.detach(),
            ),
            "clipping_raw_reverse_kl_loss": (
                comparison["raw_reverse_kl_numerator"], denominator.detach(),
            ),
            "clipping_paper_entry_rate": (
                comparison["paper_entry_clipped_count"], comparison["eligible_entries"],
            ),
            "clipping_repo_position_rate": (
                comparison["repo_position_clipped_count"], comparison["available_positions"],
            ),
        }
        return diagnostics, ratios
