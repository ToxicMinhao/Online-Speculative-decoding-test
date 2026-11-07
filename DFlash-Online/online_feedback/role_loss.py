"""Isolated MIX/ROLE objectives on the qualified native K+ selection.

The shared forward supplies its existing logits, forward-KL rows and selection.
Only additive FP32 scalars cross projection chunks. Reverse KL is evaluated on
the nonempty mixture group, using the released AdaFlash BF16 row operations.
"""

import torch

from .adaflash_repo import NUMERICAL_TOLERANCES, POSITION_CAP, repo_position_terms
from .kplus import first_error_groups, group_terms, kplus_contract, redistribute_response


DIAGNOSTIC_NAMES = (
    'role_loss_prefix_positions', 'role_loss_error_positions',
    'role_loss_prefix_weight', 'role_loss_error_weight',
    'role_loss_prefix_mixture_positions', 'role_loss_error_mixture_positions',
    'role_loss_prefix_capped_positions', 'role_loss_error_capped_positions',
    'role_loss_probability_zeros', 'role_loss_probability_entries',
    'role_loss_nonfinite_entries', 'role_loss_teacher_argmax_canonical_mismatches',
    'role_loss_prefix_teacher_argmax_canonical_mismatches',
    'role_loss_error_teacher_argmax_canonical_mismatches',
    'role_loss_normalized_rkl_before_cap',
)


def role_loss_contract(arm):
    if arm not in ('MIX', 'ROLE'):
        raise ValueError('Role-loss arm must be MIX or ROLE; KPLUS uses install_kplus')
    return dict(
        name='kplus_role_loss_pilot_v1', arm=arm, selection=kplus_contract(),
        temperature=1.0, ce_fraction=0.8, reverse_kl_fraction=0.2,
        normalized_position_cap=POSITION_CAP,
        full_denominator='D=original_full_native_weights.sum()+1e-6',
        selected_denominator='Z=original_selected_weights.sum()+1e-6',
        reverse_kl='released_repo_BF16_rows; inplace_divide_by_FP32_D; clamp_max(0.01)',
        mixture_group='S' if arm == 'MIX' else 'E',
        expression=('0.8*Nc(S)/Z+0.2*(D/Z)*Nr(S)' if arm == 'MIX' else
                    'Nf(P)/Z+0.8*Nc(E)/Z+0.2*(D/Z)*Nr(E)'),
        expression_order='left_to_right_after_response_wide_component_reductions',
        redistribution='native_Kplus_two_priority_correction_separately_for_CE_and_RKL',
        no_error_ROLE='return_original_native_Kplus_loss_graph',
        empty_mixture_group='do_not_evaluate_CE_or_RKL',
        weights='detached_native_FP32',
        checkpoint_lifetime='finish_response_backward_before_preparing_next_D',
        additional_model_forwards=0, additional_vocabulary_projections=0,
        tolerances=NUMERICAL_TOLERANCES,
    )


class RoleLossReducer:
    """Consume native K terms; never stack the K+ and point-loss installers.

Terms 0..4 are the unchanged native selected forward-KL terms. Terms 5..10
are native K+ group terms (W1,W2,Nf1,Nf2,count1,count2). The remaining terms
are prefix FKL, mixture CE/RKL, their priority-1/2 numerators, and diagnostics.
Finish backward before preparing another response: checkpoint recomputation
uses this response's saved full denominator, as in the qualified A_repo path.
"""

    def __init__(self, arm):
        role_loss_contract(arm)
        self.arm = arm
        self.denominator = None

    @staticmethod
    def validate_response(input_ids):
        if input_ids.shape[0] != 1:
            raise ValueError('Role loss requires one response per microbatch')

    def prepare_response(self, weights):
        if weights.dtype != torch.float32 or weights.requires_grad:
            raise ValueError('Role loss requires detached native FP32 weights')
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError('Role loss requires finite nonnegative native weights')
        self.denominator = (weights.detach().reshape(-1).sum() + 1e-6).clone()
        return self.denominator

    def chunk_terms(self, logits, teacher_logits, token_loss, proposals, labels,
                    weights, selection, native_terms, eos_token_ids=()):
        if self.denominator is None:
            raise RuntimeError('Prepare the full response denominator before role loss')
        weights = weights.detach()
        priority1, priority2 = first_error_groups(
            proposals, labels, weights, selection, eos_token_ids,
        )
        errors = priority1 | priority2
        prefix = selection & ~errors
        mixture = selection if self.arm == 'MIX' else errors
        groups = group_terms(token_loss, proposals, labels, weights, selection, eos_token_ids)
        # Preserve the native numerator graph in every no-error chunk. A whole
        # no-error ROLE response returns native_terms[0] directly below.
        prefix_numerator = (native_terms[0] if not bool(errors.any()) else
                            (token_loss * (weights * prefix)).sum())
        zero = weights.new_zeros(())
        ce = rkl = ce1 = ce2 = rkl1 = rkl2 = zero
        normalized_sum = zero
        prefix_mixture = error_mixture = prefix_capped = error_capped = zero
        probability_zeros = probability_entries = nonfinite_entries = zero
        if bool(mixture.any()):
            # Index before evaluating RKL: zero-weight F.kl_div can still have
            # nonfinite backward when a BF16 probability underflows to zero.
            rows = repo_position_terms(logits[mixture], teacher_logits[mixture], self.denominator)
            active_weights = weights[mixture]
            low_weights = active_weights * priority1[mixture]
            high_weights = active_weights * priority2[mixture]
            ce = (rows['ce'] * active_weights).sum()
            rkl = (rows['clipped_rkl'] * active_weights).sum()
            ce1 = (rows['ce'] * low_weights).sum()
            ce2 = (rows['ce'] * high_weights).sum()
            rkl1 = (rows['clipped_rkl'] * low_weights).sum()
            rkl2 = (rows['clipped_rkl'] * high_weights).sum()
            with torch.no_grad():
                is_prefix = prefix[mixture]
                is_error = errors[mixture]
                capped = rows['normalized_rkl'] > POSITION_CAP
                prefix_mixture = is_prefix.sum().float()
                error_mixture = is_error.sum().float()
                prefix_capped = (capped & is_prefix).sum().float()
                error_capped = (capped & is_error).sum().float()
                probability_zeros = (rows['probabilities'] == 0).sum().float()
                probability_entries = zero.new_tensor(rows['probabilities'].numel())
                nonfinite_entries = (~torch.isfinite(rows['entries'])).sum().float()
                normalized_sum = (rows['normalized_rkl'] * active_weights).sum()
        with torch.no_grad():
            mismatch = teacher_logits.detach().argmax(-1) != labels
            diagnostics = torch.stack((
                prefix.sum().float(), errors.sum().float(),
                (weights * prefix).sum(), (weights * errors).sum(),
                prefix_mixture, error_mixture, prefix_capped, error_capped,
                probability_zeros, probability_entries, nonfinite_entries,
                (mismatch & selection).sum().float(),
                (mismatch & prefix).sum().float(), (mismatch & errors).sum().float(),
                normalized_sum,
            ))
        return (*native_terms, *groups, prefix_numerator, ce, rkl, ce1, ce2, rkl1, rkl2,
                diagnostics)

    @staticmethod
    def _mixture_numerators(terms):
        weight1, weight2, _, _, count1, count2 = terms[5:11]
        ce, _ = redistribute_response(terms[12], terms[1], (
            weight1, weight2, terms[14], terms[15], count1, count2,
        ))
        rkl, _ = redistribute_response(terms[13], terms[1], (
            weight1, weight2, terms[16], terms[17], count1, count2,
        ))
        return ce, rkl

    def reduce_response(self, terms, outer_denominator=None):
        selected_denominator = (terms[1] + 1e-6 if outer_denominator is None
                                else outer_denominator)
        if not bool(terms[3] > 0) or (
                self.arm == 'ROLE' and not bool((terms[9] + terms[10]) > 0)):
            # This is the original K+ neutral branch, including its loss graph.
            return terms[0] / selected_denominator.clamp_min(torch.finfo(torch.float32).tiny)
        ce, rkl = self._mixture_numerators(terms)
        if self.arm == 'MIX':
            return 0.8 * ce / selected_denominator + 0.2 * (
                self.denominator / selected_denominator) * rkl
        return terms[11] / selected_denominator + 0.8 * ce / selected_denominator + 0.2 * (
            self.denominator / selected_denominator) * rkl

    @torch.no_grad()
    def response_metrics(self, terms, outer_denominator=None):
        selected_denominator = (terms[1] + 1e-6 if outer_denominator is None
                                else outer_denominator)
        ce, rkl = self._mixture_numerators(terms)
        _, metrics = redistribute_response(terms[0], terms[1], terms[5:11])
        metrics.update(dict(zip(DIAGNOSTIC_NAMES, terms[18])))
        one = self.denominator.new_ones(())
        metrics.update(
            role_loss_response_count=one,
            role_loss_full_denominator=self.denominator,
            role_loss_selected_denominator=selected_denominator,
            role_loss_full_to_selected_ratio=self.denominator / selected_denominator,
            role_loss_prefix_fkl_numerator=terms[11].detach(),
            role_loss_ce_numerator=ce, role_loss_clipped_rkl_numerator=rkl,
            role_loss_prefix_component=(terms[11] / selected_denominator
                                        if self.arm == 'ROLE' else one.new_zeros(())),
            role_loss_ce_component=0.8 * ce / selected_denominator,
            role_loss_rkl_component=0.2 * (self.denominator / selected_denominator) * rkl,
            role_loss_value=self.reduce_response(terms, selected_denominator),
        )
        ratios = {
            'role_loss_selected_fraction': (terms[4][1], terms[4][0]),
            'role_loss_full_to_selected': (self.denominator, selected_denominator),
            'role_loss_priority2_fraction': (terms[10], terms[9] + terms[10]),
            'role_loss_prefix_cap_fraction': (terms[18][6], terms[18][4]),
            'role_loss_error_cap_fraction': (terms[18][7], terms[18][5]),
            'role_loss_probability_zero_fraction': (terms[18][8], terms[18][9]),
            'role_loss_teacher_canonical_mismatch_fraction': (terms[18][11], terms[3]),
        }
        return metrics, ratios


def install_role_loss(model, arm):
    """Install one experimental reducer on the native BF16 K model."""
    contract = role_loss_contract(arm)
    if model.objective != 'osd_first_error' or model.loss_dtype != torch.bfloat16:
        raise ValueError('Role loss requires native BF16 osd_first_error')
    if any(getattr(model, name, None) is not None for name in (
            'prefix_guard', 'first_error_redistribution', 'point_loss_extension',
            'role_loss_extension')):
        raise ValueError('Role loss cannot combine objective extensions')
    model.role_loss_extension = RoleLossReducer(arm)
    return contract
