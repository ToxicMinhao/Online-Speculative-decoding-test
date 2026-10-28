"""Hard-CE, projected-target and OSD objectives on author DFlash blocks.

Sampling, conditioning and the trainable backbone remain in SpecForge. Only the
labels supplied to the weighted vocabulary loss change between objectives.
"""
import copy
import math
import time

import torch
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.training.strategies.base import (
    DFlashTrainStrategy, StepOutput, _cpu_max_valid_anchors,
)

from .projected_target import projected_targets
from .rng import preserve_rng
from .first_error import (
    CE_OBJECTIVES, FIRST_ERROR_OBJECTIVES, NEW_OBJECTIVES,
    first_error_mask, selection_statistics,
)
from .ranking import (
    OSD_OBJECTIVES as ORIGINAL_OSD_OBJECTIVES, RANKING_OBJECTIVES, allocation_coefficients,
    fixed_anchors, matching_prefix_mask, reference_margin_penalty, rival_softplus,
)


OSD_OBJECTIVES = ORIGINAL_OSD_OBJECTIVES | NEW_OBJECTIVES


def _model_options(model):
    return {
        'target_lm_head': model.lm_head,
        'target_embed_tokens': model.embed_tokens,
        'mask_token_id': model.mask_token_id,
        'block_size': model.block_size,
        'attention_backend': model.attention_backend,
        'num_anchors': model.num_anchors,
        'loss_decay_gamma': model.loss_decay_gamma,
        'objective_chunk_blocks': model.objective_chunk_blocks,
        'teacher_metrics': False,
    }


def _synchronize(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


class _ReferenceBlocks(OnlineDFlashModel):
    """Use the current forward's anchors, without another random draw."""

    def _sample_anchor_positions(self, seq_len, loss_mask, device, max_valid_anchors=None):
        return self.fixed_anchors


class FrozenReference:
    """Event-local reference, deliberately outside the registered module tree."""

    def __init__(self, current):
        with preserve_rng():
            draft = copy.deepcopy(current.draft_model).eval().requires_grad_(False)
            self.model = _ReferenceBlocks(draft_model=draft, **_model_options(current))
            # The copied draft is in eval mode. Do not call recursive eval() on
            # the shared, frozen head/embedding and change the current's flags.
            self.model.training = False

    def hidden(self, input_ids, hidden_states, loss_mask, anchors, keep):
        self.model.fixed_anchors = (anchors, keep)
        with preserve_rng(), torch.no_grad():
            return self.model._forward_draft_blocks(input_ids, hidden_states, loss_mask)[2]


class TargetObjectiveModel(OnlineDFlashModel):
    """Shared block forward and loss masks; no optimizer or sampler changes."""

    def __init__(self, author_model, objective, *, loss_precision='float32',
                 ranking_lambda=1.0, eos_token_ids=()):
        if objective not in {'hard_ce', 'projected', *OSD_OBJECTIVES}:
            raise ValueError(f'Unknown target objective: {objective}')
        if loss_precision not in {'float32', 'bfloat16'} or (
                loss_precision != 'float32' and objective not in OSD_OBJECTIVES):
            raise ValueError('BF16 probability/KL arithmetic is an OSD option')
        if not math.isfinite(ranking_lambda) or ranking_lambda < 0:
            raise ValueError('ranking_lambda must be finite and nonnegative')
        if (author_model.loss_type != 'dflash' or author_model.lk_loss_type is not None
                or getattr(author_model.draft_model, 'candidate_selector', None) is not None):
            raise ValueError('Target objectives require the standard DFlash CE backbone')
        if any(parameter.requires_grad for module in (author_model.lm_head, author_model.embed_tokens)
               for parameter in module.parameters()):
            raise ValueError('Target embedding and vocabulary head must remain frozen')
        if author_model.objective_chunk_blocks <= 0:
            raise ValueError('Pin positive objective_chunk_blocks to bound vocabulary memory')
        super().__init__(draft_model=author_model.draft_model, **_model_options(author_model))
        self.objective = objective
        self.is_osd = objective in OSD_OBJECTIVES
        self.uses_teacher = self.is_osd and objective not in CE_OBJECTIVES
        self.ranking_lambda = float(ranking_lambda)
        self.ranking_enabled = objective in RANKING_OBJECTIVES and ranking_lambda != 0
        self.eos_token_ids = tuple(eos_token_ids)
        self.loss_dtype = getattr(torch, loss_precision)
        self.reference = None
        self.costs = {}

    def begin_event(self):
        """Reset costs and freeze the event reference only when the loss uses it."""
        self.costs = {
            'reference_setup_s': 0.0, 'reference_forward_s': 0.0,
            'reference_head_s': 0.0, 'projection_s': 0.0,
        }
        if self.is_osd:
            self.costs['teacher_head_s'] = 0.0
        if self.objective in NEW_OBJECTIVES:
            self.costs['selection_s'] = 0.0
        if self.objective == 'projected' or self.ranking_enabled:
            weight = next(self.draft_model.parameters())
            _synchronize(weight)
            started = time.perf_counter()
            self.reference = FrozenReference(self)
            _synchronize(weight)
            self.costs['reference_setup_s'] = time.perf_counter() - started

    def end_event(self):
        self.reference = None
        return dict(self.costs)

    def _chunk_loss(self, hidden, reference_hidden, labels, weights, teacher_hidden=None,
                    ranking_coefficients=None, reference_softplus=None, reference_correct=None):
        guard = getattr(self, "prefix_guard", None)
        if guard is not None:
            guard.work['current_objective_chunks'] += 1
        logits = self.lm_head(hidden).to(self.loss_dtype)
        if self.objective in CE_OBJECTIVES:
            # Canonical CE uses precisely the qualified probability precision.
            # The legacy hard_ce/F.cross_entropy path remains separate.
            token_loss = -F.log_softmax(logits, dim=-1).gather(
                -1, labels.unsqueeze(-1),
            ).squeeze(-1)
        elif self.is_osd:
            # T=1, pure teacher forward KL; the teacher is never differentiated.
            # Checkpoint recomputation also enters this timer, so it is included.
            with torch.no_grad():
                _synchronize(teacher_hidden)
                started = time.perf_counter()
                teacher_logits = self.lm_head(teacher_hidden).to(self.loss_dtype)
                _synchronize(teacher_logits)
                self.costs['teacher_head_s'] = self.costs.get('teacher_head_s', 0.0) + (
                    time.perf_counter() - started)
                teacher_probability = F.softmax(teacher_logits, dim=-1)
            token_loss = F.kl_div(
                F.log_softmax(logits, dim=-1), teacher_probability, reduction='none',
            ).sum(dim=-1)
        elif self.objective == 'projected':
            with torch.no_grad():
                _synchronize(reference_hidden)
                started = time.perf_counter()
                reference_logits = self.lm_head(reference_hidden)
                _synchronize(reference_logits)
                self.costs['reference_head_s'] += time.perf_counter() - started
                started = time.perf_counter()
                targets = projected_targets(reference_logits, labels)
                _synchronize(reference_logits)
                self.costs['projection_s'] += time.perf_counter() - started
            token_loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        else:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none',
            ).reshape_as(labels)
        if self.objective in NEW_OBJECTIVES:
            with torch.no_grad():
                _synchronize(logits)
                started = time.perf_counter()
                proposals = logits.detach().argmax(-1)
                selection = (
                    first_error_mask(proposals, labels, weights, self.eos_token_ids)
                    if self.objective in FIRST_ERROR_OBJECTIVES
                    else torch.ones_like(labels, dtype=torch.bool)
                )
                statistics = selection_statistics(
                    proposals, labels, weights, selection, self.eos_token_ids,
                )
                _synchronize(statistics)
                self.costs['selection_s'] = self.costs.get('selection_s', 0.0) + (
                    time.perf_counter() - started)
            selected_weights = weights * selection
            eligible = selected_weights > 0
            correct = ((proposals == labels) & eligible).sum().float()
            terms = (
                (token_loss * selected_weights).sum(), selected_weights.sum(),
                correct, eligible.sum().float(), statistics,
            )
            guard = getattr(self, "prefix_guard", None)
            if guard is not None:
                return (*terms, *guard.chunk(logits, labels, weights, hidden.storage_offset()))
            return terms
        # Probability/KL rows may be BF16; position weights and reduction stay FP32.
        loss_sum = (token_loss * weights).sum()
        with torch.no_grad():
            eligible = weights > 0
            current_correct = logits.argmax(-1) == labels
            correct = (current_correct * eligible).sum().float()
        if ranking_coefficients is not None:
            ranking = rival_softplus(logits, labels)
            if reference_softplus is not None:
                ranking = reference_margin_penalty(ranking, reference_softplus)
            ranking_sum = (weights * ranking_coefficients * ranking).sum()
            combined = loss_sum + self.ranking_lambda * ranking_sum
            terms = (combined, weights.sum(), correct, eligible.sum().float(),
                     loss_sum.detach(), ranking_sum.detach())
            if reference_softplus is not None:
                # Return additive diagnostics, rather than mutating counters:
                # checkpoint recomputation must not log these positions twice.
                with torch.no_grad():
                    protected = (ranking_coefficients > 0) & eligible
                    active = (protected & (ranking > 0)).sum().float()
                    reference_correct = reference_correct & eligible
                    regressed = (reference_correct & ~current_correct).sum().float()
                return (*terms, active, regressed, reference_correct.sum().float())
            return terms
        return loss_sum, weights.sum(), correct, eligible.sum().float()

    def _block_labels_and_weights(self, input_ids, loss_mask, anchors, keep):
        """The unchanged author positions, supervision mask and decay weights."""
        offsets = torch.arange(self.block_size, device=input_ids.device).view(1, 1, -1)
        indices = anchors.unsqueeze(-1) + offsets
        in_bounds = indices < input_ids.shape[1]
        safe_indices = indices.clamp(max=input_ids.shape[1] - 1)
        labels = input_ids.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe_indices)
        response_mask = loss_mask.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe_indices)
        weights = keep.unsqueeze(-1).float() * in_bounds.float() * (offsets > 0).float() * response_mask
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            weights = weights * torch.exp(-(offsets - 1).clamp(min=0).float() / self.loss_decay_gamma)
        return labels, weights, safe_indices

    def _reference_prefix(self, input_ids, hidden_states, loss_mask, anchors, keep, labels, weights):
        """Preserve the original U/P prefix-only interface."""
        return self._reference_feedback(
            input_ids, hidden_states, loss_mask, anchors, keep, labels, weights,
        )[0]

    @torch.no_grad()
    def _reference_feedback(self, input_ids, hidden_states, loss_mask, anchors, keep, labels, weights):
        """Finish reference work before building the current model's autograd graph.

        Attention still sees complete blocks. Only vocabulary projection is
        chunked. R also retains detached scalar penalties/correctness per position;
        no vocabulary logits or reference hidden activations survive this call.
        """
        with preserve_rng():
            _synchronize(hidden_states)
            started = time.perf_counter()
            hidden = self.reference.hidden(
                input_ids, hidden_states.detach(), loss_mask, anchors, keep,
            ).reshape(input_ids.shape[0], anchors.shape[1], self.block_size, -1)
            _synchronize(hidden)
            self.costs['reference_forward_s'] += time.perf_counter() - started
            started = time.perf_counter()
            prefixes = []
            softplus_values, correct_positions = [], []
            for start in range(0, anchors.shape[1], self.objective_chunk_blocks):
                stop = start + self.objective_chunk_blocks
                logits = self.lm_head(hidden[:, start:stop])
                proposals = logits.argmax(-1)
                prefixes.append(matching_prefix_mask(
                    proposals, labels[:, start:stop], weights[:, start:stop],
                    self.eos_token_ids,
                ))
                if self.objective == 'osd_reference_margin':
                    softplus_values.append(rival_softplus(logits, labels[:, start:stop]))
                    correct_positions.append((proposals == labels[:, start:stop]) &
                                             (weights[:, start:stop] > 0))
                del logits
            _synchronize(hidden)
            self.costs['reference_head_s'] += time.perf_counter() - started
            prefix = torch.cat(prefixes, dim=1)
            if self.objective == 'osd_reference_margin':
                return prefix, torch.cat(softplus_values, dim=1), torch.cat(correct_positions, dim=1)
            return prefix, None, None

    def _forward_ranking(self, input_ids, hidden_states, loss_mask,
                         target_last_hidden_states, max_valid_anchors):
        anchors, keep = self._sample_anchor_positions(
            input_ids.shape[1], loss_mask, input_ids.device,
            max_valid_anchors=max_valid_anchors,
        )
        labels, weights, safe_indices = self._block_labels_and_weights(
            input_ids, loss_mask, anchors, keep,
        )
        reference_softplus = reference_correct = None
        if self.objective == 'osd_reference_margin':
            prefix, reference_softplus, reference_correct = self._reference_feedback(
                input_ids, hidden_states, loss_mask, anchors, keep, labels, weights,
            )
        else:
            prefix = self._reference_prefix(
                input_ids, hidden_states, loss_mask, anchors, keep, labels, weights,
            )
        coefficients, rho = allocation_coefficients(prefix, weights, self.objective)
        with fixed_anchors(self, anchors, keep):
            _, _, hidden = self._forward_draft_blocks(
                input_ids, hidden_states, loss_mask, max_valid_anchors,
            )
        hidden = hidden.reshape(input_ids.shape[0], anchors.shape[1], self.block_size, -1)
        teacher_hidden = self._aligned_target_hidden(
            target_last_hidden_states.detach(), safe_indices,
        )
        extra_inputs = ((reference_softplus, reference_correct)
                        if reference_softplus is not None else ())
        terms = checkpointed_chunk_reduce(
            self._chunk_loss, hidden, None, labels, weights, teacher_hidden, coefficients,
            *extra_inputs,
            chunk_size=self.objective_chunk_blocks, dim=1, checkpoint=True,
        )
        loss_sum, denominator, correct, positions, kl_sum, ranking_sum = terms[:6]
        denominator = denominator + 1e-6
        loss = loss_sum / denominator
        metrics = {
            'loss_terms': (loss_sum, denominator.detach()),
            'accuracy_denom': positions.detach(),
            'ratio_metrics': {
                'objective_loss': (loss_sum.detach(), denominator.detach()),
                'forward_kl': (kl_sum, denominator.detach()),
                'ranking_loss': (ranking_sum, denominator.detach()),
            },
            'sum_metrics': {
                'supervised_position_exposure': positions.detach(),
                'reference_prefix_positions': prefix.sum().float(),
                'reference_prefix_weight': (weights * prefix).sum(),
                'ranking_coefficient_weight': (weights * coefficients).sum(),
                'ranking_rho': rho,
            },
        }
        if self.objective == 'osd_reference_margin':
            active, regressed, reference_correct_count = terms[6:]
            protected_count = prefix.sum().float()
            auxiliary_sum = self.ranking_lambda * ranking_sum
            metrics['ratio_metrics'].update({
                'auxiliary_loss': (auxiliary_sum, denominator.detach()),
                'protected_activation_rate': (active, protected_count),
                'reference_correct_to_incorrect_rate': (regressed, reference_correct_count),
            })
            metrics['sum_metrics'].update({
                'kl_loss_numerator': kl_sum,
                'auxiliary_loss_numerator': auxiliary_sum,
                'loss_denominator': denominator.detach(),
                'protected_active_positions': active,
                'protected_positions': protected_count,
                'reference_correct_positions': reference_correct_count,
                'reference_correct_current_incorrect_positions': regressed,
            })
        return loss, correct / positions.clamp_min(1), metrics

    def forward(self, input_ids, hidden_states, loss_mask, target_last_hidden_states=None,
                max_valid_anchors=None, selector_loss_alpha=None, collect_detailed_metrics=True):
        # Target labels were verified against canonical frozen-target replay at
        # the feature boundary. The head/reference use exactly these same blocks.
        if (self.objective == 'projected' or self.ranking_enabled) and self.reference is None:
            raise RuntimeError('Begin the event before using a reference-dependent objective')
        if self.uses_teacher and target_last_hidden_states is None:
            raise ValueError('OSD requires frozen target_last_hidden_states')
        if self.ranking_enabled:
            return self._forward_ranking(
                input_ids, hidden_states, loss_mask, target_last_hidden_states, max_valid_anchors,
            )
        anchors, keep, output_hidden = self._forward_draft_blocks(
            input_ids, hidden_states, loss_mask, max_valid_anchors,
        )
        reference_hidden = None
        if self.reference is not None:
            _synchronize(output_hidden)
            started = time.perf_counter()
            reference_hidden = self.reference.hidden(input_ids, hidden_states, loss_mask, anchors, keep)
            _synchronize(output_hidden)
            self.costs['reference_forward_s'] += time.perf_counter() - started

        # Same-position labels and mask follow the pinned author's standard CE
        # path: exclude anchor slot zero, out-of-range slots and prompt tokens.
        labels, weights, safe_indices = self._block_labels_and_weights(
            input_ids, loss_mask, anchors, keep,
        )

        block_shape = (input_ids.shape[0], anchors.shape[1], self.block_size, -1)
        output_hidden = output_hidden.reshape(block_shape)
        if reference_hidden is not None:
            reference_hidden = reference_hidden.reshape(block_shape)
        teacher_hidden = None
        if self.uses_teacher:
            # Target row i predicts token i+1; draft slots predict at their own
            # positions. Reuse the author alignment rather than shifting labels.
            teacher_hidden = self._aligned_target_hidden(
                target_last_hidden_states.detach(), safe_indices,
            )
        guard = getattr(self, "prefix_guard", None)
        if guard is not None:
            guard.begin_response(input_ids, hidden_states, loss_mask, anchors, keep, labels, weights)
        terms = checkpointed_chunk_reduce(
            self._chunk_loss, output_hidden, reference_hidden, labels, weights, teacher_hidden,
            chunk_size=self.objective_chunk_blocks, dim=1, checkpoint=True,
        )
        loss_sum, denominator, correct, positions = terms[:4]
        if self.is_osd:
            # Apply the author's epsilon once per response, not once per chunk.
            denominator = denominator + 1e-6
        loss = loss_sum / denominator.clamp_min(torch.finfo(torch.float32).tiny)
        accuracy = correct / positions.clamp_min(1)
        metrics = {
            'loss_terms': (loss_sum, denominator.detach()),
            'accuracy_denom': positions.detach(),
            'ratio_metrics': {'objective_loss': (loss_sum.detach(), denominator.detach())},
            'sum_metrics': {'supervised_position_exposure': positions.detach()},
        }
        if self.objective in NEW_OBJECTIVES:
            statistics = terms[4]
            names = (
                'available_position_occurrences', 'selected_position_occurrences',
                'available_position_weight', 'selected_position_weight',
                'no_error_blocks', 'empty_blocks',
                *(f'first_error_position_{offset}' for offset in range(1, self.block_size)),
            )
            metrics['sum_metrics'].update(dict(zip(names, statistics)))
            metrics['sum_metrics'].update({
                'main_loss_numerator': loss_sum.detach(),
                'loss_denominator': denominator.detach(),
            })
            loss_name = 'canonical_ce' if self.objective in CE_OBJECTIVES else 'forward_kl'
            metrics['ratio_metrics'].update({
                loss_name: (loss_sum.detach(), denominator.detach()),
                'selection_rate': (statistics[1], statistics[0]),
            })
        if guard is not None:
            preservation = guard.finish_response(terms[5], terms[6])
            metrics['ratio_metrics']['prefix_preservation_H'] = (
                preservation, preservation.new_ones(()),
            )
        return loss, accuracy, metrics


class OSDTrainStrategy(DFlashTrainStrategy):
    """Forward teacher features as loss inputs even when diagnostics are off.

    The upstream DFlash strategy treats these features as optional metrics.
    OSD needs them for every update; checkpointing and the trainer stay shared.
    """

    required_features = DFlashTrainStrategy.required_features | {'target_last_hidden_states'}

    def forward_loss(self, batch, ctx=None):
        self.validate_batch(batch)
        tensors = batch.tensors
        device = self._device()
        loss, accuracy, metrics = self.dflash_model(
            **{name: tensors[name].to(device, non_blocking=True)
               for name in ('input_ids', 'hidden_states', 'loss_mask', 'target_last_hidden_states')},
            max_valid_anchors=_cpu_max_valid_anchors(tensors['loss_mask']),
            collect_detailed_metrics=ctx.collect_detailed_metrics if ctx is not None else True,
        )
        return StepOutput(
            loss=loss,
            metrics={'accuracy': accuracy.detach(), 'accuracy_denom': metrics['accuracy_denom']},
            ratio_metrics=metrics['ratio_metrics'],
            loss_terms=metrics['loss_terms'],
            sum_metrics=metrics['sum_metrics'],
        )
