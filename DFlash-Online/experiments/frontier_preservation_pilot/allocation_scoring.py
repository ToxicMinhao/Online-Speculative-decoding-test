"""Native two-response token contractions for allocation qualification.

No optimizer step occurs here. The reference preserves the authored BF16 FKL,
K+ correction, PAIR term, eight-block head shape and response reduction order.
Only the optimizer shadow and its analytic pullback use continuous FP32 algebra.
Fast tangents use the already qualified attention implementation and explicitly
represent the FP32 direction in native parameter precision before differentiation.
"""

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import _StopRecomputationError

from specforge.core.chunking import checkpointed_chunk_reduce
from online_feedback.kplus import first_error_groups, redistribute_response
from online_feedback.prefix_allocation import PrefixAllocation, allocation_summaries
from online_feedback.repair_conditional_kl import (
    repair_pair_reduce_response,
    validate_repair_conditional_kl,
)
from online_feedback.rng import preserve_rng, restore_rng
from .attention_jvp import native_attention_jvp
from .math import (
    adamw_shadow,
    allocate_prefix,
    coefficient_sensitivities,
    current_roles,
    optimizer_vjp,
    require,
)
from .native import NativeHidden, trainable_parameters


class _AssignedPrefix(PrefixAllocation):
    """An immutable chunk assignment; checkpoint replay never advances a cursor."""

    def __init__(self, coefficients, proposals, prefix):
        super().__init__()
        self.coefficients = coefficients.detach()
        self.proposals = proposals
        self.prefix = prefix

    def chunk_terms(
        self,
        log_probability,
        token_loss,
        proposals,
        labels,
        weights,
        selection,
        eos_token_ids=(),
    ):
        roles = current_roles(proposals, labels, weights, eos_token_ids)
        require(torch.equal(proposals, self.proposals), 'Current native proposals changed')
        require(torch.equal(roles['prefix'], self.prefix), 'Frozen prefix support changed')
        base = weights.detach().float()
        summaries = allocation_summaries(base, self.coefficients, roles['valid'], self.prefix)
        return ((self.coefficients - base) * token_loss).sum(), *summaries


class NativeAllocationScorer:
    """A fixed window, with at most one backbone graph live at a time.

    All public operations preserve ambient RNG, native extension state and cost
    dictionaries. They never populate live parameter .grad buffers. Metadata is
    small per-position state; neither logits nor per-token gradients are cached.
    """

    def __init__(self, trainer, responses, work=None):
        require(len(responses) == 2, 'Require exactly two frozen native responses')
        require(not trainer.core.accumulation_remainder, 'Require an accumulation boundary')
        self.trainer = trainer
        self.model = trainer.bundle.model
        self.responses = tuple(responses)
        self.work = work
        self.hidden = NativeHidden(self.model)
        named = trainable_parameters(self.hidden)
        self.parameter_names = tuple(named)
        self.parameters = tuple(named.values())
        require(
            all(p.dtype == torch.bfloat16 for p in self.parameters),
            'Require native BF16 parameters',
        )
        require(self.model.loss_dtype == torch.bfloat16, 'Require native BF16 FKL')
        require(
            self.model.block_size == 16 and self.model.objective_chunk_blocks == 8,
            'Require the authored 16-slot/eight-block head layout',
        )
        require(
            self.model.objective == 'osd_first_error'
            and self.model.first_error_redistribution is not None
            and self.model.prefix_allocation is not None
            and self.model.repair_conditional_kl,
            'Require the native K+PAIR teaching path',
        )
        require(
            not any(p.requires_grad for p in self.model.lm_head.parameters()),
            'Teacher head must be frozen',
        )
        validate_repair_conditional_kl(self.model)
        self.snapshots = None
        self.current_direct_loss_masks = None
        self.repair_coefficients = None
        self._clean()

    def _clean(self):
        require(
            all(parameter.grad is None for parameter in self.parameters),
            'Scoring requires clean native gradient buffers',
        )

    def close(self):
        """There is no persistent hook or graph to release; verify clean buffers."""
        self._clean()

    @contextmanager
    def _isolated(self):
        self._clean()
        costs = self.model.costs
        prefix = self.model.prefix_allocation
        self.model.costs = {}
        try:
            with preserve_rng():
                yield
        finally:
            self.model.costs = costs
            self.model.prefix_allocation = prefix
            self._clean()

    @contextmanager
    def _operation(self, name):
        if self.work is not None:
            self.work.start(name)
        try:
            yield
        except _StopRecomputationError:
            # The native non-reentrant checkpoint intentionally stops after its
            # required saved tensors have been rebuilt. Both vocabulary heads
            # have already executed; this is successful head work, not failure.
            if name == 'allocation_native_teaching_head_chunk' and self.work is not None:
                self.work.finish(name)
            raise
        else:
            if self.work is not None:
                self.work.finish(name)

    def _ready(self):
        require(self.snapshots is not None, 'Freeze the current window before scoring')

    def _blocks(self, response, parameters=None):
        restore_rng(response.rng_after_sampling)
        with self._operation('allocation_native_backbone_forward'):
            hidden = (
                self.hidden(response)
                if parameters is None
                else torch.func.functional_call(self.hidden, parameters, (response,))
            )
        return hidden.reshape(1, response.anchors.shape[1], self.model.block_size, -1)

    def _slices(self, tensor):
        for start in range(0, tensor.shape[1], self.model.objective_chunk_blocks):
            yield start, slice(start, start + self.model.objective_chunk_blocks)

    @torch.no_grad()
    def freeze_current(self):
        require(self.snapshots is None, 'The current window is already frozen')
        snapshots = []
        with self._isolated():
            for response in self.responses:
                ids = response.tensors['input_ids']
                labels, beta, indices = self.model._block_labels_and_weights(
                    ids, response.tensors['loss_mask'], response.anchors, response.keep
                )
                hidden = self._blocks(response)
                proposals = []
                for _, part in self._slices(hidden):
                    with self._operation('allocation_freeze_native_head_chunk'):
                        logits = self.model.lm_head(hidden[:, part]).to(self.model.loss_dtype)
                        require(torch.isfinite(logits).all(), 'Nonfinite current native logits')
                        proposals.append(logits.argmax(-1))
                        del logits
                proposals = torch.cat(proposals, dim=1)
                roles = current_roles(proposals, labels, beta, self.model.eos_token_ids)
                selection = roles['prefix'] | roles['repair']
                priority1, priority2 = first_error_groups(
                    proposals, labels, beta, selection, self.model.eos_token_ids
                )
                # Match the authored chunk-sum order independently of the loss.
                totals = None
                for _, part in self._slices(beta):
                    values = (
                        (beta[:, part] * selection[:, part]).sum(),
                        (beta[:, part] * priority1[:, part]).sum(),
                        (beta[:, part] * priority2[:, part]).sum(),
                    )
                    totals = (
                        values
                        if totals is None
                        else tuple(a + b for a, b in zip(totals, values))
                    )
                require(totals is not None, 'Native response has no anchor blocks')
                selected, weight1, weight2 = totals
                roles['denominator'] = (selected + 1e-6).detach()
                zero = selected.new_zeros(())
                _, metrics = redistribute_response(
                    zero,
                    selected,
                    (
                        weight1,
                        weight2,
                        zero,
                        zero,
                        priority1.sum().float(),
                        priority2.sum().float(),
                    ),
                )
                repair = beta * (
                    priority1 * metrics['kplus_coefficient1']
                    + priority2 * metrics['kplus_coefficient2']
                )
                teacher = self.model._aligned_target_hidden(
                    response.tensors['target_last_hidden_states'].detach(), indices
                )
                snapshots.append(
                    dict(
                        roles=roles,
                        proposals=proposals.detach(),
                        repair_coefficients=repair.detach(),
                        direct_loss_mask=selection.detach(),
                        teacher_hidden=teacher.detach(),
                        priority1=priority1,
                        priority2=priority2,
                        kplus_coefficient1=metrics['kplus_coefficient1'],
                        kplus_coefficient2=metrics['kplus_coefficient2'],
                    )
                )
                del hidden
        self.snapshots = snapshots
        self.current_direct_loss_masks = [item['direct_loss_mask'] for item in snapshots]
        self.repair_coefficients = [item['repair_coefficients'] for item in snapshots]
        return snapshots

    def _teaching_loss(self, response_index, coefficients):
        snapshot = self.snapshots[response_index]
        roles = snapshot['roles']
        require(coefficients.shape == roles['beta'].shape, 'Prefix coefficient shape changed')
        require(
            torch.equal(coefficients[~roles['prefix']], roles['beta'][~roles['prefix']]),
            'Teaching changed non-prefix coefficients',
        )
        hidden = self._blocks(self.responses[response_index])

        def chunk(hidden, labels, beta, teacher, assigned, proposals, prefix):
            original = self.model.prefix_allocation
            self.model.prefix_allocation = _AssignedPrefix(assigned, proposals, prefix)
            try:
                with self._operation('allocation_native_teaching_head_chunk'):
                    return self.model._chunk_loss(hidden, None, labels, beta, teacher)
            finally:
                self.model.prefix_allocation = original

        terms = checkpointed_chunk_reduce(
            chunk,
            hidden,
            roles['labels'],
            roles['beta'],
            snapshot['teacher_hidden'],
            coefficients.detach(),
            snapshot['proposals'],
            roles['prefix'],
            chunk_size=8,
            dim=1,
            checkpoint=True,
        )
        numerator, metrics = self.model.first_error_redistribution.reduce_response(terms[:11])
        require(
            torch.equal(metrics['kplus_coefficient1'], snapshot['kplus_coefficient1'])
            and torch.equal(metrics['kplus_coefficient2'], snapshot['kplus_coefficient2']),
            'Frozen native repair coefficients changed',
        )
        numerator, _ = PrefixAllocation().reduce_response(numerator, terms[11:15])
        pair, pair_metrics = repair_pair_reduce_response(terms[15:], metrics)
        if bool(pair_metrics['repair_pair_repair_rows'] > 0):
            numerator = numerator + pair
        denominator = terms[1] + 1e-6
        require(
            torch.equal(denominator, roles['denominator']), 'Frozen native denominator changed'
        )
        return numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)

    def reference_gradient(self, prefix_coefficients=None):
        """Two native backwards, each loss divided by two before differentiation."""
        self._ready()
        coefficients = (
            [s['roles']['beta'] for s in self.snapshots]
            if prefix_coefficients is None
            else prefix_coefficients
        )
        require(len(coefficients) == 2, 'Require two coefficient arrays')
        accumulated = None
        with self._isolated():
            for index, assigned in enumerate(coefficients):
                loss = self._teaching_loss(index, assigned) / 2
                with self._operation('allocation_native_reference_backward'):
                    gradients = torch.autograd.grad(
                        loss, self.parameters, create_graph=False, allow_unused=False
                    )
                require(
                    all(torch.isfinite(g).all() for g in gradients), 'Nonfinite native gradient'
                )
                if accumulated is None:
                    accumulated = gradients
                else:
                    # AccumulateGrad adds response two into native BF16 buffers.
                    for total, contribution in zip(accumulated, gradients):
                        total.add_(contribution)
                del loss, gradients
        require(
            all(torch.isfinite(g).all() for g in accumulated),
            'Nonfinite accumulated native gradient',
        )
        return accumulated

    def _assessment(self, hidden, roles):
        def chunk(hidden, labels, mask, weights):
            with self._operation('allocation_preview_native_head_chunk'):
                logits = self.model.lm_head(hidden).to(self.model.loss_dtype)
            selected = logits[mask]
            require(torch.isfinite(selected).all(), 'Nonfinite preview assessment logits')
            selected_labels = labels[mask]
            wrong = selected.detach().clone()
            wrong.scatter_(-1, selected_labels.unsqueeze(-1), -torch.inf)
            rivals = wrong.argmax(-1)
            canonical = selected.gather(-1, selected_labels.unsqueeze(-1)).squeeze(-1).float()
            competitor = selected.gather(-1, rivals.unsqueeze(-1)).squeeze(-1).float()
            margins = canonical - competitor
            numerator = (weights[mask] * F.softplus(-margins) / math.log(2.0)).sum()
            return (numerator,)

        (numerator,) = checkpointed_chunk_reduce(
            chunk,
            hidden,
            roles['labels'],
            roles['assessment_mask'],
            roles['assessment_weight'],
            chunk_size=8,
            dim=1,
            checkpoint=True,
        )
        count = int(roles['valid'].any(-1).sum())
        return numerator / count if count else numerator

    def preview_and_direction(self, gradients):
        """Continuous FP32 warm-Adam preview, native BF16 assessment, analytic VJP."""
        self._ready()
        wrapper = self.trainer.backend.optimizer
        optimizer = wrapper.optimizer
        require(len(optimizer.param_groups) == 1, 'Require the native single AdamW group')
        require(
            tuple(map(id, optimizer.param_groups[0]['params']))
            == tuple(map(id, self.parameters)),
            'Native optimizer parameter ordering changed',
        )
        state = [optimizer.state[p] for p in self.parameters]
        require(
            all(all(k in item for k in ('exp_avg', 'exp_avg_sq', 'step')) for item in state),
            'Require complete warm Adam state',
        )
        with self._isolated(), self._operation('allocation_fp32_adam_shadow'):
            preview, context = adamw_shadow(
                self.parameters,
                gradients,
                [s['exp_avg'] for s in state],
                [s['exp_avg_sq'] for s in state],
                [s['step'] for s in state],
                optimizer.param_groups[0],
                max_grad_norm=wrapper.max_grad_norm,
            )
            assessment_values = []
            cotangents = None
            for index, response in enumerate(self.responses):
                represented = {
                    name: value.to(parameter.dtype)
                    for name, value, parameter in zip(
                        self.parameter_names, preview, self.parameters
                    )
                }
                hidden = self._blocks(response, represented)
                value = self._assessment(hidden, self.snapshots[index]['roles'])
                assessment_values.append(float(value.detach()))
                with self._operation('allocation_preview_native_backward'):
                    current = torch.autograd.grad(
                        value / 2, preview, create_graph=False, allow_unused=False
                    )
                if cotangents is None:
                    cotangents = current
                else:
                    for total, contribution in zip(cotangents, current):
                        total.add_(contribution)
                del represented, hidden, value, current
            direction = tuple(optimizer_vjp(context, cotangents))
            diagnostics = dict(
                response_assessments=assessment_values,
                window_assessment=sum(assessment_values) / 2,
                gradient_norm=float(context['norm']),
                clip=float(context['clip']),
                clipping_active=context['clipping_active'],
                shadow_dtype='torch.float32',
                preview_parameter_dtype='torch.bfloat16',
                assessment_dtype='torch.float32',
                direction_dtype='torch.float32',
                accumulation_factor=0.5,
                native_adam_steps=0,
            )
            del preview, context, cotangents
        return dict(
            w=direction, diagnostics=diagnostics, memory_inventory=self.memory_inventory()
        )

    def _token_losses(self, hidden, teacher):
        with self._operation('allocation_native_token_head_chunk'):
            logits = self.model.lm_head(hidden).to(self.model.loss_dtype)
            with torch.no_grad():
                teacher_logits = self.model.lm_head(teacher).to(self.model.loss_dtype)
                probability = F.softmax(teacher_logits, dim=-1)
            rows = F.kl_div(F.log_softmax(logits, dim=-1), probability, reduction='none').sum(
                -1
            )
            require(torch.isfinite(rows).all(), 'Nonfinite native FKL rows')
            return rows

    def token_fkl(self, response_index, block_index, position):
        """Fresh full backbone; one original complete head chunk; one unweighted row."""
        self._ready()
        snapshot = self.snapshots[response_index]
        require(
            bool(snapshot['roles']['prefix'][0, block_index, position]),
            'Serial oracle is restricted to the sealed current prefix',
        )
        with self._isolated():
            hidden = self._blocks(self.responses[response_index])
            start = (block_index // 8) * 8
            losses = self._token_losses(
                hidden[:, start : start + 8], snapshot['teacher_hidden'][:, start : start + 8]
            )
            return losses[0, block_index - start, position]

    @torch.no_grad()
    def fast_scores(self, direction):
        self._ready()
        require(len(direction) == len(self.parameters), 'Direction parameter count changed')
        require(
            all(
                w.dtype == torch.float32 and w.shape == p.shape and torch.isfinite(w).all()
                for w, p in zip(direction, self.parameters)
            ),
            'Invalid FP32 optimizer direction',
        )
        effective = tuple(w.detach().to(p.dtype) for w, p in zip(direction, self.parameters))
        require(
            all(torch.isfinite(w).all() for w in effective), 'Nonfinite represented direction'
        )
        scores, allocations, diagnostics = [], [], []
        with self._isolated(), native_attention_jvp() as attention_work:
            for index, response in enumerate(self.responses):
                with torch.autograd.forward_ad.dual_level():
                    duals = {
                        name: torch.autograd.forward_ad.make_dual(p, tangent)
                        for name, p, tangent in zip(
                            self.parameter_names, self.parameters, effective
                        )
                    }
                    hidden = self._blocks(response, duals)
                    derivatives = []
                    for _, part in self._slices(hidden):
                        losses = self._token_losses(
                            hidden[:, part], self.snapshots[index]['teacher_hidden'][:, part]
                        )
                        _, tangent = torch.autograd.forward_ad.unpack_dual(losses)
                        require(
                            tangent is not None and torch.isfinite(tangent).all(),
                            'Native FKL directional derivative is missing or nonfinite',
                        )
                        derivatives.append(tangent.detach())
                    rows = torch.cat(derivatives, dim=1)
                    roles = self.snapshots[index]['roles']
                    score = coefficient_sensitivities(rows, roles)
                    allocation, diagnostic = allocate_prefix(
                        score, roles['beta'], roles['prefix']
                    )
                    scores.append(score)
                    allocations.append(allocation)
                    diagnostics.append(diagnostic)
                    del hidden, duals, losses, rows, derivatives, tangent
        return dict(
            scores=scores,
            allocations=allocations,
            allocation_diagnostics=diagnostics,
            effective_direction=effective,
            direction_scale=1.0,
            direction_policy='FP32 w cast once per parameter to native BF16; no rescaling',
            direct_loss_masks=self.current_direct_loss_masks,
            repair_coefficients=self.repair_coefficients,
            accumulation_factor=0.5,
            response_denominators=[item['roles']['denominator'] for item in self.snapshots],
            attention_work=attention_work,
        )

    def memory_inventory(self):
        """Tensor sizes, not an assertion that the unmeasured composition fits VRAM."""
        native = sum(p.numel() * p.element_size() for p in self.parameters)
        fp32 = sum(p.numel() * 4 for p in self.parameters)
        return dict(
            trainable_parameters=sum(p.numel() for p in self.parameters),
            native_vector_bytes=native,
            fp32_vector_bytes=fp32,
            shadow_persistent_vectors=dict(
                fp32_gradient=fp32, fp32_diagonal=fp32, fp32_preview=fp32
            ),
            assessment_vectors=dict(native_functional_parameters=native, fp32_cotangent=fp32),
            fast_persistent_vectors=dict(fp32_direction=fp32, native_direction=native),
            reference_gradient_bytes=native,
            live_backbone_graphs_maximum=1,
            per_token_parameter_banks=0,
            full_fp32_moment_banks=0,
            note='Shadow/r/context released before fast scoring; model activations, head checkpoints, allocator scratch and compiler workspace must be measured separately.',
        )
