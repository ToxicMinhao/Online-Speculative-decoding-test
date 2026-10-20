"""Response-level agreement filtering over the qualified AdaFlash OSD recipe.

The historical trainer is loaded privately and remains unchanged. H/L use its
ordinary backward, clipping and AdamW step. G/N split the *same* sampled forward
into chronological partial losses; neither group is resampled or normalized by
its own size. Native parameters and Adam moments remain BF16. Raw autograd
group gradients are copied to FP32 for product-free sign tests in ``core``;
this does not claim to recover precision lost by native BF16 differentiation.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    '_agreement_qualified_per_turn_training', HERE.parent / 'dflash_per_turn/training.py')
qualified = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qualified)
_core_spec = importlib.util.spec_from_file_location('_agreement_filter_step_core', HERE / 'core.py')
core = importlib.util.module_from_spec(_core_spec)
_core_spec.loader.exec_module(core)
author = qualified.author
require, digest, audit_response = qualified.require, qualified.digest, qualified.audit_response
RECIPE = dict(qualified.RECIPE)
rng_state, restore_rng, state_digest = qualified.rng_state, qualified.restore_rng, qualified.state_digest
GROUP_PROTOCOL = dict(
    schema='chronological_response_groups_v1',
    eligible_positions='response_start+1 through response_end_exclusive-1',
    midpoint='floor(unique_eligible_count/2); odd extra position belongs to B',
    overlapping_occurrences='same absolute output position always has the same group',
    across_passes='groups fixed; qualified anchor sampler advances normally each pass',
    denominator='both partial losses use the qualified whole-response weighted denominator',
    raw_gradient_storage='native autograd gradients converted to FP32 without clipping',
    aggregate_storage='FP32 group sum cast to native parameter dtype before one global clip',
    gradient_sum_tolerance='abs(sum-reference) <= 8*eps_native*(abs(A)+abs(B)+abs(reference))+1e-8',
    loss_sum_rtol=2e-6, loss_sum_atol=1e-7,
    qualified_parameter_dtypes=['bfloat16', 'float32'],
    finite_check='all raw gradients, losses, aggregate and parameter updates',
    empty_group='correctness failure; no resampling or ordinary-update fallback',
)
EQUIVALENCE_TOLERANCES = dict(
    parameter=dict(native_ulp_multiplier=1, absolute=1e-7),
    exp_avg=dict(native_epsilon_multiplier=8, absolute=1e-8),
    exp_avg_sq=dict(native_epsilon_multiplier=16, absolute=1e-12),
    parameter_formula='abs(grouped-reference) <= max(outward native ULP(grouped), outward native ULP(reference))+1e-7',
    moment_formula='abs(grouped-reference) <= multiplier*eps_native*max(abs(grouped),abs(reference))+absolute',
    displacement_relative_error_max=0.10,
    displacement_rule='global norm(grouped_delta-reference_delta)/norm(reference_delta) <=0.10; zero/nonzero mismatch fails',
    optimizer_counters='exact', anchor_exposure='exact', sampler_rng='exact', stochastic_rng='exact',
    reduction_chunk_elements=1_048_576,
)

COMPILER_POLICY = dict(
    schema='agreement_reusable_backward_buffers_v1',
    setting='torch._functorch.config.donated_buffer',
    donated_buffer=False,
    scope='All H/L/G/N trainer workers, before model setup or compiled forward',
    reason='One shared forward supplies multiple retained-graph backwards; saved buffers must remain reusable.',
)


def compiler_runtime():
    """Check the effective compiler policy without silently repairing drift."""
    import torch
    import torch._functorch.config as config
    require(getattr(config, 'donated_buffer', None) is False,
            'Agreement training requires donated_buffer=False before compilation')
    return {**COMPILER_POLICY, 'torch_version': str(torch.__version__),
            'effective_donated_buffer': config.donated_buffer}


def configure_compiler_runtime():
    """Disable single-use saved-buffer donation, equally for every learning arm.

    Native workers call this before the inherited model/runtime constructor;
    direct trainers call it before any training forward. This is a compiler
    buffer-lifetime compatibility setting, with no loss or precision change.
    """
    import torch._functorch.config as config
    require(type(getattr(config, 'donated_buffer', None)) is bool,
            'Installed Torch lacks the required donated_buffer compiler setting')
    config.donated_buffer = False
    return compiler_runtime()


def response_groups(audit):
    """All potentially supervised positions, independent of either pass's draw."""
    start, end = audit['response_start'], audit['response_end_exclusive']
    positions = list(range(start + 1, end))
    split = len(positions) // 2
    require(split > 0 and split < len(positions), 'Response has an empty chronological supervision group')
    result = dict(a=positions[:split], b=positions[split:], response_start=start,
                  response_end_exclusive=end, midpoint=split, protocol=dict(GROUP_PROTOCOL))
    result['identity_sha256'] = digest(result)
    return result


def _group_rows(labels, binary, groups):
    import torch
    expected = list(range(groups['response_start'] + 1, groups['response_end_exclusive']))
    require(groups['a'] + groups['b'] == expected and groups['a'] and groups['b'] and
            groups['midpoint'] == len(expected) // 2 and len(groups['a']) == groups['midpoint'],
            'Fixed chronological position groups changed')
    require(groups['protocol'] == GROUP_PROTOCOL and
            groups['identity_sha256'] == digest({k: v for k, v in groups.items() if k != 'identity_sha256'}),
            'Position group protocol/hash changed')
    first_b = groups['b'][0]
    group_a = (labels < first_b) & binary.bool()
    group_b = (labels >= first_b) & binary.bool()
    require(bool(torch.all((~binary.bool()) | ((labels > groups['response_start']) &
                (labels < groups['response_end_exclusive'])))), 'Loss outside newest response')
    require(bool(group_a.any()) and bool(group_b.any()),
            'Sampled objective has an empty supervision group; no resampling permitted')
    return group_a, group_b


def grouped_response_loss(draft, embed_tokens, lm_head, input_ids, hidden_states, teacher_hidden_states,
                          loss_mask, *, groups, mask_token_id, generator=None, attention_backend='sdpa',
                          projection_chunk=64, checkpoint_projection=True):
    """Faithful author-response forward with two additional masked loss sums.

    Sampling, parallel-block attention, teacher row alignment, native head dtype,
    exponential weights, global denominator and head checkpointing are copied
    from the qualified ``author_response_loss``. One stochastic backbone forward
    feeds all three losses. Only the newest response supplies loss; earlier
    responses may condition it without receiving repeated supervision.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    require(type(projection_chunk) is int and projection_chunk > 0, 'Invalid projection chunk')
    require(input_ids.ndim == 2 and input_ids.shape == loss_mask.shape and input_ids.shape[0] == 1,
            'Training uses one complete response per microbatch')
    require(hidden_states.shape[:2] == input_ids.shape and teacher_hidden_states.shape[:2] == input_ids.shape,
            'Teacher feature sequence differs from the complete sequence')
    require(not hidden_states.requires_grad and not teacher_hidden_states.requires_grad, 'Teacher must be frozen')
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    anchors, keep = author.sample_anchor_positions(seq_len, loss_mask, generator=generator)
    n = anchors.shape[1]
    offsets = torch.arange(author.BLOCK_SIZE, device=device).view(1, 1, -1)
    labels = anchors.unsqueeze(-1) + offsets
    safe_labels = labels.clamp(max=seq_len - 1)
    noise_ids = torch.full((bsz, n, author.BLOCK_SIZE), mask_token_id, dtype=torch.long, device=device)
    noise_ids[:, :, 0] = torch.where(keep, input_ids.gather(1, anchors.clamp(0, seq_len - 1)), mask_token_id)
    with torch.no_grad():
        noise_embedding = embed_tokens(noise_ids.reshape(bsz, -1))
    positions = torch.cat((torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1),
                           labels.reshape(bsz, -1)), dim=1)
    attention = author.create_attention_mask(anchors, keep, seq_len, backend=attention_backend)
    output = draft(position_ids=positions, noise_embedding=noise_embedding,
                   target_hidden=hidden_states, attention_mask=attention)
    if isinstance(output, tuple):
        require(len(output) == 2 and output[1] is None, 'Adaptive head must remain disabled')
        output = output[0]
    require(output.shape[:2] == (bsz, n * author.BLOCK_SIZE), 'Draft output shape differs from sampled blocks')
    binary = keep.unsqueeze(-1).expand(-1, -1, author.BLOCK_SIZE).float() * (labels < seq_len).float()
    binary = binary * (offsets > 0).float()
    binary = binary * torch.gather(loss_mask.unsqueeze(1).expand(-1, n, -1), 2, safe_labels)
    group_a, group_b = _group_rows(labels, binary, groups)
    decay = torch.exp(-(offsets - 1).clamp(min=0).float() / RECIPE['loss_decay_gamma'])
    weights = binary * decay
    denominator = weights.sum() + 1e-6
    selected = (weights.reshape(-1) > 0).nonzero(as_tuple=True)[0]
    flat_output = output.reshape(-1, output.shape[-1])
    teacher_rows = (safe_labels.reshape(-1) - 1).clamp(min=0)
    totals = [output.sum() * 0.0 for _ in range(3)]

    def chunk_loss(student_hidden, teacher_hidden, row_weights, in_a):
        student_logits = lm_head(student_hidden)
        with torch.no_grad():
            teacher_logits = lm_head(teacher_hidden.detach().to(student_logits.dtype))
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        rows = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction='none').sum(-1)
        weighted = rows * row_weights
        return weighted.sum(), (weighted * in_a).sum(), (weighted * (~in_a)).sum()

    for start in range(0, selected.numel(), projection_chunk):
        ix = selected[start:start + projection_chunk]
        args = (flat_output[ix], teacher_hidden_states[0, teacher_rows[ix]],
                weights.reshape(-1)[ix], group_a.reshape(-1)[ix])
        chunk = checkpoint(chunk_loss, *args, use_reentrant=False) if checkpoint_projection else chunk_loss(*args)
        totals = [total + value for total, value in zip(totals, chunk)]
    loss, loss_a, loss_b = (total / denominator for total in totals)
    require(all(bool(torch.isfinite(value)) for value in (loss, loss_a, loss_b)), 'Nonfinite forward KL')
    require(bool(torch.isclose(loss_a + loss_b, loss, rtol=GROUP_PROTOCOL['loss_sum_rtol'],
                               atol=GROUP_PROTOCOL['loss_sum_atol'])), 'Partial loss sum does not match full loss')
    positions_with_loss = labels[binary.bool()].detach().cpu().tolist()
    metadata = dict(anchor_positions=anchors[0].detach().cpu().tolist(),
                    block_keep_mask=keep[0].detach().cpu().tolist(),
                    label_positions=labels[0].detach().cpu().tolist(),
                    teacher_positions=(safe_labels[0] - 1).clamp(min=0).detach().cpu().tolist(),
                    binary_loss_mask=binary[0].detach().cpu().tolist(),
                    position_weights=weights[0].detach().cpu().tolist(),
                    sampled_blocks=n, eligible_blocks=int((binary.sum(-1) > 0).sum().item()),
                    eligible_positions=int(binary.sum().item()),
                    unique_supervised_positions=len(set(positions_with_loss)),
                    unique_supervised_position_ids=sorted(set(positions_with_loss)),
                    weighted_position_sum=float(weights.sum().item()),
                    normalization_denominator=float(denominator.item()),
                    excluded_anchor_positions=n, zero_supervision=False, backbone_forward_calls=1,
                    head_projection_chunks=(selected.numel() + projection_chunk - 1) // projection_chunk)
    # Preserve the qualified sampler/weight hash exactly; group metadata is a
    # separate object, so cross-arm anchor/exposure checks remain meaningful.
    metadata['sampling_and_weights_sha256'] = digest(metadata)
    metadata['grouping'] = dict(identity_sha256=groups['identity_sha256'],
        group_a_positions=list(groups['a']), group_b_positions=list(groups['b']),
        group_a_presentations=int(group_a.sum().item()), group_b_presentations=int(group_b.sum().item()),
        group_a_unique_positions=sorted(set(labels[group_a].detach().cpu().tolist())),
        group_b_unique_positions=sorted(set(labels[group_b].detach().cpu().tolist())),
        group_a_weight_sum=float(weights[group_a].sum()), group_b_weight_sum=float(weights[group_b].sum()),
        shared_denominator=float(denominator.item()))
    return loss, loss_a, loss_b, metadata


def check_gradient_sum(raw_a, raw_b, reference, parameters, *, context=None, report_path=None, observer=None):
    """Collect complete compact evidence before enforcing the unchanged gate."""
    from experiments.dflash_osd_agreement_filter.gradient_diagnostics import (
        summarize_gradient_sum, durable_write, enforce_gradient_sum)
    summary = summarize_gradient_sum(raw_a, raw_b, reference, parameters, context=context)
    if report_path is not None:
        durable_write(report_path, summary)
    if observer is not None:
        observer(summary, raw_a, raw_b, reference)
    enforce_gradient_sum(summary)
    return summary


def _compare_native_tensor(reference, current, kind):
    """Chunked qualification comparison avoids whole-parameter FP64 copies."""
    import torch
    require(reference.shape == current.shape and reference.dtype == current.dtype,
            'Disabled-filter qualification tensor shape/dtype differs')
    policy = EQUIVALENCE_TOLERANCES[kind]
    eps = torch.finfo(current.dtype).eps
    left, right = reference.reshape(-1), current.detach().reshape(-1)
    max_abs = max_scaled = 0.0
    chunk = EQUIVALENCE_TOLERANCES['reduction_chunk_elements']
    for start in range(0, left.numel(), chunk):
        # CPU snapshots are authoritative; only one chunk is moved at a time.
        native_ref = left[start:start+chunk].to(device=current.device)
        native_observed = right[start:start+chunk]
        ref, observed = native_ref.float(), native_observed.float()
        require(bool(torch.isfinite(ref).all()) and bool(torch.isfinite(observed).all()),
                'Nonfinite disabled-filter qualification tensor')
        delta = (ref-observed).abs()
        if kind == 'parameter':
            def outward_ulp(value):
                upper = torch.nextafter(value, torch.full_like(value, float('inf'))).float()
                lower = torch.nextafter(value, torch.full_like(value, -float('inf'))).float()
                return torch.maximum((upper-value.float()).abs(), (value.float()-lower).abs())
            tolerance = torch.maximum(outward_ulp(native_ref), outward_ulp(native_observed)) + policy['absolute']
        else:
            tolerance = policy['native_epsilon_multiplier'] * eps * torch.maximum(ref.abs(), observed.abs()) + policy['absolute']
        max_abs = max(max_abs, float(delta.max()))
        max_scaled = max(max_scaled, float((delta/tolerance).max()))
        require(bool(torch.all(delta <= tolerance)), f'Disabled filter differs from qualified OSD: {kind}')
    return dict(max_abs_error=max_abs, max_error_over_tolerance=max_scaled, elements=left.numel())


def _compare_displacements(initial_weights, reference_weights, current_parameters):
    """A weight tolerance alone must not admit a missing ordinary update."""
    import torch
    reference_sq = observed_sq = error_sq = dot = 0.0
    count = differing = reference_nonzero = observed_nonzero = lost_nonzero = introduced_nonzero = 0
    chunk = EQUIVALENCE_TOLERANCES['reduction_chunk_elements']
    for name, parameter in current_parameters.items():
        initial = initial_weights[name].reshape(-1)
        reference = reference_weights[name].reshape(-1)
        current = parameter.detach().reshape(-1)
        for start in range(0, parameter.numel(), chunk):
            base = initial[start:start+chunk].to(device=parameter.device, dtype=torch.float64)
            expected = reference[start:start+chunk].to(device=parameter.device, dtype=torch.float64) - base
            actual = current[start:start+chunk].double() - base
            error = actual - expected
            reference_sq += float(expected.dot(expected)); observed_sq += float(actual.dot(actual))
            error_sq += float(error.dot(error)); dot += float(expected.dot(actual))
            count += error.numel(); differing += int((error != 0).sum())
            reference_nonzero += int((expected != 0).sum()); observed_nonzero += int((actual != 0).sum())
            lost_nonzero += int(((expected != 0) & (actual == 0)).sum())
            introduced_nonzero += int(((expected == 0) & (actual != 0)).sum())
    reference_norm, observed_norm, error_norm = reference_sq**.5, observed_sq**.5, error_sq**.5
    require((reference_norm == 0) == (observed_norm == 0),
            'Disabled-filter reference/applied update has zero/nonzero mismatch')
    relative_error = error_norm/reference_norm if reference_norm else 0.0
    require(relative_error <= EQUIVALENCE_TOLERANCES['displacement_relative_error_max'],
            'Disabled-filter displacement exceeds fixed relative-error ceiling')
    return dict(reference_update_norm=reference_norm, grouped_update_norm=observed_norm,
                error_norm=error_norm, relative_error=relative_error,
                cosine=(dot/(reference_norm*observed_norm) if reference_norm else None),
                coordinates=count, differing_update_coordinates=differing,
                differing_update_fraction=differing/count,
                reference_nonzero_coordinates=reference_nonzero, grouped_nonzero_coordinates=observed_nonzero,
                reference_nonzero_became_zero=lost_nonzero, reference_zero_became_nonzero=introduced_nonzero,
                rule=EQUIVALENCE_TOLERANCES['displacement_rule'])


class AgreementTrainer(qualified.PersistentPerTurnTrainer):
    """Independent two-event arm; ordinary optimizer history is always retained."""
    def __init__(self, *args, arm='H', qualification=False, filter_enabled=True, **kwargs):
        require(arm in ('H', 'L', 'G', 'N'), 'Unknown learning arm')
        require(type(qualification) is bool and type(filter_enabled) is bool, 'Boolean qualification/filter required')
        require(filter_enabled or (qualification and arm == 'G'), 'Disabled filter is qualification-only G')
        self.compiler_runtime = configure_compiler_runtime()
        self.arm, self.qualification, self.filter_enabled = arm, qualification, filter_enabled
        self._initializing = True
        super().__init__(*args, **kwargs)
        self.learning_rate = 1e-4 if arm == 'L' else 3e-4
        self.optimizer.param_groups[0]['lr'] = self.learning_rate
        self._initializing = False
        self._check_owner()

    def _check_owner(self):
        import torch
        require(compiler_runtime() == self.compiler_runtime, 'Trainer compiler configuration changed')
        super()._check_owner()
        require(all(p.dtype in (torch.bfloat16, torch.float32) for p in self.parameters.values()) and
                len({p.dtype for p in self.parameters.values()}) == 1, 'Qualified parameter precision changed')
        if not self._initializing:
            require(self.learning_rate == (1e-4 if self.arm == 'L' else 3e-4), 'Arm learning rate changed')

    def _grouped_response(self, response, audit, groups):
        import torch
        self.synchronize()
        started = time.perf_counter()
        ids = torch.tensor([response['input_ids'] + response['output_ids']], dtype=torch.long, device=self.device)
        with torch.no_grad():
            features = self.teacher_features(ids)
        require(isinstance(features, (tuple, list)) and len(features) == 2, 'Teacher callback must return two tensors')
        hidden, teacher = features
        for feature in features:
            require(feature.ndim == 3 and feature.shape[:2] == ids.shape and feature.device == self.device and
                    not feature.requires_grad and bool(torch.isfinite(feature).all()), 'Invalid complete teacher features')
        del feature, features
        self.synchronize()
        teacher_s = time.perf_counter() - started
        started = time.perf_counter()
        pad = author.BLOCK_SIZE - 1
        ids = torch.cat((ids, ids.new_full((1, pad), self.mask_token_id)), dim=1)
        hidden = torch.cat((hidden, hidden.new_zeros((1, pad, hidden.shape[-1]))), dim=1)
        teacher = torch.cat((teacher, teacher.new_zeros((1, pad, teacher.shape[-1]))), dim=1)
        loss_mask = torch.zeros_like(ids, dtype=torch.float32)
        loss_mask[:, audit['response_start']:audit['response_end_exclusive']] = 1
        loss, loss_a, loss_b, sampling = grouped_response_loss(
            self.draft, self.embed_tokens, self.lm_head, ids, hidden, teacher, loss_mask, groups=groups,
            mask_token_id=self.mask_token_id, generator=self.generator, attention_backend=self.attention_backend,
            projection_chunk=self.projection_chunk)
        self.synchronize()
        forward_s = time.perf_counter() - started
        started = time.perf_counter()
        params = tuple(self.parameters.values())
        raw_a = dict(zip(self.parameters, (v.detach().float() for v in
            torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=False))))
        raw_b = dict(zip(self.parameters, (v.detach().float() for v in
            torch.autograd.grad(loss_b, params, retain_graph=self.qualification, allow_unused=False))))
        reference_check = None
        if self.qualification:
            reference = dict(zip(self.parameters, torch.autograd.grad(loss, params, allow_unused=False)))
            context = dict(getattr(self, '_gradient_check_context', {}))
            context.update(record_id=response['record_id'], response_audit=audit,
                           sampling=sampling, groups=groups,
                           losses=dict(full=float(loss.detach()), a=float(loss_a.detach()), b=float(loss_b.detach())),
                           active_checkpoint_sha256=self.active_checkpoint_sha256,
                           anchor_rng_after_sha256=state_digest(self.generator.get_state()),
                           global_rng_after_sha256=state_digest(rng_state()))
            directory = getattr(self, 'gradient_diagnostics_dir', None)
            report_path = (Path(directory) / ('event_%02d_pass_%02d_%s.json' %
                           (context['event'], context['pass_index'], context['trial_type']))) if directory else None
            reference_check = check_gradient_sum(raw_a, raw_b, reference, self.parameters,
                context=context, report_path=report_path,
                observer=getattr(self, 'gradient_diagnostics_observer', None))
            del reference
        require(all(bool(torch.isfinite(v).all()) for row in (raw_a, raw_b) for v in row.values()),
                'Nonfinite raw group gradient')
        self.synchronize()
        backward_s = time.perf_counter() - started
        metadata = dict(record_id=response['record_id'], turn=response['turn'], loss=float(loss.detach()),
            partial_loss_a=float(loss_a.detach()), partial_loss_b=float(loss_b.detach()),
            response_start=audit['response_start'], response_end_exclusive=audit['response_end_exclusive'],
            sequence_sha256=audit['sequence_sha256'], loss_mask_sha256=audit['loss_mask_sha256'], **sampling,
            gradient_sum_qualification=reference_check,
            gradient_backward_calls=3 if self.qualification else 2,
            raw_gradient_storage_dtype='torch.float32 promoted from native autograd dtype',
            phase_seconds=dict(teacher_preparation_s=teacher_s, forward_and_loss_s=forward_s, backward_s=backward_s))
        return metadata, raw_a, raw_b

    def train_event(self, event, responses):
        import torch
        require(type(event) is int and event in (1, 2), 'Bounded pilot permits exactly two update events')
        require(not self.failed and not self.awaiting_checkpoint, 'Trainer failed or checkpoint is not bound')
        if self.arm in ('H', 'L'):
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            result = super().train_event(event, responses)
            result.update(agreement_arm=self.arm, qualification=self.qualification,
                          filter_enabled=False, group_protocol=None,
                          compiler_runtime=dict(self.compiler_runtime),
                          peak_memory_allocated_bytes=(torch.cuda.max_memory_allocated(self.device)
                                                       if self.device.type == 'cuda' else None),
                          peak_memory_reserved_bytes=(torch.cuda.max_memory_reserved(self.device)
                                                      if self.device.type == 'cuda' else None))
            return result
        self._check_owner()
        require(event == self.completed_events + 1, 'Updates must execute once in chronological order')
        qualified.event_schedule(event, responses)
        response = responses[0]
        require(response.get('episode_id') == self.episode_id and response.get('target_id') == self.target_id and
                response.get('target_identity_sha256') == self.target_identity_sha256, 'Response/target ownership differs')
        require(response.get('capture_checkpoint_sha256') == self.active_checkpoint_sha256,
                'Train only this arm\'s currently served response')
        require(response.get('record_id') not in self.seen_record_ids, 'Old response cannot be trained again')
        started = time.perf_counter()
        audit = audit_response(response)
        groups = response_groups(audit)
        audit_s = time.perf_counter() - started
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        batches = []
        initial_steps = self.optimizer_steps
        training_started = time.perf_counter()
        try:
            self.draft.train()
            for pass_index in range(2):
                self.optimizer.zero_grad(set_to_none=True)
                batch_started = time.perf_counter()
                self._gradient_check_context = dict(event=event, pass_index=pass_index,
                    trial_type='filtered' if self.filter_enabled else 'disabled_filter',
                    optimizer_steps_before=self.optimizer_steps,
                    anchor_rng_before_sha256=state_digest(self.generator.get_state()),
                    global_rng_before_sha256=state_digest(rng_state()))
                if getattr(self, 'gradient_diagnostics_dir', None) is not None:
                    self._gradient_check_context['state_identity_sha256'] = state_digest(self.snapshot())
                value, raw_a, raw_b = self._grouped_response(response, audit, groups)
                self._check_owner()
                optimizer_started = time.perf_counter()
                stats = core.apply_adamw_step(self.parameters, self.optimizer, raw_a, raw_b,
                    mode=self.arm if self.filter_enabled else 'unfiltered', clip_norm=1.0)
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_steps += 1
                del raw_a, raw_b
                self._check_owner()
                self.synchronize()
                self.response_presentations += 1
                self.eligible_position_presentations += value['eligible_positions']
                self.eligible_block_presentations += value['eligible_blocks']
                batches.append(dict(pass_index=pass_index, optimizer_steps_after=self.optimizer_steps,
                    presentations=[value], loss=value['loss'], agreement=stats,
                    optimizer_s=time.perf_counter()-optimizer_started, wall_s=time.perf_counter()-batch_started))
        except BaseException:
            self.failed = True
            raise
        self.completed_events = event
        self.seen_record_ids.add(response['record_id'])
        self.awaiting_checkpoint = True
        training_s = time.perf_counter() - training_started
        presentations = [b['presentations'][0] for b in batches]
        teacher_s = sum(v['phase_seconds']['teacher_preparation_s'] for v in presentations)
        return dict(event=event, episode_id=self.episode_id,
            arm_name=author.ARM_NAME + ' with agreement filter' if self.arm == 'G' else
                     author.ARM_NAME + ' with own-state uniform step scaling',
            agreement_arm=self.arm, author_commit=author.AUTHOR_COMMIT, qualification=self.qualification,
            compiler_runtime=dict(self.compiler_runtime),
            filter_enabled=self.filter_enabled, recipe={**RECIPE, 'learning_rate': self.learning_rate},
            group_protocol=dict(GROUP_PROTOCOL), fixed_groups=groups, response_audits=[audit],
            capture_checkpoint_sha256=self.active_checkpoint_sha256, support_turns=[event],
            optimizer_steps_this_event=self.optimizer_steps-initial_steps, optimizer_steps_total=self.optimizer_steps,
            response_presentations_this_event=2, response_presentations_total=self.response_presentations,
            eligible_position_presentations_this_event=sum(v['eligible_positions'] for v in presentations),
            eligible_block_presentations_this_event=sum(v['eligible_blocks'] for v in presentations),
            response_order=[response['record_id']]*2, batches=batches,
            anchor_exposure_sha256=digest([v['sampling_and_weights_sha256'] for v in presentations]),
            peak_memory_allocated_bytes=(torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None),
            peak_memory_reserved_bytes=(torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None),
            costs=dict(mask_audit_s=audit_s, training_s=training_s, teacher_preparation_s=teacher_s,
                       draft_training_and_optimizer_s=training_s-teacher_s,
                       training_event_wall_s=time.perf_counter()-started,
                       accounting='training_s includes teacher preparation and group/optimizer work; do not add twice'))

    def snapshot(self, *, include_draft=True):
        state = super().snapshot(include_draft=include_draft)
        state['schema_version'] = 'dflash_agreement_filter_state_v1'
        state['agreement'] = dict(arm=self.arm, qualification=self.qualification, filter_enabled=self.filter_enabled,
                                  group_protocol=copy.deepcopy(GROUP_PROTOCOL), numerics=copy.deepcopy(core.NUMERICS),
                                  equivalence_tolerances=copy.deepcopy(EQUIVALENCE_TOLERANCES),
                                  compiler_runtime=dict(self.compiler_runtime))
        return state

    def restore(self, state, *, draft_state=None):
        require(state.get('schema_version') == 'dflash_agreement_filter_state_v1', 'Foreign recovery schema')
        require(state.get('agreement') == dict(arm=self.arm, qualification=self.qualification,
                filter_enabled=self.filter_enabled, group_protocol=GROUP_PROTOCOL, numerics=core.NUMERICS,
                equivalence_tolerances=EQUIVALENCE_TOLERANCES,
                compiler_runtime=self.compiler_runtime), 'Recovery arm/filter protocol differs')
        require(state.get('learning_rate') == (1e-4 if self.arm == 'L' else 3e-4), 'Recovery learning rate differs')
        restored = dict(state)
        restored['schema_version'] = 'dflash_per_turn_state_v1'
        try:
            super().restore(restored, draft_state=draft_state)
        except BaseException:
            self.failed = True
            raise

    def qualify_disabled_filter(self, responses):
        """Two isolated two-step trials, then restore the exact pre-trial state.

        Only qualification G may call this. The ordinary reference is the
        unchanged parent implementation; the second trial disables G's mask.
        This costs four temporary optimizer steps and extra CPU state copies,
        all explicitly separated from the main pilot's 96 steps. No generated
        answers are changed and no qualification learning survives restoration.
        """
        import torch
        require(self.qualification and self.arm == 'G' and self.filter_enabled,
                'Disabled-filter equivalence is available only for qualification G')
        require(not self.failed and not self.awaiting_checkpoint, 'Qualification requires complete bound state')
        event = self.completed_events + 1
        require(event in (1, 2), 'Qualification must precede a bounded update')
        started = time.perf_counter()
        initial = self.snapshot()
        identity = state_digest(initial)
        initial_steps = self.optimizer_steps
        success = False
        result = None
        try:
            reference = qualified.PersistentPerTurnTrainer.train_event(self, event, responses)
            reference_weights = {name: value.detach().cpu().clone() for name, value in self.parameters.items()}
            reference_optimizer = {}
            for name, parameter in self.parameters.items():
                reference_optimizer[name] = {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor)
                                             else copy.deepcopy(value)
                                             for key, value in self.optimizer.state[parameter].items()}
            reference_anchor_rng = self.generator.get_state().cpu().clone()
            reference_global_rng = rng_state()
            self.restore(initial)
            self.filter_enabled = False
            observed = self.train_event(event, responses)
            require(reference['anchor_exposure_sha256'] == observed['anchor_exposure_sha256'],
                    'Disabled-filter trial changed anchor/exposure sequence')
            require(torch.equal(reference_anchor_rng, self.generator.get_state().cpu()) and
                    state_digest(reference_global_rng) == state_digest(rng_state()),
                    'Disabled-filter trial changed sampler or stochastic RNG consumption')
            comparisons = {}
            for name, parameter in self.parameters.items():
                item = dict(parameter=_compare_native_tensor(reference_weights[name], parameter, 'parameter'))
                current = self.optimizer.state[parameter]
                expected = reference_optimizer[name]
                require(set(current) == set(expected), 'Disabled-filter Adam state inventory differs')
                for key, value in expected.items():
                    if key in ('exp_avg', 'exp_avg_sq'):
                        item[key] = _compare_native_tensor(value, current[key], key)
                    elif isinstance(value, torch.Tensor):
                        require(torch.equal(value, current[key].detach().cpu()),
                                'Disabled-filter Adam counter or other tensor differs')
                    else:
                        require(value == current[key], 'Disabled-filter Adam metadata differs')
                comparisons[name] = item
            displacement_comparison = _compare_displacements(initial['draft'], reference_weights, self.parameters)
            result = dict(status='passed', event=event, tolerances=copy.deepcopy(EQUIVALENCE_TOLERANCES),
                          compiler_runtime=dict(self.compiler_runtime),
                          raw_gradient_sum_rule=GROUP_PROTOCOL['gradient_sum_tolerance'],
                          reference_anchor_exposure_sha256=reference['anchor_exposure_sha256'],
                          grouped_anchor_exposure_sha256=observed['anchor_exposure_sha256'],
                          reference_accounting=reference, grouped_accounting=observed,
                          comparisons=comparisons, displacement_comparison=displacement_comparison,
                          temporary_optimizer_steps=4,
                          persistent_optimizer_steps_before=initial_steps,
                          initial_state_identity_sha256=identity)
            success = True
        finally:
            self.filter_enabled = True
            self.restore(initial)
            # A rejected qualification must not silently make the trainer usable.
            if not success:
                self.failed = True
        if state_digest(self.snapshot()) != identity:
            self.failed = True
            raise ValueError('Qualification failed to restore exact initial state')
        result.update(restored_state_identity_sha256=identity,
                      persistent_optimizer_steps_after=self.optimizer_steps,
                      qualification_wall_s=time.perf_counter()-started,
                      interpretation='Finite-precision recipe equivalence; not exact BF16 bit equality or performance evidence')
        return result
