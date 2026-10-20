"""Fixed four-response learning curve using the qualified ordinary author loss.

Only the feedback schedule changes: independently normalized response losses are
averaged in pairs, with one ordinary clipped AdamW step per pair. Historical
modules are imported privately and their files remain untouched. Source-loss
checks use separate fixed anchors and restore training randomness exactly.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    '_baseline_curve_author_training', HERE.parent / 'dflash_two_update_followup/author_training.py')
author = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(author)
author.OUTPUT_CAP = 1024
# The divisor is the same qualified effective batch of two responses. The four
# passes are the entire trajectory, not four additional passes after step 2.
author.RECIPE = {k: v for k, v in author.RECIPE.items() if k != 'optimizer_steps_per_event'}
author.RECIPE.update(passes=4, responses_per_step=2, optimizer_steps_per_trajectory=8)
RECIPE = dict(author.RECIPE)
PROTOCOL = dict(schema='dflash_fixed_feedback_learning_curve_v1', feedback_responses=4,
    checkpoint_steps=[0, 2, 8], total_passes=4, total_steps=8,
    training_response_presentations=16, response_order='saved order, repeated unchanged each pass',
    loss_reduction='mean of two independently normalized response losses',
    source_anchor_rule='separate device generator seeded from seed 17 and immutable response identity',
    source_model_mode='eval, no_grad; global RNG and all module modes restored',
    source_diagnostic_checkpoints=[0, 2, 8], native_parameter_dtype='torch.bfloat16')
require, digest, audit_response = author.require, author.digest, author.audit_response
_rng_spec = importlib.util.spec_from_file_location(
    '_baseline_curve_rng_helpers', HERE.parent / 'dflash_per_turn/training.py')
_rng = importlib.util.module_from_spec(_rng_spec)
_rng_spec.loader.exec_module(_rng)
rng_state, restore_rng, state_digest = _rng.rng_state, _rng.restore_rng, _rng.state_digest


def _seed_global(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _clone_cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_cpu(v) for v in value)
    return copy.deepcopy(value)


def step_schedule(current_step, target_step, responses):
    require(type(current_step) is int and type(target_step) is int and
            (current_step, target_step) in ((0, 2), (2, 8)),
            'Keep the continuous checkpoint progression 0 -> 2 -> 8')
    require(len(responses) == 4 and len({r.get('record_id') for r in responses}) == 4,
            'Exactly four distinct frozen-feedback responses are required')
    return [(step // 2, step, responses[(step % 2)*2:(step % 2)*2+2])
            for step in range(current_step, target_step)]


class FixedFeedbackTrainer(author.PersistentAuthorOSDTrainer):
    """One group, one fixed feedback pool, one persistent ordinary optimizer.

    Call source_losses at released step 0, train_to_step(2), bind_checkpoint(1),
    source_losses, train_to_step(8), bind_checkpoint(2), then source_losses.
    Export/activation remains the native worker's responsibility. Optional
    artifact_dir writes fsynced per-step and source-diagnostic receipts, including
    partial progress on failure, without overwriting historical evidence.
    """
    def __init__(self, *args, artifact_dir=None, **kwargs):
        _seed_global(17)
        super().__init__(*args, **kwargs)
        self.support_bindings = None
        self.support_sha256 = None
        self.source_anchor_specs = None
        self.source_diagnostic_steps = []
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            require(not any((self.artifact_dir / name).exists() for name in
                    ('TRAIN_STEPS.jsonl', 'SOURCE_LOSSES.jsonl', 'FAILURES.jsonl')),
                    'Use a fresh trainer evidence directory')

    def _check_owner(self):
        import torch
        super()._check_owner()
        require(all(p.dtype in (torch.float32, torch.bfloat16) for p in self.parameters.values()) and
                len({p.dtype for p in self.parameters.values()}) == 1,
                'Homogeneous qualified native parameter precision required')
        require(self.device.type != 'cuda' or all(p.dtype == torch.bfloat16 for p in self.parameters.values()),
                'Native GPU training must retain BF16 parameters; no master-weight substitution')

    def _record(self, name, value):
        if self.artifact_dir is None:
            return
        line = json.dumps(value, sort_keys=True, allow_nan=False) + '\n'
        with (self.artifact_dir / name).open('a', encoding='utf-8') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _support(self, responses):
        require(len(responses) == 4 and len({r.get('record_id') for r in responses}) == 4,
                'Exactly four distinct fixed feedback responses are required')
        bindings, audits = [], []
        for response in responses:
            require(response.get('episode_id') == self.episode_id and
                    response.get('target_id') == self.target_id and
                    response.get('target_identity_sha256') == self.target_identity_sha256,
                    'Fixed feedback group/target ownership differs')
            require(response.get('capture_checkpoint_sha256') == self.released_checkpoint_sha256,
                    'All fixed feedback must be generated by released frozen DFlash')
            require(response.get('role', 'feedback') == 'feedback', 'Evaluation requests cannot enter training')
            audit = audit_response(response)
            audits.append(audit)
            bindings.append(dict(record_id=response['record_id'], turn=response['turn'],
                episode_id=self.episode_id, target_id=self.target_id,
                target_identity_sha256=self.target_identity_sha256,
                capture_checkpoint_sha256=self.released_checkpoint_sha256,
                input_ids_sha256=audit['input_ids_sha256'], output_ids_sha256=audit['output_ids_sha256'],
                sequence_sha256=audit['sequence_sha256'], loss_mask_sha256=audit['loss_mask_sha256']))
        require(len({b['input_ids_sha256'] for b in bindings}) == 4,
                'Independent feedback requests cannot duplicate exact input tokens')
        identity = digest(bindings)
        if self.support_bindings is None:
            require(self.optimizer_steps == 0, 'Bind the feedback pool before learning')
            self.support_bindings = copy.deepcopy(bindings)
            self.support_sha256 = identity
        require(bindings == self.support_bindings and identity == self.support_sha256,
                'Frozen feedback tokens, identities, or saved order changed')
        return audits

    def train_event(self, *args, **kwargs):
        raise ValueError('Use train_to_step for the fixed four-response learning curve')

    def train_to_step(self, target_step, responses):
        import torch
        require(not self.failed and not self.awaiting_checkpoint,
                'Trainer failed or pending exported checkpoint binding')
        self._check_owner()
        started = time.perf_counter()
        schedule = step_schedule(self.optimizer_steps, target_step, responses)
        audits = self._support(responses)
        require(self.source_anchor_specs is not None and 0 in self.source_diagnostic_steps,
                'Measure fixed-anchor released source loss before training')
        if self.optimizer_steps == 2:
            require(2 in self.source_diagnostic_steps, 'Measure source loss at step 2 before continuing')
        lookup = {a['record_id']: a for a in audits}
        audit_s = time.perf_counter()-started
        initial_step = self.optimizer_steps
        initial_checkpoint = self.active_checkpoint_sha256
        anchor_before = state_digest(self.generator.get_state())
        training_started = time.perf_counter()
        batches, order = [], []
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            self.draft.train()
            self.optimizer.zero_grad(set_to_none=True)
            for pass_index, before_step, pair in schedule:
                batch_started = time.perf_counter()
                # Exact qualified parent: each loss has its own weighted-slot
                # denominator; its backward is scaled by responses_per_step=2.
                values = [self._train_response(r, lookup[r['record_id']]) for r in pair]
                self._check_owner()
                self.synchronize()
                optimizer_started = time.perf_counter()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in self.parameters.values()), 'Missing/nonfinite full-weight gradient')
                norm = torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1.0,
                                                     error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_steps += 1
                self._check_owner()
                require(all(bool(torch.isfinite(p).all()) for p in self.parameters.values()),
                        'Nonfinite draft update')
                self.synchronize()
                optimizer_s = time.perf_counter()-optimizer_started
                self.response_presentations += 2
                self.eligible_position_presentations += sum(v['eligible_positions'] for v in values)
                self.eligible_block_presentations += sum(v['eligible_blocks'] for v in values)
                order.extend(v['record_id'] for v in values)
                batch = dict(pass_index=pass_index, optimizer_steps_before=before_step,
                    optimizer_steps_after=self.optimizer_steps, presentations=values,
                    loss=sum(v['loss'] for v in values)/2, backward_response_scale=0.5,
                    gradient_norm_before_clip=float(norm), optimizer_s=optimizer_s,
                    wall_s=time.perf_counter()-batch_started,
                    anchor_rng_after_sha256=state_digest(self.generator.get_state()))
                batches.append(batch)
                self._record('TRAIN_STEPS.jsonl', dict(episode_id=self.episode_id,
                    support_sha256=self.support_sha256, **batch))
        except BaseException as error:
            self.failed = True
            self._record('FAILURES.jsonl', dict(operation='train_to_step', target_step=target_step,
                optimizer_steps=self.optimizer_steps, response_presentations=self.response_presentations,
                error_type=type(error).__name__, error=str(error), completed_step_records=len(batches)))
            raise
        training_s = time.perf_counter()-training_started
        self.completed_events += 1
        self.seen_record_ids.update(r['record_id'] for r in responses)
        self.awaiting_checkpoint = True
        presentations = [p for b in batches for p in b['presentations']]
        teacher_s = sum(p['phase_seconds']['teacher_preparation_s'] for p in presentations)
        return dict(event=self.completed_events, episode_id=self.episode_id,
            arm_name=author.ARM_NAME, author_commit=author.AUTHOR_COMMIT, protocol=dict(PROTOCOL), recipe=dict(RECIPE),
            target_step=target_step, starting_step=initial_step, feedback_checkpoint_sha256=self.released_checkpoint_sha256,
            prior_checkpoint_sha256=initial_checkpoint, support_sha256=self.support_sha256,
            support_bindings=copy.deepcopy(self.support_bindings), response_audits=audits,
            optimizer_steps_this_event=self.optimizer_steps-initial_step, optimizer_steps_total=self.optimizer_steps,
            response_presentations_this_event=len(presentations), response_presentations_total=self.response_presentations,
            eligible_position_presentations_this_event=sum(p['eligible_positions'] for p in presentations),
            eligible_position_presentations_total=self.eligible_position_presentations,
            eligible_block_presentations_this_event=sum(p['eligible_blocks'] for p in presentations),
            eligible_block_presentations_total=self.eligible_block_presentations,
            response_order=order, batches=batches,
            anchor_exposure_sha256=digest([p['sampling_and_weights_sha256'] for p in presentations]),
            anchor_rng_before_sha256=anchor_before, anchor_rng_after_sha256=state_digest(self.generator.get_state()),
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None,
            costs=dict(mask_audit_s=audit_s, training_s=training_s, teacher_preparation_s=teacher_s,
                draft_training_and_optimizer_s=training_s-teacher_s,
                training_event_wall_s=time.perf_counter()-started,
                accounting='training_s includes teacher, backwards, optimizer and step receipts; source checks and export excluded'))

    def _source_response(self, response, audit, generator):
        import torch
        self.synchronize()
        started = time.perf_counter()
        ids = torch.tensor([response['input_ids']+response['output_ids']], dtype=torch.long, device=self.device)
        features = self.teacher_features(ids)
        require(isinstance(features, (tuple, list)) and len(features)==2,
                'Teacher callback must return two complete causal feature tensors')
        hidden, teacher = features
        for feature in features:
            require(feature.ndim==3 and feature.shape[:2]==ids.shape and feature.device==self.device and
                    not feature.requires_grad and bool(torch.isfinite(feature).all()),
                    'Missing/nonfinite/unfrozen complete teacher features')
        del feature, features
        self.synchronize()
        teacher_s = time.perf_counter()-started
        started = time.perf_counter()
        pad = author.BLOCK_SIZE-1
        ids = torch.cat((ids, ids.new_full((1,pad), self.mask_token_id)), dim=1)
        hidden = torch.cat((hidden, hidden.new_zeros((1,pad,hidden.shape[-1]))), dim=1)
        teacher = torch.cat((teacher, teacher.new_zeros((1,pad,teacher.shape[-1]))), dim=1)
        loss_mask = torch.zeros_like(ids, dtype=torch.float32)
        loss_mask[:,audit['response_start']:audit['response_end_exclusive']] = 1
        loss, sampling = author.author_response_loss(self.draft, self.embed_tokens, self.lm_head,
            ids, hidden, teacher, loss_mask, mask_token_id=self.mask_token_id, generator=generator,
            attention_backend=self.attention_backend, projection_chunk=self.projection_chunk,
            checkpoint_projection=False)
        self.synchronize()
        return dict(record_id=response['record_id'], turn=response['turn'], loss=float(loss), **sampling,
            phase_seconds=dict(teacher_preparation_s=teacher_s, forward_and_loss_s=time.perf_counter()-started))

    def source_losses(self, responses):
        import torch
        require(not self.failed and not self.awaiting_checkpoint and self.optimizer_steps in (0,2,8),
                'Source diagnostics require a complete bound checkpoint at 0, 2, or 8')
        self._check_owner()
        require(all(p.grad is None for p in self.parameters.values()), 'Source diagnostic cannot consume pending gradients')
        started = time.perf_counter()
        audits = self._support(responses)
        global_before = rng_state()
        sampler_before = self.generator.get_state().clone()
        versions = [p._version for p in self.parameters.values()]
        modes = [(module, module.training) for module in self.draft.modules()]
        optimizer_steps = self.optimizer_steps
        values, specs = [], []
        success = False
        failure = None
        try:
            _seed_global(17)
            self.draft.eval()
            with torch.no_grad():
                for response, audit, binding in zip(responses, audits, self.support_bindings):
                    seed = int(digest(dict(seed=17, purpose='fixed_source_loss', response=binding))[:16],16) % (2**63)
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    value = self._source_response(response, audit, generator)
                    value['anchor_seed'] = seed
                    value['anchor_rng_after_sha256'] = state_digest(generator.get_state())
                    values.append(value)
                    specs.append(dict(record_id=response['record_id'], seed=seed,
                        sampling_and_weights_sha256=value['sampling_and_weights_sha256'],
                        anchor_rng_after_sha256=value['anchor_rng_after_sha256']))
            if self.source_anchor_specs is None:
                require(self.optimizer_steps==0, 'Fixed diagnostic anchors must first be established at released step 0')
                self.source_anchor_specs = copy.deepcopy(specs)
            require(specs==self.source_anchor_specs, 'Fixed diagnostic anchors or weights changed between checkpoints')
            require(versions==[p._version for p in self.parameters.values()] and
                    self.optimizer_steps==optimizer_steps and all(p.grad is None for p in self.parameters.values()),
                    'Source diagnostics mutated weights, optimizer progress, or gradients')
            success = True
        except BaseException as error:
            failure = dict(operation='source_losses', step=self.optimizer_steps,
                error_type=type(error).__name__, error=str(error), completed_responses=len(values))
            raise
        finally:
            for module, training in modes:
                module.training = training
            restore_rng(global_before)
            # The diagnostic never borrows the training generator. Compare before
            # restoring it so an accidental use is visible instead of concealed.
            sampler_unchanged = torch.equal(sampler_before, self.generator.get_state())
            self.generator.set_state(sampler_before.cpu())
            if not success or not sampler_unchanged:
                self.failed = True
            if failure is not None:
                self._record('FAILURES.jsonl', {**failure,
                    'global_rng_restored':state_digest(global_before)==state_digest(rng_state()),
                    'training_anchor_rng_restored':torch.equal(sampler_before,self.generator.get_state()),
                    'source_diagnostic_wall_s':time.perf_counter()-started})
            require(sampler_unchanged, 'Source-loss diagnostic consumed training anchor randomness')
        if state_digest(global_before)!=state_digest(rng_state()):
            self.failed = True
            raise ValueError('Source diagnostic did not restore global RNG')
        self._check_owner()
        self.source_diagnostic_steps = sorted(set(self.source_diagnostic_steps+[self.optimizer_steps]))
        diagnostic_wall_s = time.perf_counter()-started
        result = dict(status='passed', episode_id=self.episode_id, step=self.optimizer_steps,
            checkpoint_sha256=self.active_checkpoint_sha256, support_sha256=self.support_sha256,
            source_loss_mean=sum(v['loss'] for v in values)/4, loss=sum(v['loss'] for v in values)/4, response_losses=values,
            reduction='unweighted mean of four independently normalized response losses',
            source_anchor_specs=copy.deepcopy(specs), fixed_anchor_set_sha256=digest(specs),
            fixed_anchor_identity_sha256=digest(specs), training_rng_preserved=True,
            training_anchor_rng_before_sha256=state_digest(sampler_before),
            training_anchor_rng_after_sha256=state_digest(self.generator.get_state()),
            global_rng_before_sha256=state_digest(global_before), global_rng_after_sha256=state_digest(rng_state()),
            optimizer_steps_before=optimizer_steps, optimizer_steps_after=self.optimizer_steps,
            weights_version_unchanged=True, diagnostic_model_mode='eval_no_grad',
            costs=dict(source_diagnostic_wall_s=diagnostic_wall_s, wall_s=diagnostic_wall_s,
                teacher_preparation_s=sum(v['phase_seconds']['teacher_preparation_s'] for v in values),
                forward_and_loss_s=sum(v['phase_seconds']['forward_and_loss_s'] for v in values),
                accounting='Diagnostic costs exclude training/export and require no generated answers'))
        self._record('SOURCE_LOSSES.jsonl', result)
        return result

    def snapshot(self, *, include_draft=True):
        self._check_owner()
        require(not self.failed and not self.awaiting_checkpoint, 'Snapshot requires a complete bound state')
        state = dict(schema_version=PROTOCOL['schema'], protocol=dict(PROTOCOL), recipe=dict(RECIPE),
            episode_id=self.episode_id, target_id=self.target_id, target_identity_sha256=self.target_identity_sha256,
            released_checkpoint_sha256=self.released_checkpoint_sha256, checkpoint_sha256=self.active_checkpoint_sha256,
            completed_events=self.completed_events, optimizer_steps=self.optimizer_steps,
            response_presentations=self.response_presentations,
            eligible_position_presentations=self.eligible_position_presentations,
            eligible_block_presentations=self.eligible_block_presentations,
            seen_record_ids=sorted(self.seen_record_ids), support_bindings=self.support_bindings,
            support_sha256=self.support_sha256, source_anchor_specs=self.source_anchor_specs,
            source_diagnostic_steps=self.source_diagnostic_steps, optimizer=self.optimizer.state_dict(),
            anchor_rng_state=self.generator.get_state(), global_rng_state=rng_state(),
            draft_module_modes={name:module.training for name,module in self.draft.named_modules()},
            includes_draft=include_draft)
        if include_draft:
            state['draft'] = self.draft.state_dict()
        return _clone_cpu(state)

    def restore(self, state, *, draft_state=None):
        require(state.get('schema_version')==PROTOCOL['schema'] and state.get('protocol')==PROTOCOL and
                state.get('recipe')==RECIPE and state.get('episode_id')==self.episode_id and
                state.get('target_id')==self.target_id and state.get('target_identity_sha256')==self.target_identity_sha256 and
                state.get('released_checkpoint_sha256')==self.released_checkpoint_sha256,
                'Recovery belongs to a different group/model/numerical protocol')
        require(state['includes_draft'] or draft_state is not None, 'Explicit bound draft weights required')
        self.draft.load_state_dict(state['draft'] if state['includes_draft'] else draft_state, strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.optimizer.zero_grad(set_to_none=True)
        for name in ('completed_events','optimizer_steps','response_presentations','eligible_position_presentations',
                     'eligible_block_presentations','support_bindings','support_sha256','source_anchor_specs',
                     'source_diagnostic_steps'):
            setattr(self,name,copy.deepcopy(state[name]))
        self.active_checkpoint_sha256 = state['checkpoint_sha256']
        self.seen_record_ids = set(state['seen_record_ids'])
        self.generator.set_state(state['anchor_rng_state'].cpu())
        require(set(state['draft_module_modes'])==set(dict(self.draft.named_modules())), 'Draft module inventory changed')
        for name,module in self.draft.named_modules():
            module.training = state['draft_module_modes'][name]
        restore_rng(state['global_rng_state'])
        self.failed = self.awaiting_checkpoint = False
        self._check_owner()

    def save_checkpoint_state(self, path, *, include_draft=True):
        import torch
        started = time.perf_counter()
        path = Path(path)
        require(not path.exists() and not path.is_symlink(), 'Recovery state must use a fresh path')
        state = self.snapshot(include_draft=include_draft)
        with path.open('xb') as handle:
            torch.save(state,handle)
            handle.flush();os.fsync(handle.fileno())
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(4*1024*1024),b''):
                h.update(chunk)
        return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=h.hexdigest(),
            includes_draft=include_draft,checkpoint_sha256=self.active_checkpoint_sha256,
            optimizer_steps=self.optimizer_steps,checkpoint_persistence_s=time.perf_counter()-started)
