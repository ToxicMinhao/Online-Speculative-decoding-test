"""Fresh pooled P8 training with the qualified ordinary author response loss.

Pool ownership never replaces original source ownership. Eight source responses
are presented in sealed A0/B0/.../A3/B3 order twice. Each backward divides its
individually normalized response loss by two; one clip and AdamW step follows
each pair. Historical four-response and L8 entry points remain unchanged.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

from experiments.dflash_baseline_learning_curve import training as curve
from .evidence import composite_identity, validate_pool

author = curve.author
require, digest = curve.require, curve.digest
LR = 3e-4
RECIPE = dict(curve.RECIPE, passes=2)
PROTOCOL = dict(schema='dflash_feedback_allocation_control_v1', checkpoint_id='P8',
    feedback_responses=8, total_passes=2, total_steps=8, training_response_presentations=16,
    checkpoint_steps=[0, 8], learning_rate=LR, response_order='A0,B0,A1,B1,A2,B2,A3,B3 twice',
    loss_reduction='mean of two independently normalized response losses',
    initialization='released draft, fresh AdamW, fresh normal seed-17 training sampler per dataset',
    source_diagnostics='original PR9 source anchors only after step 8',
    response_only_supervision=True)
ORIGINAL_FIELDS = ('domain', 'group_id', 'request_id', 'record_id', 'episode_id', 'turn',
    'role', 'target_id', 'target_identity_sha256', 'capture_checkpoint_sha256')
TOKEN_FIELDS = ('input_ids_sha256', 'output_ids_sha256', 'sequence_sha256', 'loss_mask_sha256')
SAMPLING_FIELDS = ('anchor_positions', 'block_keep_mask', 'label_positions', 'teacher_positions',
    'binary_loss_mask', 'position_weights', 'sampled_blocks', 'eligible_blocks', 'eligible_positions',
    'unique_supervised_positions', 'unique_supervised_position_ids', 'weighted_position_sum',
    'normalization_denominator', 'excluded_anchor_positions', 'zero_supervision',
    'backbone_forward_calls', 'head_projection_chunks', 'sampling_and_weights_sha256')


def _bindings(responses):
    audits, bindings = [], []
    for response in responses:
        audit = author.audit_response(response)
        owner = {key: response[key] for key in (*ORIGINAL_FIELDS, 'pool_id', 'pool_request_id')}
        audits.append(dict(audit, **owner,
            response_loss_mask=[0]*len(response['input_ids'])+[1]*len(response['output_ids'])+[0]*9))
        bindings.append(dict(owner, **{key: audit[key] for key in TOKEN_FIELDS}))
    return audits, bindings


def _validate_sampling(value, audit):
    """Reconcile reported slots with response-only supervision and author decay."""
    anchors = value['anchor_positions']
    start, end = audit['response_start'], audit['response_end_exclusive']
    require(isinstance(anchors, list) and anchors == sorted(set(anchors)) and
        len(anchors) == value['sampled_blocks'] == min(512, end-start-1) and
        all(type(a) is int and start <= a < end for a in anchors), 'P8 sampled anchors differ')
    require(value['block_keep_mask'] == [True]*len(anchors), 'P8 keep mask differs')
    fields = ('label_positions', 'teacher_positions', 'binary_loss_mask', 'position_weights')
    require(all(len(value[field]) == len(anchors) for field in fields), 'P8 sampling block counts differ')
    positions, weights, blocks = [], [], 0
    for index, anchor in enumerate(anchors):
        labels = list(range(anchor, anchor+10))
        binary = [float(offset > 0 and start <= label < end) for offset, label in enumerate(labels)]
        require(value['label_positions'][index] == labels and
            value['teacher_positions'][index] == [label-1 for label in labels] and
            value['binary_loss_mask'][index] == binary, 'P8 label/teacher/mask positions differ')
        observed = value['position_weights'][index]
        expected = [active*math.exp(-max(0, offset-1)/7) for offset, active in enumerate(binary)]
        require(len(observed) == 10 and all(math.isfinite(w) and math.isclose(w, e, rel_tol=2e-6, abs_tol=1e-7)
            for w, e in zip(observed, expected)), 'P8 position weights differ')
        weights.extend(observed)
        positions.extend(label for label, active in zip(labels, binary) if active)
        blocks += int(any(binary))
    require(value['unique_supervised_position_ids'] == sorted(set(positions)) and
        value['unique_supervised_positions'] == len(set(positions)) and
        value['eligible_positions'] == len(positions) and value['eligible_blocks'] == blocks and
        value['excluded_anchor_positions'] == len(anchors), 'P8 slot exposure counts differ')
    require(value['zero_supervision'] is (len(positions) == 0) and value['backbone_forward_calls'] == 1,
        'P8 zero-supervision/forward accounting differs')
    require(math.isclose(value['weighted_position_sum'], sum(weights), rel_tol=3e-6, abs_tol=1e-7) and
        math.isclose(value['normalization_denominator'], value['weighted_position_sum']+1e-6,
            rel_tol=3e-6, abs_tol=1e-10), 'P8 separate response denominator differs')


def validate_training_accounting(account, pool):
    """Validate completed P8 accounting without importing Torch or old training state."""
    validate_pool(pool)
    responses = pool['support']
    audits, bindings = _bindings(responses)
    require(account.get('status') == 'passed' and account.get('checkpoint_id') == 'P8' and
        account.get('pool_id') == pool['pool_id'] and account.get('episode_id') == pool['pool_id'],
        'P8 accounting/checkpoint ownership differs')
    require(account.get('pool_payload_sha256') == pool['pool_payload_sha256'], 'Pool seal changed')
    require(digest(account.get('recipe')) == digest(RECIPE) and account.get('protocol') == PROTOCOL,
        'Qualified P8 recipe/protocol differs')
    for field, expected in dict(starting_step=0, target_step=8, optimizer_steps_total=8,
            optimizer_steps_this_event=8, response_presentations_total=16,
            response_presentations_this_event=16, attempted_response_calls=16,
            completed_backward_presentations=16, uncommitted_backward_presentations=0,
            unknown_backward_response_calls=0, unattempted_response_calls=0, optimizer_steps_unknown=0).items():
        require(type(account.get(field)) is int and account[field] == expected,
            'P8 training accounting differs: '+field)
    require(account.get('source_diagnostic_steps') == [] and account.get('optimizer_state_exported') is False,
        'No intermediate diagnostics or optimizer export permitted')
    require(account.get('support_bindings') == bindings and account.get('support_sha256') == digest(bindings)
        and account.get('response_audits') == audits, 'Original support bindings/masks differ')
    require(account.get('feedback_checkpoint_sha256') ==
        pool['checkpoints']['F']['checkpoint_identity_sha256'], 'Frozen feedback checkpoint differs')
    order = [response['pool_request_id'] for response in responses]*2
    require(account.get('response_order') == order, 'P8 response order differs')
    require(account.get('training_anchor_rule') == 'fresh_normal_seed17' and
        account.get('historical_training_anchor_equality_required') is False,
        'P8 must use fresh normal training anchors')
    batches = account.get('batches', [])
    require(len(batches) == 8, 'Exactly eight P8 accumulation pairs required')
    previous_rng = account.get('anchor_rng_before_sha256')
    require(author._sha256(previous_rng), 'Initial sampler receipt missing')
    all_values = []
    for step, batch in enumerate(batches):
        require(batch.get('pass_index') == step//4 and batch.get('optimizer_steps_before') == step and
            batch.get('optimizer_steps_after') == step+1, 'P8 pair schedule differs')
        require(batch.get('effective_learning_rate') == LR and batch.get('backward_response_scale') == .5,
            'P8 LR/response averaging differs')
        values = batch.get('presentations', [])
        require(len(values) == 2, 'P8 needs two separately normalized responses per step')
        for offset, value in enumerate(values):
            index = (step*2+offset) % 8
            binding, audit = bindings[index], audits[index]
            require(all(value.get(k) == v for k, v in binding.items()),
                'Original P8 response identity/token binding differs')
            require(value.get('response_start') == audit['response_start'] and
                value.get('response_end_exclusive') == audit['response_end_exclusive'],
                'P8 response-only supervision boundary differs')
            require(all(k in value for k in SAMPLING_FIELDS), 'Missing observed P8 anchor/exposure fields')
            sampling = {k: value[k] for k in SAMPLING_FIELDS if k != 'sampling_and_weights_sha256'}
            require(value['sampling_and_weights_sha256'] == digest(sampling), 'P8 sampling receipt hash differs')
            _validate_sampling(value, audit)
            require(value.get('anchor_rng_before_sha256') == previous_rng and
                author._sha256(value.get('anchor_rng_after_sha256')), 'P8 sampler trajectory changed')
            previous_rng = value['anchor_rng_after_sha256']
            require(math.isfinite(value.get('loss', math.nan)) and
                math.isfinite(value['normalization_denominator']) and value['normalization_denominator'] > 0,
                'Nonfinite P8 loss/denominator')
            require(all(audit['response_start'] < p < audit['response_end_exclusive']
                for p in value['unique_supervised_position_ids']), 'P8 supervised prompt or padding')
        require(batch.get('anchor_rng_after_sha256') == previous_rng and
            batch.get('loss') == sum(v['loss'] for v in values)/2, 'P8 pair mean/sampler differs')
        all_values.extend(values)
    require(account.get('anchor_rng_after_sha256') == previous_rng and
        account.get('anchor_exposure_sha256') == digest([v['sampling_and_weights_sha256'] for v in all_values]),
        'Final P8 sampler/exposure digest differs')
    for field, row_field in (('eligible_position_presentations_total', 'eligible_positions'),
            ('eligible_block_presentations_total', 'eligible_blocks'),
            ('zero_supervision_presentations', 'zero_supervision')):
        require(account.get(field) == sum(v[row_field] for v in all_values), 'P8 exposure total differs: '+field)
    require(account.get('input_token_presentations_total') == 2*sum(len(r['input_ids']) for r in responses) and
        account.get('output_token_presentations_total') == 2*sum(len(r['output_ids']) for r in responses),
        'P8 token presentation totals differ')
    return dict(status='passed', pool_id=pool['pool_id'], optimizer_steps=8, response_presentations=16,
        training_anchor_rule='fresh_normal_seed17', support_sha256=digest(bindings))


class PooledFeedbackTrainer(curve.FixedFeedbackTrainer):
    def __init__(self, draft, embed_tokens, lm_head, teacher_features, *, original_pool, target_id,
                 target_identity_sha256, released_checkpoint_sha256, mask_token_id,
                 attention_backend='flex_attention', projection_chunk=64,
                 synchronize=None, check_frozen=None, artifact_dir=None):
        from .loss import validate_pool_diagnostics
        validate_pool(original_pool)
        validate_pool_diagnostics(original_pool)
        self.original_pool = copy.deepcopy(original_pool)
        self.pool_id = original_pool['pool_id']
        require(artifact_dir is None or not (Path(artifact_dir)/'POOL_BINDING.jsonl').exists(),
            'Fresh P8 evidence directory required')
        require(original_pool['checkpoints']['F']['checkpoint_identity_sha256'] == released_checkpoint_sha256,
            'P8 must initialize from the sealed released checkpoint')
        super().__init__(draft, embed_tokens, lm_head, teacher_features, episode_id=self.pool_id,
            target_id=target_id, target_identity_sha256=target_identity_sha256,
            released_checkpoint_sha256=released_checkpoint_sha256, mask_token_id=mask_token_id,
            attention_backend=attention_backend, projection_chunk=projection_chunk,
            synchronize=synchronize, check_frozen=check_frozen, artifact_dir=artifact_dir)
        self._support(original_pool['support'])
        self.initial_anchor_rng_sha256 = curve.state_digest(self.generator.get_state())
        require(not self.optimizer.state and self.optimizer.param_groups[0]['foreach'] is None and
            self.optimizer.param_groups[0]['fused'] is None, 'P8 requires fresh qualified AdamW')
        self._record('POOL_BINDING.jsonl', dict(pool_id=self.pool_id,
            pool_payload_sha256=original_pool['pool_payload_sha256'], support_sha256=self.support_sha256,
            original_group_ids=original_pool['group_ids'], fresh_optimizer=True, seed=17))

    def _check_owner(self):
        super()._check_owner()
        require(self.optimizer.param_groups[0]['foreach'] is None and self.optimizer.param_groups[0]['fused'] is None,
            'Qualified P8 optimizer execution settings changed')

    def _support(self, responses):
        responses = list(responses)
        require(responses == self.original_pool['support'], 'P8 responses differ from sealed original feedback/order')
        require(len(responses) == 8 and len({composite_identity(r) for r in responses}) == 8 and
            len({r['pool_request_id'] for r in responses}) == 8, 'Eight distinct composite source identities required')
        require(all(r['pool_id'] == self.pool_id and r['role'] == 'feedback' and
            r['group_id'] == r['episode_id'] and r['target_id'] == self.target_id and
            r['target_identity_sha256'] == self.target_identity_sha256 and
            r['capture_checkpoint_sha256'] == self.released_checkpoint_sha256 for r in responses),
            'P8 source/target ownership differs')
        require([r['group_id'] for r in responses] == self.original_pool['group_ids']*4,
            'P8 pairs must interleave original groups')
        audits, bindings = _bindings(responses)
        if self.support_bindings is None:
            require(self.optimizer_steps == 0, 'Bind P8 pool before learning')
            self.support_bindings, self.support_sha256 = copy.deepcopy(bindings), digest(bindings)
        require(bindings == self.support_bindings and digest(bindings) == self.support_sha256,
            'P8 support binding changed')
        return audits

    def _train_response(self, response, audit):
        before = curve.state_digest(self.generator.get_state())
        value = author.PersistentAuthorOSDTrainer._train_response(self, response, audit)
        value.update({k: response[k] for k in (*ORIGINAL_FIELDS, 'pool_id', 'pool_request_id')})
        value.update({k: audit[k] for k in TOKEN_FIELDS})
        value.update(anchor_rng_before_sha256=before,
            anchor_rng_after_sha256=curve.state_digest(self.generator.get_state()))
        _validate_sampling(value, audit)
        return value

    def train_eight(self, responses):
        import torch
        require(not self.failed and not self.awaiting_checkpoint and self.optimizer_steps == 0 and
            self.completed_events == 0 and not self.optimizer.state, 'P8 must start once from fresh released state')
        self._check_owner()
        started = time.perf_counter()
        responses = list(responses)
        audits = self._support(responses)
        lookup = {a['pool_request_id']: a for a in audits}
        before = curve.state_digest(self.generator.get_state())
        require(before == self.initial_anchor_rng_sha256, 'Fresh seed-17 training sampler was changed before P8')
        audit_s = time.perf_counter()-started
        training_started = time.perf_counter()
        batches, pending_presentations = [], []
        attempted_response_calls = completed_backward_presentations = 0
        optimizer_step_pending = False
        pending_response_identity = None
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            self.draft.train()
            self.optimizer.zero_grad(set_to_none=True)
            for step in range(8):
                tick = time.perf_counter()
                values = []
                pending_presentations = values
                for response in responses[(step % 4)*2:(step % 4)*2+2]:
                    pending_response_identity = {key: response[key] for key in
                        ('pool_id', 'pool_request_id', 'episode_id', 'record_id', 'turn')}
                    attempted_response_calls += 1
                    values.append(self._train_response(response, lookup[response['pool_request_id']]))
                    completed_backward_presentations += 1
                    pending_response_identity = None
                self._check_owner()
                self.synchronize()
                opt_started = time.perf_counter()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.parameters.values()),
                    'Missing/nonfinite full-weight gradient')
                norm = torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1., error_if_nonfinite=True)
                optimizer_step_pending = True
                self.optimizer.step()
                optimizer_step_pending = False
                self.optimizer_steps += 1
                self.response_presentations += 2
                self.eligible_position_presentations += sum(v['eligible_positions'] for v in values)
                self.eligible_block_presentations += sum(v['eligible_blocks'] for v in values)
                self.optimizer.zero_grad(set_to_none=True)
                self._check_owner()
                require(all(bool(torch.isfinite(p).all()) for p in self.parameters.values()), 'Nonfinite P8 update')
                self.synchronize()
                batch = dict(pass_index=step//4, optimizer_steps_before=step, optimizer_steps_after=step+1,
                    presentations=values, loss=sum(v['loss'] for v in values)/2, backward_response_scale=.5,
                    effective_learning_rate=LR, gradient_norm_before_clip=float(norm),
                    optimizer_s=time.perf_counter()-opt_started, wall_s=time.perf_counter()-tick,
                    anchor_rng_after_sha256=curve.state_digest(self.generator.get_state()))
                batches.append(batch)
                self._record('TRAIN_STEPS.jsonl', dict(pool_id=self.pool_id, **batch))
                pending_presentations = []
        except BaseException as error:
            self.failed = True
            failure = dict(status='failed', operation='train_eight', pool_id=self.pool_id,
                optimizer_steps=self.optimizer_steps, response_presentations=self.response_presentations,
                optimizer_steps_unknown=int(optimizer_step_pending),
                error_type=type(error).__name__, error=str(error), completed_batches=batches,
                pending_presentations=pending_presentations, attempted_response_calls=attempted_response_calls,
                completed_backward_presentations=completed_backward_presentations,
                unknown_backward_response_calls=attempted_response_calls-completed_backward_presentations,
                unattempted_response_calls=16-attempted_response_calls,
                pending_response_identity=pending_response_identity,
                uncommitted_backward_presentations=max(0, completed_backward_presentations-2*self.optimizer_steps))
            self._record('FAILURES.jsonl', failure)
            error.training_partial = failure
            raise
        elapsed = time.perf_counter()-training_started
        self.completed_events = 1
        self.awaiting_checkpoint = True
        self.seen_record_ids.update(r['pool_request_id'] for r in responses)
        presentations = [v for b in batches for v in b['presentations']]
        teacher_s = sum(v['phase_seconds']['teacher_preparation_s'] for v in presentations)
        account = dict(status='passed', checkpoint_id='P8', pool_id=self.pool_id, episode_id=self.pool_id,
            pool_payload_sha256=self.original_pool['pool_payload_sha256'], domain=self.original_pool['domain'],
            protocol=copy.deepcopy(PROTOCOL), recipe=dict(RECIPE), starting_step=0, target_step=8,
            optimizer_steps_total=8, optimizer_steps_this_event=8,
            response_presentations_total=16, response_presentations_this_event=16,
            attempted_response_calls=attempted_response_calls,
            completed_backward_presentations=completed_backward_presentations, uncommitted_backward_presentations=0,
            unknown_backward_response_calls=0, unattempted_response_calls=0, optimizer_steps_unknown=0,
            attention_backend=self.attention_backend, projection_chunk=self.projection_chunk,
            response_order=[v['pool_request_id'] for v in presentations],
            original_response_order=[{k: v[k] for k in ('episode_id', 'record_id', 'turn')} for v in presentations],
            batches=batches, response_audits=audits, support_bindings=copy.deepcopy(self.support_bindings),
            support_sha256=self.support_sha256, feedback_checkpoint_sha256=self.released_checkpoint_sha256,
            anchor_rng_before_sha256=before, anchor_rng_after_sha256=curve.state_digest(self.generator.get_state()),
            anchor_exposure_sha256=digest([v['sampling_and_weights_sha256'] for v in presentations]),
            training_anchor_rule='fresh_normal_seed17', historical_training_anchor_equality_required=False,
            eligible_position_presentations_total=self.eligible_position_presentations,
            eligible_block_presentations_total=self.eligible_block_presentations,
            zero_supervision_presentations=sum(v['zero_supervision'] for v in presentations),
            input_token_presentations_total=2*sum(len(r['input_ids']) for r in responses),
            output_token_presentations_total=2*sum(len(r['output_ids']) for r in responses),
            source_diagnostic_steps=[], optimizer_state_exported=False,
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None,
            costs=dict(mask_audit_s=audit_s, training_s=elapsed, teacher_preparation_s=teacher_s,
                draft_training_and_optimizer_s=elapsed-teacher_s, training_event_wall_s=time.perf_counter()-started,
                accounting='Training includes teacher, forward/backward, optimizer and receipts; nested costs must not be added'))
        try:
            validate_training_accounting(account, self.original_pool)
        except BaseException as error:
            self.failed = True
            failure = dict(account, status='failed', operation='validate_completed_training',
                optimizer_steps=self.optimizer_steps, response_presentations=self.response_presentations,
                completed_batches=batches, pending_presentations=[], optimizer_steps_unknown=0,
                error_type=type(error).__name__, error=str(error))
            self._record('FAILURES.jsonl', failure)
            error.training_partial = failure
            raise
        return account

    def bind_p8_checkpoint(self, checkpoint_sha256):
        require(self.optimizer_steps == 8 and self.completed_events == 1 and self.awaiting_checkpoint,
            'Only the completed pending P8 checkpoint can be bound once')
        require(checkpoint_sha256 != self.released_checkpoint_sha256, 'P8 checkpoint must be distinct from released F')
        author.PersistentAuthorOSDTrainer.bind_checkpoint(self, 1, checkpoint_sha256)

    def train_to_step(self, *args, **kwargs):
        raise ValueError('Use explicit P8 train_eight; historical progression remains unchanged')

    def train_event(self, *args, **kwargs):
        raise ValueError('Use explicit P8 train_eight')

    def source_losses(self, *args, **kwargs):
        raise ValueError('Use frozen P8 diagnostics only after the completed eight-step checkpoint')

    def bind_checkpoint(self, *args, **kwargs):
        raise ValueError('Bind the final checkpoint explicitly as P8')

    def snapshot(self, *args, **kwargs):
        raise ValueError('P8 does not export optimizer/master state or intermediate checkpoints')

    def restore(self, *args, **kwargs):
        raise ValueError('P8 starts from released weights and fresh AdamW; no old optimizer loading')
