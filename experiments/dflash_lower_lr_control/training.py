"""Explicit fresh 0→8 conservative control; historical 0→2→8 stays intact.

Only the learning rate differs. Ordinary training keeps the qualified sampler,
BF16 weights/moments, separate response denominators and pair-mean backward.
Retained anchor/exposure receipts are checks, never replacement training samples.
"""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import time

_spec = importlib.util.spec_from_file_location('_lower_lr_qualified_curve',
    Path(__file__).resolve().parents[1] / 'dflash_baseline_learning_curve/training.py')
curve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(curve)
from .loss import SAMPLING_FIELDS

author = curve.author
require, digest = curve.require, curve.digest
LR = 3e-5
RECIPE = dict(curve.RECIPE, learning_rate=LR)
PROTOCOL = dict(schema='dflash_lower_lr_control_v1', checkpoint_steps=[0,8],
    feedback_responses=4, total_passes=4, total_steps=8, training_response_presentations=16,
    learning_rate=LR, initialization='released draft and fresh AdamW in each group',
    source_diagnostics='reuse validated retained PR9 anchor bindings; no new F/S2 evaluations',
    response_only_supervision=True, checkpoint_id='L8')
TRAIN_FIELDS = ('record_id','turn','response_start','response_end_exclusive',
                'sequence_sha256','loss_mask_sha256', *SAMPLING_FIELDS)


class LowerLRTrainer(curve.FixedFeedbackTrainer):
    def __init__(self, draft, embed_tokens, lm_head, teacher_features, *, episode_id, target_id,
                 target_identity_sha256, released_checkpoint_sha256, mask_token_id,
                 retained_training, source_references=None, heldout_references=None,
                 attention_backend='flex_attention', projection_chunk=64,
                 synchronize=None, check_frozen=None, artifact_dir=None):
        import torch
        curve._seed_global(17)
        require(isinstance(episode_id,str) and episode_id and isinstance(target_id,str) and target_id,
                'Episode/target identity required')
        require(author._sha256(target_identity_sha256) and author._sha256(released_checkpoint_sha256),
                'Invalid model identity')
        require(type(mask_token_id) is int and mask_token_id >= 0, 'Invalid mask token')
        require(attention_backend in ('flex_attention','sdpa') and type(projection_chunk) is int and
                projection_chunk > 0, 'Invalid qualified loss execution settings')
        require(not author._has_adaptive_head(draft), 'Adaptive heads must remain disabled')
        self.draft,self.embed_tokens,self.lm_head = draft,embed_tokens,lm_head
        self.teacher_features = teacher_features
        self.episode_id,self.target_id = episode_id,target_id
        self.target_identity_sha256 = target_identity_sha256
        self.released_checkpoint_sha256 = self.active_checkpoint_sha256 = released_checkpoint_sha256
        self.mask_token_id,self.attention_backend,self.projection_chunk = mask_token_id,attention_backend,projection_chunk
        self.parameters = dict(draft.named_parameters())
        require(self.parameters and all(p.requires_grad for p in self.parameters.values()),
                'Every draft parameter must be trainable')
        self._parameter_ids = tuple(id(p) for p in self.parameters.values())
        self.device = next(iter(self.parameters.values())).device
        require(all(p.device == self.device for p in self.parameters.values()) and callable(teacher_features),
                'Single-device draft and frozen teacher callback required')
        self.synchronize = synchronize or (lambda: torch.cuda.synchronize(self.device) if self.device.type=='cuda' else None)
        self.check_frozen = check_frozen or (lambda: None)
        self.optimizer = torch.optim.AdamW(list(self.parameters.values()),lr=LR,betas=(.9,.999),
            eps=1e-8,weight_decay=.01,foreach=None,fused=None)
        self._optimizer_id = id(self.optimizer)
        self.generator = torch.Generator(device=self.device).manual_seed(17)
        self.completed_events = self.optimizer_steps = self.response_presentations = 0
        self.eligible_position_presentations = self.eligible_block_presentations = 0
        self.seen_record_ids = set()
        self.awaiting_checkpoint = self.failed = False
        self.support_bindings = self.support_sha256 = None
        # This path does not pretend to have executed the historical diagnostics.
        self.source_anchor_specs = None
        self.source_diagnostic_steps = []
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True,exist_ok=True)
            require(not any((self.artifact_dir/x).exists() for x in
                ('TRAIN_STEPS.jsonl','FAILURES.jsonl','ANCHOR_CHECKS.jsonl','RETAINED_BINDINGS.jsonl')),
                'Fresh L8 trainer evidence directory required')
        self.retained_training = copy.deepcopy(retained_training)
        self.retained_source_references = copy.deepcopy(source_references)
        self.retained_heldout_references = copy.deepcopy(heldout_references)
        self._validate_retained()
        self._check_owner()

    def _check_owner(self):
        import torch
        require(id(self.optimizer)==self._optimizer_id and tuple(id(p) for p in self.draft.parameters())==
            self._parameter_ids and tuple(id(p) for g in self.optimizer.param_groups for p in g['params'])==
            self._parameter_ids, 'Draft/optimizer ownership changed')
        require(all(p.requires_grad for p in self.parameters.values()) and
            not author._has_adaptive_head(self.draft), 'Qualified trainable parameter set changed')
        require(len({p.dtype for p in self.parameters.values()})==1 and
            all(p.dtype in (torch.float32,torch.bfloat16) for p in self.parameters.values()) and
            (self.device.type!='cuda' or all(p.dtype==torch.bfloat16 for p in self.parameters.values())),
            'Retain native BF16 precision; no master-weight substitution')
        for module in (self.embed_tokens,self.lm_head):
            require(not module.training and all(not p.requires_grad and p.grad is None and
                p.device==self.device for p in module.parameters()), 'Teacher embedding/head changed')
        require(len(self.optimizer.param_groups)==1, 'Adam parameter groups changed')
        group=self.optimizer.param_groups[0]
        require(group['lr']==LR and group['betas']==(.9,.999) and group['eps']==1e-8 and
            group['weight_decay']==.01 and group['foreach'] is None and group['fused'] is None,
            'Qualified lower-LR AdamW recipe changed')
        states=self.optimizer.state
        require((not states and self.optimizer_steps==0) or (set(states)==set(self.parameters.values()) and
            {int(s['step'].item()) for s in states.values()}=={self.optimizer_steps}),
            'Adam state inventory/counters changed')
        self.check_frozen()

    def _validate_retained(self):
        t=self.retained_training
        require(t.get('status')=='passed' and len(t.get('updates',[]))==2 and
            len(t.get('source_losses',[]))==3, 'Complete original training/diagnostic receipts required')
        updates=[u['accounting'] for u in t['updates']]
        require([(u['starting_step'],u['target_step']) for u in updates]==[(0,2),(2,8)],
            'Original 0→2→8 trajectory binding differs')
        require(all(u['episode_id']==self.episode_id and
            u['feedback_checkpoint_sha256']==self.released_checkpoint_sha256 for u in updates),
            'Original training belongs to another group/released draft')
        require(updates[0]['support_bindings']==updates[1]['support_bindings'] and
            updates[0]['support_sha256']==updates[1]['support_sha256'], 'Original support changes across updates')
        self.original_support_bindings=copy.deepcopy(updates[0]['support_bindings'])
        self.original_batches=copy.deepcopy([b for u in updates for b in u['batches']])
        require(len(self.original_batches)==8 and all(len(b['presentations'])==2 for b in self.original_batches),
            'Original exposure must contain eight pairs')
        require([b['optimizer_steps_before'] for b in self.original_batches]==list(range(8)) and
            [b['optimizer_steps_after'] for b in self.original_batches]==list(range(1,9)) and
            [b['pass_index'] for b in self.original_batches]==[0,0,1,1,2,2,3,3],
            'Original presentation schedule differs')
        order=[p['record_id'] for b in self.original_batches for p in b['presentations']]
        require(order==[b['record_id'] for b in self.original_support_bindings]*4,
            'Original saved response order differs')
        for diagnostic,step in zip(t['source_losses'],(0,2,8)):
            require(diagnostic['status']=='passed' and diagnostic['step']==step and
                diagnostic['support_sha256']==updates[0]['support_sha256'] and
                len(diagnostic['response_losses'])==4, 'Original source diagnostic binding differs')
            require(diagnostic['source_anchor_specs']==t['source_losses'][0]['source_anchor_specs'],
                'Original fixed source diagnostic anchors differ')
        first=t['source_losses'][0]
        for binding,row,spec in zip(self.original_support_bindings,first['response_losses'],first['source_anchor_specs']):
            seed=int(digest(dict(seed=17,purpose='fixed_source_loss',response=binding))[:16],16)%(2**63)
            require(row['record_id']==spec['record_id']==binding['record_id'] and
                row['anchor_seed']==spec['seed']==seed and
                row['sampling_and_weights_sha256']==spec['sampling_and_weights_sha256'] and
                row['anchor_rng_after_sha256']==spec['anchor_rng_after_sha256'],
                'Original source diagnostic seed/sampling binding differs')
        if self.retained_source_references is not None:
            require(len(self.retained_source_references)==4, 'Four retained source references required')
            for row,reference in zip(first['response_losses'],self.retained_source_references):
                for key in ('record_id','anchor_seed','anchor_rng_after_sha256',*SAMPLING_FIELDS):
                    require(reference.get(key)==row[key], 'Source reference differs: '+key)
        self._record('RETAINED_BINDINGS.jsonl',dict(episode_id=self.episode_id,
            retained_training_sha256=digest(t),original_support_sha256=updates[0]['support_sha256'],
            original_diagnostics_reused=True,new_source_diagnostics_executed=False))

    def train_to_step8(self,responses):
        import torch
        require(not self.failed and not self.awaiting_checkpoint and self.optimizer_steps==0 and
            self.completed_events==0 and not self.optimizer.state, 'L8 must start once from released fresh state')
        self._check_owner()
        started=time.perf_counter()
        audits=self._support(responses)
        require(self.support_bindings==self.original_support_bindings,
            'Training responses differ from retained fixed feedback')
        before=curve.state_digest(self.generator.get_state())
        require(before==self.retained_training['updates'][0]['accounting']['anchor_rng_before_sha256'],
            'Training sampler initial state differs')
        lookup={a['record_id']:a for a in audits}
        audit_s=time.perf_counter()-started
        training_started=time.perf_counter()
        batches=[]
        attempted_response_calls = completed_backward_presentations = 0
        pending_presentations = []
        if self.device.type=='cuda': torch.cuda.reset_peak_memory_stats(self.device)
        try:
            self.draft.train()
            self.optimizer.zero_grad(set_to_none=True)
            for step in range(8):
                tick=time.perf_counter()
                pair=responses[(step%2)*2:(step%2)*2+2]
                values=[]
                pending_presentations=values
                for response in pair:
                    attempted_response_calls += 1
                    values.append(self._train_response(response,lookup[response['record_id']]))
                    completed_backward_presentations += 1
                original=self.original_batches[step]
                errors=[dict(presentation=i,field=k) for i,(v,ref) in enumerate(zip(values,original['presentations']))
                        for k in TRAIN_FIELDS if v.get(k)!=ref.get(k)]
                rng_after=curve.state_digest(self.generator.get_state())
                if rng_after!=original['anchor_rng_after_sha256']:
                    errors.append(dict(field='anchor_rng_after_sha256'))
                receipt=dict(step=step+1,status='failed' if errors else 'passed',errors=errors,
                    original_sampling_sha256=[v['sampling_and_weights_sha256'] for v in original['presentations']],
                    observed_presentations=values,anchor_rng_after_sha256=rng_after,
                    effective_learning_rate=self.optimizer.param_groups[0]['lr'])
                self._record('ANCHOR_CHECKS.jsonl',receipt)
                require(not errors, 'Original training anchors/exposure mismatch before optimizer step')
                self._check_owner();self.synchronize();opt_started=time.perf_counter()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.parameters.values()),
                    'Missing/nonfinite full-weight gradient')
                norm=torch.nn.utils.clip_grad_norm_(list(self.parameters.values()),1.,error_if_nonfinite=True)
                self.optimizer.step();self.optimizer.zero_grad(set_to_none=True);self.optimizer_steps+=1
                self._check_owner()
                require(all(bool(torch.isfinite(p).all()) for p in self.parameters.values()), 'Nonfinite update')
                self.synchronize()
                self.response_presentations+=2
                self.eligible_position_presentations+=sum(v['eligible_positions'] for v in values)
                self.eligible_block_presentations+=sum(v['eligible_blocks'] for v in values)
                batch=dict(pass_index=step//2,optimizer_steps_before=step,optimizer_steps_after=step+1,
                    presentations=values,loss=sum(v['loss'] for v in values)/2,backward_response_scale=.5,
                    effective_learning_rate=LR,gradient_norm_before_clip=float(norm),
                    optimizer_s=time.perf_counter()-opt_started,wall_s=time.perf_counter()-tick,
                    anchor_rng_after_sha256=rng_after,original_exposure_match=True)
                batches.append(batch)
                self._record('TRAIN_STEPS.jsonl',dict(episode_id=self.episode_id,**batch))
                pending_presentations=[]
        except BaseException as error:
            self.failed=True
            failure=dict(operation='train_to_step8',optimizer_steps=self.optimizer_steps,
                response_presentations=self.response_presentations,error_type=type(error).__name__,error=str(error),
                completed_batches=batches, pending_presentations=pending_presentations,
                attempted_response_calls=attempted_response_calls,
                completed_backward_presentations=completed_backward_presentations,
                uncommitted_backward_presentations=max(0,completed_backward_presentations-2*self.optimizer_steps))
            self._record('FAILURES.jsonl',failure)
            error.training_partial=failure
            raise
        elapsed=time.perf_counter()-training_started
        self.completed_events=1;self.awaiting_checkpoint=True
        self.seen_record_ids.update(r['record_id'] for r in responses)
        presentations=[v for b in batches for v in b['presentations']]
        teacher_s=sum(v['phase_seconds']['teacher_preparation_s'] for v in presentations)
        return dict(status='passed',checkpoint_id='L8',episode_id=self.episode_id,protocol=dict(PROTOCOL),recipe=dict(RECIPE),
            starting_step=0,target_step=8,optimizer_steps_total=8,optimizer_steps_this_event=8,
            response_presentations_total=16,response_presentations_this_event=16,
            attempted_response_calls=attempted_response_calls,
            completed_backward_presentations=completed_backward_presentations,uncommitted_backward_presentations=0,
            response_order=[v['record_id'] for v in presentations],batches=batches,response_audits=audits,
            support_bindings=copy.deepcopy(self.support_bindings),support_sha256=self.support_sha256,
            feedback_checkpoint_sha256=self.released_checkpoint_sha256,original_exposure_match=True,
            anchor_rng_before_sha256=before,anchor_rng_after_sha256=curve.state_digest(self.generator.get_state()),
            anchor_exposure_sha256=digest([v['sampling_and_weights_sha256'] for v in presentations]),
            eligible_position_presentations_total=self.eligible_position_presentations,
            eligible_block_presentations_total=self.eligible_block_presentations,
            zero_supervision_presentations=sum(v['zero_supervision'] for v in presentations),
            retained_diagnostics_binding_sha256=digest(self.retained_training['source_losses']),
            source_diagnostic_steps=[],optimizer_state_exported=False,
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None,
            costs=dict(mask_audit_s=audit_s,training_s=elapsed,teacher_preparation_s=teacher_s,
                draft_training_and_optimizer_s=elapsed-teacher_s,training_event_wall_s=time.perf_counter()-started,
                accounting='training_s includes teacher, forward/backward, optimizer and receipts; do not add nested components'))

    def bind_l8_checkpoint(self,checkpoint_sha256):
        require(self.optimizer_steps==8 and self.completed_events==1 and self.awaiting_checkpoint,
                'Only the completed pending L8 checkpoint can be bound once')
        author.PersistentAuthorOSDTrainer.bind_checkpoint(self,1,checkpoint_sha256)

    def train_to_step(self,*args,**kwargs):
        raise ValueError('Use the explicit fresh train_to_step8 control path')

    def source_losses(self,*args,**kwargs):
        raise ValueError('Use read-only L8 loss evaluation against retained source anchors')

    def bind_checkpoint(self,*args,**kwargs):
        raise ValueError('Bind the final checkpoint explicitly as L8')

    def snapshot(self,*args,**kwargs):
        raise ValueError('Optimizer/master-state exports are outside this control')

    def restore(self,*args,**kwargs):
        raise ValueError('Each L8 group starts from released weights with fresh AdamW')
