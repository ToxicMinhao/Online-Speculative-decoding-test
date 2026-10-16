"""Per-response scheduling over the qualified AdaFlash OSD loss and sampler.

No new objective: only one-response/two-step scheduling, durable state, and an
isolated lower-LR diagnostic are added. The historical eight-response module is
loaded privately; its source bytes and all old experiments remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_per_turn_author_loss', HERE.parent / 'dflash_two_update_followup/author_training.py')
author = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(author)
author.OUTPUT_CAP = 1024
# Only response accumulation changes. All sampler, weighting and KL constants
# remain byte-identical to the qualified author-path implementation.
author.RECIPE = {**author.RECIPE, 'responses_per_step': 1, 'optimizer_steps_per_event': 2}
RECIPE = dict(author.RECIPE)
require, digest, audit_response = author.require, author.digest, author.audit_response


def rng_state():
    import torch
    result = {'python': random.getstate(), 'torch': torch.get_rng_state().clone()}
    try:
        import numpy as np
        result['numpy'] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        result['cuda'] = [v.clone() for v in torch.cuda.get_rng_state_all()]
    return result


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def state_digest(value):
    """Content identity for nested tensor/Python/NumPy state, independent of pickle."""
    import torch
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif hasattr(item, 'dtype') and hasattr(item, 'tobytes'):
            h.update(str((str(item.dtype), tuple(item.shape))).encode()); h.update(item.tobytes())
        else:
            h.update(repr(item).encode())
        h.update(b'\0')
    visit(value)
    return h.hexdigest()


def event_schedule(event, responses):
    require(type(event) is int and 1 <= event <= 23, 'Per-turn event must be a nonfinal turn')
    require(len(responses) == 1 and responses[0].get('turn') == event,
            'Each update uses only its own newest response')
    return [(0, responses), (1, responses)]


class PersistentPerTurnTrainer(author.PersistentAuthorOSDTrainer):
    def __init__(self, *args, **kwargs):
        import torch
        self.learning_rate = 3e-4
        random.seed(17)
        torch.manual_seed(17)
        try:
            import numpy as np
            np.random.seed(17)
        except ImportError:
            pass
        super().__init__(*args, **kwargs)

    def _check_owner(self):
        # Parent hard-codes 3e-4. Validate the complete same optimizer contract,
        # permitting 1e-4 exclusively inside the isolated diagnostic branch.
        require(id(self.optimizer) == self._optimizer_id and tuple(id(p) for p in self.draft.parameters()) ==
                self._parameter_ids and tuple(id(p) for g in self.optimizer.param_groups for p in g['params']) ==
                self._parameter_ids, 'Draft/optimizer identity changed')
        require(all(p.requires_grad for p in self.parameters.values()) and not author._has_adaptive_head(self.draft),
                'Full-weight/no-adaptive-head contract changed')
        for module in (self.embed_tokens, self.lm_head):
            require(not module.training and all(not p.requires_grad and p.grad is None and p.device == self.device
                                               for p in module.parameters()), 'Target components must remain frozen')
        require(len(self.optimizer.param_groups) == 1, 'Optimizer group count changed')
        group = self.optimizer.param_groups[0]
        require(self.learning_rate in (3e-4, 1e-4) and group['lr'] == self.learning_rate and
                group['betas'] == (0.9, 0.999) and group['eps'] == 1e-8 and group['weight_decay'] == .01,
                'Qualified AdamW recipe changed')
        states = self.optimizer.state
        require((not states and self.optimizer_steps == 0) or
                (set(states) == set(self.parameters.values()) and
                 {int(s['step'].item()) for s in states.values()} == {self.optimizer_steps}),
                'Adam history/counter changed')
        self.check_frozen()

    def train_event(self, event, responses):
        import torch
        require(not self.failed and not self.awaiting_checkpoint, 'Trainer failed or pending checkpoint binding')
        require(event == self.completed_events + 1, 'Updates must execute once in chronological order')
        self._check_owner()
        started = time.perf_counter()
        schedule = event_schedule(event, responses)
        response = responses[0]
        require(response.get('episode_id') == self.episode_id and response.get('target_id') == self.target_id and
                response.get('target_identity_sha256') == self.target_identity_sha256, 'Response/target ownership differs')
        require(response.get('capture_checkpoint_sha256') == self.active_checkpoint_sha256,
                'Training answer must come from current online checkpoint')
        require(response.get('record_id') not in self.seen_record_ids, 'Old response cannot be trained again')
        audit = audit_response(response)
        audit_seconds = time.perf_counter() - started
        batches = []
        initial_steps = self.optimizer_steps
        training_started = time.perf_counter()
        try:
            self.draft.train()
            for pass_index, _ in schedule:
                self.optimizer.zero_grad(set_to_none=True)
                batch_started = time.perf_counter()
                # Parent calls the unchanged author_response_loss and backward;
                # its private RECIPE now divides by effective response batch=1.
                value = self._train_response(response, audit)
                self._check_owner()
                self.synchronize()
                optimizer_started = time.perf_counter()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.parameters.values()),
                        'Missing/nonfinite full-weight gradient')
                norm = torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1., error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_steps += 1
                self._check_owner()
                require(all(bool(torch.isfinite(p).all()) for p in self.parameters.values()), 'Nonfinite draft update')
                self.synchronize()
                self.response_presentations += 1
                self.eligible_position_presentations += value['eligible_positions']
                self.eligible_block_presentations += value['eligible_blocks']
                batches.append(dict(pass_index=pass_index, optimizer_steps_after=self.optimizer_steps,
                                    presentations=[value], loss=value['loss'], gradient_norm_before_clip=float(norm),
                                    optimizer_s=time.perf_counter() - optimizer_started,
                                    wall_s=time.perf_counter() - batch_started))
        except BaseException:
            self.failed = True
            raise
        self.completed_events = event
        self.seen_record_ids.add(response['record_id'])
        self.awaiting_checkpoint = True
        training_seconds = time.perf_counter() - training_started
        presentations = [b['presentations'][0] for b in batches]
        teacher_seconds = sum(v['phase_seconds']['teacher_preparation_s'] for v in presentations)
        return dict(event=event, episode_id=self.episode_id, arm_name=author.ARM_NAME, author_commit=author.AUTHOR_COMMIT,
                    recipe={**RECIPE, 'learning_rate': self.learning_rate}, response_audits=[audit],
                    capture_checkpoint_sha256=self.active_checkpoint_sha256, support_turns=[event],
                    optimizer_steps_this_event=self.optimizer_steps-initial_steps, optimizer_steps_total=self.optimizer_steps,
                    response_presentations_this_event=2, response_presentations_total=self.response_presentations,
                    eligible_position_presentations_this_event=sum(v['eligible_positions'] for v in presentations),
                    eligible_block_presentations_this_event=sum(v['eligible_blocks'] for v in presentations),
                    response_order=[response['record_id']]*2, batches=batches,
                    anchor_exposure_sha256=digest([v['sampling_and_weights_sha256'] for v in presentations]),
                    costs=dict(mask_audit_s=audit_seconds, training_s=training_seconds,
                               teacher_preparation_s=teacher_seconds,
                               draft_training_and_optimizer_s=training_seconds-teacher_seconds,
                               training_event_wall_s=time.perf_counter()-started,
                               accounting='training_s includes teacher preparation; do not add twice'))

    def snapshot(self, *, include_draft=True):
        self._check_owner()
        require(not self.failed and not self.awaiting_checkpoint, 'Only complete bound states can be saved')
        result = dict(schema_version='dflash_per_turn_state_v1', episode_id=self.episode_id,
                      target_identity_sha256=self.target_identity_sha256,
                      released_checkpoint_sha256=self.released_checkpoint_sha256,
                      active_checkpoint_sha256=self.active_checkpoint_sha256, recipe=dict(RECIPE),
                      learning_rate=self.learning_rate, completed_events=self.completed_events,
                      optimizer_steps=self.optimizer_steps, response_presentations=self.response_presentations,
                      eligible_position_presentations=self.eligible_position_presentations,
                      eligible_block_presentations=self.eligible_block_presentations,
                      seen_record_ids=sorted(self.seen_record_ids), optimizer=self.optimizer.state_dict(),
                      anchor_rng_state=self.generator.get_state(), global_rng_state=rng_state(),
                      includes_draft=include_draft, draft_training=self.draft.training)
        if include_draft:
            result['draft'] = self.draft.state_dict()
        # Checkpoints must not alias live tensors that mutate during a branch.
        def clone(v):
            import torch
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().clone()
            if isinstance(v, dict):
                return {k: clone(x) for k, x in v.items()}
            if isinstance(v, (tuple, list)):
                return type(v)(clone(x) for x in v)
            return copy.deepcopy(v)
        return clone(result)

    def restore(self, state, *, draft_state=None):
        require(state['schema_version'] == 'dflash_per_turn_state_v1' and state['episode_id'] == self.episode_id and
                state['target_identity_sha256'] == self.target_identity_sha256 and
                state['released_checkpoint_sha256'] == self.released_checkpoint_sha256 and state['recipe'] == RECIPE,
                'Recovery state belongs to different episode/models/recipe')
        require(state['includes_draft'] or draft_state is not None, 'Explicit bound draft weights required')
        self.draft.load_state_dict(state['draft'] if state['includes_draft'] else draft_state, strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.optimizer.zero_grad(set_to_none=True)
        for name in ('learning_rate', 'completed_events', 'optimizer_steps', 'response_presentations',
                     'eligible_position_presentations', 'eligible_block_presentations', 'active_checkpoint_sha256'):
            setattr(self, name, state[name])
        self.seen_record_ids = set(state['seen_record_ids'])
        self.generator.set_state(state['anchor_rng_state'].cpu())
        self.draft.train(state['draft_training'])
        self.failed = self.awaiting_checkpoint = False
        restore_rng(state['global_rng_state'])
        self._check_owner()

    def save_state(self, path):
        import torch
        path = Path(path)
        require(not path.exists(), 'Never overwrite a recovery state')
        state = self.snapshot(include_draft=False)
        with path.open('xb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*2**20), b''):
                h.update(chunk)
        return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=h.hexdigest(),
                    state_identity_sha256=state_digest(state), includes_draft=False,
                    checkpoint_sha256=state['active_checkpoint_sha256'])
