"""KW-v1: observe ordinary K training and allocate its two native gradients.

The upstream forward/backward, response reduction, DDP wrapper and native
optimizer remain in use. A small observer captures one StepOutput per response;
the optimizer wrapper performs isolated previews before one live AdamW step.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch

from .training import SpecForgeEventTrainer, SCHEMA
from .rng import restore_rng
from .response_weighting import (
    KW_CONTRACT, ResponseWeightController, HalfGradientCapture,
    response_features, apply_weighted_gradients_, optimizer_secant, gradient_norm,
)


CONTRACT_KEY = 'response_weight_controller'


def validate_kw_snapshot_metadata(state, *, completed_event):
    """KW has one migration at K8; later restores require its whole controller."""
    if completed_event not in range(9, 13):
        raise ValueError('A KW snapshot must finish event 9..12')
    if state['algorithm_contract'].get(CONTRACT_KEY) != KW_CONTRACT:
        raise ValueError('KW algorithm contract differs')
    controller = ResponseWeightController()
    controller.load_state_dict(state['kw_controller'])
    events = completed_event - 8
    if (controller.n != events * 126 or controller.windows != events * 128
            or controller.previews != events * 256):
        raise ValueError('KW controller counts differ from completed event boundary')
    for number, event in enumerate(state['events'], 1):
        if number <= 8:
            if 'kw' in event:
                raise ValueError('Original K8 history cannot contain KW updates')
            continue
        recorded = event.get('kw', {})
        if (recorded.get('contract') != KW_CONTRACT or recorded.get('windows') != 128
                or recorded.get('previews') != 256 or recorded.get('credit_opportunities') != 126):
            raise ValueError('KW event history has inconsistent controller accounting')
    migration = state['kw_migration']
    if (migration['from_arm'] != 'K' or migration['completed_event'] != 8
            or migration['global_step'] != 1024 or migration['presentations'] != 2048
            or len(migration['source_sha256']) != 64):
        raise ValueError('KW migration must identify the original complete K8')
    return {'status': 'passed', 'controller_windows': controller.windows,
            'credit_opportunities': controller.n, 'optimizer_previews': controller.previews}


def verify_kw_live_resume(trainer, state, *, receipt_path=None):
    """Verify the same scientific state, plus the explicit KW metadata transition."""
    from experiments.ok_continuation.resume import verify_live_resume, differences
    from .storage import write_json
    compared = dict(state)
    if trainer.completed_event == 8:
        base_contract = dict(trainer.contract)
        if base_contract.pop(CONTRACT_KEY, None) != KW_CONTRACT or base_contract != state['algorithm_contract']:
            raise ValueError('K8 migration changes an existing scientific contract')
        compared['algorithm_contract'] = trainer.contract
        expected_controller = ResponseWeightController().state_dict()
        if differences(trainer.kw_checkpoint_state(), expected_controller):
            raise ValueError('K8 migration must start with exactly zero controller state')
    else:
        validate_kw_snapshot_metadata(state, completed_event=trainer.completed_event)
        if differences(trainer.kw_checkpoint_state(), state['kw_controller']):
            raise ValueError('KW controller did not restore exactly')
        if differences(trainer.kw_migration, state['kw_migration']):
            raise ValueError('KW original migration identity changed')
    receipt = verify_live_resume(trainer, compared)
    receipt.update(algorithm_arm='KW', controller_exact=True,
                   migration_from_K8=trainer.completed_event == 8)
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    return receipt


class _ObservedStrategy:
    """Observe original response output, outside checkpointed recomputation."""
    def __init__(self, strategy, owner):
        self.strategy, self.owner = strategy, owner

    def __getattr__(self, name):
        return getattr(self.strategy, name)

    def forward_loss(self, batch, ctx=None):
        output = self.strategy.forward_loss(batch, ctx)
        feature = response_features(output.loss, output.sum_metrics, self.owner._response_length)
        self.owner._features.append(feature)
        self.owner._losses.append(float(output.loss.detach().float().cpu()))
        return output


class _ObservedCore:
    """Keep TrainerCore's native accumulation and boundary normalization."""
    def __init__(self, core, owner):
        self.core, self.owner = core, owner

    def __getattr__(self, name):
        return getattr(self.core, name)

    def train_step(self, batch, ctx=None):
        owner = self.owner
        if self.core.accumulation_remainder == 0:
            owner._begin_window()
        owner._response_ids.append(owner._response_id)
        owner._capture.begin_response()
        result = self.core.train_step(batch, ctx)
        if not result.optimizer_stepped:
            owner._contributions.append(owner._capture.end_response())
        return result


class _ObservedOptimizer:
    """Leave native state serialization unchanged; KW state lives beside it."""
    def __init__(self, optimizer, owner):
        self.native, self.owner = optimizer, owner

    def __getattr__(self, name):
        return getattr(self.native, name)

    def step(self):
        return self.owner._weighted_step(self.native)


class KWContinuationTrainer(SpecForgeEventTrainer):
    def __init__(self, config_path, output_root, *, resume_state, completed_event, arm='KW'):
        if arm != 'KW' or completed_event not in range(8, 12) or resume_state is None:
            raise ValueError('KW requires its complete boundary 8..11')
        self.completed_event = completed_event
        self.arm = arm
        self.controller = ResponseWeightController()
        self.qualification_observer = None
        self.optimizer_application_observer = None
        self.window_rows = []
        self._features, self._losses, self._contributions, self._response_ids = [], [], [], []
        self._window_log = None
        self.work_attempts = dict(previews=0, live_steps=0, credits=0)
        super().__init__(config_path, output_root, resume_state=resume_state,
                         reduction='response_mean', objective='osd_first_error', ranking_lambda=0)
        if self.core.accumulation_remainder or self.horizon != 7424:
            raise ValueError('KW must begin at a qualified complete accumulation boundary')
        self.named_parameters = tuple((name, p) for name, p in
                                      self.bundle.draft_model.named_parameters() if p.requires_grad)
        expected_names = tuple(row[0] for row in self.backend.optimizer.recipe['parameters'])
        if tuple(name for name, _ in self.named_parameters) != expected_names:
            raise ValueError('KW trainable parameter ordering differs from native optimizer')
        self._capture = HalfGradientCapture(self.named_parameters)
        self.core.strategy = _ObservedStrategy(self.core.strategy, self)
        self.core = _ObservedCore(self.core, self)
        self.backend.optimizer = _ObservedOptimizer(self.backend.optimizer, self)

    def _restore(self, path):
        from experiments.kw_v1.resume import validate_snapshot
        from experiments.ok_continuation.resume import verify_live_resume

        state = torch.load(path, map_location='cpu', weights_only=False)
        migrating = self.completed_event == 8
        validate_snapshot(state, objective='osd_first_error', config_text=self.config_text,
                          completed_event=self.completed_event,
                          algorithm_arm='K' if migrating else 'KW')
        if not migrating:
            self.contract[CONTRACT_KEY] = deepcopy(KW_CONTRACT)
        if (state['event_schema'] != SCHEMA or state['algorithm_contract'] != self.contract
                or state['config_text'] != self.config_text or state['reduction'] != self.reduction
                or state['horizon'] != self.horizon or state['accumulation_steps'] != self.accumulation):
            raise ValueError('KW restore scientific contract differs')
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        self.backend.load_state_dict(state['backend'])
        restore_rng(state['rng'])
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = state['events']
        self.core._micro = self.presentations
        # This comparison occurs before the sole K8->KW metadata migration.
        self.restore_receipt = verify_live_resume(self, state)
        if migrating:
            with Path(path).open('rb') as stream:
                source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            self.kw_migration = dict(from_arm='K', completed_event=8, global_step=1024,
                                     presentations=2048, source_state=str(path), source_sha256=source_hash)
            self.contract[CONTRACT_KEY] = deepcopy(KW_CONTRACT)
        else:
            self.controller.load_state_dict(state['kw_controller'])
            self.kw_migration = deepcopy(state['kw_migration'])

    def _prepare_event(self, event_id, feature_paths, passes):
        expected = self.completed_event + 1
        if (event_id != f'event_{expected:03d}' or len(feature_paths) != 128 or passes != 2
                or self.global_step != self.completed_event * 128
                or self.presentations != self.completed_event * 256
                or len(self.events) != self.completed_event or self.controller.pending is not None):
            raise ValueError('KW permits exactly its next 128-response, two-pass event')
        self._event_start_presentations = self.presentations
        self._window_log = self.events_root / event_id / 'KW_WINDOWS.jsonl'
        self.window_rows = []
        return super()._prepare_event(event_id, feature_paths, passes)

    def _batch(self, raw):
        self._response_id = raw['record_id']
        self._response_length = raw['input_ids'].shape[-1] - int(raw['prompt_length'])
        return super()._batch(raw)

    def _begin_window(self):
        self._phi_entry = self.controller.phi.clone()
        self._features, self._losses, self._contributions, self._response_ids = [], [], [], []
        self._window_started = time.perf_counter()
        self._capture_host_start = self._capture.capture_host_s

    def _observe(self, stage, payload):
        if self.qualification_observer is not None:
            self.qualification_observer(stage, self, payload)

    def _work(self, kind, phase, **identity):
        if phase == 'started':
            self.work_attempts[kind] += 1
        if self._window_log is not None:
            with self._window_log.with_name('KW_WORK.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(kind=kind, phase=phase, **identity), allow_nan=False)+'\n')
                stream.flush()

    def _weighted_step(self, native):
        preoptimizer_wall_s = time.perf_counter() - self._window_started
        self._contributions.append(self._capture.end_response())
        if len(self._features) != 2 or len(self._contributions) != 2:
            raise ValueError('KW requires exactly two separate ordinary response observations')
        names = tuple(name for name, _ in self.named_parameters)
        gradients = [p.grad for _, p in self.named_parameters]
        first, second = self._contributions
        started = time.perf_counter()
        allocation = self.controller.allocate(*self._features, phi_entry=self._phi_entry)
        row_id = dict(event=self.completed_event + 1,
                      pass_index=(self.presentations - self._event_start_presentations) // 128,
                      window_in_pass=((self.presentations - self._event_start_presentations) % 128) // 2 + 1,
                      global_step=self.global_step + 1, response_ids=list(self._response_ids),
                      anchor_binding_source='RESULT.json.records',
                      presentation_ordinals=[self.presentations-self._event_start_presentations-1,
                                             self.presentations-self._event_start_presentations])
        self._observe('pre_step', dict(native=gradients, first=first, second=second,
                                     allocation=allocation, identity=row_id))
        ordinary_norm = gradient_norm(gradients)
        has_credit = self.controller.pending is not None
        if has_credit:
            self._work('credits', 'started', **row_id)
        credit = self.controller.credit(gradients, names)
        if has_credit:
            self._work('credits', 'completed', **row_id)
        credit_s = time.perf_counter() - started
        preview_started = time.perf_counter()
        def preview_attempt(a):
            self._work('previews', 'started', allocation=a, **row_id)
            if self.optimizer_application_observer is not None:
                self.optimizer_application_observer(a)
        secant = optimizer_secant(native.optimizer, self.named_parameters, gradients, first, second,
                                  allocation.a, max_grad_norm=native.max_grad_norm,
                                  on_optimizer_step=preview_attempt)
        self._work('previews', 'completed_pair', **row_id)
        preview_s = time.perf_counter() - preview_started
        weighting_started = time.perf_counter()
        apply_weighted_gradients_(self.named_parameters, gradients, first, second, allocation.a)
        weighted_norm = gradient_norm([p.grad for _, p in self.named_parameters])
        weighting_s = time.perf_counter() - weighting_started
        update_started = time.perf_counter()
        self._work('live_steps', 'started', **row_id)
        if self.optimizer_application_observer is not None:
            self.optimizer_application_observer(allocation.a)
        native_norm = native.step()
        self._work('live_steps', 'completed', **row_id)
        from .response_weighting import _finite
        for name, parameter in self.named_parameters:
            _finite(parameter, 'live parameter ' + name)
            for key, value in native.optimizer.state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor):
                    _finite(value, 'live Adam ' + name + '.' + key)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        update_s = time.perf_counter() - update_started
        self._observe('post_step', dict(identity=row_id, allocation=allocation))
        end_of_pass = row_id['window_in_pass'] == 64
        self.controller.save_pending(secant.eligibility, allocation.b, row_id,
                                     names=names, end_of_pass=end_of_pass)
        row = dict(**row_id, features=[z.tolist() for z in self._features],
                   ordinary_response_losses=list(self._losses), a=allocation.a,
                   weights=list(allocation.weights), phi_entry=self._phi_entry.tolist(),
                   phi_after=self.controller.phi.tolist(), cached_b=allocation.b.tolist(),
                   credit=credit, rms_v=self.controller.v, credits_n=self.controller.n,
                   ordinary_gradient_norm=ordinary_norm, weighted_gradient_norm=weighted_norm,
                   native_preclip_norm=float(native_norm.cpu()), **secant.stats,
                   allocation_saturated=abs(allocation.a) == .25,
                   secant_endpoint_clipped=abs(allocation.a) + 1/16 > .25,
                   live_gradient_clipped=float(native_norm.cpu()) > native.max_grad_norm,
                   pending_discarded_at_pass_end=end_of_pass,
                   costs=dict(ordinary_training_s=preoptimizer_wall_s,
                              capture_host_s=self._capture.capture_host_s-self._capture_host_start,
                              credit_and_controller_s=credit_s, previews_s=preview_s,
                              weighting_s=weighting_s, live_update_s=update_s,
                              window_observed_s=time.perf_counter()-self._window_started),
                   memory=dict(allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,
                               reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0))
        self.window_rows.append(row)
        if self._window_log is not None:
            with self._window_log.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
        self._features, self._losses, self._contributions, self._response_ids = [], [], [], []
        return native_norm

    def kw_checkpoint_state(self):
        return self.controller.state_dict()

    def _checkpoint_state(self, event):
        if len(self.window_rows) != 128 or self.controller.pending is not None:
            raise ValueError('KW cannot save incomplete event/controller history')
        event['kw'] = dict(contract=deepcopy(KW_CONTRACT), windows=len(self.window_rows),
                           previews=2*len(self.window_rows), credit_opportunities=126,
                           controller_final=dict(phi=self.controller.phi.tolist(), v=self.controller.v,
                                                 n=self.controller.n),
                           log=str(self._window_log),
                           costs={name:sum(row['costs'][name] for row in self.window_rows)
                                  for name in self.window_rows[0]['costs']})
        state = super()._checkpoint_state(event)
        state['kw_controller'] = self.kw_checkpoint_state()
        state['kw_migration'] = deepcopy(self.kw_migration)
        validate_kw_snapshot_metadata(state, completed_event=self.completed_event+1)
        return state

    def status(self):
        return {**super().status(), 'algorithm_arm': 'KW',
                'kw_work_attempts': dict(self.work_attempts)}

    def close(self):
        if hasattr(self, '_capture'):
            self._capture.close()
        self.controller.clear_pending()
        self._features, self._losses, self._contributions, self._response_ids = [], [], [], []
        super().close()
