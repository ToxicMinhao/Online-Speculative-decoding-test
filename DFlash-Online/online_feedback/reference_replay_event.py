"""Ordinary K updates on an explicit alternating learning/reference schedule.

KR changes response exposure and optimizer-step count, not the K loss. Each
scheduled pair enters the existing TrainerCore twice and advances the same Adam
state. Retained anchors select the blocks; ordinary sampler calls still advance
training RNG once per presentation. There is no reference model or correction.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import time

from .storage import write_json
from .training import SpecForgeEventTrainer

SCHEMA = 'ordinary_reference_replay_v1'
RNG_POLICY = ('Consume one ordinary shuffle per pass and one author anchor draw per '
              'presentation; use sealed order and anchors, with no RNG restoration.')


def expand_reference_schedule(reference):
    """Turn each sealed A/R window into two consecutive ordinary optimizer steps."""
    updates = []
    for window in reference['windows']:
        for role, source in (('A', 'learning'), ('R', 'reference')):
            updates.append(dict(
                optimizer_step=len(updates) + 1, outer_window=window['step'],
                role=role, pass_index=window['pass_index'], batch_index=window['batch_index'],
                source_pair_index=(window['batch_index'] if role == 'A'
                                   else window['reference_batch_index']),
                records=deepcopy(window[source]),
            ))
    return dict(schema=SCHEMA, arm='KR', passes=reference['passes'],
                features=deepcopy(reference['features']),
                distinct_responses=len(reference['features']),
                outer_windows=len(reference['windows']), optimizer_steps=len(updates),
                response_presentations=2 * len(updates), rng_policy=RNG_POLICY,
                updates=updates)


def diagnostic_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(item) for item in value]
    return value


class ReferenceReplayEventTrainer(SpecForgeEventTrainer):
    """A separate event schedule using the unchanged K strategy and optimizer."""

    def __init__(self, *args, **kwargs):
        self.replay_schedule = None
        self.replay_rows = []
        self.replay_counts = Counter()
        self.replay_passes = []
        self.replay_updates = []
        self.replay_costs = Counter()
        self.replay_status = 'not_started'
        self._active_replay = None
        self._last_update_identity = None
        self._active_binding = None
        self._draw_matches = []
        self._first_update_check = None
        super().__init__(*args, **kwargs)

    def train_replay_event(self, event_id, feature_paths, replay_schedule, *, first_update_check=None):
        """Run the sealed trajectory once; first-A checking uses its real forwards."""
        if (self.objective != 'osd_first_error' or self.reduction != 'response_mean'
                or self.accumulation != 2 or self.global_step or self.presentations or self.events):
            raise ValueError('KR requires fresh ordinary K with two-response averaging')
        if (replay_schedule['schema'] != SCHEMA or replay_schedule['arm'] != 'KR'
                or replay_schedule['passes'] != 2
                or replay_schedule['optimizer_steps'] > self.horizon):
            raise ValueError('KR schedule or inherited optimizer horizon differs')
        self.replay_schedule = deepcopy(replay_schedule)
        self._first_update_check = first_update_check
        self.contract['reference_replay_event'] = dict(
            arm='KR', schema=SCHEMA, ordinary_objective='osd_first_error',
            outer_windows=replay_schedule['outer_windows'],
            optimizer_steps=replay_schedule['optimizer_steps'],
            response_presentations=replay_schedule['response_presentations'],
            rng_policy=RNG_POLICY, correction=False, reference_model=False,
        )
        self._replay_directory = self.events_root / event_id
        self.replay_status = 'running'
        try:
            return super().train_event(event_id, feature_paths, passes=2)
        except BaseException as error:
            self.replay_status = 'failed'
            if self._replay_directory.exists():
                try:
                    write_json(self._replay_directory / 'REPLAY_FAILED.json', diagnostic_json(
                        dict(error_type=type(error).__name__, error=str(error), **self.replay_report())))
                except BaseException as reporting_error:
                    error.add_note('Could not preserve KR failure report: ' + str(reporting_error))
            raise
        finally:
            self._first_update_check = None
            self._active_binding = None
            self._active_replay = None

    def _prepare_event(self, event_id, feature_paths, passes):
        features = super()._prepare_event(event_id, feature_paths, passes)
        if self.replay_schedule is not None and self.replay_schedule['features'] != features:
            raise ValueError('KR feature paths or metadata differ from the sealed pool')
        return features

    def _sample_and_record_anchors(self, *args, **kwargs):
        if self._active_binding is None:
            return super()._sample_and_record_anchors(*args, **kwargs)
        import torch

        # Retain normal RNG advancement even though the selected blocks are sealed.
        drawn, drawn_keep = self._original_sampler(*args, **kwargs)
        self.replay_counts['anchor_rng_draws'] += 1
        binding, = self._active_binding['sampled_anchors']
        observation = dict(positions=drawn.detach().cpu().tolist(),
                           keep_mask=drawn_keep.detach().cpu().tolist())
        matches = observation == binding
        if self._active_replay['optimizer_step'] == 1 and not matches:
            raise ValueError('First A anchor RNG draw differs from retained K')
        anchors = torch.tensor(binding['positions'], device=drawn.device, dtype=drawn.dtype)
        keep = torch.tensor(binding['keep_mask'], device=drawn_keep.device, dtype=drawn_keep.dtype)
        if anchors.shape != drawn.shape or keep.shape != drawn_keep.shape:
            raise ValueError('Saved anchor geometry differs from the ordinary response sampler')
        self._anchor_observations.append(deepcopy(binding))
        self._draw_matches.append(matches)
        return anchors, keep

    @contextmanager
    def _observe_replay_work(self):
        """Count real native work; hooks do not retain activations or gradients."""
        backward = self.backend.backward
        adam = self.backend.optimizer.optimizer
        adam_step = adam.step
        handles = []

        def counted(operation, name, *args, **kwargs):
            role = self._active_replay['role']
            self.replay_counts[name + '_attempted'] += 1
            self.replay_counts[role + '_' + name + '_attempted'] += 1
            try:
                result = operation(*args, **kwargs)
            except BaseException:
                self.replay_counts[name + '_failed'] += 1
                self.replay_counts[role + '_' + name + '_failed'] += 1
                raise
            self.replay_counts[name + '_completed'] += 1
            self.replay_counts[role + '_' + name + '_completed'] += 1
            return result

        def before_forward(*args):
            self.replay_counts['response_forwards_attempted'] += 1
            self.replay_counts[self._active_replay['role'] + '_response_forwards_attempted'] += 1

        def after_forward(*args):
            self.replay_counts['response_forwards_completed'] += 1
            self.replay_counts[self._active_replay['role'] + '_response_forwards_completed'] += 1

        self.backend.backward = lambda *a, **k: counted(backward, 'backward', *a, **k)
        adam.step = lambda *a, **k: counted(adam_step, 'Adam', *a, **k)
        model = self.bundle.model
        handles.append(model.register_forward_pre_hook(before_forward))
        handles.append(model.register_forward_hook(after_forward))
        for module, name in [(model.draft_model, 'draft_forwards'),
                             (model.lm_head, 'head_forwards_including_recompute'),
                             *((layer, 'layer_forwards_including_recompute') for layer in model.draft_model.layers)]:
            def count_module(*args, key=name):
                self.replay_counts[key] += 1
            handles.append(module.register_forward_pre_hook(count_module))
        try:
            yield
        finally:
            self.backend.backward = backward
            adam.step = adam_step
            for handle in handles:
                handle.remove()

    def _train_passes(self, features, passes):
        import torch
        from specforge.training.strategies.base import StepContext

        paths = {row['record_id']: row['path'] for row in features}
        current_pass = None
        rows = self.replay_rows
        native_optimizer = self.backend.optimizer.optimizer
        with self._observe_replay_work():
            for update in self.replay_schedule['updates']:
                update_started = time.perf_counter()
                journal_before = self.replay_costs['journal_s']
                self._active_replay = update
                self._last_update_identity = {
                    key: update[key] for key in ('optimizer_step', 'outer_window', 'role', 'pass_index')}
                if update['pass_index'] != current_pass:
                    current_pass = update['pass_index']
                    draw = torch.randperm(len(features)).tolist()
                    self.replay_counts['shuffle_rng_draws'] += 1
                    drawn_ids = [features[index]['record_id'] for index in draw]
                    sealed_ids = [row['record_id'] for entry in self.replay_schedule['updates']
                                  if entry['pass_index'] == current_pass and entry['role'] == 'A'
                                  for row in entry['records']]
                    self.replay_passes.append(dict(pass_index=current_pass,
                                                  ordinary_shuffle_matches_sealed=drawn_ids == sealed_ids))
                    if current_pass == 0 and drawn_ids != sealed_ids:
                        raise ValueError('Initial ordinary shuffle differs from sealed K order')
                for response_index, binding in enumerate(update['records']):
                    self._active_binding = binding
                    loading_started = time.perf_counter()
                    raw = torch.load(paths[binding['record_id']], map_location='cpu', weights_only=True)
                    feature_load_s = time.perf_counter() - loading_started
                    self.replay_costs[update['role'] + '_feature_load_s'] += feature_load_s
                    self._anchor_observations.clear()
                    self._draw_matches.clear()
                    started = time.perf_counter()
                    learning_rate = self.backend.optimizer.get_learning_rate()
                    result = self.core.train_step(self._batch(raw), StepContext(
                        global_step=self.global_step, total_steps=self.horizon, collect_detailed_metrics=True))
                    metrics = result.materialize_metrics()
                    elapsed = time.perf_counter() - started
                    self.presentations += 1
                    self.global_step += int(result.optimizer_stepped)
                    self.replay_counts['presentations_completed'] += 1
                    self.replay_counts[update['role'] + '_presentations_completed'] += 1
                    row = dict(record_id=binding['record_id'], pass_index=current_pass,
                               outer_window=update['outer_window'], role=update['role'],
                               scheduled_optimizer_step=update['optimizer_step'],
                               optimizer_step_after=self.global_step, optimizer_stepped=result.optimizer_stepped,
                               learning_rate=learning_rate if result.optimizer_stepped else None,
                               metrics=metrics, training_s=elapsed, feature_load_s=feature_load_s,
                               sampled_anchors=list(self._anchor_observations),
                               anchor_rng_draw_matches_sealed=list(self._draw_matches),
                               metrics_scope='accumulation_window' if result.optimizer_stepped else 'presentation')
                    row['pass'] = current_pass
                    rows.append(row)
                    self.replay_costs[update['role'] + '_training_s'] += elapsed
                    self._journal(row)
                    if (result.optimizer_stepped != (response_index == 1)
                            or self.global_step != update['optimizer_step'] - int(response_index == 0)):
                        raise ValueError('KR optimizer boundary differs from its two-response schedule')
                    if self._anchor_observations != binding['sampled_anchors'] or len(self._draw_matches) != 1:
                        raise ValueError('KR requires exactly one saved anchor selection per presentation')
                    if not all(math.isfinite(float(value)) for value in metrics.values()):
                        raise ValueError('Nonfinite ordinary K training metric')
                self.replay_counts['updates_completed'] += 1
                if self.backend.optimizer.optimizer is not native_optimizer:
                    raise ValueError('KR must retain one continuous native optimizer')
                step_counter = int(next(iter(native_optimizer.state.values()))['step'])
                if step_counter != update['optimizer_step']:
                    raise ValueError('Native Adam counter differs from the KR trajectory')
                update_wall_s = time.perf_counter() - update_started
                journal_s = self.replay_costs['journal_s'] - journal_before
                self.replay_costs[update['role'] + '_update_wall_s'] += update_wall_s
                self.replay_updates.append(dict(
                    optimizer_step=update['optimizer_step'], outer_window=update['outer_window'],
                    role=update['role'], native_Adam_step=step_counter,
                    record_ids=[row['record_id'] for row in update['records']],
                    update_wall_s=update_wall_s, journal_s=journal_s,
                    wall_scope='Feature loading, ordinary K work, metric materialization and row journal; first-A gate excluded.',
                ))
                if update['role'] == 'R':
                    self.replay_counts['outer_windows_completed'] += 1
                if update['optimizer_step'] == 1 and self._first_update_check is not None:
                    checked = time.perf_counter()
                    try:
                        self._first_update_check(deepcopy(rows[:2]))
                        self.replay_counts['first_A_checks_completed'] += 1
                    finally:
                        self.replay_costs['first_update_check_s'] += time.perf_counter() - checked
                if update['optimizer_step'] == 2:
                    self._record_first_window()
                write_json(self._replay_directory / 'REPLAY_PROGRESS.json', self.replay_report())
        counters = [int(state['step']) for state in native_optimizer.state.values()]
        if not counters or any(step != self.global_step for step in counters):
            raise ValueError('Final native Adam counters do not match the complete KR event')
        self.replay_status = 'completed'
        return rows

    def _record_first_window(self):
        """Validate the first real A/R pair, using existing rows and counters."""
        expected = dict(Adam_completed=2, backward_completed=4,
                        response_forwards_completed=4, presentations_completed=4,
                        A_Adam_completed=1, R_Adam_completed=1,
                        updates_completed=2, outer_windows_completed=1)
        actual = {key: self.replay_counts[key] for key in expected}
        passed = actual == expected and [row['role'] for row in self.replay_rows] == ['A', 'A', 'R', 'R']
        receipt = dict(status='passed' if passed else 'failed', actual_work=actual,
                       expected_work=expected, native_Adam_steps=[row['native_Adam_step'] for row in self.replay_updates],
                       first_A_record_gate_passed=self.replay_counts['first_A_checks_completed'] == 1,
                       records=[{key: row[key] for key in (
                           'record_id', 'pass', 'role', 'outer_window', 'sampled_anchors',
                           'optimizer_stepped', 'optimizer_step_after')} for row in self.replay_rows],
                       continuous_optimizer=True, rng_restored_between_updates=False,
                       additional_forwards=0, additional_optimizer_steps=0)
        write_json(self._replay_directory / 'FIRST_AR_WINDOW.json', receipt)
        if not passed:
            raise ValueError('First complete A/R window did not reconcile')

    def _journal(self, row):
        started = time.perf_counter()
        with (self._replay_directory / 'REPLAY_PRESENTATIONS.jsonl').open('a') as stream:
            stream.write(json.dumps(diagnostic_json(row), allow_nan=False) + '\n')
            stream.flush()
        self.replay_costs['journal_s'] += time.perf_counter() - started

    def replay_report(self):
        active = self._active_replay
        active_identity = self._last_update_identity if active is None else {
            key: active[key] for key in ('optimizer_step', 'outer_window', 'role', 'pass_index')}
        return dict(arm='KR', schema=SCHEMA, status=self.replay_status, rng_policy=RNG_POLICY,
                    counts=dict(self.replay_counts), costs=dict(self.replay_costs),
                    pass_draws=list(self.replay_passes), updates=list(self.replay_updates),
                    global_step=self.global_step,
                    presentations=self.presentations,
                    active_update=active_identity,
                    correction_steps=0, protected_capture=False, reference_model=False)

    def _checkpoint_state(self, event):
        event['training_variant'] = 'KR'
        event['reference_replay'] = self.replay_report()
        return super()._checkpoint_state(event)
