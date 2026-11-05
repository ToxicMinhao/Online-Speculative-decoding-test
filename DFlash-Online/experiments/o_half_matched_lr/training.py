"""Ordinary O mathematics with exact source-state migration and schedule checks.

This wrapper does not inherit the K-specific protection trainer. The native
model, loss, sampling, accumulation, optimizer and exporter remain unchanged.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from experiments.ok_coding import training as original
from experiments.ok_continuation import resume
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.training import SpecForgeEventTrainer, SCHEMA
from .contract import ARM, SEEDS, LEARNING_RATE, OBJECTIVE, CONTRACT_KEY


def experiment_contract(seed, binding):
    if type(seed) is not int or seed not in SEEDS or not isinstance(binding, dict) or not binding:
        raise ValueError('Require the declared O_HALF seed and sealed binding')
    return dict(schema=CONTRACT_KEY, arm=ARM, seed=seed,
                learning_rate=LEARNING_RATE, base_objective=OBJECTIVE,
                initialization='original_O_seed_step_zero',
                binding=json.loads(json.dumps(binding, allow_nan=False)),
                maximum_completed_events=4, reference=None, ranking_lambda=0.)


def validate_config_text(config_text, seed, original_config_text):
    """Permit only LR and output identity changes from the actual O config."""
    import yaml
    current = yaml.safe_load(config_text)
    source = yaml.safe_load(original_config_text)
    original.validate_config_text(original_config_text, seed)
    if current['training']['learning_rate'] != LEARNING_RATE:
        raise ValueError('O_HALF requires the fixed constant learning rate1.5e-4')
    checked = deepcopy(current)
    checked['training']['learning_rate'] = 3e-4
    for key in ('run_id', 'output_dir'):
        if key in source:
            checked[key] = source[key]
        else:
            checked.pop(key, None)
    if resume.differences(checked, source):
        raise ValueError('O_HALF configuration changed beyond LR/run/output identity')
    original.validate_config_text(yaml.safe_dump(checked), seed)


def _recipe_json(recipe):
    # PLAN uses JSON lists; native Adam uses tuples. Keep parameter order exact.
    return json.loads(json.dumps(recipe, allow_nan=False))


def native_algorithm_contract(transport):
    """Restore only the pinned provider's tuple fields after PLAN JSON transport.

    JSON lists elsewhere (for example coding-event bindings) remain lists.
    Actual source, live and saved contracts are never normalized or rewritten;
    they still pass the historical exact, type-sensitive comparison.
    """
    def tuples(value):
        if isinstance(value, (list, tuple)):
            return tuple(tuples(item) for item in value)
        if isinstance(value, dict):
            return {key: tuples(item) for key, item in value.items()}
        return value

    native = deepcopy(transport)
    for key in ('dflash_target_layer_ids', 'specforge_step_options', 'specforge_model_provenance'):
        if key in native:
            if not isinstance(native[key], (list, tuple)):
                raise ValueError('Malformed tuple-valued original contract field: ' + key)
            native[key] = tuples(native[key])
    return native


def _optimizer_view(optimizer, learning_rate):
    """Copy only small mutable containers; never duplicate model-sized moments."""
    recipe = deepcopy(optimizer['recipe'])
    recipe['optimizer_options']['lr'] = learning_rate
    state = optimizer['optimizer_state_dict']
    groups = deepcopy(state['param_groups'])
    if len(groups) != 1:
        raise ValueError('Require one native Adam group')
    groups[0]['lr'] = learning_rate
    return dict(recipe=recipe, optimizer_state_dict=dict(state, param_groups=groups))


def validate_snapshot(state, *, seed, binding, original_contract,
                      original_algorithm_contract, original_optimizer_recipe,
                      original_config_text, config_text, completed_event,
                      expected_cuda_rng_states=1):
    """Validate own state with a read-only LR-normalized historical O view."""
    if type(completed_event) is not int or completed_event not in range(5):
        raise ValueError('An O_HALF boundary must be event0..4')
    validate_config_text(config_text, seed, original_config_text)
    original_algorithm_contract = native_algorithm_contract(original_algorithm_contract)
    expected = deepcopy(original_algorithm_contract)
    expected[CONTRACT_KEY] = experiment_contract(seed, binding)
    if resume.differences(state['algorithm_contract'], expected):
        raise ValueError('Foreign arm, objective, mechanism or qualification contract')
    if state['config_text'] != config_text:
        raise ValueError('Own saved configuration identity differs')
    required = {'event_schema', 'draft_state_dict', 'backend', 'rng', 'global_step',
        'presentations', 'events', 'strategy', 'run_id', 'world_size', 'config_text',
        'algorithm_contract', 'reduction', 'accumulation_steps', 'horizon'}
    if set(state) != required or set(state['backend']) != {'optimizer'}:
        raise ValueError('Unexpected or missing scientific state fields')
    for event in state['events']:
        if event.get('arm', ARM) != ARM:
            raise ValueError('Foreign arm in own event history')
    optimizer = state['backend']['optimizer']
    expected_recipe = deepcopy(original_optimizer_recipe)
    expected_recipe['optimizer_options']['lr'] = LEARNING_RATE
    if _recipe_json(optimizer['recipe']) != _recipe_json(expected_recipe):
        raise ValueError('Own native optimizer recipe or parameter ordering differs')
    groups = optimizer['optimizer_state_dict']['param_groups']
    if len(groups) != 1 or groups[0]['lr'] != LEARNING_RATE:
        raise ValueError('Own native Adam group learning rate differs')
    # The old gate remains unchanged and still rejects actual half-LR states.
    # Only this already-checked temporary view uses the historical LR/contract.
    historical = dict(state, config_text=original_config_text,
                      algorithm_contract=deepcopy(original_algorithm_contract),
                      backend={'optimizer': _optimizer_view(optimizer, 3e-4)})
    checked = original.validate_snapshot(historical, arm='O', seed=seed,
        campaign_contract=original_contract, config_text=original_config_text,
        completed_event=completed_event, expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(checked, arm=ARM, seed=seed, exact_contract=True,
                adam_state_entries=len(optimizer['optimizer_state_dict']['state']),
                qualification_steps=0, reference_operations=0)


def migrate_source(source, *, seed, binding, original_contract,
                   original_algorithm_contract, original_optimizer_recipe,
                   original_config_text, config_text, new_contract,
                   expected_cuda_rng_states=1):
    """Change only the declared LR and metadata; source remains immutable."""
    original.validate_snapshot(source, arm='O', seed=seed,
        campaign_contract=original_contract, config_text=original_config_text,
        completed_event=0, expected_cuda_rng_states=expected_cuda_rng_states)
    original_algorithm_contract = native_algorithm_contract(original_algorithm_contract)
    expected = deepcopy(original_algorithm_contract)
    expected[CONTRACT_KEY] = experiment_contract(seed, binding)
    if (resume.differences(source['algorithm_contract'], original_algorithm_contract)
            or resume.differences(expected, new_contract)
            or _recipe_json(source['backend']['optimizer']['recipe']) != _recipe_json(original_optimizer_recipe)):
        raise ValueError('Original O identity or runtime contract differs')
    candidate = dict(source, config_text=config_text, algorithm_contract=deepcopy(new_contract),
                     backend={'optimizer': _optimizer_view(source['backend']['optimizer'], LEARNING_RATE)})
    validate_snapshot(candidate, seed=seed, binding=binding, original_contract=original_contract,
        original_algorithm_contract=original_algorithm_contract,
        original_optimizer_recipe=original_optimizer_recipe,
        original_config_text=original_config_text, config_text=config_text,
        completed_event=0, expected_cuda_rng_states=expected_cuda_rng_states)
    fields = ('draft_state_dict', 'rng', 'global_step', 'presentations', 'events')
    if resume.differences({k: source[k] for k in fields}, {k: candidate[k] for k in fields}):
        raise ValueError('Initialization changed scientific weights, RNG or history')
    return candidate, dict(status='passed', original_arm='O', original_seed=seed,
        original_steps=0, source_weights_rng_moments_counters_history_exact=True,
        changes=['additive O_HALF contract', 'configuration/run/output identity',
                 'learning rate3e-4→1.5e-4 in recipe and native Adam group'],
        qualification_only=False, no_reseed_after_restore=True)


def live_state(trainer):
    full = trainer.backend.state_dict()
    return dict(event_schema=SCHEMA,
        draft_state_dict=trainer.strategy.checkpoint_state_filter(full['model']),
        backend={'optimizer': full['optimizer']}, rng=capture_rng(),
        global_step=trainer.global_step, presentations=trainer.presentations,
        events=trainer.events, strategy='dflash', run_id=trainer.cfg.run_id, world_size=1,
        config_text=trainer.config_text, algorithm_contract=trainer.contract,
        reduction=trainer.reduction, accumulation_steps=trainer.accumulation, horizon=trainer.horizon)


def verify_live(trainer, expected, *, receipt_path=None):
    """Check a genuine empty accumulation boundary, including post-event state.

    The last anchor observation is retained metadata, not pending gradients.
    Unlike an initial-resume gate, a completed-event check must allow that row.
    """
    from online_feedback.storage import write_json
    started = time.perf_counter()
    actual = live_state(trainer)
    fields = ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations',
              'events', 'config_text', 'algorithm_contract', 'reduction',
              'accumulation_steps', 'horizon')
    mismatch = resume.differences({k: actual[k] for k in fields}, {k: expected[k] for k in fields})
    model = trainer.bundle.model
    clean = (not trainer.closed and not trainer.failed
        and trainer.core._micro == expected['presentations']
        and trainer.core.accumulation_remainder == 0
        and not trainer.core._ratio_totals and not trainer.core._sum_totals
        and trainer._presentation_schedule is None and model.training
        and model.objective == OBJECTIVE and model.reference is None and not model.ranking_enabled
        and all(p.grad is None for p in trainer.bundle.draft_model.parameters()))
    if not clean:
        mismatch.append('runtime_boundary_or_objective_or_reference_or_gradients')
    receipt = dict(status='passed' if not mismatch else 'failed', exact_resume=not mismatch,
                   mismatches=mismatch[:32], global_step=trainer.global_step,
                   presentations=trainer.presentations, reference_operations=0,
                   no_rng_reset=True, wall_s=time.perf_counter() - started)
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    if mismatch:
        raise ValueError('Own-state restoration differs: ' + str(mismatch[:8]))
    return receipt


class OHalfTrainer(SpecForgeEventTrainer):
    """One own-state ordinary O event, checked against naturally sampled history."""

    def __init__(self, config_path, output_root, *, seed, binding, original_contract,
                 original_algorithm_contract, original_optimizer_recipe,
                 original_config_text, resume_state=None, source_state=None,
                 completed_event=0, expected_cuda_rng_states=1):
        if type(completed_event) is not int or completed_event not in range(4):
            raise ValueError('Start only at O_HALF boundary0..3')
        if ((completed_event == 0) != (source_state is not None)
                or (completed_event > 0) != (resume_state is not None)):
            raise ValueError('Require original O initialization or own complete continuation')
        self.seed, self.binding = seed, deepcopy(binding)
        self.original_contract = deepcopy(original_contract)
        self.original_algorithm_contract = native_algorithm_contract(original_algorithm_contract)
        self.original_optimizer_recipe = deepcopy(original_optimizer_recipe)
        self.original_config_text = original_config_text
        self.completed_event = completed_event
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.response_records = []
        self.response_observer_path = None
        self.work = {}
        self._work_handles = []
        super().__init__(config_path, output_root, resume_state=resume_state,
                         reduction='response_mean', objective=OBJECTIVE, ranking_lambda=0.)
        if source_state is not None:
            migrated, self.migration_receipt = migrate_source(source_state,
                **self.validation_arguments(), new_contract=self.contract)
            self._load_decoded(migrated)
            verify_live(self, migrated)
        if self.global_step != completed_event * 128 or self.presentations != completed_event * 256:
            raise ValueError('Own restored boundary counters differ')

    def validation_arguments(self):
        return dict(seed=self.seed, binding=self.binding, original_contract=self.original_contract,
                    original_algorithm_contract=self.original_algorithm_contract,
                    original_optimizer_recipe=self.original_optimizer_recipe,
                    original_config_text=self.original_config_text, config_text=self.config_text,
                    expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _validate_config(self):
        super()._validate_config()
        validate_config_text(self.config_text, self.seed, self.original_config_text)

    def _build_runtime(self):
        super()._build_runtime()
        self.contract[original.CONTRACT_KEY] = original.coding_contract(self.seed, self.original_contract)
        if resume.differences(self.contract, self.original_algorithm_contract):
            raise ValueError('Ordinary O runtime mathematics/recipe contract differs from source')
        self.contract[CONTRACT_KEY] = experiment_contract(self.seed, self.binding)
        model = self.bundle.model
        if model.objective != OBJECTIVE or model.reference is not None or model.ranking_enabled:
            raise ValueError('O_HALF must route to ordinary all-valid forward KL')
        def count(name):
            def observe(_module, _inputs):
                self.work[name] = self.work.get(name, 0) + 1
            return observe
        self._work_handles.append(model.draft_model.register_forward_pre_hook(count('current_draft_forwards')))
        for layer in model.draft_model.layers:
            self._work_handles.append(layer.register_forward_pre_hook(count('current_layer_forwards')))
        self._work_handles.append(model.lm_head.register_forward_pre_hook(count('vocabulary_head_calls')))
        native_chunk = model._chunk_loss
        native_end = model.end_event
        native_optimizer_step = self.backend.optimizer.step
        self._native_chunk, self._native_end = native_chunk, native_end
        self._native_optimizer_step = native_optimizer_step
        def counted_chunk(*args, **kwargs):
            self.work['objective_chunks'] = self.work.get('objective_chunks', 0) + 1
            return native_chunk(*args, **kwargs)
        def observed_end():
            return dict(native_end(), work=dict(self.work), reference_operations=0,
                        optimizer_step_host_s=self.optimizer_step_host_s,
                        response_observation_s=self.response_observation_s,
                        response_record_write_s=self.response_record_write_s,
                        optimizer_timer_scope='Host call including native clip/Adam/clear; not a kernel profile')
        def timed_optimizer_step():
            started = time.perf_counter()
            try:
                return native_optimizer_step()
            finally:
                self.optimizer_step_host_s += time.perf_counter() - started
        self.optimizer_step_host_s = 0.
        self.response_observation_s = 0.
        self.response_record_write_s = 0.
        self.backend.optimizer.step = timed_optimizer_step
        model._chunk_loss, model.end_event = counted_chunk, observed_end
        native_forward = self.author_strategy.forward_loss
        def observed_forward(batch, ctx=None):
            output = native_forward(batch, ctx)
            observed_started = time.perf_counter()
            def scalar(value):
                return float(value.detach().item()) if hasattr(value, 'detach') else float(value)
            record = dict(record_id=str(batch.sample_ids[0]), scope='one_original_response_forward',
                loss=scalar(output.loss),
                ratio_metrics={name: [scalar(a), scalar(b)] for name, (a, b) in output.ratio_metrics.items()},
                sum_metrics={name: scalar(value) for name, value in output.sum_metrics.items()})
            self.response_records.append(record)
            self.work['response_forwards'] = self.work.get('response_forwards', 0) + 1
            if self.response_observer_path is not None:
                from experiments.kw_v1.evaluate import append_json, safe_scalars
                write_started = time.perf_counter()
                append_json(self.response_observer_path, safe_scalars(record))
                self.response_record_write_s += time.perf_counter() - write_started
            self.response_observation_s += time.perf_counter() - observed_started
            return output
        self.author_strategy.forward_loss = observed_forward

    def _load_decoded(self, state):
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        self.backend.load_state_dict(state['backend'])
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        restore_rng(state['rng'])

    def _restore(self, path):
        import torch
        state = torch.load(path, map_location='cpu', weights_only=False)
        validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self._load_decoded(state)

    def _prepare_event(self, event_id, feature_paths, passes):
        number = self.completed_event + 1
        if (event_id != f'event_{number:03d}' or len(self.events) != self.completed_event
                or self.global_step != self.completed_event * 128
                or self.presentations != self.completed_event * 256
                or len(feature_paths) != 128 or type(passes) is not int or passes != 2):
            raise ValueError('Only the next original O event with128responses/two passes is admitted')
        self.response_records.clear()
        self.work.clear()
        self.optimizer_step_host_s = 0.
        self.response_observation_s = 0.
        self.response_record_write_s = 0.
        return super()._prepare_event(event_id, feature_paths, passes)

    def _train_passes(self, features, passes):
        records = super()._train_passes(features, passes)
        if len(records) != len(self.response_records):
            raise ValueError('Original response-observation count differs')
        for row, observed in zip(records, self.response_records):
            if row['record_id'] != observed['record_id']:
                raise ValueError('Response observation identity differs')
            row['response_observation'] = observed
            row['metrics_scope'] = 'accumulation_window' if row['optimizer_stepped'] else 'presentation'
        return records

    def train_event(self, event_id, feature_paths, passes=2, *, presentation_schedule=None,
                    reference_check=None):
        if reference_check is not None or presentation_schedule is None:
            raise ValueError('Require original O natural schedule; no additional reference checks')
        if isinstance(presentation_schedule, (str, Path)):
            presentation_schedule = json.loads(Path(presentation_schedule).read_text())
        original.validate_event_receipt(presentation_schedule,
            event_number=self.completed_event + 1, objective=OBJECTIVE)
        return super().train_event(event_id, feature_paths, passes,
                                  presentation_schedule=presentation_schedule)

    def close(self):
        for handle in self._work_handles:
            handle.remove()
        self._work_handles.clear()
        if hasattr(self, '_native_chunk'):
            self.bundle.model._chunk_loss = self._native_chunk
            self.bundle.model.end_event = self._native_end
            self.backend.optimizer.step = self._native_optimizer_step
            del self._native_chunk, self._native_end, self._native_optimizer_step
        return super().close()
