"""Own-state K_HALF/KP trajectories, without relaxing historical O/K gates.

Only the learning rate and the declared KP auxiliary objective differ. Model
assembly, natural anchor sampling, native accumulation, AdamW and publication
remain the established DFlash-Online paths.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from experiments.ok_coding import training as original
from experiments.ok_continuation import resume
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer, SCHEMA

ARMS = ('K_HALF', 'KP')
LEARNING_RATES = {'K_HALF': 1.5e-4, 'KP': 3e-4}
CONTRACT_KEY = 'k_prefix_protection_v1'
HORIZON = 7424


def experiment_contract(arm, seed, binding, *, qualification=None):
    if arm not in ARMS or seed not in (17, 29):
        raise ValueError('Require a declared K_HALF/KP arm and seed17/29')
    if not isinstance(binding, dict) or not binding:
        raise ValueError('Require the sealed original-data/execution binding')
    contract = dict(schema=CONTRACT_KEY, arm=arm, seed=seed,
        learning_rate=LEARNING_RATES[arm], binding=json.loads(json.dumps(binding, allow_nan=False)),
        base_objective='osd_first_error', initialization='original_seed_step_zero',
        reference='fixed_released_F' if arm == 'KP' else None,
        coefficient=.1 if arm == 'KP' else 0., margin_ratio=.5, reference_cap=1.,
        denominator_epsilon=1e-6, ordinary_response_mean=True,
        independent_reference_denominator=True, maximum_completed_events=4)
    if qualification is not None:
        contract['qualification_only'] = qualification
        if qualification.get('native_only'):
            contract.update(reference=None, coefficient=0.)
        elif 'coefficient' in qualification:
            contract['coefficient'] = qualification['coefficient']
    return contract


def validate_config_text(config_text, seed, arm):
    """Use the original configuration gate after checking the sole LR exception."""
    import yaml
    raw = yaml.safe_load(config_text)
    if arm not in ARMS or raw['training']['learning_rate'] != LEARNING_RATES[arm]:
        raise ValueError('New-arm learning rate differs from the declared fixed value')
    # A validation-only view: no model, optimizer, saved config or historical gate
    # is modified. The original gate still rejects half-LR O/K snapshots.
    raw['training']['learning_rate'] = 3e-4
    original.validate_config_text(yaml.safe_dump(raw), seed)


def _validate_rng(rng, expected_cuda_rng_states):
    import random
    import numpy as np
    import torch
    if set(rng) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('Require all four saved training RNG components')
    if expected_cuda_rng_states not in (0, 1) or len(rng['cuda']) != expected_cuda_rng_states:
        raise ValueError('Saved CUDA RNG stream count differs')
    for value in [rng['torch'], *rng['cuda']]:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel():
            raise ValueError('Malformed saved Torch RNG')
    random.Random(0).setstate(rng['python'])
    np.random.RandomState(0).set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch'])


def validate_snapshot(state, *, arm, seed, binding, original_contract, config_text,
                      completed_event, expected_cuda_rng_states=1, qualification=None,
                      qualification_steps=0):
    """Validate new-arm states directly, including truthful disposable windows.

    A qualification window retains four completed historical events and advances
    512→513 steps. It is never represented as a completed fifth event.
    """
    import torch
    from experiments.osd_components.checkpoints import EXPECTED_KEYS
    require = resume._require
    require(type(completed_event) is int and 0 <= completed_event <= 4, 'Completed event must be0..4')
    require(qualification_steps in (0, 1), 'Qualification permits exactly one temporary window')
    require(not qualification_steps or qualification is not None, 'Production cannot contain a partial extra event')
    validate_config_text(config_text, seed, arm)
    expected_contract = experiment_contract(arm, seed, binding, qualification=qualification)
    require(state.get('algorithm_contract', {}).get(CONTRACT_KEY) == expected_contract, 'Own arm/seed/reference contract differs')
    require(state['algorithm_contract'].get(original.CONTRACT_KEY) == original.coding_contract(seed, original_contract),
            'Original sealed coding identity differs')
    forbidden = ('reference_correction_event', 'prefix_correction_event', 'prefix_guard', 'response_weight_controller')
    require(not any(name in state['algorithm_contract'] for name in forbidden), 'Unrequested mechanism state')
    require(state['algorithm_contract']['online_target_objective']['name'] == 'osd_first_error', 'K base objective differs')
    require(state['event_schema'] == SCHEMA and state['strategy'] == 'dflash' and state['world_size'] == 1, 'Runtime schema differs')
    require(state['config_text'] == config_text and state['reduction'] == 'response_mean'
            and state['accumulation_steps'] == 2 and state['horizon'] == HORIZON, 'Configuration or normalization differs')
    steps = completed_event * 128 + qualification_steps
    require(type(state['global_step']) is int and state['global_step'] == steps
            and type(state['presentations']) is int and state['presentations'] == steps * 2, 'Own counters differ')
    require(len(state['events']) == completed_event, 'Completed-event history differs')
    for number, event in enumerate(state['events'], 1):
        original.validate_event_receipt(event, event_number=number, objective='osd_first_error', require_completed=False)
    weights = state['draft_state_dict']
    require(set(weights) == EXPECTED_KEYS, 'Require the58 qualified draft tensors')
    for name, value in weights.items():
        require(isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16 and value.numel() > 0,
                'Invalid BF16 draft tensor: ' + name)
        resume._finite(value, name)
    require(set(state['backend']) == {'optimizer'}, 'Require only the native optimizer backend state')
    optimizer = state['backend']['optimizer']
    recipe, saved = optimizer['recipe'], optimizer['optimizer_state_dict']
    require(recipe['type'] == 'direct_bf16_adamw' and recipe['version'] == 1 and recipe['max_grad_norm'] == 1.,
            'Native optimizer recipe differs')
    parameters = recipe['parameters']
    names = [row[0] for row in parameters]
    require(len(names) == 58 and set(names) == set(weights), 'Optimizer includes missing/extra trainable tensors')
    require(len(saved['param_groups']) == 1, 'Require one continuous native Adam group')
    group = saved['param_groups'][0]
    ids = group['params']
    require(len(ids) == 58 and len(set(ids)) == 58, 'Optimizer parameter IDs differ')
    options = {key: value for key, value in group.items() if key != 'params'}
    require(not resume.differences(options, recipe['optimizer_options']), 'Adam group/recipe options differ')
    require(options['lr'] == LEARNING_RATES[arm] and options['weight_decay'] == .01
            and options['betas'] == (.9, .999) and options['eps'] == 1e-8 and options['amsgrad'] is False,
            'Declared optimizer settings differ')
    require(set(saved['state']) == (set(ids) if steps else set()), 'Adam moments missing or nonempty at initialization')
    for (name, shape, dtype), parameter_id in zip(parameters, ids):
        weight = weights[name]
        require(tuple(shape) == tuple(weight.shape) and dtype == str(weight.dtype), 'Adam tensor binding differs: ' + name)
        if not steps:
            continue
        moments = saved['state'][parameter_id]
        require(set(moments) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Incomplete Adam state: ' + name)
        counter = moments['step']
        require(isinstance(counter, torch.Tensor) and counter.numel() == 1
                and bool(torch.isfinite(counter).all()) and counter.item() == steps, 'Adam counter differs: ' + name)
        for key in ('exp_avg', 'exp_avg_sq'):
            value = moments[key]
            require(value.dtype == torch.bfloat16 and value.shape == weight.shape, 'Adam precision/shape differs: ' + name)
            resume._finite(value, name + '.' + key)
    _validate_rng(state['rng'], expected_cuda_rng_states)
    return dict(status='passed', arm=arm, seed=seed, completed_event=completed_event,
                global_step=steps, presentations=steps * 2, exact_contract=True,
                draft_tensors=58, adam_state_entries=len(saved['state']),
                qualification_steps=qualification_steps, scientific_extra_steps=0)


def migrate_source(source, *, arm, seed, binding, original_contract, config_text,
                   new_contract, completed_event=0, qualification=None,
                   expected_cuda_rng_states=1):
    """Declare the narrow metadata/LR migration; leave source tensors and RNG exact."""
    original.validate_snapshot(source, arm='K', seed=seed, campaign_contract=original_contract,
        config_text=source['config_text'], completed_event=completed_event,
        expected_cuda_rng_states=expected_cuda_rng_states)
    if completed_event and qualification is None:
        raise ValueError('Production initialization must use original step-zero state')
    expected_base = deepcopy(source['algorithm_contract'])
    expected_base[CONTRACT_KEY] = experiment_contract(arm, seed, binding, qualification=qualification)
    if resume.differences(expected_base, new_contract):
        raise ValueError('Runtime migration changes more than the declared additive contract')
    candidate = dict(source, config_text=config_text, algorithm_contract=deepcopy(new_contract))
    candidate['backend'] = {'optimizer': deepcopy(source['backend']['optimizer'])}
    optimizer = candidate['backend']['optimizer']
    optimizer['recipe']['optimizer_options']['lr'] = LEARNING_RATES[arm]
    optimizer['optimizer_state_dict']['param_groups'][0]['lr'] = LEARNING_RATES[arm]
    validate_snapshot(candidate, arm=arm, seed=seed, binding=binding, original_contract=original_contract,
        config_text=config_text, completed_event=completed_event, qualification=qualification,
        expected_cuda_rng_states=expected_cuda_rng_states)
    # Scientific state comparison excludes exactly the two declared option values.
    original_optimizer = deepcopy(source['backend']['optimizer'])
    original_optimizer['recipe']['optimizer_options']['lr'] = LEARNING_RATES[arm]
    original_optimizer['optimizer_state_dict']['param_groups'][0]['lr'] = LEARNING_RATES[arm]
    expected = {key: source[key] for key in ('draft_state_dict', 'rng', 'global_step', 'presentations', 'events')}
    actual = {key: candidate[key] for key in expected}
    changes = resume.differences(expected, actual) + resume.differences(original_optimizer, optimizer)
    if changes:
        raise ValueError('Scientific state changed during initialization migration: ' + str(changes[:8]))
    receipt = dict(status='passed', original_seed=seed, original_steps=source['global_step'],
        source_weights_rng_moments_counters_history_exact=True,
        changes=['additive algorithm/reference contract', 'configuration identity']
                + (['declared learning rate3e-4→1.5e-4 in recipe and Adam group'] if arm == 'K_HALF' else []),
        qualification_only=qualification is not None, no_reseed_after_restore=True)
    return candidate, receipt


def live_state(trainer):
    """Return native complete state at an empty accumulation boundary."""
    full = trainer.backend.state_dict()
    return dict(event_schema=SCHEMA,
        draft_state_dict=trainer.strategy.checkpoint_state_filter(full['model']),
        backend={'optimizer': full['optimizer']}, rng=capture_rng(),
        global_step=trainer.global_step, presentations=trainer.presentations,
        events=trainer.events, strategy='dflash', run_id=trainer.cfg.run_id, world_size=1,
        config_text=trainer.config_text, algorithm_contract=trainer.contract,
        reduction=trainer.reduction, accumulation_steps=trainer.accumulation, horizon=trainer.horizon)


def verify_live(trainer, expected, receipt_path=None):
    started = time.perf_counter()
    actual = live_state(trainer)
    keys = ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations', 'events',
            'config_text', 'algorithm_contract', 'reduction', 'accumulation_steps', 'horizon')
    mismatch = resume.differences({key: actual[key] for key in keys}, {key: expected[key] for key in keys})
    clean = (trainer.core._micro == expected['presentations'] and trainer.core.accumulation_remainder == 0
        and not trainer.core._ratio_totals and not trainer.core._sum_totals
        and trainer._presentation_schedule is None
        and trainer.bundle.model.training and all(p.grad is None for p in trainer.bundle.draft_model.parameters()))
    if not clean:
        mismatch.append('runtime_boundary_or_mode_or_gradients')
    reference = getattr(trainer.bundle.model, 'released_reference', None)
    if reference is not None:
        reference_ids = {id(p) for p in reference.model.draft_model.parameters()}
        optimized = {id(p) for p in trainer.backend.optimizer.model_params}
        if reference.model.draft_model.training or reference_ids & optimized:
            mismatch.append('reference_mode_or_optimizer_membership')
    receipt = dict(status='passed' if not mismatch else 'failed', exact_resume=not mismatch,
                   mismatches=mismatch[:32], global_step=trainer.global_step,
                   presentations=trainer.presentations, wall_s=time.perf_counter() - started)
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    if mismatch:
        raise ValueError('Own-state restoration differs: ' + str(mismatch[:8]))
    return receipt


class PrefixProtectionTrainer(SpecForgeEventTrainer):
    """Add the KP loss around the same current model and native TrainerCore."""

    def __init__(self, config_path, output_root, *, arm, seed, binding, original_contract,
                 resume_state=None, source_state=None, completed_event=0,
                 qualification=None, coefficient=.1, forced_active=False, native_only=False):
        self.arm, self.seed, self.binding = arm, seed, binding
        self.original_contract = original_contract
        self.completed_event = completed_event
        self.qualification = qualification
        self.coefficient, self.forced_active = coefficient, forced_active
        self.native_only = native_only
        self.response_records = []
        self.response_observer_path = None
        self.reference_setup_s = 0.
        if arm not in ARMS or (qualification is None and (coefficient != .1 or forced_active or native_only)):
            raise ValueError('Only fixed production arms or explicit disposable qualification are permitted')
        super().__init__(config_path, output_root, resume_state=resume_state,
                         reduction='response_mean', objective='osd_first_error', ranking_lambda=0.)
        if source_state is not None:
            migrated, self.migration_receipt = migrate_source(source_state, arm=arm, seed=seed,
                binding=binding, original_contract=original_contract, config_text=self.config_text,
                new_contract=self.contract, completed_event=completed_event, qualification=qualification)
            self._load_decoded(migrated)
            verify_live(self, migrated)
        if self.global_step != completed_event * 128 or self.presentations != completed_event * 256:
            raise ValueError('Runtime did not start from the declared own boundary')

    def _validate_config(self):
        super()._validate_config()
        validate_config_text(self.config_text, self.seed, self.arm)

    def _build_runtime(self):
        super()._build_runtime()
        self.contract[original.CONTRACT_KEY] = original.coding_contract(self.seed, self.original_contract)
        self.contract[CONTRACT_KEY] = experiment_contract(self.arm, self.seed, self.binding,
                                                         qualification=self.qualification)
        if self.arm == 'KP' and not self.native_only:
            from online_feedback.prefix_protection import ReleasedPrefixObjective
            started = time.perf_counter()
            old = self.bundle.model
            old._sample_anchor_positions = self._original_sampler
            model = ReleasedPrefixObjective(old, self.cfg.model.draft_checkpoint_path,
                coefficient=self.coefficient, eos_token_ids=self.eos_token_ids,
                qualification=self.qualification is not None, forced_active=self.forced_active)
            # NO_SHARD uses DDP. All current draft/head/embedding Parameters are
            # identical objects, so its existing reducer and Adam remain valid.
            if self.backend._wrapper_kind != 'ddp':
                raise ValueError('The qualified single-GPU DDP wrapper is required')
            self.backend.module.module = model
            self.bundle.model = model
            self._original_sampler = model._sample_anchor_positions
            model._sample_anchor_positions = self._sample_and_record_anchors
            model.train()
            self.reference_setup_s = time.perf_counter() - started
        self.objective_model = self.bundle.model
        original_forward = self.author_strategy.forward_loss

        def observed_forward(batch, ctx=None):
            output = original_forward(batch, ctx)
            # Observe before ReductionAdapter/TrainerCore merges the two rows.
            def scalar(value):
                return float(value.detach().item()) if hasattr(value, 'detach') else float(value)
            record = dict(record_id=str(batch.sample_ids[0]), scope='one_original_response_forward',
                loss=scalar(output.loss),
                ratio_metrics={name: [scalar(pair[0]), scalar(pair[1])]
                               for name, pair in output.ratio_metrics.items()},
                sum_metrics={name: scalar(value) for name, value in output.sum_metrics.items()})
            if self.arm == 'KP' and not self.native_only:
                record['prefix_protection'] = deepcopy(self.bundle.model.last_response_record)
            self.response_records.append(record)
            if self.response_observer_path is not None:
                from experiments.kw_v1.evaluate import append_json, safe_scalars
                append_json(self.response_observer_path, safe_scalars(record))
            return output
        self.author_strategy.forward_loss = observed_forward

    def _load_decoded(self, state):
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        self.backend.load_state_dict(state['backend'])
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        # Final stochastic operation: construction/reference/optimizer setup is
        # complete. No random draw or reseed follows this exact restore.
        restore_rng(state['rng'])

    def _restore(self, path):
        import torch
        state = torch.load(path, map_location='cpu', weights_only=False)
        validate_snapshot(state, arm=self.arm, seed=self.seed, binding=self.binding,
            original_contract=self.original_contract, config_text=self.config_text,
            completed_event=self.completed_event, qualification=self.qualification)
        if state['algorithm_contract'] != self.contract:
            raise ValueError('Own runtime contract differs from saved contract')
        self._load_decoded(state)

    def _prepare_event(self, event_id, feature_paths, passes):
        number = len(self.events) + 1
        if self.qualification is not None or not 1 <= number <= 4 or event_id != f'event_{number:03d}':
            raise ValueError('Production permits exactly its next event1..4')
        if len(feature_paths) != 128 or passes != 2:
            raise ValueError('Require128 responses and two passes')
        return super()._prepare_event(event_id, feature_paths, passes)

    def _train_passes(self, features, passes):
        self.response_records.clear()
        rows = super()._train_passes(features, passes)
        if len(rows) != len(self.response_records):
            raise ValueError('Per-response observation count differs from native forwards')
        for row, response in zip(rows, self.response_records):
            if row['record_id'] != response['record_id']:
                raise ValueError('Per-response metric identity differs')
            row['response_observation'] = response
        return rows

    def train_event(self, event_id, feature_paths, passes=2, *, presentation_schedule=None,
                    reference_check=False):
        if reference_check or presentation_schedule is None:
            raise ValueError('Require original natural schedule; no additional reference checks')
        schedule = json.loads(Path(presentation_schedule).read_text()) if isinstance(presentation_schedule, (str, Path)) else presentation_schedule
        original.validate_event_receipt(schedule, event_number=len(self.events)+1, objective='osd')
        return super().train_event(event_id, feature_paths, passes=passes,
                                  presentation_schedule=schedule, reference_check=None)

    def close(self):
        model = getattr(self, 'objective_model', None)
        if model is not None and hasattr(model, 'close_counters'):
            model.close_counters()
        self.objective_model = None
        return super().close()
