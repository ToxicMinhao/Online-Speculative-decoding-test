"""Fresh O/K initialization and own-state continuation for coding events 1..4.

The numerical training, sampling, optimizer and export paths are inherited.
These additive gates admit genuine empty Adam at event zero and check event-one
natural schedules without changing historical continuation validators.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from experiments.ok_continuation import resume as inherited
from experiments.ok_events_5_8.resume import verify_saved_boundary
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.storage import save_torch
from online_feedback.training import SpecForgeEventTrainer, SCHEMA

OBJECTIVES = inherited.OBJECTIVES
HORIZON = inherited.HORIZON
CONTRACT_KEY = 'coding_campaign'
verify_live_resume = inherited.verify_live_resume


def coding_contract(seed, campaign_contract):
    if type(seed) is not int or seed not in (17, 29):
        raise ValueError('Coding training seed must be 17 or 29')
    if not isinstance(campaign_contract, dict) or not campaign_contract:
        raise ValueError('Require the sealed coding campaign contract')
    if campaign_contract.get('training_seed', seed) != seed:
        raise ValueError('Campaign contract belongs to a different training seed')
    # JSON round-trip also rejects tensors or nonfinite values in provenance.
    binding = json.loads(json.dumps(campaign_contract, allow_nan=False))
    return dict(schema='ok_coding_v1', training_seed=seed, binding=binding)


def validate_config_text(config_text, seed):
    """Pin the inherited numerical recipe, separately from the four-event cap."""
    import yaml
    raw = yaml.safe_load(config_text)
    expected = {
        'model': dict(torch_dtype='bfloat16'),
        'data': dict(max_length=2272),
        'training': dict(seed=seed, strategy='dflash', max_steps=7424,
            batch_size=1, accumulation_steps=2, fsdp_sharding='NO_SHARD',
            optimizer_cpu_offload=False, learning_rate=3e-4,
            lr_scheduler='constant', warmup_ratio=0., max_grad_norm=1.,
            attention_backend='flex_attention', num_anchors=512,
            objective_chunk_blocks=8, loss_decay_gamma=7., loss_type='dflash',
            lk_loss_type=None),
        'online_feedback': dict(responses_per_event=128, passes_per_event=2,
            max_events=58, optimizer='bf16_adamw', weight_decay=.01,
            shuffle_responses=True, kl_precision='bfloat16'),
    }
    for section, values in expected.items():
        for key, expected_value in values.items():
            if raw.get(section, {}).get(key) != expected_value:
                raise ValueError(f'Coding recipe differs: {section}.{key}')
    if not raw['model'].get('draft_checkpoint_path'):
        raise ValueError('A released draft weights path is required')


def validate_event_receipt(receipt, *, event_number, objective,
                           reference_schedule=None, trainer_status=None,
                           require_completed=True):
    """Validate events 1..4, including event one's natural O/K draws."""
    result = inherited.validate_event_receipt(
        receipt, event_number=event_number, objective=objective,
        trainer_status=trainer_status, require_completed=require_completed)
    if reference_schedule is not None:
        inherited.validate_event_receipt(reference_schedule, event_number=event_number,
                                         objective='osd')
        if inherited.differences(receipt['features'], reference_schedule['features']):
            raise ValueError('O/K feature paths or metadata differ')
        for row, ordinary in zip(receipt['records'], reference_schedule['records']):
            if any(row[key] != ordinary[key] for key in ('record_id', 'pass', 'sampled_anchors')):
                raise ValueError('O/K natural response order or anchors differ')
        result['schedule_matched'] = True
    return result


def _validate_zero(state, objective, config_text, expected_cuda_rng_states):
    """Validate empty state directly, without inventing an event or moments."""
    import random
    import numpy as np
    import torch
    from experiments.osd_components.checkpoints import EXPECTED_KEYS
    require = inherited._require
    required = {'event_schema', 'draft_state_dict', 'backend', 'rng', 'global_step',
        'presentations', 'events', 'strategy', 'run_id', 'world_size', 'config_text',
        'algorithm_contract', 'reduction', 'accumulation_steps', 'horizon'}
    require(required <= set(state), 'Incomplete initialization snapshot')
    require(state['event_schema'] == SCHEMA and state['strategy'] == 'dflash'
            and state['world_size'] == 1, 'Initialization schema or runtime differs')
    require(state['config_text'] == config_text and state['reduction'] == 'response_mean'
            and state['accumulation_steps'] == 2 and state['horizon'] == HORIZON,
            'Initialization config or training recipe differs')
    require(type(state['global_step']) is int and state['global_step'] == 0
            and type(state['presentations']) is int and state['presentations'] == 0
            and state['events'] == [], 'Initialization must have genuinely empty history')
    require(state['algorithm_contract']['online_target_objective']['name'] == objective,
            'Initialization objective differs')
    weights = state['draft_state_dict']
    require(set(weights) == EXPECTED_KEYS, 'Require exactly 58 released draft tensors')
    for name, value in weights.items():
        require(isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16
                and value.numel() > 0, 'Invalid initialized draft tensor: ' + name)
        inherited._finite(value, name)
    require(set(state['backend']) == {'optimizer'}, 'Unexpected backend state')
    optimizer = state['backend']['optimizer']
    recipe, saved = optimizer['recipe'], optimizer['optimizer_state_dict']
    require(recipe['type'] == 'direct_bf16_adamw' and recipe['version'] == 1
            and recipe['max_grad_norm'] == 1., 'Initialized Adam recipe differs')
    names = [row[0] for row in recipe['parameters']]
    require(len(names) == 58 and set(names) == set(weights), 'Initialized trainable membership differs')
    require(saved['state'] == {} and len(saved['param_groups']) == 1,
            'Initialization must contain empty native Adam state')
    group = saved['param_groups'][0]
    require(len(group['params']) == 58 and len(set(group['params'])) == 58,
            'Initialized Adam parameter group differs')
    options = {key: value for key, value in group.items() if key != 'params'}
    require(not inherited.differences(options, recipe['optimizer_options']), 'Adam options differ')
    require(options['lr'] == 3e-4 and options['weight_decay'] == .01
            and options['betas'] == (.9, .999) and options['eps'] == 1e-8
            and options['amsgrad'] is False, 'Initialized Adam hyperparameters differ')
    for name, shape, dtype in recipe['parameters']:
        require(tuple(shape) == tuple(weights[name].shape) and dtype == str(weights[name].dtype),
                'Initialized Adam parameter binding differs: ' + name)
    rng = state['rng']
    require(set(rng) == {'python', 'numpy', 'torch', 'cuda'}, 'Require complete initialization RNG')
    require(type(expected_cuda_rng_states) is int and expected_cuda_rng_states in (0, 1)
            and isinstance(rng['cuda'], list) and len(rng['cuda']) == expected_cuda_rng_states,
            'Initialization CUDA RNG stream count differs')
    for value in [rng['torch'], *rng['cuda']]:
        require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                and value.ndim == 1 and value.numel() > 0, 'Malformed initialization Torch RNG')
    try:
        random.Random(0).setstate(rng['python'])
        np.random.RandomState(0).set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['torch'])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Malformed initialization RNG') from error
    return dict(status='passed', completed_event=0, global_step=0, presentations=0,
                objective=objective, horizon=HORIZON, draft_tensors=58,
                optimizer_parameters=58, adam_state_entries=0,
                cuda_rng_states=len(rng['cuda']), scientific_extra_steps=0)


def validate_snapshot(state, *, arm, seed, campaign_contract, config_text,
                      completed_event, expected_cuda_rng_states=1):
    if arm not in OBJECTIVES or type(completed_event) is not int or not 0 <= completed_event <= 4:
        raise ValueError('Require an own O/K coding boundary 0..4')
    expected = coding_contract(seed, campaign_contract)
    validate_config_text(config_text, seed)
    if state.get('algorithm_contract', {}).get(CONTRACT_KEY) != expected:
        raise ValueError('Saved coding seed/data/source contract differs')
    if any(key in state['algorithm_contract'] for key in
           ('reference_correction_event', 'prefix_correction_event', 'prefix_guard',
            'response_weight_controller')):
        raise ValueError('Only ordinary O/K state is admitted')
    if completed_event == 0:
        result = _validate_zero(state, OBJECTIVES[arm], config_text, expected_cuda_rng_states)
    else:
        result = inherited.validate_snapshot(state, objective=OBJECTIVES[arm],
            config_text=config_text, completed_event=completed_event,
            expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(result, arm=arm, training_seed=seed, campaign_contract_exact=True)


def _export_equal(state, export_path):
    from safetensors import safe_open
    seen = set()
    for file in sorted(Path(export_path).glob('*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as source:
            for name in source.keys():
                if name in seen or name not in state['draft_state_dict']:
                    raise ValueError('Unexpected or duplicate export tensor: ' + name)
                if not inherited._tensor_equal(source.get_tensor(name), state['draft_state_dict'][name]):
                    raise ValueError('Draft differs from exact export/released weights: ' + name)
                seen.add(name)
    if seen != set(state['draft_state_dict']):
        raise ValueError('Export/released tensor membership differs')


def load_snapshot(path, export_path, *, arm, seed, campaign_contract, config_text,
                  completed_event, expected_cuda_rng_states=1):
    import torch
    started = time.perf_counter()
    state = torch.load(path, map_location='cpu', weights_only=False)
    receipt = validate_snapshot(state, arm=arm, seed=seed, campaign_contract=campaign_contract,
        config_text=config_text, completed_event=completed_event,
        expected_cuda_rng_states=expected_cuda_rng_states)
    _export_equal(state, export_path)
    receipt.update(state_path=str(path), export_path=str(export_path), export_exact=True,
                   wall_s=time.perf_counter()-started)
    return state, receipt


def compare_initial_states(ordinary, candidate, *, seed, campaign_contract,
                           expected_cuda_rng_states=1):
    """Compare full ready-to-train state; objective metadata intentionally differs."""
    import torch
    states = []
    for arm, value in (('O', ordinary), ('K', candidate)):
        state = torch.load(value, map_location='cpu', weights_only=False) if isinstance(value, (str, Path)) else value
        validate_snapshot(state, arm=arm, seed=seed, campaign_contract=campaign_contract,
            config_text=state['config_text'], completed_event=0,
            expected_cuda_rng_states=expected_cuda_rng_states)
        common_contract = deepcopy(state['algorithm_contract'])
        common_contract.pop('online_target_objective')
        states.append(dict(weights=state['draft_state_dict'], optimizer=state['backend']['optimizer'],
            rng=state['rng'], global_step=state['global_step'], presentations=state['presentations'],
            events=state['events'], contract=common_contract, horizon=state['horizon'],
            accumulation=state['accumulation_steps'], reduction=state['reduction']))
    mismatch = inherited.differences(*states)
    if mismatch:
        raise ValueError('O/K initialization differs: ' + str(mismatch[:12]))
    return dict(status='passed', training_seed=seed, exact_initial_state=True,
                empty_adam=True, scientific_extra_steps=0)


class CodingTrainer(SpecForgeEventTrainer):
    """One fresh process admits exactly one coding event and then must close."""

    def __init__(self, config_path, output_root, *, arm, seed, campaign_contract,
                 resume_state=None, completed_event=0):
        if arm not in OBJECTIVES or type(completed_event) is not int or completed_event not in range(4):
            raise ValueError('Coding trainer starts at own boundary 0..3')
        if completed_event and resume_state is None:
            raise ValueError('Later coding events require an own complete restart state')
        self.arm, self.seed, self.completed_event = arm, seed, completed_event
        self.campaign_contract = deepcopy(campaign_contract)
        self.coding_binding = coding_contract(seed, campaign_contract)
        super().__init__(config_path, output_root, resume_state=resume_state,
                         objective=OBJECTIVES[arm], reduction='response_mean', ranking_lambda=0)
        if (self.global_step != completed_event*128 or self.presentations != completed_event*256
                or len(self.events) != completed_event or self.core.accumulation_remainder
                or self.core._micro != self.presentations):
            raise ValueError('Coding own-arm boundary counters or history differ')

    def _validate_config(self):
        super()._validate_config()
        validate_config_text(self.config_text, self.seed)

    def _build_runtime(self):
        super()._build_runtime()
        self.contract[CONTRACT_KEY] = deepcopy(self.coding_binding)

    def initial_state(self):
        if self.global_step or self.presentations or self.events:
            raise ValueError('Initial state is available only before ordinary training')
        full = self.backend.state_dict()
        return dict(event_schema=SCHEMA,
            draft_state_dict=self.strategy.checkpoint_state_filter(full['model']),
            backend={'optimizer': full['optimizer']}, rng=capture_rng(), global_step=0,
            presentations=0, events=[], strategy='dflash', run_id=self.cfg.run_id,
            world_size=1, config_text=self.config_text, algorithm_contract=deepcopy(self.contract),
            reduction=self.reduction, accumulation_steps=self.accumulation, horizon=self.horizon)

    def save_initial_state(self, path, *, expected_cuda_rng_states=1):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Refusing to replace initialized state: ' + str(path))
        with preserve_rng():
            state = self.initial_state()
            receipt = validate_snapshot(state, arm=self.arm, seed=self.seed,
                campaign_contract=self.campaign_contract, config_text=self.config_text,
                completed_event=0, expected_cuda_rng_states=expected_cuda_rng_states)
            verify_live_resume(self, state)
            _export_equal(state, self.cfg.model.draft_checkpoint_path)
            save_torch(path, state)
        verify_live_resume(self, state)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(receipt, path=str(path), state_path=str(path), sha256=digest,
                    bytes=path.stat().st_size, released_weights_exact=True)

    def _prepare_event(self, event_id, feature_paths, passes):
        number = self.completed_event + 1
        if (event_id != f'event_{number:03d}' or len(self.events) != self.completed_event
                or self.global_step != self.completed_event*128
                or self.presentations != self.completed_event*256
                or len(feature_paths) != 128 or type(passes) is not int or passes != 2):
            raise ValueError('Only the next sealed coding event with 128 responses/two passes is admitted')
        return super()._prepare_event(event_id, feature_paths, passes)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None,
                    presentation_schedule=None):
        if reference_check is not None:
            raise ValueError('Coding O/K adds no reference or mechanism probes')
        if (self.arm == 'K') != (presentation_schedule is not None):
            raise ValueError('O samples normally; K checks the completed O natural schedule')
        if isinstance(presentation_schedule, (str, Path)):
            presentation_schedule = json.loads(Path(presentation_schedule).read_text())
        if presentation_schedule is not None:
            validate_event_receipt(presentation_schedule, event_number=self.completed_event+1,
                                   objective='osd')
        return super().train_event(event_id, feature_paths, passes,
                                  presentation_schedule=presentation_schedule)
