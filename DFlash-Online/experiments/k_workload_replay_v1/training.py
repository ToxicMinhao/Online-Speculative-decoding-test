"""Track2 pass allocation around unchanged native K training operations.

The two explicit input lists are the only training change. Native construction,
anchor sampling, train_step, Adam, checkpoint publication and export are reused.
Historical full-K validators and their no-replay guards remain untouched.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from experiments.full_k_gsm8k import training as full_k
from experiments.ok_continuation.resume import differences, _finite
from experiments.ok_coding.training import _export_equal
from experiments.o_half_matched_lr.training import native_algorithm_contract
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer, SCHEMA

ARMS = ('K', 'U', 'R')
CONTRACT_KEY = 'k_workload_replay_v1'
OBJECTIVE = 'osd_first_error'
MAX_EVENTS = 8
PACKET_KEYS = ('record_id', 'global_ordinal', 'path', 'sha256')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _file_identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def experiment_contract(arm, binding):
    if arm not in ARMS or not _sha256(binding):
        raise ValueError('Require K/U/R and the sealed SHA-256 campaign binding')
    return dict(schema=CONTRACT_KEY, arm=arm, binding=binding, training_seed=17,
                learning_rate=3e-4, maximum_events=8, committed_steps=1024,
                native_horizon=7424, native_max_events=58,
                initialization='independent_released_seed17_empty_adam',
                feedback='shared_immutable_K_native_packets',
                pass_allocation='128fresh_then_128fresh' if arm == 'K'
                    else '128fresh_then_96fresh_32past_except_event1',
                reference=None)


def validate_config_text(config_text, arm, original_config_text):
    if arm not in ARMS:
        raise ValueError('Unknown replay arm')
    full_k.validate_config_text(config_text, 'K', original_config_text)


def packet_binding(record):
    """Separate immutable identity from reporting and decoded-length metadata."""
    try:
        result = {key: record[key] for key in PACKET_KEYS}
    except (TypeError, KeyError) as error:
        raise ValueError('Require explicit packet identity, ordinal, path and hash') from error
    if (not isinstance(result['record_id'], str) or not result['record_id']
            or type(result['global_ordinal']) is not int
            or not 0 <= result['global_ordinal'] < 1024
            or not isinstance(result['path'], str) or not Path(result['path']).is_absolute()
            or not _sha256(result['sha256'])):
        raise ValueError('Malformed immutable packet binding')
    return result


def validate_pass_plan(pass_plan, *, arm, event_number, previous_records=None):
    """Validate two sets of 128, then sort before each native torch.randperm(128)."""
    if arm not in ARMS or type(event_number) is not int or not 1 <= event_number <= 8:
        raise ValueError('Require a declared K/U/R event 1..8')
    if not isinstance(pass_plan, (list, tuple)) or len(pass_plan) != 2:
        raise ValueError('Require exactly two explicit pass input lists')
    plans = []
    for records in pass_plan:
        if not isinstance(records, (list, tuple)) or len(records) != 128:
            raise ValueError('Each pass must contain exactly 128 complete packets')
        rows = [packet_binding(record) for record in records]
        if len({row['record_id'] for row in rows}) != 128 or len({row['global_ordinal'] for row in rows}) != 128:
            raise ValueError('Each pass must contain distinct record identities and ordinals')
        plans.append(sorted(rows, key=lambda row: row['global_ordinal']))
    first, second = plans
    beginning = 128 * (event_number - 1)
    if [row['global_ordinal'] for row in first] != list(range(beginning, beginning + 128)):
        raise ValueError('First pass must contain exactly this event\'s 128 fresh records')
    fresh = {row['record_id']: row for row in first}
    fresh_second = [row for row in second if row['record_id'] in fresh]
    old = [row for row in second if row['record_id'] not in fresh]
    if any(row != fresh[row['record_id']] for row in fresh_second):
        raise ValueError('A fresh packet binding changed between passes')
    if arm == 'K' or event_number == 1:
        if first != second:
            raise ValueError('K and neutral event 1 require two identical fresh input lists')
    elif len(fresh_second) != 96 or len(old) != 32:
        raise ValueError('Replay requires exactly 96 fresh and 32 distinct past packets')
    if any(row['global_ordinal'] >= beginning for row in old):
        raise ValueError('Replay packets must be strictly past records')
    if previous_records is not None:
        previous = {row['record_id']: packet_binding(row) for row in previous_records}
        if set(fresh).intersection(previous):
            raise ValueError('Fresh feedback cannot reuse an earlier request identity')
        for row in old:
            if row != previous.get(row['record_id']):
                raise ValueError('Replay must preserve its original complete immutable packet binding')
    return plans


def validate_event_receipt(receipt, *, arm, event_number, require_completed=True,
                           trainer_status=None, previous_records=None):
    plan = validate_pass_plan(receipt['pass_plan'], arm=arm, event_number=event_number,
                              previous_records=previous_records)
    expected = dict(event_id=f'event_{event_number:03d}', objective=OBJECTIVE,
                    passes=2, response_presentations=256, optimizer_steps=128,
                    start_step=(event_number - 1) * 128, end_step=event_number * 128,
                    reduction='response_mean')
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Event identity, objective or budget differs')
    if require_completed and receipt.get('status') != 'completed':
        raise ValueError('Event is incomplete')
    if [packet_binding(row) for row in receipt['features']] != plan[0]:
        raise ValueError('Fresh event inventory differs from the explicit first pass')
    records = receipt['records']
    if len(records) != 256:
        raise ValueError('Require 256 observed native presentations')
    metadata = {row['record_id']: row for inputs in receipt['pass_plan'] for row in inputs}
    fresh_ids = {row['record_id'] for row in plan[0]}
    for index, row in enumerate(records):
        stepped = index % 2 == 1
        if (row['pass'] != index // 128 or type(row['optimizer_stepped']) is not bool
                or row['optimizer_stepped'] != stepped
                or row['optimizer_step_after'] != expected['start_step'] + (index + 1) // 2
                or row['learning_rate'] != (3e-4 if stepped else None)):
            raise ValueError('Native pass, accumulation or Adam boundary differs')
        draws = row['sampled_anchors']
        if len(draws) != 1:
            raise ValueError('Require one native anchor draw per presentation')
        positions, keep = draws[0]['positions'], draws[0]['keep_mask']
        if (len(positions) != 1 or len(keep) != 1 or not 0 < len(positions[0]) <= 512
                or len(positions[0]) != len(keep[0])
                or any(type(value) is not int or value < 0 for value in positions[0])
                or any(type(value) is not bool for value in keep[0])):
            raise ValueError('Malformed native anchor draw')
        if any(not math.isfinite(float(value)) for value in row['metrics'].values()):
            raise ValueError('Nonfinite native training metric')
        packet = metadata[row['record_id']]
        expected_exposure = dict(global_ordinal=packet['global_ordinal'],
            replay=row['record_id'] not in fresh_ids, prompt_tokens=packet['prompt_length'],
            response_tokens=packet['response_tokens'], active_anchors=sum(keep[0]))
        if any(row.get(key) != value for key, value in expected_exposure.items()):
            raise ValueError('Presentation token, replay or active-anchor exposure differs')
    for pass_index in (0, 1):
        observed = records[128 * pass_index:128 * (pass_index + 1)]
        if sorted(row['record_id'] for row in observed) != sorted(row['record_id'] for row in plan[pass_index]):
            raise ValueError('Observed exposure differs from the explicit pass input')
    if receipt.get('pass_exposure') != pass_exposure(records):
        raise ValueError('Per-pass exposure accounting differs from actual presentations')
    if trainer_status is not None:
        wanted = dict(global_step=128 * event_number, presentations=256 * event_number,
                      completed_events=[f'event_{n:03d}' for n in range(1, event_number + 1)],
                      accumulation_remainder=0, objective=OBJECTIVE, horizon=7424)
        if any(trainer_status.get(key) != value for key, value in wanted.items()):
            raise ValueError('Trainer counters do not match the complete event')
    return dict(status='passed', arm=arm, **expected, anchor_draws_checked=256,
                extra_optimizer_steps=0, historical_anchors_forced=False)


def pass_exposure(records):
    summaries = []
    for pass_index in (0, 1):
        rows = [row for row in records if row['pass'] == pass_index]
        summaries.append(dict(pass_index=pass_index, presentations=len(rows),
            replay_presentations=sum(row['replay'] for row in rows),
            prompt_tokens=sum(row['prompt_tokens'] for row in rows),
            response_tokens=sum(row['response_tokens'] for row in rows),
            active_anchors=sum(row['active_anchors'] for row in rows),
            response_forward_backward_presentations=len(rows),
            optimizer_steps=sum(row['optimizer_stepped'] for row in rows)))
    return summaries


def validate_snapshot(state, *, arm, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe,
                      expected_cuda_rng_states=1):
    """Validate replay history separately; keep all native weight/Adam/RNG checks."""
    import random
    import numpy as np
    import torch
    from experiments.osd_components.checkpoints import EXPECTED_KEYS
    if type(completed_event) is not int or not 0 <= completed_event <= MAX_EVENTS:
        raise ValueError('Complete replay boundary must be event 0..8')
    validate_config_text(config_text, arm, original_config_text)
    required = {'event_schema', 'draft_state_dict', 'backend', 'rng', 'global_step',
                'presentations', 'events', 'strategy', 'run_id', 'world_size', 'config_text',
                'algorithm_contract', 'reduction', 'accumulation_steps', 'horizon'}
    if set(state) != required or set(state['backend']) != {'optimizer'}:
        raise ValueError('Unexpected or missing scientific state fields')
    contract = native_algorithm_contract(expected_contract)
    if (contract.get(CONTRACT_KEY) != experiment_contract(arm, binding)
            or contract['online_target_objective']['name'] != OBJECTIVE
            or differences(state['algorithm_contract'], contract)):
        raise ValueError('Foreign replay arm, objective or scientific contract')
    expected = dict(event_schema=SCHEMA, strategy='dflash', world_size=1,
                    config_text=config_text, reduction='response_mean', accumulation_steps=2,
                    horizon=7424, global_step=128 * completed_event,
                    presentations=256 * completed_event)
    if any(state.get(key) != value for key, value in expected.items()) or len(state['events']) != completed_event:
        raise ValueError('Own complete state identity or counters differ')
    previous = []
    for number, event in enumerate(state['events'], 1):
        validate_event_receipt(event, arm=arm, event_number=number,
                               require_completed=False, previous_records=previous)
        previous.extend(event['features'])
    weights = state['draft_state_dict']
    if set(weights) != EXPECTED_KEYS:
        raise ValueError('Require the exact 58 draft parameter names')
    for name, value in weights.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or not value.numel():
            raise ValueError('Invalid native BF16 draft tensor: ' + name)
        _finite(value, name)
    optimizer = state['backend']['optimizer']
    recipe, saved = optimizer['recipe'], optimizer['optimizer_state_dict']
    if differences(recipe, expected_recipe) or recipe['type'] != 'direct_bf16_adamw' or recipe['version'] != 1:
        raise ValueError('Native optimizer recipe or parameter ordering differs')
    if len(saved['param_groups']) != 1:
        raise ValueError('Require one native Adam group')
    group = saved['param_groups'][0]
    options = {key: value for key, value in group.items() if key != 'params'}
    if (differences(options, recipe['optimizer_options']) or recipe['max_grad_norm'] != 1.
            or options['lr'] != 3e-4 or options['weight_decay'] != .01
            or options['betas'] != (.9, .999) or options['eps'] != 1e-8 or options['amsgrad'] is not False):
        raise ValueError('Qualified native Adam hyperparameters differ')
    parameters, ids = recipe['parameters'], group['params']
    if (len(parameters) != len(ids) or len(ids) != 58 or len(set(ids)) != 58
            or {item[0] for item in parameters} != set(weights)
            or set(saved['state']) != (set(ids) if completed_event else set())):
        raise ValueError('Native Adam parameter or moment membership differs')
    for (name, shape, dtype), parameter_id in zip(parameters, ids):
        weight = weights[name]
        if tuple(shape) != tuple(weight.shape) or dtype != str(weight.dtype):
            raise ValueError('Optimizer binding differs: ' + name)
        if not completed_event:
            continue
        moments = saved['state'][parameter_id]
        if set(moments) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Incomplete native Adam state: ' + name)
        counter = moments['step']
        if not isinstance(counter, torch.Tensor) or counter.numel() != 1 or counter.item() != state['global_step']:
            raise ValueError('Native Adam counter differs: ' + name)
        for key in ('exp_avg', 'exp_avg_sq'):
            value = moments[key]
            if value.dtype != torch.bfloat16 or value.shape != weight.shape:
                raise ValueError('Native Adam moment precision/shape differs: ' + name)
            _finite(value, name + '.' + key)
    rng = state['rng']
    if (set(rng) != {'python', 'numpy', 'torch', 'cuda'} or expected_cuda_rng_states not in (0, 1)
            or not isinstance(rng['cuda'], list) or len(rng['cuda']) != expected_cuda_rng_states):
        raise ValueError('Require complete qualified RNG state')
    for value in [rng['torch'], *rng['cuda']]:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel():
            raise ValueError('Malformed Torch RNG payload')
    random.Random(0).setstate(rng['python'])
    np.random.RandomState(0).set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch'])
    return dict(status='passed', arm=arm, completed_event=completed_event,
                global_step=state['global_step'], presentations=state['presentations'],
                adam_state_entries=len(saved['state']), cuda_rng_states=len(rng['cuda']),
                extra_model_passes=0, extra_optimizer_steps=0)


def compare_neutral_states(first, second, *, completed_event=1):
    """Exact decoded scientific equality, excluding arm/path/timing metadata."""
    if completed_event not in (0, 1):
        raise ValueError('Neutral equality is defined only for initialization and event 1')
    views = []
    bindings = []
    for state in (first, second):
        contract = deepcopy(state['algorithm_contract'])
        own = contract.pop(CONTRACT_KEY)
        bindings.append(own['binding'])
        if own != experiment_contract(own['arm'], own['binding']):
            raise ValueError('Foreign neutral-state contract')
        if (len(state['events']) != completed_event or state['global_step'] != completed_event * 128
                or state['presentations'] != completed_event * 256):
            raise ValueError('Neutral comparison requires its exact event boundary')
        schedules = [[{key: row[key] for key in ('record_id', 'pass', 'sampled_anchors',
                     'optimizer_step_after', 'optimizer_stepped', 'learning_rate', 'metrics')}
                      for row in event['records']] for event in state['events']]
        views.append(dict(weights=state['draft_state_dict'], optimizer=state['backend'],
                          rng=state['rng'], contract=contract, schedules=schedules,
                          global_step=state['global_step'], presentations=state['presentations'],
                          horizon=state['horizon'], accumulation_steps=state['accumulation_steps'],
                          reduction=state['reduction']))
    mismatch = differences(*views)
    if bindings[0] != bindings[1] or mismatch:
        raise ValueError('Neutral decoded scientific state or native anchors differ: ' + str(mismatch[:8]))
    return dict(status='passed', completed_event=completed_event,
                exact_weights_adam_rng_counters_anchors=True,
                allowed_difference='arm/run/output/path/timing metadata only', rng_overwritten=False)


def compare_initial_states(first, second):
    result = compare_neutral_states(first, second, completed_event=0)
    result['exact_released_weights_empty_adam_rng'] = True
    return result


def materialize_export(state_path, config_path, export_path, receipt_path=None):
    """Publish a committed state's native export without constructing a trainer.

    The caller first verifies the committed successor and retires its unneeded
    predecessor. Export workspace therefore does not overlap a pending state.
    """
    import torch
    import yaml
    from specforge.export.to_hf import export_to_hf
    state_path, export_path = Path(state_path).absolute(), Path(export_path).absolute()
    if export_path.exists():
        raise ValueError('A deferred export must use a new destination')
    configuration = Path(config_path).read_text()
    raw = yaml.safe_load(configuration)
    draft_config = Path(raw['model']['draft_model_config'])
    if draft_config.is_dir():
        draft_config /= 'config.json'
    started = time.perf_counter()
    with preserve_rng():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['config_text'] != configuration or state['algorithm_contract'].get(CONTRACT_KEY) is None:
            raise ValueError('Export configuration or replay state identity differs')
        export_to_hf(str(state_path), str(draft_config), str(export_path),
                     embedding_source=raw['model']['target_model_path'],
                     embedding_key=raw['model']['embedding_key'])
        _export_equal(state, export_path)
    receipt = dict(status='passed', export_status='materialized', export_exact=True,
                   state_path=str(state_path), export_path=str(export_path),
                   global_step=state['global_step'], export_and_verify_s=time.perf_counter() - started,
                   extra_model_passes=0, extra_optimizer_steps=0)
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    return receipt


class WorkloadReplayTrainer(SpecForgeEventTrainer):
    """One native runtime at a time; restore each arm's complete boundary."""

    # These helpers observe or verify state only; none allocates presentations.
    _append_observation = full_k.FullKTrainer._append_observation
    _install_observations = full_k.FullKTrainer._install_observations

    def __init__(self, config_path, output_root, *, arm, binding, original_config_text,
                 resume_state=None, completed_event=0, expected_cuda_rng_states=1,
                 defer_export=True):
        if type(completed_event) is not int or not 0 <= completed_event <= MAX_EVENTS:
            raise ValueError('Require complete replay boundary 0..8')
        if completed_event and resume_state is None:
            raise ValueError('Continuation requires its exact own recovery state')
        experiment_contract(arm, binding)
        self.arm, self.binding = arm, binding
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.completed_event = completed_event
        self.defer_export = defer_export
        self.response_records, self.window_records = [], []
        self.work, self._work_handles = {}, []
        self._pass_plan = None
        self._packet_stats = {}
        self._initial_verified = False
        self.restore_s = 0.
        started = time.perf_counter()
        super().__init__(config_path, output_root, resume_state=resume_state,
                         objective=OBJECTIVE, reduction='response_mean', ranking_lambda=0.)
        self.construction_and_restore_s = time.perf_counter() - started
        self.construction_s = self.construction_and_restore_s - self.restore_s

    def _validate_config(self):
        super()._validate_config()
        validate_config_text(self.config_text, self.arm, self.original_config_text)

    def _build_runtime(self):
        super()._build_runtime()
        self.contract[CONTRACT_KEY] = experiment_contract(self.arm, self.binding)
        self._install_observations()

    def _export(self, export_path):
        if not self.defer_export:
            return super()._export(export_path)

    def materialize_export(self, export_path=None, receipt_path=None):
        if not self.events:
            raise ValueError('Require a committed event before export')
        event_dir = self.events_root / self.events[-1]['event_id']
        export_path = export_path or event_dir / 'export'
        receipt = materialize_export(self.latest, self.config_path, export_path,
                                     receipt_path or event_dir / 'EXPORT_CHECK.json')
        result_path = event_dir / 'RESULT.json'
        result = json.loads(result_path.read_text())
        result.update(export_status='materialized', export_path=str(export_path),
                      checkpoint_path=str(export_path), deferred_export_receipt=receipt)
        write_json(result_path, result)
        return receipt

    def validation_arguments(self):
        return dict(arm=self.arm, binding=self.binding, config_text=self.config_text,
                    original_config_text=self.original_config_text, expected_contract=self.contract,
                    expected_recipe=self.backend.optimizer.recipe,
                    expected_cuda_rng_states=self.expected_cuda_rng_states)

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        started = time.perf_counter()
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state, export_path)
                receipt['export_exact'] = True
            receipt['live'] = full_k.verify_live(self, state)
        receipt['wall_s'] = time.perf_counter() - started
        if receipt_path is not None:
            write_json(receipt_path, receipt)
        return receipt

    def close(self):
        if self.closed:
            return
        for handle in self._work_handles:
            handle.remove()
        self._work_handles.clear()
        if hasattr(self, '_native_forward'):
            self.author_strategy.forward_loss = self._native_forward
            self.backend.optimizer.step = self._native_step
            self.bundle.model._chunk_loss = self._native_chunk
            self.bundle.model.end_event = self._native_end
        super().close()
        # The borrowed observers retain bound model methods. Drop those owners
        # as well so serial arm construction really releases the old runtime.
        for name in ('_native_forward', '_native_step', '_native_chunk', '_native_end'):
            if hasattr(self, name):
                delattr(self, name)
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def _restore(self, path):
        import torch
        started = time.perf_counter()
        path = Path(path)
        if path.suffix == '.json':
            reference = json.loads(path.read_text())
            if (self.completed_event != 0 or reference['schema'] != 'track2_shared_initial_v1'
                    or reference['arm'] != self.arm or reference['binding'] != self.binding
                    or reference['config_text'] != self.config_text
                    or reference['run_id'] != self.cfg.run_id
                    or reference['comparison']['exact_released_weights_empty_adam_rng'] is not True
                    or file_sha256(reference['state_path']) != reference['sha256']):
                raise ValueError('Shared initial reference identity or content differs')
            state = torch.load(reference['state_path'], map_location='cpu', weights_only=False)
            if state['algorithm_contract'][CONTRACT_KEY] != experiment_contract('K', self.binding):
                raise ValueError('Shared initial payload must be this campaign\'s new K initialization')
            # Explicit per-arm restore metadata accompanies the common tensor payload.
            state = dict(state, config_text=self.config_text, run_id=self.cfg.run_id,
                         algorithm_contract=self.contract)
        else:
            state = torch.load(path, map_location='cpu', weights_only=False)
        validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        self.backend.load_state_dict(state['backend'])
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        restore_rng(state['rng'])
        full_k.verify_live(self, state, receipt_path=self.output_root / 'RESUME_CHECK.json')
        self._initial_verified = self.completed_event == 0
        self.restore_s = time.perf_counter() - started

    def save_initial(self, comparator_path=None, *, shared_snapshot=False):
        """Independently verify each initialization; optionally retain one tensor copy."""
        import torch
        destination = self.output_root / ('initial_reference.json' if shared_snapshot else 'initial_state.pt')
        if destination.exists() or self.global_step or self.presentations or self.events or self._initial_verified:
            raise ValueError('Initial snapshot must be genuinely new and empty')
        if (self.arm != 'K') != (comparator_path is not None) or (shared_snapshot and self.arm == 'K'):
            raise ValueError('U/R must compare against this campaign\'s independently initialized K')
        with preserve_rng():
            state = full_k.live_state(self)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=0)
            _export_equal(state, self.cfg.model.draft_checkpoint_path)
            if comparator_path is not None:
                first = torch.load(comparator_path, map_location='cpu', weights_only=False)
                if first['algorithm_contract'][CONTRACT_KEY] != experiment_contract('K', self.binding):
                    raise ValueError('Initial comparator is not the current released K')
                receipt['paired_initialization'] = compare_initial_states(first, state)
            if shared_snapshot:
                write_json(destination, dict(schema='track2_shared_initial_v1', arm=self.arm,
                    binding=self.binding, state_path=str(Path(comparator_path).absolute()),
                    sha256=file_sha256(comparator_path), config_text=self.config_text,
                    run_id=self.cfg.run_id, comparison=receipt['paired_initialization']))
            else:
                save_torch(destination, state)
                decoded = torch.load(destination, map_location='cpu', weights_only=False)
                full_k.verify_live(self, decoded)
            write_json(self.output_root / 'INITIAL_CHECK.json', dict(receipt, export_exact=True,
                       state_path=str(destination), shared_tensor_payload=shared_snapshot))
        self._initial_verified = True
        return dict(receipt, state_path=str(destination), shared_tensor_payload=shared_snapshot)

    def _prepare_event(self, event_id, feature_paths, passes):
        started = time.perf_counter()
        number = len(self.events) + 1
        if (not 1 <= number <= MAX_EVENTS or event_id != f'event_{number:03d}'
                or self.global_step != (number - 1) * 128 or self.presentations != (number - 1) * 256
                or type(passes) is not int or passes != 2):
            raise ValueError('Require the next complete 128-response/two-pass replay event')
        if not self.events and not self._initial_verified:
            raise ValueError('Verify and retain initial state before serving or training')
        previous = [row for event in self.events for row in event['features']]
        self._pass_plan = validate_pass_plan(self._pass_plan, arm=self.arm,
                                             event_number=number, previous_records=previous)
        self._active_event = event_id
        self.response_records.clear()
        self.window_records.clear()
        self.work.clear()
        unique = {row['record_id']: row for rows in self._pass_plan for row in rows}
        self._packet_stats = {}
        for record_id, row in unique.items():
            if file_sha256(row['path']) != row['sha256']:
                raise ValueError('Prepared packet differs from its immutable content binding: ' + record_id)
            self._packet_stats[record_id] = _file_identity(row['path'])
        decoded = super()._prepare_event(event_id, [row['path'] for row in unique.values()], passes)
        metadata = {}
        for row, bound in zip(decoded, unique.values()):
            if row['record_id'] != bound['record_id']:
                raise ValueError('Prepared packet original record identity differs from binding')
            metadata[row['record_id']] = {**row, **bound}
        self._pass_plan = [[metadata[row['record_id']] for row in rows] for rows in self._pass_plan]
        self._feature_admission_s = time.perf_counter() - started
        return self._pass_plan[0]

    def _batch(self, raw):
        # The expensive content check happened once at admission. Detect subsequent
        # replacement or writes on every native load, without rehashing large tensors.
        record_id = raw['record_id']
        records = {row['record_id']: row for rows in self._pass_plan for row in rows}
        if record_id not in records or _file_identity(records[record_id]['path']) != self._packet_stats.get(record_id):
            raise ValueError('An admitted immutable prepared packet changed during the event')
        return super()._batch(raw)

    def _train_passes(self, features, passes):
        write_json(self.events_root / self._active_event / 'PASS_PLAN.json', self._pass_plan)
        rows = []
        self._pass_loop_costs = []
        for pass_index, input_list in enumerate(self._pass_plan):
            # Each call executes exactly the original one-pass loop, including
            # its own randperm(128), fresh anchor draws and native train_step.
            started = time.perf_counter()
            observed = super()._train_passes(input_list, 1)
            elapsed = time.perf_counter() - started
            self._pass_loop_costs.append(dict(pass_index=pass_index, total_loop_s=elapsed,
                input_load_and_loop_bookkeeping_s=elapsed - sum(row['training_s'] for row in observed)))
            for row in observed:
                row['pass'] = pass_index
            rows.extend(observed)
        if len(rows) != len(self.response_records) or len(self.window_records) != 128:
            raise ValueError('Native forward/window observation accounting differs')
        metadata = {row['record_id']: row for inputs in self._pass_plan for row in inputs}
        fresh = {row['record_id'] for row in features}
        for row, observed in zip(rows, self.response_records):
            if (row['record_id'] != observed['record_id'] or row['pass'] != observed['pass_index']
                    or row['sampled_anchors'] != observed['sampled_anchors']):
                raise ValueError('Response observation identity or native anchor binding differs')
            record = metadata[row['record_id']]
            row.update(response_observation=observed, global_ordinal=record['global_ordinal'],
                       replay=row['record_id'] not in fresh, prompt_tokens=record['prompt_length'],
                       response_tokens=record['response_tokens'],
                       active_anchors=sum(sum(draw['keep_mask'][0]) for draw in row['sampled_anchors']))
        return rows

    def _checkpoint_state(self, event):
        event['pass_plan'] = self._pass_plan
        event['pass_exposure'] = pass_exposure(event['records'])
        event['runtime_costs'] = dict(construction_s=self.construction_s, restore_s=self.restore_s,
            construction_and_restore_s=self.construction_and_restore_s,
            feature_admission_io_and_validation_s=self._feature_admission_s)
        event['pass_loop_costs'] = self._pass_loop_costs
        event['export_status'] = 'deferred' if self.defer_export else 'materialized'
        validate_event_receipt(event, arm=self.arm, event_number=len(self.events) + 1,
                               require_completed=False,
                               previous_records=[row for prior in self.events for row in prior['features']])
        return super()._checkpoint_state(event)

    def train_event(self, event_id, feature_paths=None, passes=2, *, pass_plan=None,
                    second_pass=None, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('Replay permits no reference probe or forced historical schedule')
        if pass_plan is not None and (feature_paths is not None or second_pass is not None):
            raise ValueError('Supply one explicit pass plan or two packet-binding lists')
        self._pass_plan = pass_plan if pass_plan is not None else [feature_paths, second_pass]
        result = super().train_event(event_id, [], passes)
        if self.defer_export:
            result['checkpoint_path'] = None
            result['costs']['export_s'] = 0.
            write_json(self.events_root / event_id / 'RESULT.json', result)
        validate_event_receipt(result, arm=self.arm, event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result
