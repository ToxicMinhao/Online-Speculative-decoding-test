"""Three isolated CODE arms on the existing native online training engine.

Only experiment contracts, observations and complete-state checks live here.
SpecForgeEventTrainer owns forwards, sampling, response averaging, backwards,
clipping, BF16 AdamW, atomic recovery and serving exports.
"""
from copy import deepcopy
import math
import pickle
import time

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.mathqa_fixed_lr_transfer.training import FORBIDDEN_CONTROLLERS
from experiments.persistent_prefix_withholding.qualification import (
    isolated_backend_for_load, tensor_identity,
)
from experiments.unchanged_c_mathqa_transfer.state import validate_allocation_observation
from online_feedback.adaflash_repo import install_adaflash_repo, adaflash_repo_contract, ReleasedAdaFlashOPD
from online_feedback.kplus import install_kplus, kplus_contract, KPlusRedistribution
from online_feedback.prefix_allocation import install_prefix_allocation, prefix_allocation_contract
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

from .configuration import historical_view, validate_config_text
from .contract import ARMS, EVENTS, WINDOW, PASSES

CONTRACT_KEY = 'opencode_training_v1'
WITNESS_SCHEMA = 'opencode_initial_witness_v1'
WEIGHT_FILE_BYTES = 2_097_259_104
RECOVERY_METADATA_BYTES = 500_000_000
MAX_RECOVERY_BYTES = 3 * WEIGHT_FILE_BYTES + RECOVERY_METADATA_BYTES
OBJECTIVES = {'O_CODE58': 'osd', 'AREPO_CODE58': 'osd', 'C_CODE58': 'osd_first_error'}


def recovery_size_plan(state):
    """Count metadata without materializing a model-sized serialized buffer."""
    import torch
    class Counter:
        size = 0
        def write(self, value):
            self.size += len(value)
            return len(value)
        def flush(self):
            pass

    tensor_bytes = 0
    def descriptor(value):
        nonlocal tensor_bytes
        if not torch.is_tensor(value):
            return value
        tensor_bytes += value.numel() * value.element_size()
        return dict(tensor_shape=list(value.shape), tensor_dtype=str(value.dtype))
    optimizer = state['backend']['optimizer']
    saved = optimizer['optimizer_state_dict']
    metadata = dict(state,
        draft_state_dict={key: descriptor(value) for key, value in state['draft_state_dict'].items()},
        backend=dict(state['backend'], optimizer=dict(optimizer, optimizer_state_dict=dict(saved,
            state={key: {field: descriptor(value) for field, value in entry.items()}
                   for key, entry in saved['state'].items()}))))
    counter = Counter()
    pickle.dump(metadata, counter, protocol=2)
    metadata_with_headers = counter.size + 5_000_000
    if tensor_bytes > 3 * WEIGHT_FILE_BYTES or metadata_with_headers > RECOVERY_METADATA_BYTES:
        raise RuntimeError('Recovery exceeds the declared tensor/500MB metadata envelope')
    return dict(status='passed', tensor_payload_bytes=tensor_bytes,
        metadata_pickle_bytes=counter.size, serialization_header_reserve_bytes=5_000_000,
        metadata_envelope_bytes=RECOVERY_METADATA_BYTES, complete_state_bound_bytes=MAX_RECOVERY_BYTES)


def validate_recovery_file(path):
    from pathlib import Path
    size = Path(path).stat().st_size
    if size > MAX_RECOVERY_BYTES:
        raise RuntimeError('Serialized recovery exceeds the declared complete-state envelope')
    return dict(status='passed', path=str(path), bytes=size, maximum_bytes=MAX_RECOVERY_BYTES)


def experiment_contract(arm, binding):
    if arm not in ARMS:
        raise ValueError('Unknown CODE training arm')
    native.experiment_contract('K', binding)  # Existing strict binding syntax.
    return dict(schema=CONTRACT_KEY, arm=arm, binding=binding,
        dataset='OpenCodeInstruct', seed=17, context_limit=4096,
        initialization='independent_released_weights_fresh_Adam',
        feedback='own_native_responses', maximum_events=58, horizon=7424,
        learning_rate=3e-4, objective=OBJECTIVES[arm], reference=None,
        loss=adaflash_repo_contract() if arm == 'AREPO_CODE58' else None,
        redistribution=kplus_contract() if arm == 'C_CODE58' else None,
        allocation=prefix_allocation_contract(.25) if arm == 'C_CODE58' else None)


def install_arm(model, arm, *, allocation_cap=.25):
    """Select existing numerical components; O installs no extension."""
    if arm not in ARMS or model.objective != OBJECTIVES[arm]:
        raise ValueError('Training objective does not belong to this CODE arm')
    if arm == 'AREPO_CODE58':
        install_adaflash_repo(model)
    elif arm == 'C_CODE58':
        install_kplus(model)
        if allocation_cap is not None:
            install_prefix_allocation(model, relative_cap=allocation_cap)


def _native_contract(contract, *, arm, binding):
    """Checked metadata view for the unchanged complete-state structural gate."""
    if (contract.get(CONTRACT_KEY) != experiment_contract(arm, binding)
            or native.CONTRACT_KEY in contract
            or contract['online_target_objective']['name'] != OBJECTIVES[arm]):
        raise ValueError('Changed CODE objective, configuration or arm contract')
    result = deepcopy(contract)
    del result[CONTRACT_KEY]
    result[native.CONTRACT_KEY] = native.experiment_contract('K', binding)
    result['online_target_objective']['name'] = native.OBJECTIVE
    return result


def validate_event_receipt(event, *, arm, binding, event_number,
                           require_completed=True, trainer_status=None):
    if (event.get('method') != arm or event.get('dataset') != 'OpenCodeInstruct'
            or event.get('objective') != OBJECTIVES[arm]
            or event.get(CONTRACT_KEY) != experiment_contract(arm, binding)):
        raise ValueError('Event identity or CODE numerical contract changed')
    for row in event['records']:
        observation = row['response_observation']
        if (observation['record_id'] != row['record_id']
                or observation['pass_index'] != row['pass']
                or observation['sampled_anchors'] != row['sampled_anchors']):
            raise ValueError('Original-forward observation lost its response/anchor binding')
        scalars = [observation['loss'], *observation['sum_metrics'].values()]
        scalars.extend(value for pair in observation['ratio_metrics'].values() for value in pair)
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError('Nonfinite original-forward CODE observation')
        if arm == 'C_CODE58':
            validate_allocation_observation(observation)
        elif arm == 'AREPO_CODE58':
            if observation['sum_metrics']['adaflash_response_count'] != 1:
                raise ValueError('AREPO observation must count each original response once')
    status = dict(trainer_status, objective=native.OBJECTIVE) if trainer_status else None
    checked = native.validate_event_receipt(dict(event, objective=native.OBJECTIVE),
        arm='K', event_number=event_number, require_completed=require_completed,
        trainer_status=status)
    return dict(checked, arm=arm, objective=OBJECTIVES[arm], dataset='OpenCodeInstruct')


def validate_snapshot(state, *, arm, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe,
                      expected_cuda_rng_states=1):
    """Keep actual CODE fields immutable while reusing native tensor/RNG checks."""
    validate_config_text(config_text, original_config_text)
    if state['config_text'] != config_text or differences(state['algorithm_contract'], expected_contract):
        raise ValueError('Saved CODE configuration/contract differs from this own trajectory')
    for number, event in enumerate(state['events'], 1):
        validate_event_receipt(event, arm=arm, binding=binding, event_number=number,
                               require_completed=False)
    view_config = historical_view(config_text)
    view_contract = _native_contract(state['algorithm_contract'], arm=arm, binding=binding)
    view = dict(state, config_text=view_config, algorithm_contract=view_contract,
        events=[dict(event, objective=native.OBJECTIVE) for event in state['events']])
    checked = native.validate_snapshot(view, arm='K', binding=binding,
        config_text=view_config, original_config_text=historical_view(original_config_text),
        completed_event=completed_event, expected_contract=view_contract,
        expected_recipe=expected_recipe, expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(checked, arm=arm, context_limit=4096, actual_CODE_state_unchanged=True)


def initial_witness(state):
    """Exact initialization evidence, without another copy of released tensors."""
    optimizer = state['backend']['optimizer']['optimizer_state_dict']
    if (state['global_step'] or state['presentations'] or state['events'] or optimizer['state']):
        raise ValueError('Initial witness requires genuinely empty Adam and event history')
    keys = ('backend', 'rng', 'global_step', 'presentations', 'events', 'strategy',
            'world_size', 'config_text', 'algorithm_contract', 'reduction',
            'accumulation_steps', 'horizon')
    return dict(schema=WITNESS_SCHEMA,
        weights={name: tensor_identity(value) for name, value in state['draft_state_dict'].items()},
        **{key: deepcopy(state[key]) for key in keys})


def compare_initial_witnesses(first, second):
    """Both independently validated arms must share real tensors/Adam/RNG."""
    import yaml
    views = []
    for witness in (first, second):
        if witness['schema'] != WITNESS_SCHEMA:
            raise ValueError('Require a compact CODE initialization witness')
        contract = deepcopy(witness['algorithm_contract'])
        identity = contract[CONTRACT_KEY]
        _native_contract(contract, arm=identity['arm'], binding=identity['binding'])
        if (witness['global_step'] or witness['presentations'] or witness['events']
                or witness['backend']['optimizer']['optimizer_state_dict']['state']):
            raise ValueError('Initial witness contains trained state')
        del contract[CONTRACT_KEY]
        del contract['online_target_objective']  # Checked intended objective only.
        config = yaml.safe_load(witness['config_text'])
        if config['data']['max_length'] != 4096:
            raise ValueError('Initial CODE witness has a different context')
        for key in ('run_id', 'output_dir'):
            config.pop(key, None)
        views.append({**{key: witness[key] for key in (
            'weights', 'backend', 'rng', 'global_step', 'presentations', 'events',
            'strategy', 'world_size', 'reduction', 'accumulation_steps', 'horizon')},
            'contract': contract, 'config': config})
    mismatch = differences(*views)
    if mismatch:
        raise ValueError('Independent CODE initialization differs: ' + repr(mismatch[:12]))
    return dict(status='passed', exact_weights_empty_Adam_RNG=True,
        allowed_changes=['arm/objective contract', 'run_id', 'output_dir'], rng_overwritten=False)


def compare_historical_initial(reference, current, *, original_config_text,
                               expected_cuda_rng_states=1):
    """Authenticate historical K0, then compare only declared CODE differences."""
    import yaml
    historical = reference['algorithm_contract'][native.CONTRACT_KEY]
    if historical['arm'] != 'K':
        raise ValueError('The historical comparator must be released native K step zero')
    native.validate_snapshot(reference, arm='K', binding=historical['binding'],
        config_text=reference['config_text'], original_config_text=historical_view(original_config_text),
        completed_event=0, expected_contract=reference['algorithm_contract'],
        expected_recipe=reference['backend']['optimizer']['recipe'],
        expected_cuda_rng_states=expected_cuda_rng_states)
    current_identity = current['algorithm_contract'][CONTRACT_KEY]
    _native_contract(current['algorithm_contract'], arm=current_identity['arm'],
                     binding=current_identity['binding'])
    views = []
    for state, key in ((reference, native.CONTRACT_KEY), (current, CONTRACT_KEY)):
        witness = initial_witness(state)
        config = yaml.safe_load(historical_view(state['config_text']))
        for field in ('run_id', 'output_dir'):
            config.pop(field, None)
        contract = deepcopy(state['algorithm_contract'])
        del contract[key]
        del contract['online_target_objective']
        views.append({**{field: witness[field] for field in (
            'weights', 'backend', 'rng', 'global_step', 'presentations', 'events',
            'strategy', 'world_size', 'reduction', 'accumulation_steps', 'horizon')},
            'contract': contract, 'config': config})
    mismatch = differences(*views)
    if mismatch:
        raise ValueError('Historical released initialization differs: ' + repr(mismatch[:12]))
    return dict(status='passed', exact_weights_empty_Adam_RNG=True,
        comparator='independently validated historical native K0',
        allowed_changes=['declared CODE arm/objective contract', 'data.max_length:2272-to-4096',
                         'run_id', 'output_dir'], rng_overwritten=False)


def verify_live(trainer, state, receipt_path=None):
    mismatch = differences(native.live_state(trainer), state)
    model = trainer.bundle.model
    trainer.check_installed_arm()
    if (trainer.closed or trainer.failed or trainer.core.accumulation_remainder
            or trainer.core._micro != trainer.presentations or trainer.core._ratio_totals
            or trainer.core._sum_totals or trainer._presentation_schedule is not None
            or not model.training or model.reference is not None or model.ranking_enabled
            or any(parameter.grad is not None for parameter in trainer.bundle.draft_model.parameters())):
        mismatch.append('unclean_live_boundary')
    result = dict(status='failed' if mismatch else 'passed', exact_resume=not mismatch,
        mismatches=mismatch[:32], global_step=trainer.global_step,
        presentations=trainer.presentations, extra_model_passes=0, extra_optimizer_steps=0)
    if receipt_path is not None:
        write_json(receipt_path, result)
    if mismatch:
        raise ValueError('Live CODE state differs: ' + repr(mismatch[:8]))
    return result


def authenticate_resume(*, state_path, export_path, initial_witness_path,
                        arm, binding, config_text, original_config_text, completed_event,
                        state_identity, export_identity, expected_cuda_rng_states=1):
    """Read-only own-state authentication; this does not authorize a restart.

    The caller must separately reconcile local/shared work ledgers, the complete
    publication marker and available constructor/time/storage budgets. No model
    runtime is constructed, no RNG is restored, and no optimizer is applied.
    """
    import torch
    from pathlib import Path
    from experiments.ok_coding.storage import identity
    if identity(state_path) != state_identity or identity(Path(export_path)/'model.safetensors') != export_identity:
        raise ValueError('Own recovery or serving export identity changed')
    with preserve_rng():
        witness = torch.load(initial_witness_path, map_location='cpu', weights_only=False)
        if witness.get('schema') != WITNESS_SCHEMA:
            raise ValueError('Resume needs this CODE trajectory\'s compact initialization witness')
        compare_initial_witnesses(witness, witness)
        if witness['config_text'] != config_text:
            raise ValueError('Resume configuration differs from its initial witness')
        state = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
        result = validate_snapshot(state, arm=arm, binding=binding, config_text=config_text,
            original_config_text=original_config_text, completed_event=completed_event,
            expected_contract=witness['algorithm_contract'],
            expected_recipe=witness['backend']['optimizer']['recipe'],
            expected_cuda_rng_states=expected_cuda_rng_states)
        _export_equal(state, export_path)
        size = validate_recovery_file(state_path)
    return dict(result, serialized_size=size, exact_export=True, automatic_resume=False,
                optimizer_applications=0, trainer_initializations=0, RNG_restored=False)


class CodeTrainer(native.FullKTrainer):
    """One own-feedback trajectory; fresh processes isolate different arms."""
    def __init__(self, config_path, output_root, *, arm, binding, original_config_text,
                 resume_state=None, completed_event=0, expected_cuda_rng_states=1,
                 qualification_only=False):
        experiment_contract(arm, binding)
        if (type(completed_event) is not int or not 0 <= completed_event <= EVENTS
                or (completed_event > 0) != (resume_state is not None)):
            raise ValueError('Resume requires an exact own complete event boundary')
        self.arm, self.binding = arm, binding
        self.original_config_text = original_config_text
        self.completed_event = completed_event
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.qualification_only = qualification_only
        self.allocation_cap = .25
        self.step_observer = None
        self.response_records, self.window_records = [], []
        self.work, self._work_handles = {}, []
        SpecForgeEventTrainer.__init__(self, config_path, output_root, resume_state=resume_state,
            objective=OBJECTIVES[arm], reduction='response_mean', ranking_lambda=0.)

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        validate_config_text(self.config_text, self.original_config_text)

    def _build_runtime(self):
        import rich.style  # Materialize logging identifiers before native seed17.
        SpecForgeEventTrainer._build_runtime(self)
        install_arm(self.bundle.model, self.arm)
        self.contract[CONTRACT_KEY] = experiment_contract(self.arm, self.binding)
        self._install_observations()
        self._ordinary_forward, self._ordinary_step = self._native_forward, self._native_step

        def checked_forward(*args, **kwargs):
            import torch
            self.check_installed_arm()
            output = self._ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(output.loss).all()):
                raise ValueError('Nonfinite CODE response loss')
            return output

        self._native_forward = checked_forward

        def checked_step():
            if self.step_observer is not None:
                if not self.qualification_only:
                    raise ValueError('Preclip gradient observer is qualification-only')
                self.step_observer(self)
            return self._ordinary_step()

        self._native_step = checked_step

    def check_installed_arm(self):
        model = self.bundle.model
        actual = {name: getattr(model, name, None) for name in
            ('first_error_redistribution', 'point_loss_extension', 'prefix_allocation')}
        wanted = {'O_CODE58': (False, False, False), 'AREPO_CODE58': (False, True, False),
                  'C_CODE58': (True, False, self.allocation_cap is not None)}[self.arm]
        if (model.objective != OBJECTIVES[self.arm]
                or tuple(value is not None for value in actual.values()) != wanted):
            raise ValueError('CODE arm has missing or foreign objective extensions')
        forbidden = set(FORBIDDEN_CONTROLLERS) - {'point_loss_extension', 'prefix_allocation'}
        if any(getattr(model, name, None) is not None for name in forbidden):
            raise ValueError('Foreign controller is active in CODE training')
        if self.arm == 'AREPO_CODE58' and not isinstance(actual['point_loss_extension'], ReleasedAdaFlashOPD):
            raise ValueError('AREPO must use the literal released-loss implementation')
        if self.arm == 'C_CODE58' and not isinstance(actual['first_error_redistribution'], KPlusRedistribution):
            raise ValueError('C must use the unchanged K+ implementation')
        if self.arm == 'C_CODE58' and self.allocation_cap is not None:
            if actual['prefix_allocation'].contract != prefix_allocation_contract(self.allocation_cap):
                raise ValueError('C allocation changed during training')

    def qualification_allocation(self, cap):
        if not self.qualification_only or self.arm != 'C_CODE58' or cap not in (None, 0., .25):
            raise ValueError('Allocation controls are restricted to disposable C qualification')
        self.bundle.model.prefix_allocation = None
        self.allocation_cap = cap
        if cap is not None:
            install_prefix_allocation(self.bundle.model, relative_cap=cap)

    def _append_observation(self, name, value):
        if name == 'RESPONSE_OBSERVATIONS.jsonl' and self.arm == 'C_CODE58':
            controller = self.bundle.model.prefix_allocation
            value['prefix_allocation'] = deepcopy(controller.last_summary) if controller else {'enabled': False}
        return native.FullKTrainer._append_observation(self, name, value)

    def validation_arguments(self):
        return dict(arm=self.arm, binding=self.binding, config_text=self.config_text,
            original_config_text=self.original_config_text, expected_contract=self.contract,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def save_initial(self, comparator_path=None):
        import torch
        destination = self.output_root / 'INITIAL_WITNESS.pt'
        if destination.exists():
            raise ValueError('Initialization witness already exists')
        with preserve_rng():
            state = native.live_state(self)
            checked = validate_snapshot(state, **self.validation_arguments(), completed_event=0)
            witness = initial_witness(state)
            save_torch(destination, witness)
            checked.update(status='checking', witness_path=str(destination),
                full_released_tensor_copy_saved=False, rng_overwritten=False)
            write_json(self.output_root / 'INITIAL_CHECK.json', checked)
            try:
                _export_equal(state, self.cfg.model.draft_checkpoint_path)
                if comparator_path is not None:
                    first = torch.load(comparator_path, map_location='cpu', weights_only=False)
                    checked['paired_initialization'] = (
                        compare_initial_witnesses(first, witness) if first.get('schema') == WITNESS_SCHEMA
                        else compare_historical_initial(first, state,
                            original_config_text=self.original_config_text,
                            expected_cuda_rng_states=self.expected_cuda_rng_states))
                decoded = torch.load(destination, map_location='cpu', weights_only=False)
                if differences(witness, decoded):
                    raise ValueError('Compact initialization witness did not serialize exactly')
                verify_live(self, state)
            except BaseException as error:
                checked.update(status='failed', error=repr(error))
                write_json(self.output_root / 'INITIAL_CHECK.json', checked)
                raise
            checked.update(status='passed', export_exact=True)
            write_json(self.output_root / 'INITIAL_CHECK.json', checked)
        return checked

    def _restore(self, path):
        import torch
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        device = next(self.bundle.draft_model.parameters()).device
        self.backend.load_state_dict(isolated_backend_for_load(state['backend'], device))
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        restore_rng(state['rng'])
        verify_live(self, state, self.output_root / 'RESUME_CHECK.json')

    def _prepare_event(self, event_id, feature_paths, passes):
        number = len(self.events) + 1
        if (self.qualification_only or not 1 <= number <= EVENTS
                or event_id != f'event_{number:03d}' or len(feature_paths) != WINDOW
                or type(passes) is not int or passes != PASSES or self.global_step != (number - 1) * 128
                or self.presentations != (number - 1) * 256):
            raise ValueError('Require the next complete own 128-response CODE event')
        if not self.events and not (self.output_root / 'INITIAL_WITNESS.pt').exists():
            raise ValueError('Authenticate released initialization before collecting feedback')
        self._active_event = event_id
        self.response_records.clear()
        self.window_records.clear()
        self.work.clear()
        features = SpecForgeEventTrainer._prepare_event(self, event_id, feature_paths, passes)
        if any(len(item['record_id'].encode()) > 1024 or len(item['path'].encode()) > 4095
               for item in features):
            raise ValueError('Feature identity/path exceeds the declared recovery metadata envelope')
        previous = {item['record_id'] for event in self.events for item in event['features']}
        if previous.intersection(item['record_id'] for item in features):
            raise ValueError('Own-feedback CODE trajectory reuses an earlier response identity')
        return features

    def _checkpoint_state(self, event):
        event.update(method=self.arm, dataset='OpenCodeInstruct',
                     **{CONTRACT_KEY: experiment_contract(self.arm, self.binding)})
        validate_event_receipt(event, arm=self.arm, binding=self.binding,
                               event_number=len(self.events) + 1, require_completed=False)
        state = SpecForgeEventTrainer._checkpoint_state(self, event)
        write_json(self.events_root/event['event_id']/'RECOVERY_SIZE_PLAN.json', recovery_size_plan(state))
        return state

    def _train_passes(self, features, passes):
        rows = native.FullKTrainer._train_passes(self, features, passes)
        metadata = {item['record_id']: item for item in features}
        for row in rows:
            source = metadata[row['record_id']]
            row['prompt_tokens'] = source['prompt_length']
            row['response_tokens'] = source['response_tokens']
            row['active_anchors'] = sum(sum(draw['keep_mask'][0]) for draw in row['sampled_anchors'])
        return rows

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None,
                    presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('CODE uses natural own-response sampling without reference probes')
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        write_json(self.events_root/event_id/'RECOVERY_SIZE_ACTUAL.json', validate_recovery_file(self.latest))
        validate_event_receipt(result, arm=self.arm, binding=self.binding,
                               event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def run_pair(self, pair, pass_index=0, ordinal=0):
        from experiments.bounded_prefix_allocation.training import CandidateTrainer
        if not self.qualification_only or self._presentation_schedule is not None:
            raise ValueError('Disposable pairs require qualification and natural sampling')
        return CandidateTrainer.run_pair(self, pair, pass_index, ordinal)

    def save_qualification(self, path):
        """Full temporary state, explicitly distinct from production recovery."""
        from experiments.prefix_ranking.qualify import cpu_copy
        if not self.qualification_only or self.core.accumulation_remainder:
            raise ValueError('Require a disposable empty-accumulation boundary')
        state = cpu_copy(native.live_state(self))
        controller = getattr(self.bundle.model, 'prefix_allocation', None)
        metadata = dict(schema='opencode_disposable_v1', arm=self.arm,
            allocation_cap=self.allocation_cap, active_event=getattr(self, '_active_event', None),
            response_records=deepcopy(self.response_records), window_records=deepcopy(self.window_records),
            anchor_observations=deepcopy(self._anchor_observations), model_costs=deepcopy(self.bundle.model.costs),
            allocation_summary=deepcopy(controller.last_summary) if controller else None)
        save_torch(path, dict(state, qualification_state=metadata))
        return dict(path=str(path), global_step=self.global_step, presentations=self.presentations,
                    production_recovery=False, extra_optimizer_steps=0)

    def restore_qualification(self, path, *, runtime=None):
        import torch
        if not self.qualification_only:
            raise ValueError('Disposable restoration is forbidden in production')
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        metadata = payload.pop('qualification_state')
        if (metadata['schema'] != 'opencode_disposable_v1' or metadata['arm'] != self.arm
                or differences(payload['algorithm_contract'], self.contract)
                or payload['config_text'] != self.config_text or payload['events']
                or payload['presentations'] != 2 * payload['global_step']
                or not 0 <= payload['global_step'] <= 3):
            raise ValueError('Foreign or incomplete disposable CODE state')
        self.backend.optimizer.zero_grad()
        self.core._ratio_totals.clear()
        self.core._sum_totals.clear()
        self.bundle.draft_model.load_state_dict(payload['draft_state_dict'], strict=True)
        device = next(self.bundle.draft_model.parameters()).device
        self.backend.load_state_dict(isolated_backend_for_load(payload['backend'], device))
        self.global_step, self.presentations = payload['global_step'], payload['presentations']
        self.core._micro = self.presentations
        self.events = []
        if self.arm == 'C_CODE58':
            self.qualification_allocation(metadata['allocation_cap'])
        self.response_records = deepcopy(metadata['response_records'])
        self.window_records = deepcopy(metadata['window_records'])
        self._anchor_observations[:] = deepcopy(metadata['anchor_observations'])
        self._active_event = metadata['active_event']
        self.bundle.model.costs = deepcopy(metadata['model_costs'])
        if self.arm == 'C_CODE58' and self.bundle.model.prefix_allocation is not None:
            self.bundle.model.prefix_allocation.last_summary = deepcopy(metadata['allocation_summary'])
        self._presentation_schedule = None
        restore_rng(payload['rng'])
        if runtime is not None:
            from experiments.functional_step_geometry.qualify import restore_runtime
            restore_runtime(self.bundle.model, runtime)
        result = verify_live(self, payload)
        return dict(result, production_recovery=False)

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        started = time.perf_counter()
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False, mmap=True)
            result = validate_snapshot(state, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state, export_path)
            result.update(live=verify_live(self, state), export_exact=export_path is not None,
                          serialized_size=validate_recovery_file(self.latest))
        result['wall_s'] = time.perf_counter() - started
        if receipt_path is not None:
            write_json(receipt_path, result)
        return result
