"""Two-event BASE/PAIR wrapper around the qualified native event trainer.

The inherited numerical horizon remains 7,424 constant-LR steps. This wrapper
independently refuses production work beyond the prospectively sealed endpoint.
No historical arm validator or trainer is modified.
"""
from copy import deepcopy
import math

from experiments.full_k_gsm8k import training as native
from experiments.mathqa_fixed_lr_transfer.training import FORBIDDEN_CONTROLLERS
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.opencode_native_comparison.configuration import historical_view
from experiments.opencode_native_comparison.training import initial_witness
from experiments.persistent_prefix_withholding.qualification import isolated_backend_for_load
from experiments.unchanged_c_mathqa_transfer.state import validate_allocation_observation
from online_feedback.kplus import install_kplus, kplus_contract, KPlusRedistribution
from online_feedback.prefix_allocation import install_prefix_allocation, prefix_allocation_contract
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

from .configuration import validate_config_text

CONTRACT_KEY = 'repair_conditional_kl_pilot_v1'
ARMS = ('BASE', 'PAIR')


def discarded_startup(trainer, feature_paths, output):
    """Qualified two-forward/backward compile startup; zero Adam and exact RNG restore."""
    from pathlib import Path
    from experiments.functional_step_geometry.native_pilot import discarded_startup as run
    output = Path(output)
    result = run(trainer, feature_paths, output)
    # Native startup writes observations in an uncommitted pseudo-event. Keep
    # that evidence outside the production event namespace so event-two resume
    # retains the generic native constructor's strict committed-event check.
    source = trainer.events_root / 'disposable_startup'
    source.rename(output / 'response_observations')
    return result


def experiment_contract(arm, binding, maximum_events=2):
    from online_feedback.repair_conditional_kl import repair_conditional_kl_contract
    native.experiment_contract('K', binding)
    if arm not in ARMS or type(maximum_events) is not int or maximum_events not in (1, 2):
        raise ValueError('Require BASE/PAIR and a prospectively sealed one/two-event endpoint')
    return dict(schema=CONTRACT_KEY, arm=arm, binding=binding,
        maximum_events=maximum_events, seed=17, dataset='OpenCodeInstruct',
        initialization='independent_released_weights_fresh_Adam',
        feedback='own_native_responses', inherited_constant_lr_horizon=7424,
        response_window=128, passes=2, redistribution=kplus_contract(),
        allocation=prefix_allocation_contract(.25),
        auxiliary=repair_conditional_kl_contract(enabled=arm == 'PAIR'))


def validate_event_receipt(event, *, arm, binding, event_number, maximum_events=2,
                           require_completed=True, trainer_status=None):
    if (type(event_number) is not int or not 1 <= event_number <= maximum_events
            or event.get('method') != arm or event.get('dataset') != 'OpenCodeInstruct'
            or event.get(CONTRACT_KEY) != experiment_contract(arm, binding, maximum_events)):
        raise ValueError('Pilot event identity or declared endpoint changed')
    checked = native.validate_event_receipt(event, arm='K', event_number=event_number,
        require_completed=require_completed, trainer_status=trainer_status)
    for row in event['records']:
        observation = row['response_observation']
        if (observation['record_id'] != row['record_id']
                or observation['pass_index'] != row['pass']
                or observation['sampled_anchors'] != row['sampled_anchors']):
            raise ValueError('Original-forward observation lost its response binding')
        values = [observation['loss'], *observation['sum_metrics'].values()]
        values += [value for pair in observation['ratio_metrics'].values() for value in pair]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError('Nonfinite original-forward pilot observation')
        validate_allocation_observation(observation)
        has_auxiliary = 'repair_pair_auxiliary' in observation['ratio_metrics']
        if has_auxiliary != (arm == 'PAIR'):
            raise ValueError('Production auxiliary flag or original-forward logging is missing')
    return dict(checked, arm=arm, maximum_events=maximum_events)


def validate_snapshot(state, *, arm, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe,
                      maximum_events=2, expected_cuda_rng_states=1):
    """Check actual pilot metadata, then reuse unchanged native tensor/RNG gates."""
    validate_config_text(config_text, original_config_text, maximum_events=maximum_events)
    if (type(completed_event) is not int or not 0 <= completed_event <= maximum_events
            or state['config_text'] != config_text
            or differences(state['algorithm_contract'], expected_contract)
            or expected_contract.get(CONTRACT_KEY) != experiment_contract(arm, binding, maximum_events)
            or native.CONTRACT_KEY in expected_contract):
        raise ValueError('Foreign pilot recovery, configuration or endpoint')
    for number, event in enumerate(state['events'], 1):
        validate_event_receipt(event, arm=arm, binding=binding, event_number=number,
            maximum_events=maximum_events, require_completed=False)
    contract = deepcopy(expected_contract)
    del contract[CONTRACT_KEY]
    contract[native.CONTRACT_KEY] = native.experiment_contract('K', binding)
    config = historical_view(config_text)
    result = native.validate_snapshot(dict(state, algorithm_contract=contract, config_text=config),
        arm='K', binding=binding, config_text=config,
        original_config_text=historical_view(original_config_text), completed_event=completed_event,
        expected_contract=contract, expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(result, arm=arm, maximum_events=maximum_events, actual_state_unchanged=True)


def compare_initial_witnesses(first, second):
    """Exact released weights, empty Adam and RNG; only declared arm differs."""
    import yaml
    views = []
    bindings = []
    for witness in (first, second):
        if (witness['global_step'] or witness['presentations'] or witness['events']
                or witness['backend']['optimizer']['optimizer_state_dict']['state']):
            raise ValueError('Initial comparison requires genuinely empty Adam')
        contract = deepcopy(witness['algorithm_contract'])
        identity = contract.pop(CONTRACT_KEY)
        if identity != experiment_contract(identity['arm'], identity['binding'], identity['maximum_events']):
            raise ValueError('Malformed initial pilot contract')
        bindings.append((identity['binding'], identity['maximum_events']))
        config = yaml.safe_load(witness['config_text'])
        for key in ('run_id', 'output_dir'):
            config.pop(key, None)
        views.append({**{key: witness[key] for key in ('weights', 'backend', 'rng', 'global_step',
            'presentations', 'events', 'strategy', 'world_size', 'reduction', 'accumulation_steps', 'horizon')},
            'contract': contract, 'config': config})
    mismatch = differences(*views)
    if bindings[0] != bindings[1] or mismatch:
        raise ValueError('Released BASE/PAIR initialization differs: ' + repr(mismatch[:8]))
    return dict(status='passed', exact_weights_empty_Adam_RNG=True, rng_overwritten=False)


class PilotTrainer(native.FullKTrainer):
    """Persistent native Adam, with strict own-response one/two-event admission."""
    def __init__(self, config_path, output_root, *, arm, binding, original_config_text,
                 maximum_events=2, resume_state=None, completed_event=0,
                 expected_cuda_rng_states=1, qualification_only=False):
        experiment_contract(arm, binding, maximum_events)
        if (type(completed_event) is not int or not 0 <= completed_event <= maximum_events
                or (completed_event > 0) != (resume_state is not None)):
            raise ValueError('Resume requires the exact own complete event boundary')
        self.arm, self.binding, self.maximum_events = arm, binding, maximum_events
        self.original_config_text, self.completed_event = original_config_text, completed_event
        self.expected_cuda_rng_states, self.qualification_only = expected_cuda_rng_states, qualification_only
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        SpecForgeEventTrainer.__init__(self, config_path, output_root, resume_state=resume_state,
            objective=native.OBJECTIVE, reduction='response_mean', ranking_lambda=0.)

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        validate_config_text(self.config_text, self.original_config_text, maximum_events=self.maximum_events)

    def _build_runtime(self):
        import rich.style  # Match the qualified pre-seed logging initialization.
        from online_feedback.repair_conditional_kl import install_repair_conditional_kl
        SpecForgeEventTrainer._build_runtime(self)
        model = self.bundle.model
        install_kplus(model)
        install_prefix_allocation(model, relative_cap=.25)
        if self.arm == 'PAIR':
            install_repair_conditional_kl(model, enabled=True)
        self.contract[CONTRACT_KEY] = experiment_contract(self.arm, self.binding, self.maximum_events)
        self._install_observations()
        ordinary_forward = self._native_forward
        def checked_forward(*args, **kwargs):
            import torch
            self.check_installed_arm()
            result = ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError('Nonfinite pilot response loss')
            return result
        self._native_forward = checked_forward

    def check_installed_arm(self):
        model = self.bundle.model
        if (model.objective != native.OBJECTIVE
                or not isinstance(getattr(model, 'first_error_redistribution', None), KPlusRedistribution)
                or getattr(model, 'prefix_allocation', None) is None
                or model.prefix_allocation.contract != prefix_allocation_contract(.25)
                or bool(getattr(model, 'repair_conditional_kl', False)) != (self.arm == 'PAIR')
                or any(getattr(model, name, None) is not None
                       for name in set(FORBIDDEN_CONTROLLERS) - {'prefix_allocation'})):
            raise ValueError('Missing pilot loss or foreign objective extension')

    def _append_observation(self, name, value):
        if name == 'RESPONSE_OBSERVATIONS.jsonl':
            value['prefix_allocation'] = deepcopy(self.bundle.model.prefix_allocation.last_summary)
        if name == 'WINDOW_OBSERVATIONS.jsonl':
            value['clipped'] = value['gradient_norm'] > 1.
        return native.FullKTrainer._append_observation(self, name, value)

    def validation_arguments(self):
        return dict(arm=self.arm, binding=self.binding, maximum_events=self.maximum_events,
            config_text=self.config_text, original_config_text=self.original_config_text,
            expected_contract=self.contract, expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def save_initial(self, comparator_path=None):
        import torch
        destination = self.output_root / 'INITIAL_WITNESS.pt'
        if destination.exists():
            raise FileExistsError('Initial pilot witness already exists')
        with preserve_rng():
            state = native.live_state(self)
            result = validate_snapshot(state, **self.validation_arguments(), completed_event=0)
            _export_equal(state, self.cfg.model.draft_checkpoint_path)
            witness = initial_witness(state)
            if comparator_path is not None:
                result['paired_initialization'] = compare_initial_witnesses(
                    torch.load(comparator_path, map_location='cpu', weights_only=False), witness)
            save_torch(destination, witness)
            if differences(witness, torch.load(destination, map_location='cpu', weights_only=False)):
                raise ValueError('Initial pilot witness did not serialize exactly')
            native.verify_live(self, state)
            result.update(witness_path=str(destination), export_exact=True)
            write_json(self.output_root / 'INITIAL_CHECK.json', result)
        return result

    def _restore(self, path):
        import torch
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self._load_state(state)
        native.verify_live(self, state, self.output_root / 'RESUME_CHECK.json')

    def _load_state(self, state):
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        device = next(self.bundle.draft_model.parameters()).device
        self.backend.load_state_dict(isolated_backend_for_load(state['backend'], device))
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        restore_rng(state['rng'])

    def _prepare_event(self, event_id, feature_paths, passes):
        number = len(self.events) + 1
        if self.qualification_only or number > self.maximum_events:
            raise ValueError('Pilot endpoint reached or disposable trainer used for production')
        if not self.events and not (self.output_root / 'INITIAL_WITNESS.pt').exists():
            raise ValueError('Authenticate released initialization before production')
        # This existing gate checks native event shape and counters. Its initial
        # filename is not reused; apply the shared generic feature gate directly.
        if (event_id != f'event_{number:03d}' or len(feature_paths) != 128 or type(passes) is not int
                or passes != 2 or self.global_step != (number - 1) * 128
                or self.presentations != (number - 1) * 256):
            raise ValueError('Require the next own 128-response/two-pass event')
        self._active_event = event_id
        self.response_records.clear()
        self.window_records.clear()
        self.work.clear()
        features = SpecForgeEventTrainer._prepare_event(self, event_id, feature_paths, passes)
        previous = {item['record_id'] for event in self.events for item in event['features']}
        if previous.intersection(item['record_id'] for item in features):
            raise ValueError('An own-feedback event cannot reuse earlier response identities')
        return features

    def _checkpoint_state(self, event):
        event.update(method=self.arm, dataset='OpenCodeInstruct', **{CONTRACT_KEY: self.contract[CONTRACT_KEY]})
        validate_event_receipt(event, arm=self.arm, binding=self.binding, event_number=len(self.events) + 1,
            maximum_events=self.maximum_events, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('Production uses natural native sampling, without reference probes')
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        validate_event_receipt(result, arm=self.arm, binding=self.binding, event_number=len(self.events),
            maximum_events=self.maximum_events, trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False, mmap=True)
            result = validate_snapshot(state, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state, export_path)
            self.check_installed_arm()
            result.update(live=native.verify_live(self, state), export_exact=export_path is not None)
        if receipt_path is not None:
            write_json(receipt_path, result)
        return result

    def run_pair(self, pair, pass_index=0, ordinal=0):
        from experiments.bounded_prefix_allocation.training import CandidateTrainer
        if not self.qualification_only or self.global_step or self.presentations:
            raise ValueError('Disposable qualification permits exactly one native Adam update')
        return CandidateTrainer.run_pair(self, pair, pass_index, ordinal)
