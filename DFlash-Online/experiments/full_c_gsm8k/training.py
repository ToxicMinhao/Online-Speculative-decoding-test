"""Released-start, persistent K+ training with the unchanged C allocator.

The PR25 trainer still owns sampling, response reduction, accumulation, AdamW,
checkpoint publication and the 58-event horizon. This adapter adds only the C
contract, coefficient observations and exact own-state restoration.
"""
from copy import deepcopy
import time

from experiments.bounded_prefix_allocation.training import CandidateTrainer
from experiments.full_k_gsm8k import training as native
from experiments.kplus_full_gsm8k import training as kplus
from experiments.mathqa_fixed_lr_transfer.training import FORBIDDEN_CONTROLLERS
from experiments.ok_coding.training import _export_equal
from experiments.persistent_prefix_withholding.qualification import isolated_backend_for_load
from experiments.unchanged_c_mathqa_transfer.state import validate_allocation_observation
from online_feedback.kplus import KPlusRedistribution, kplus_contract
from online_feedback.prefix_allocation import prefix_allocation_contract
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

CONTRACT_KEY = 'full_c_gsm8k_v1'
ARM = 'KPLUS_C'
RELATIVE_CAP = .25


def experiment_contract(binding):
    native.experiment_contract('K', binding)
    return dict(schema=CONTRACT_KEY, arm=ARM, dataset='gsm8k', binding=binding,
        training_seed=17, initialization='released_seed17_exact_original_K_initial_scientific_state',
        feedback='own_fresh_native_responses', maximum_events=58, horizon=7424,
        learning_rate=3e-4, native_loss=kplus_contract(),
        allocation=prefix_allocation_contract(RELATIVE_CAP), reference=None,
        continuation_rng_fork=None, additional_model_forwards=0)


def kplus_contract_view(contract, binding):
    """Project only the experiment label after checking the exact C contract."""
    if (contract.get(CONTRACT_KEY) != experiment_contract(binding)
            or kplus.CONTRACT_KEY in contract or native.CONTRACT_KEY in contract):
        raise ValueError('Foreign or changed full C scientific contract')
    view = dict(contract)
    del view[CONTRACT_KEY]
    view[kplus.CONTRACT_KEY] = kplus.experiment_contract(binding)
    return view


def validate_event_receipt(event, *, binding, event_number, require_completed=True,
                           trainer_status=None):
    if (event.get('method') != ARM or event.get('dataset') != 'gsm8k'
            or event.get(CONTRACT_KEY) != experiment_contract(binding)):
        raise ValueError('Full C event lost its fixed method or released-start ancestry')
    result = kplus.validate_event_receipt(dict(event, method=kplus.ARM),
        event_number=event_number, require_completed=require_completed, trainer_status=trainer_status)
    for row in event['records']:
        observation = row['response_observation']
        if observation['sampled_anchors'] != row['sampled_anchors']:
            raise ValueError('C observation does not describe the original natural anchor draw')
        validate_allocation_observation(observation)
    return dict(result, arm=ARM, allocation_contract_preserved=True)


def validate_snapshot(state, *, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe,
                      expected_cuda_rng_states=1):
    for number, event in enumerate(state['events'], 1):
        validate_event_receipt(event, binding=binding, event_number=number, require_completed=False)
    view = dict(state,
        algorithm_contract=kplus_contract_view(state['algorithm_contract'], binding),
        events=[dict(event, method=kplus.ARM) for event in state['events']])
    result = kplus.validate_snapshot(view, binding=binding, config_text=config_text,
        original_config_text=original_config_text, completed_event=completed_event,
        expected_contract=kplus_contract_view(expected_contract, binding), expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(result, arm=ARM, allocation_contract_preserved=True, continuation_rng_fork=None)


class FullCTrainer(kplus.KPlusTrainer):
    """One continuous seed-17 trajectory; no reference model or RNG fork."""

    def __init__(self, config_path, output_root, *, binding, original_config_text,
                 resume_state=None, completed_event=0, expected_cuda_rng_states=1,
                 qualification_only=False):
        self.qualification_only = qualification_only
        self.relative_cap = self._qualification_cap = RELATIVE_CAP
        self.step_observer = None
        super().__init__(config_path, output_root, binding=binding,
            original_config_text=original_config_text, resume_state=resume_state,
            completed_event=completed_event, expected_cuda_rng_states=expected_cuda_rng_states)
        self.arm = ARM

    def _build_runtime(self):
        super()._build_runtime()
        del self.contract[kplus.CONTRACT_KEY]
        self.contract[CONTRACT_KEY] = experiment_contract(self.binding)
        self._install_allocation()
        self._check_native_path()
        self._ordinary_forward, self._ordinary_step = self._native_forward, self._native_step

        def checked_forward(*args, **kwargs):
            import torch
            self._check_native_path()
            result = self._ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError('Nonfinite full C response loss')
            return result

        def checked_step():
            self._check_actual_lr()
            if self.step_observer is not None:
                if not self.qualification_only:
                    raise ValueError('Preclip gradient inspection is qualification-only')
                self.step_observer(self)
            result = self._ordinary_step()
            self._check_actual_lr()
            return result

        self._native_forward, self._native_step = checked_forward, checked_step

    _install_allocation = CandidateTrainer._install_allocation
    _check_allocation = CandidateTrainer._check_allocation
    qualification_allocation = CandidateTrainer.qualification_allocation
    _append_observation = CandidateTrainer._append_observation
    save_qualification = CandidateTrainer.save_qualification
    restore_qualification = CandidateTrainer.restore_qualification

    def _check_native_path(self):
        model = self.bundle.model
        forbidden = tuple(name for name in FORBIDDEN_CONTROLLERS if name != 'prefix_allocation')
        if (not isinstance(getattr(model, 'first_error_redistribution', None), KPlusRedistribution)
                or any(getattr(model, name, None) is not None for name in forbidden)):
            raise ValueError('Only the original K+ rule and unchanged C may be active')
        self._check_allocation()
        controller = getattr(model, 'prefix_allocation', None)
        if controller is not None and controller.contract != prefix_allocation_contract(controller.relative_cap):
            raise ValueError('C allocation contract changed')

    def _check_actual_lr(self):
        optimizer = self.backend.optimizer
        if (self.cfg.training.learning_rate != 3e-4
                or optimizer.recipe['optimizer_options']['lr'] != 3e-4
                or len(optimizer.optimizer.param_groups) != 1
                or optimizer.optimizer.param_groups[0]['lr'] != 3e-4
                or optimizer.get_learning_rate() != 3e-4):
            raise ValueError('Actual native constant 3e-4 learning rate changed')

    def _load_scientific_state(self, state):
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        device = next(self.bundle.draft_model.parameters()).device
        # Adam load_state_dict may retain CPU counter aliases. Isolate the saved
        # payload so a live update cannot mutate its restoration witness.
        self.backend.load_state_dict(isolated_backend_for_load(state['backend'], device))
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        self._install_allocation()
        restore_rng(state['rng'])

    def _restore(self, path):
        import torch
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self._load_scientific_state(state)
        native.verify_live(self, state, receipt_path=self.output_root/'RESUME_CHECK.json')

    def save_initial(self, comparator_path):
        """Authenticate independently constructed released weights and seed-17 state."""
        import torch
        destination = self.output_root/'initial_state.pt'
        if destination.exists() or self.global_step or self.presentations or self.events:
            raise ValueError('Initial state must be genuinely new and empty')
        with preserve_rng():
            state = native.live_state(self)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=0)
            _export_equal(state, self.cfg.model.draft_checkpoint_path)
            original = torch.load(comparator_path, map_location='cpu', weights_only=False, mmap=True)
            original_contract = original['algorithm_contract'][native.CONTRACT_KEY]
            if original_contract['arm'] != 'K':
                raise ValueError('Initial comparator must be original K step zero')
            native.validate_snapshot(original, arm='K', binding=original_contract['binding'],
                config_text=original['config_text'], original_config_text=self.original_config_text,
                completed_event=0, expected_contract=original['algorithm_contract'],
                expected_recipe=original['backend']['optimizer']['recipe'],
                expected_cuda_rng_states=self.expected_cuda_rng_states)
            projected = dict(state, algorithm_contract=kplus.native_contract_view(
                kplus_contract_view(state['algorithm_contract'], self.binding), self.binding))
            receipt['original_K_initialization'] = native.compare_initial_states(original, projected)
            save_torch(destination, state)
            native.verify_live(self, torch.load(destination, map_location='cpu', weights_only=False, mmap=True))
            receipt.update(export_exact=True, state_path=str(destination), comparator_path=str(comparator_path),
                original_weights_loaded_into_candidate=False, continuation_rng_fork=None)
            write_json(self.output_root/'INITIAL_CHECK.json', receipt)
        return receipt

    def _prepare_event(self, event_id, feature_paths, passes):
        if self.qualification_only:
            raise ValueError('Disposable qualification cannot publish production events')
        return super()._prepare_event(event_id, feature_paths, passes)

    def _checkpoint_state(self, event):
        event.update(method=ARM, dataset='gsm8k', redistribution=kplus_contract(),
                     **{CONTRACT_KEY:self.contract[CONTRACT_KEY]})
        validate_event_receipt(event, binding=self.binding, event_number=len(self.events)+1,
                               require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('Full C uses natural sampling without historical schedules or reference probes')
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        validate_event_receipt(result, binding=self.binding, event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        began = time.perf_counter()
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False, mmap=True)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state, export_path)
            receipt.update(live=native.verify_live(self, state), export_exact=export_path is not None)
        receipt['wall_s'] = time.perf_counter()-began
        if receipt_path is not None:
            write_json(receipt_path, receipt)
        return receipt

    def reset_qualification(self, *, relative_cap=.25, runtime=None):
        import torch
        if not self.qualification_only:
            raise ValueError('Disposable reset is forbidden in production')
        state = torch.load(self.output_root/'initial_state.pt', map_location='cpu', weights_only=False, mmap=True)
        validate_snapshot(state, **self.validation_arguments(), completed_event=0)
        self.qualification_allocation(relative_cap)
        self.backend.optimizer.zero_grad()
        self.core._ratio_totals.clear()
        self.core._sum_totals.clear()
        self._load_scientific_state(state)
        self.completed_event = 0
        self.response_records.clear()
        self.window_records.clear()
        self._anchor_observations.clear()
        self._presentation_schedule = None
        if runtime is not None:
            from experiments.functional_step_geometry.qualify import restore_runtime
            restore_runtime(self.bundle.model, runtime)
        return native.verify_live(self, state)

    def run_pair(self, pair, pass_index=0, ordinal=0):
        if not self.qualification_only or self._presentation_schedule is not None:
            raise ValueError('Disposable pairs require native sampling and qualification mode')
        return CandidateTrainer.run_pair(self, pair, pass_index, ordinal)
