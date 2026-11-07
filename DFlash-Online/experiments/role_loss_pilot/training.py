"""K+ method binding around the qualified persistent native K trainer.

The native state validator sees a metadata-only K projection after the new
method contract is checked. Tensors, RNG, Adam, counters and event history are
never projected, rewritten or weakened. Historical experiment guards are intact.
"""
from copy import deepcopy
import math
import time

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from online_feedback.kplus import install_kplus, kplus_contract
from .contract import ARMS, DATASETS, loss_contract
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

CONTRACT_KEY = 'role_loss_pilot_v1'
OBJECTIVE = 'osd_first_error'


def experiment_contract(binding, arm, dataset):
    if arm not in ARMS or dataset not in DATASETS:
        raise ValueError('Unrecognized pilot trajectory')
    # Reuse the qualified digest check, not the old experiment identity.
    native.experiment_contract('K', binding)
    return dict(schema=CONTRACT_KEY, arm=arm, dataset=dataset, training_seed=17, binding=binding,
                learning_rate=3e-4, maximum_events=4,
                initialization='released_seed17_exact_original_K_initial_scientific_state',
                feedback='own_fresh_native_responses', reference=None,
                loss=loss_contract(arm))


def validate_config_text(config_text, original_config_text):
    native.validate_config_text(config_text, 'K', original_config_text)


def native_contract_view(contract, binding, arm, dataset):
    """Check K+ first, then substitute only experiment identity for native checks."""
    if (contract.get(CONTRACT_KEY) != experiment_contract(binding, arm, dataset)
            or native.CONTRACT_KEY in contract):
        raise ValueError('Foreign or changed KPLUS scientific contract')
    view = dict(contract)
    del view[CONTRACT_KEY]
    view[native.CONTRACT_KEY] = native.experiment_contract('K', binding)
    return view


def validate_event_receipt(receipt, *, arm, dataset, event_number, require_completed=True, trainer_status=None):
    if (arm not in ARMS or dataset not in DATASETS or not 1 <= event_number <= 4
            or receipt.get('method') != arm or receipt.get('dataset') != dataset
            or receipt.get('pilot_loss') != loss_contract(arm)):
        raise ValueError('Event is not bound to the fixed KPLUS method')
    result = native.validate_event_receipt(receipt, arm='K', event_number=event_number,
        require_completed=require_completed, trainer_status=trainer_status)
    for row in receipt['records']:
        observation = row['response_observation']
        sums = observation['sum_metrics']
        if (observation['record_id'] != row['record_id'] or observation['pass_index'] != row['pass']
                or sums['kplus_response_count'] != 1
                or not all(math.isfinite(float(value)) for value in sums.values())):
            raise ValueError('K+ observations must describe each original forward exactly once')
        weight = sums['kplus_W1'] + sums['kplus_W2']
        tolerance = kplus_contract()['mass_conservation_atol'] + kplus_contract()['mass_conservation_rtol'] * weight
        if abs(sums['kplus_first_error_weight_after'] - weight) > tolerance:
            raise ValueError('K+ first-error mass is not conserved within the declared tolerance')
    return dict(result, arm=arm, dataset=dataset, loss=loss_contract(arm))


def validate_snapshot(state, *, arm, dataset, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe,
                      expected_cuda_rng_states=1):
    if type(completed_event) is not int or not 0 <= completed_event <= 4:
        raise ValueError('Pilot complete boundary is 0..4')
    projected = dict(state)
    projected['algorithm_contract'] = native_contract_view(state['algorithm_contract'], binding, arm, dataset)
    expected = native_contract_view(expected_contract, binding, arm, dataset)
    result = native.validate_snapshot(projected, arm='K', binding=binding,
        config_text=config_text, original_config_text=original_config_text,
        completed_event=completed_event, expected_contract=expected,
        expected_recipe=expected_recipe, expected_cuda_rng_states=expected_cuda_rng_states)
    for number, event in enumerate(state['events'], 1):
        validate_event_receipt(event, arm=arm, dataset=dataset, event_number=number, require_completed=False)
    return dict(result, arm=arm, dataset=dataset, loss=loss_contract(arm))


class RoleLossTrainer(native.FullKTrainer):
    """Persistent one-arm trainer; only the response numerator changes."""

    def __init__(self, config_path, output_root, *, binding, arm, dataset, original_config_text,
                 resume_state=None, completed_event=0, expected_cuda_rng_states=1):
        if type(completed_event) is not int or not 0 <= completed_event <= 4:
            raise ValueError('Require a complete pilot boundary 0..4')
        if (completed_event > 0) != (resume_state is not None):
            raise ValueError('Continuation requires its exact own complete state')
        self.arm, self.dataset = arm, dataset
        self.binding = binding
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.completed_event = completed_event
        self.response_records, self.window_records = [], []
        self.work, self._work_handles = {}, []
        experiment_contract(binding, arm, dataset)
        SpecForgeEventTrainer.__init__(self, config_path, output_root, resume_state=resume_state,
            objective=OBJECTIVE, reduction='response_mean', ranking_lambda=0.)

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        validate_config_text(self.config_text, self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        if self.arm == 'KPLUS':
            installed = install_kplus(self.bundle.model)
        else:
            from online_feedback.role_loss import install_role_loss
            installed = install_role_loss(self.bundle.model, self.arm)
        if installed != loss_contract(self.arm):
            raise ValueError('Installed K+ rule differs from the fixed experiment')
        self.contract[CONTRACT_KEY] = experiment_contract(self.binding, self.arm, self.dataset)
        self._install_observations()
        ordinary_forward = self._native_forward
        def finite_forward(*args, **kwargs):
            import torch
            result = ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                write_json(self.output_root/'NONFINITE_LOSS.json', dict(status='failed',
                    presentations=self.presentations, optimizer_called=False))
                raise ValueError('Nonfinite production loss; stop before backward and Adam')
            if self.arm == 'KPLUS':
                from dataclasses import replace
                sums = result.sum_metrics
                full = sums['available_position_weight'] + 1e-6
                selected = sums['selected_position_weight'] + 1e-6
                error = sums['kplus_W1'] + sums['kplus_W2']
                metrics = dict(sums,
                    role_loss_response_count=full.new_ones(()),
                    role_loss_full_denominator=full,
                    role_loss_selected_denominator=selected,
                    role_loss_full_to_selected_ratio=full/selected,
                    role_loss_prefix_weight=sums['selected_position_weight']-error,
                    role_loss_error_weight=error,
                    role_loss_value=result.loss.detach())
                ratios = dict(result.ratio_metrics,
                    role_loss_selected_fraction=(sums['selected_position_occurrences'],sums['available_position_occurrences']),
                    role_loss_full_to_selected=(full,selected),
                    role_loss_priority2_fraction=(sums['kplus_priority2_error_count'],
                        sums['kplus_priority1_error_count']+sums['kplus_priority2_error_count']))
                result=replace(result,sum_metrics=metrics,ratio_metrics=ratios)
            return result
        self._native_forward = finite_forward
        ordinary_step = self._native_step
        def finite_step():
            from .qualify import finite_gradient_check
            finite_gradient_check({n:p for n,p in self.bundle.draft_model.named_parameters() if p.requires_grad},
                                  self.output_root, self.presentations)
            return ordinary_step()
        self._native_step = finite_step

    def validation_arguments(self):
        return dict(arm=self.arm, dataset=self.dataset, binding=self.binding, config_text=self.config_text,
            original_config_text=self.original_config_text, expected_contract=self.contract,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states)

    def _restore(self, path):
        import torch
        state = torch.load(path, map_location='cpu', weights_only=False)
        validate_snapshot(state, **self.validation_arguments(), completed_event=self.completed_event)
        self.bundle.draft_model.load_state_dict(state['draft_state_dict'], strict=True)
        self.backend.load_state_dict(state['backend'])
        self.global_step, self.presentations = state['global_step'], state['presentations']
        self.events = deepcopy(state['events'])
        self.core._micro = self.presentations
        restore_rng(state['rng'])
        native.verify_live(self, state, receipt_path=self.output_root / 'RESUME_CHECK.json')

    def save_initial(self, comparator_path):
        """Compare independent initialization without loading K weights into K+."""
        import torch
        destination = self.output_root / 'initial_state.pt'
        if destination.exists() or self.global_step or self.presentations or self.events:
            raise ValueError('Initial state must be genuinely new and empty')
        with preserve_rng():
            state = native.live_state(self)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=0)
            _export_equal(state, self.cfg.model.draft_checkpoint_path)
            original = torch.load(comparator_path, map_location='cpu', weights_only=False)
            contract = original['algorithm_contract'][native.CONTRACT_KEY]
            if contract['arm'] != 'K':
                raise ValueError('The original initialization comparator must be K, not K58 or another arm')
            native.validate_snapshot(original, arm='K', binding=contract['binding'],
                config_text=original['config_text'], original_config_text=self.original_config_text,
                completed_event=0, expected_contract=original['algorithm_contract'],
                expected_recipe=original['backend']['optimizer']['recipe'],
                expected_cuda_rng_states=self.expected_cuda_rng_states)
            projected = dict(state, algorithm_contract=native_contract_view(state['algorithm_contract'], self.binding, self.arm, self.dataset))
            receipt['original_K_initialization'] = native.compare_initial_states(original, projected)
            save_torch(destination, state)
            decoded = torch.load(destination, map_location='cpu', weights_only=False)
            native.verify_live(self, decoded)
            receipt.update(export_exact=True, state_path=str(destination),
                           comparator_path=str(comparator_path), original_weights_loaded_into_candidate=False)
            write_json(self.output_root / 'INITIAL_CHECK.json', receipt)
        return receipt

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        started = time.perf_counter()
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state, export_path)
                receipt['export_exact'] = True
            receipt['live'] = native.verify_live(self, state)
        receipt['wall_s'] = time.perf_counter() - started
        if receipt_path is not None:
            write_json(receipt_path, receipt)
        return receipt

    def _prepare_event(self, event_id, feature_paths, passes):
        if len(self.events) >= 4:
            raise ValueError('The pilot ends at event 4; no automatic continuation')
        return native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('No reference probes or forced historical anchor schedule')
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        validate_event_receipt(result, arm=self.arm, dataset=self.dataset, event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def _train_passes(self, features, passes):
        rows = native.FullKTrainer._train_passes(self, features, passes)
        metadata = {row['record_id']: row for row in features}
        for row in rows:
            feature = metadata[row['record_id']]
            row['prompt_tokens'] = feature['prompt_length']
            row['response_tokens'] = feature['response_tokens']
            row['active_anchors'] = sum(sum(draw['keep_mask'][0]) for draw in row['sampled_anchors'])
        return rows

    def _checkpoint_state(self, event):
        event['method'] = self.arm
        event['dataset'] = self.dataset
        event['pilot_loss'] = loss_contract(self.arm)
        validate_event_receipt(event, arm=self.arm, dataset=self.dataset, event_number=len(self.events) + 1, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)
