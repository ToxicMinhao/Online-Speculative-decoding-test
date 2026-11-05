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
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

CONTRACT_KEY = 'kplus_full_gsm8k_v1'
ARM = 'KPLUS'
OBJECTIVE = 'osd_first_error'


def experiment_contract(binding):
    # Reuse the qualified digest check, not the old experiment identity.
    native.experiment_contract('K', binding)
    return dict(schema=CONTRACT_KEY, arm=ARM, training_seed=17, binding=binding,
                learning_rate=3e-4, maximum_events=58,
                initialization='released_seed17_exact_original_K_initial_scientific_state',
                feedback='own_fresh_native_responses', reference=None,
                redistribution=kplus_contract())


def validate_config_text(config_text, original_config_text):
    native.validate_config_text(config_text, 'K', original_config_text)


def native_contract_view(contract, binding):
    """Check K+ first, then substitute only experiment identity for native checks."""
    if (contract.get(CONTRACT_KEY) != experiment_contract(binding)
            or native.CONTRACT_KEY in contract):
        raise ValueError('Foreign or changed KPLUS scientific contract')
    view = dict(contract)
    del view[CONTRACT_KEY]
    view[native.CONTRACT_KEY] = native.experiment_contract('K', binding)
    return view


def validate_event_receipt(receipt, *, event_number, require_completed=True, trainer_status=None):
    if receipt.get('method') != ARM or receipt.get('redistribution') != kplus_contract():
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
    return dict(result, arm=ARM, treatment='first_error_weight_redistribution')


def validate_snapshot(state, *, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe,
                      expected_cuda_rng_states=1):
    projected = dict(state)
    projected['algorithm_contract'] = native_contract_view(state['algorithm_contract'], binding)
    expected = native_contract_view(expected_contract, binding)
    result = native.validate_snapshot(projected, arm='K', binding=binding,
        config_text=config_text, original_config_text=original_config_text,
        completed_event=completed_event, expected_contract=expected,
        expected_recipe=expected_recipe, expected_cuda_rng_states=expected_cuda_rng_states)
    for number, event in enumerate(state['events'], 1):
        validate_event_receipt(event, event_number=number, require_completed=False)
    return dict(result, arm=ARM, treatment='first_error_weight_redistribution')


class KPlusTrainer(native.FullKTrainer):
    """Persistent one-arm trainer; only the response numerator changes."""

    def __init__(self, config_path, output_root, *, binding, original_config_text,
                 resume_state=None, completed_event=0, expected_cuda_rng_states=1):
        if type(completed_event) is not int or not 0 <= completed_event <= 58:
            raise ValueError('Require a complete KPLUS boundary 0..58')
        if (completed_event > 0) != (resume_state is not None):
            raise ValueError('Continuation requires its exact own complete state')
        self.arm = ARM
        self.binding = binding
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.completed_event = completed_event
        self.response_records, self.window_records = [], []
        self.work, self._work_handles = {}, []
        experiment_contract(binding)
        SpecForgeEventTrainer.__init__(self, config_path, output_root, resume_state=resume_state,
            objective=OBJECTIVE, reduction='response_mean', ranking_lambda=0.)

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        validate_config_text(self.config_text, self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        installed = install_kplus(self.bundle.model)
        if installed != kplus_contract():
            raise ValueError('Installed K+ rule differs from the fixed experiment')
        self.contract[CONTRACT_KEY] = experiment_contract(self.binding)
        self._install_observations()

    def validation_arguments(self):
        return dict(binding=self.binding, config_text=self.config_text,
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
            projected = dict(state, algorithm_contract=native_contract_view(state['algorithm_contract'], self.binding))
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

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('No reference probes or forced historical anchor schedule')
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        validate_event_receipt(result, event_number=len(self.events), trainer_status=self.status())
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
        event['method'] = ARM
        event['redistribution'] = kplus_contract()
        validate_event_receipt(event, event_number=len(self.events) + 1, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)
