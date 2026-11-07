"""Distinct released-loss state binding on the unchanged persistent OSD loop."""
from copy import deepcopy
import math
import time

from experiments.adaflash_c1 import training as paper
from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from online_feedback.adaflash_repo import (
    ReleasedAdaFlashOPD, install_adaflash_repo, adaflash_repo_contract,
)
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

CONTRACT_KEY = 'adaflash_arepo_v1'
ARM = 'AREPO'
OBJECTIVE = 'osd'
validate_config_text = paper.validate_config_text


def experiment_contract(binding):
    native.experiment_contract('K', binding)
    return dict(schema=CONTRACT_KEY, arm=ARM, training_seed=17, binding=binding,
        learning_rate=3e-4, maximum_events=58,
        initialization='released_seed17_exact_original_K_initial_scientific_state',
        feedback='own_fresh_native_responses', reference=None,
        loss=adaflash_repo_contract(), adaptive_length_head=False)


def native_contract_view(contract, binding):
    expected = dict(name='osd', version=1, loss_precision='bfloat16',
        teacher='frozen_target_predecessor_hidden', temperature=1., hard_label_mix=0., denominator_epsilon=1e-6)
    if (contract.get(CONTRACT_KEY) != experiment_contract(binding)
            or native.CONTRACT_KEY in contract or paper.CONTRACT_KEY in contract
            or contract.get('online_target_objective') != expected):
        raise ValueError('Foreign or changed released AdaFlash loss contract')
    view = dict(contract)
    del view[CONTRACT_KEY]
    view[native.CONTRACT_KEY] = native.experiment_contract('K', binding)
    view['online_target_objective'] = dict(expected, name=native.OBJECTIVE)
    return view


def native_receipt_view(receipt):
    if (receipt.get('method') != ARM or receipt.get('objective') != OBJECTIVE
            or receipt.get('released_loss') != adaflash_repo_contract()):
        raise ValueError('Event is not the fixed released AdaFlash loss')
    for row in receipt['records']:
        observation = row['response_observation']
        sums = observation['sum_metrics']
        if (observation['record_id'] != row['record_id'] or observation['pass_index'] != row['pass']
                or sums['adaflash_response_count'] != 1
                or not all(math.isfinite(float(value)) for value in sums.values())):
            raise ValueError('Released-loss diagnostics must count each original forward exactly once')
    return dict(receipt, objective=native.OBJECTIVE)


def validate_event_receipt(receipt, *, event_number, require_completed=True, trainer_status=None):
    status = dict(trainer_status, objective=native.OBJECTIVE) if trainer_status is not None else None
    checked = native.validate_event_receipt(native_receipt_view(receipt), arm='K', event_number=event_number,
        require_completed=require_completed, trainer_status=status)
    return dict(checked, arm=ARM, objective=OBJECTIVE, released_loss=adaflash_repo_contract())


def validate_snapshot(state, *, binding, config_text, original_config_text,
                      completed_event, expected_contract, expected_recipe, expected_cuda_rng_states=1):
    # Only a read-only metadata view uses the old structural validator. Actual
    # snapshots and receipts retain AREPO and its already-checked exact loss.
    view = dict(state, algorithm_contract=native_contract_view(state['algorithm_contract'], binding),
                events=[native_receipt_view(event) for event in state['events']])
    result = native.validate_snapshot(view, arm='K', binding=binding, config_text=config_text,
        original_config_text=original_config_text, completed_event=completed_event,
        expected_contract=native_contract_view(expected_contract, binding),
        expected_recipe=expected_recipe, expected_cuda_rng_states=expected_cuda_rng_states)
    return dict(result, arm=ARM, released_loss=adaflash_repo_contract())


def verify_live(trainer, state, receipt_path=None):
    mismatch = differences(native.live_state(trainer), state)
    model = trainer.bundle.model
    if (trainer.closed or trainer.failed or trainer.core.accumulation_remainder
            or trainer.core._micro != trainer.presentations or trainer.core._ratio_totals
            or trainer.core._sum_totals or trainer._presentation_schedule is not None
            or not model.training or model.reference is not None or model.ranking_enabled
            or model.objective != OBJECTIVE
            or not isinstance(getattr(model,'point_loss_extension',None), ReleasedAdaFlashOPD)
            or any(p.grad is not None for p in trainer.bundle.draft_model.parameters())):
        mismatch.append('unclean_live_boundary')
    result = dict(status='failed' if mismatch else 'passed', exact_resume=not mismatch,
        mismatches=mismatch[:32], global_step=trainer.global_step, presentations=trainer.presentations,
        no_rng_reset=True, extra_model_passes=0, extra_optimizer_steps=0)
    if receipt_path is not None:
        write_json(receipt_path, result)
    if mismatch:
        raise ValueError('Live AREPO state differs: '+repr(mismatch[:8]))
    return result


class AdaFlashRepoTrainer(paper.AdaFlashTrainer):
    """Reuse PR29 observations and batching; keep a separate loss/state identity."""
    def __init__(self, config_path, output_root, *, binding, original_config_text,
                 resume_state=None, completed_event=0, expected_cuda_rng_states=1):
        if type(completed_event) is not int or not 0 <= completed_event <= 58:
            raise ValueError('Require a complete AREPO boundary 0..58')
        if (completed_event > 0) != (resume_state is not None):
            raise ValueError('Continuation requires exact own complete state')
        self.arm, self.binding = ARM, binding
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.completed_event = completed_event
        self.response_records, self.window_records = [], []
        self.work, self._work_handles = {}, []
        experiment_contract(binding)
        SpecForgeEventTrainer.__init__(self, config_path, output_root, resume_state=resume_state,
            objective=OBJECTIVE, reduction='response_mean', ranking_lambda=0.)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        if install_adaflash_repo(self.bundle.model) != adaflash_repo_contract():
            raise ValueError('Installed released loss differs from its contract')
        self.contract[CONTRACT_KEY] = experiment_contract(self.binding)
        self._install_observations()

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
        verify_live(self, state, self.output_root/'RESUME_CHECK.json')

    def save_initial(self, comparator_path):
        import torch
        destination = self.output_root/'initial_state.pt'
        if destination.exists() or self.global_step or self.presentations or self.events:
            raise ValueError('Initial AREPO state must be new and empty')
        with preserve_rng():
            state = native.live_state(self)
            receipt = validate_snapshot(state, **self.validation_arguments(), completed_event=0)
            _export_equal(state, self.cfg.model.draft_checkpoint_path)
            original = torch.load(comparator_path, map_location='cpu', weights_only=False)
            contract = original['algorithm_contract'][native.CONTRACT_KEY]
            native.validate_snapshot(original, arm='K', binding=contract['binding'],
                config_text=original['config_text'], original_config_text=self.original_config_text,
                completed_event=0, expected_contract=original['algorithm_contract'],
                expected_recipe=original['backend']['optimizer']['recipe'],
                expected_cuda_rng_states=self.expected_cuda_rng_states)
            fields = ('draft_state_dict','backend','rng','global_step','presentations','events','horizon')
            mismatch = differences({k:original[k] for k in fields}, {k:state[k] for k in fields})
            if mismatch:
                raise ValueError('Fresh released initialization differs: '+repr(mismatch[:8]))
            receipt['original_K_initialization'] = dict(status='passed',
                exact_released_weights_empty_adam_rng=True, rng_overwritten=False,
                changed_contract='Only installed loss and experiment identity')
            save_torch(destination, state)
            verify_live(self, torch.load(destination, map_location='cpu', weights_only=False))
            receipt.update(export_exact=True, state_path=str(destination), comparator_path=str(comparator_path))
            write_json(self.output_root/'INITIAL_CHECK.json', receipt)
        return receipt

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        started = time.perf_counter()
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False)
            result = validate_snapshot(state, **self.validation_arguments(), completed_event=len(self.events))
            if export_path is not None:
                _export_equal(state, export_path)
                result['export_exact'] = True
            result['live'] = verify_live(self, state)
        result['wall_s'] = time.perf_counter()-started
        if receipt_path is not None:
            write_json(receipt_path, result)
        return result

    def train_event(self, event_id, feature_paths, passes=2, *, reference_check=None, presentation_schedule=None):
        if reference_check is not None or presentation_schedule is not None:
            raise ValueError('Own-feedback AREPO uses natural shuffling and anchors')
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        validate_event_receipt(result, event_number=len(self.events), trainer_status=self.status())
        self.completed_event = len(self.events)
        return result

    def _checkpoint_state(self, event):
        event.update(method=ARM, released_loss=adaflash_repo_contract())
        validate_event_receipt(event, event_number=len(self.events)+1, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)
