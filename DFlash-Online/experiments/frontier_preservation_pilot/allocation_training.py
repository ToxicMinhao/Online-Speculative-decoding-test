"""One explicitly bound warm event, preserving native K+PAIR update mechanics."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.configuration import historical_view
from experiments.repair_conditional_kl_pilot import training as parent_training
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer

from .allocation_replay import CachedPrefixAllocation
from .fork import (FORK_CONTRACT_KEY, OLD_CONTRACT_KEY, PARENT_BINDING, PARENT_IDENTITY,
                   bind_fork_metadata, fork_contract, import_parent_state)
from .math import require
from .native import WorkReceipt, freeze_two_responses


def history_digest(events):
    return hashlib.sha256(json.dumps(events, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def production_contract(arm, binding):
    return dict(fork_contract(arm, source_binding=binding),
                numerical_contract='native_token_contraction_v2')


def validate_event(event, *, arm, binding, require_completed=True, trainer_status=None):
    require(event.get('method') == arm and event.get('dataset') == 'OpenCodeInstruct'
            and event.get(FORK_CONTRACT_KEY) == production_contract(arm, binding),
            'New event lost its explicit warm-fork identity')
    result = native.validate_event_receipt(event, arm='K', event_number=2,
                require_completed=require_completed, trainer_status=trainer_status)
    for row in event['records']:
        observed = row['response_observation']
        require(observed['record_id'] == row['record_id']
                and observed['sampled_anchors'] == row['sampled_anchors']
                and 'repair_pair_auxiliary' in observed['ratio_metrics'],
                'New event must retain native PAIR and original response observations')
        values = [observed['loss'], *observed['sum_metrics'].values()]
        values.extend(value for pair in observed['ratio_metrics'].values() for value in pair)
        require(all(math.isfinite(float(value)) for value in values), 'Nonfinite original-forward observation')
        parent_training.validate_allocation_observation(observed)
    return dict(result, fork_arm=arm, shared_feedback=True)


def validate_boundary(state, *, trainer):
    """Validate original history separately; native shape checks see a read-only view."""
    require(state['config_text'] == trainer.config_text
            and not native.differences(state['algorithm_contract'], trainer.contract),
            'Fork recovery configuration or algorithm identity changed')
    require(len(state['events']) == 2 and history_digest(state['events'][:1]) ==
            trainer.parent_validation['parent_history_digest'],
            'Immutable PR59 event-one history changed')
    parent_training.validate_event_receipt(state['events'][0], arm='PAIR',
        binding=PARENT_BINDING, event_number=1, maximum_events=1, require_completed=False)
    validate_event(state['events'][1], arm=trainer.fork_arm, binding=trainer.fork_binding,
                   require_completed=False)
    # The original full-K tensor/optimizer/RNG validator is reused on a transient
    # validation view, never on a rewritten parent or checkpoint artifact.
    contract = deepcopy(trainer.contract)
    del contract[FORK_CONTRACT_KEY]
    del contract[OLD_CONTRACT_KEY]
    contract[native.CONTRACT_KEY] = native.experiment_contract('K', trainer.fork_binding)
    config = historical_view(trainer.config_text)
    result = native.validate_snapshot(dict(state, algorithm_contract=contract, config_text=config),
        arm='K', binding=trainer.fork_binding, config_text=config,
        original_config_text=historical_view(trainer.original_config_text), completed_event=2,
        expected_contract=contract, expected_recipe=trainer.backend.optimizer.recipe,
        expected_cuda_rng_states=trainer.expected_cuda_rng_states)
    return dict(result, fork_arm=trainer.fork_arm, immutable_parent_history=True)


class ForkTrainer(parent_training.PilotTrainer):
    """Both arms restore PAIR1; only FRONTIER_P computes new prefix coefficients."""

    def __init__(self, config_path, output_root, *, arm, binding, parent_plan):
        from .native import select_parent
        parent, recovery, _ = select_parent(parent_plan)
        self.fork_arm, self.fork_binding = arm, binding
        self.parent_plan = parent
        self.parent_config_text = Path(parent['configs']['PAIR']).read_text()
        production_contract(arm, binding)
        super().__init__(config_path, output_root, arm='PAIR', binding=PARENT_BINDING,
            original_config_text=Path(parent['original_config']).read_text(), maximum_events=1,
            resume_state=recovery, completed_event=1)
        metadata = bind_fork_metadata(self.parent_validation,
            contract=fork_contract(arm, source_binding=binding),
            parent_config_text=self.parent_config_text, new_config_text=self.config_text)
        self.contract[FORK_CONTRACT_KEY] = production_contract(arm, binding)
        metadata.update(runtime='native K+PAIR; cached prefix allocation only for FRONTIER_P',
                        numerical_contract='native_token_contraction_v2')
        write_json(self.output_root / 'FORK_BINDING.json', metadata)

    def _restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        receipt = import_parent_state(self, state,
            contract=fork_contract(self.fork_arm, source_binding=self.fork_binding),
            parent_identity=identity(path), parent_config_text=self.parent_config_text,
            original_config_text=self.original_config_text, expected_contract=self.contract,
            expected_recipe=self.backend.optimizer.recipe, expected_cuda_rng_states=1)
        self.parent_validation = receipt['parent_validation']
        write_json(self.output_root / 'PARENT_RESTORATION.json', receipt)

    def check_installed_arm(self):
        extension = self.bundle.model.prefix_allocation
        if isinstance(extension, CachedPrefixAllocation):
            from .allocation_integration import validate_cached_teaching
            require(self.fork_arm == 'FRONTIER_P', 'CONTROL cannot install cached allocation')
            validate_cached_teaching(self)
        else:
            super().check_installed_arm()

    def _prepare_event(self, event_id, feature_paths, passes):
        require(not self.qualification_only and event_id == 'event_002' and passes == 2
                and len(feature_paths) == 128 and self.global_step == 128
                and self.presentations == 256 and len(self.events) == 1,
                'Only one warm continuation event is authorized')
        self._active_event = event_id
        self.response_records.clear()
        self.window_records.clear()
        self.work.clear()
        features = SpecForgeEventTrainer._prepare_event(self, event_id, feature_paths, passes)
        previous = {f['record_id'] for f in self.events[0]['features']}
        require(not previous.intersection(f['record_id'] for f in features),
                'New shared feedback overlaps the parent event')
        return features

    def discarded_startup(self, feature_paths, output):
        # The qualified startup calls strategy.forward_loss/backward directly;
        # it does not enter _train_passes or compute candidate scores.
        return parent_training.discarded_startup(self, feature_paths, output)

    def _train_passes(self, features, passes):
        if self.fork_arm == 'CONTROL':
            return native.FullKTrainer._train_passes(self, features, passes)
        from .allocation_integration import train_window
        from .allocation_scoring import NativeAllocationScorer

        rows = []
        for pass_index in range(passes):
            order = torch.randperm(len(features)).tolist()
            for start in range(0, len(order), 2):
                pair = [features[index] for index in order[start:start + 2]]
                window = self.global_step - 128 + 1
                output = self.events_root / self._active_event / f'allocation_{window:03d}'
                work = WorkReceipt(output / 'WORK.json')
                started = time.perf_counter()
                responses = freeze_two_responses(self, [row['path'] for row in pair])
                scorer = NativeAllocationScorer(self, responses, work=work,
                    collect_diagnostics=window in (1, 64, 128))
                snapshots = scorer.freeze_current()
                gradients = scorer.reference_gradient()
                preview = scorer.preview_and_direction(gradients)
                fast = scorer.fast_scores(preview['w'])
                torch.cuda.synchronize()
                diagnostics = dict(window=window, score_s=time.perf_counter()-started,
                    preview=preview['diagnostics'], direction_policy=fast['direction_policy'],
                    allocation=[{key: value.detach().cpu().tolist()
                                 if isinstance(value, torch.Tensor) else value
                                 for key, value in d.items()}
                                for d in fast['allocation_diagnostics']])
                coefficients = fast['allocations']
                if window in (1, 64, 128):
                    diagnostics['response_scores'] = [dict(
                        minimum=float(scores[snapshot['roles']['prefix']].min())
                            if bool(snapshot['roles']['prefix'].any()) else None,
                        maximum=float(scores[snapshot['roles']['prefix']].max())
                            if bool(snapshot['roles']['prefix'].any()) else None,
                        mass_moved_from_beta=float((assigned-snapshot['roles']['beta']).abs().sum()/2),
                        mass_moved_from_old_C=float((assigned-snapshot['old_C_coefficients']).abs().sum()/2))
                        for snapshot, assigned, scores in zip(snapshots, coefficients, fast['scores'])]
                del gradients, preview, fast
                scorer.close()
                del scorer
                result = train_window(self, pair, enabled=True, responses=responses,
                    snapshots=snapshots, coefficients=coefficients, work=work,
                    pass_index=pass_index)
                work.value['native_adam_steps'] = 1
                work.save()
                diagnostics.update(total_window_s=time.perf_counter()-started,
                                   native=result['window'])
                write_json(output / 'RESULT.json', diagnostics)
                # Candidate cost includes scoring once per two-response window.
                result['rows'][-1]['training_s'] += diagnostics['score_s']
                rows.extend(result['rows'])
                del responses, snapshots, coefficients
        require(len(rows) == 256 and len(self.response_records) == 256
                and len(self.window_records) == 128, 'Complete event accounting differs')
        return rows

    def _checkpoint_state(self, event):
        event.update(method=self.fork_arm, dataset='OpenCodeInstruct',
                     **{FORK_CONTRACT_KEY: production_contract(self.fork_arm, self.fork_binding)})
        validate_event(event, arm=self.fork_arm, binding=self.fork_binding, require_completed=False)
        state = SpecForgeEventTrainer._checkpoint_state(self, event)
        validate_boundary(state, trainer=self)
        return state

    def train_event(self, event_id, feature_paths, passes=2):
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        validate_event(result, arm=self.fork_arm, binding=self.fork_binding,
                       trainer_status=self.status())
        self.completed_event = 2
        return result

    def verify_boundary(self, export_path=None, receipt_path=None):
        from experiments.ok_coding.training import _export_equal
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False, mmap=True)
            result = validate_boundary(state, trainer=self)
            if export_path is not None:
                _export_equal(state, export_path)
            result.update(live=native.verify_live(self, state), export_exact=export_path is not None)
        if receipt_path is not None:
            write_json(receipt_path, result)
        return result
