"""One warm Point3 continuation per SUM/MAX arm; no replacement training loop.

Sampling, native candidate teaching, optimizer behavior and persistence reuse the
qualified event infrastructure. Only the isolated C-reference and assessment
choice differ from the earlier beta-referenced allocation runner.
"""
from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import time
import traceback

import torch
import yaml

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.configuration import historical_view
from experiments.repair_conditional_kl_pilot import training as parent_training
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer
from .allocation_integration import train_window, validate_cached_teaching
from .allocation_replay import CachedPrefixAllocation
from .allocation_scoring import NativeAllocationScorer
from .allocation_qualify import vector_identity
from .allocation_training import ForkTrainer, history_digest
from .fork import OLD_CONTRACT_KEY, PARENT_BINDING, validate_parent_snapshot, scientific_state
from .math import require
from .native import WorkReceipt, freeze_two_responses
from .point3_native import native_c_reference_gradient, score_native_C

CONTRACT_KEY = 'point3_cumulative_max_v1'
ARMS = ('F_C_SUM', 'F_C_MAX')


def production_contract(arm, binding):
    require(arm in ARMS, 'Unknown Point3 sensitivity arm')
    native.experiment_contract('K', binding)
    return dict(name=CONTRACT_KEY, arm=arm, binding=binding,
        parent_binding=PARENT_BINDING, parent_step=128, parent_presentations=256,
        event='event_002', new_presentations=256, new_adam_steps=128,
        final_step=256, final_presentations=512, completed_event=2,
        inherited_horizon=7424, assessment=arm.removeprefix('F_C_'),
        reference='actual_native_C_preclip_gradient_no_step',
        allocator='beta_centered_signed_sensitivity_cap_0.25',
        direct_teaching='native_FKL_Kplus_PAIR', numerical_contract='native_token_contraction_v2',
        reset_optimizer=False, reset_seed=False, reference_adam_steps=0)


def validate_event(event, *, arm, binding, require_completed=True, trainer_status=None):
    require(event.get('method') == arm and event.get('dataset') == 'OpenCodeInstruct'
            and event.get(CONTRACT_KEY) == production_contract(arm, binding),
            'Point3 event identity changed')
    result = native.validate_event_receipt(event, arm='K', event_number=2,
        require_completed=require_completed, trainer_status=trainer_status)
    for row in event['records']:
        observed = row['response_observation']
        require(observed['record_id'] == row['record_id']
                and observed['sampled_anchors'] == row['sampled_anchors']
                and 'repair_pair_auxiliary' in observed['ratio_metrics'],
                'Point3 lost an original response or native PAIR observation')
        values = [observed['loss'], *observed['sum_metrics'].values()]
        values.extend(value for pair in observed['ratio_metrics'].values() for value in pair)
        require(all(math.isfinite(float(value)) for value in values), 'Nonfinite native observation')
        parent_training.validate_allocation_observation(observed)
    return dict(result, point3_arm=arm, shared_feedback=True)


def validate_boundary(state, *, trainer):
    require(state['config_text'] == trainer.config_text
            and not native.differences(state['algorithm_contract'], trainer.contract),
            'Point3 recovery recipe or algorithm identity changed')
    require(len(state['events']) == 2 and history_digest(state['events'][:1]) ==
            trainer.parent_validation['parent_history_digest'], 'Original PAIR1 history changed')
    parent_training.validate_event_receipt(state['events'][0], arm='PAIR',
        binding=PARENT_BINDING, event_number=1, maximum_events=1, require_completed=False)
    validate_event(state['events'][1], arm=trainer.point3_arm, binding=trainer.point3_binding,
                   require_completed=False)
    contract = deepcopy(trainer.contract)
    del contract[CONTRACT_KEY]
    del contract[OLD_CONTRACT_KEY]
    contract[native.CONTRACT_KEY] = native.experiment_contract('K', trainer.point3_binding)
    config = historical_view(trainer.config_text)
    result = native.validate_snapshot(dict(state, algorithm_contract=contract, config_text=config),
        arm='K', binding=trainer.point3_binding, config_text=config,
        original_config_text=historical_view(trainer.original_config_text), completed_event=2,
        expected_contract=contract, expected_recipe=trainer.backend.optimizer.recipe,
        expected_cuda_rng_states=trainer.expected_cuda_rng_states)
    return dict(result, point3_arm=trainer.point3_arm, immutable_parent_history=True)


def compare_schedule(rows, expected):
    fields = ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after', 'optimizer_stepped')
    require(len(rows) == len(expected), 'Retained C presentation count changed')
    for index, (actual, reference) in enumerate(zip(rows, expected)):
        require(all(actual[key] == reference[key] for key in fields),
                f'Point3 sampling differs from retained C at presentation {index}')
    return dict(status='passed', matched_presentations=len(rows), fields=list(fields),
                forced_anchors=False, rng_reset=False)


class Point3Trainer(ForkTrainer):
    """Explicit SUM/MAX contracts over the same persistent warm native trainer."""

    def __init__(self, config_path, output_root, *, arm, binding, parent_plan,
                 retained_C_records, adam_ledger=None):
        from .native import select_parent
        parent, recovery, _ = select_parent(parent_plan)
        self.point3_arm, self.point3_binding = arm, binding
        self.retained_C_records = retained_C_records
        self.adam_ledger = adam_ledger
        self.parent_plan = parent
        self.parent_config_text = Path(parent['configs']['PAIR']).read_text()
        production_contract(arm, binding)
        parent_training.PilotTrainer.__init__(self, config_path, output_root, arm='PAIR',
            binding=PARENT_BINDING, original_config_text=Path(parent['original_config']).read_text(),
            maximum_events=1, resume_state=recovery, completed_event=1)
        before, after = yaml.safe_load(self.parent_config_text), yaml.safe_load(self.config_text)
        for key in ('run_id', 'output_dir'):
            after.pop(key, None)
            before.pop(key, None)
        require(before == after, 'Point3 configuration changed beyond run/output metadata')
        self.contract[CONTRACT_KEY] = production_contract(arm, binding)
        write_json(self.output_root / 'FORK_BINDING.json', dict(
            status='bound_not_runtime_qualified', contract=self.contract[CONTRACT_KEY],
            parent_validation=self.parent_validation, original_history_retained=True))

    def _restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        self.parent_validation = validate_parent_snapshot(state,
            parent_identity=identity(path), parent_config_text=self.parent_config_text,
            original_config_text=self.original_config_text, expected_contract=self.contract,
            expected_recipe=self.backend.optimizer.recipe, expected_cuda_rng_states=1)
        parent_training.PilotTrainer._load_state(self, state)
        mismatch = native.differences(scientific_state(native.live_state(self)), scientific_state(state))
        require(not mismatch, 'Point3 parent restoration differs: ' + repr(mismatch[:8]))
        write_json(self.output_root / 'PARENT_RESTORATION.json', dict(
            status='passed', exact_weights_optimizer_rng_counters_history=True,
            parent_validation=self.parent_validation))

    def check_installed_arm(self):
        if isinstance(self.bundle.model.prefix_allocation, CachedPrefixAllocation):
            require(self.point3_arm in ARMS, 'Cached allocation requires a Point3 arm')
            return validate_cached_teaching(self)
        return parent_training.PilotTrainer.check_installed_arm(self)

    def _train_passes(self, features, passes):
        rows = []
        self.point3_reference_work = dict(windows=0, response_forwards=0,
                                         response_backwards=0, adam_steps=0, physical={})
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
                    collect_diagnostics=window in (1, 64, 128),
                    assessment=self.point3_arm.removeprefix('F_C_'),
                    collect_assessment_diagnostics=window in (1, 64, 128))
                snapshots = scorer.freeze_current()
                gradients, reference = native_c_reference_gradient(self, pair, responses, work)
                self.point3_reference_work['windows'] += 1
                self.point3_reference_work['response_forwards'] += reference['response_forwards']
                self.point3_reference_work['response_backwards'] += reference['response_backwards']
                for key, value in reference['physical_work'].items():
                    physical = self.point3_reference_work['physical']
                    physical[key] = physical.get(key, 0) + value
                direction, preview = score_native_C(scorer, gradients)
                fast = scorer.fast_scores(direction)
                torch.cuda.synchronize()
                diagnostics = dict(window=window, score_s=time.perf_counter()-started,
                    reference=reference, preview=preview, direction_policy=fast['direction_policy'],
                    allocation=[{key: value.detach().cpu().tolist()
                                 if isinstance(value, torch.Tensor) else value
                                 for key, value in item.items()}
                                for item in fast['allocation_diagnostics']])
                coefficients = fast['allocations']
                if window in (1, 64, 128):
                    diagnostics['responses'] = sparse_diagnostics(scorer, fast)
                diagnostics['coefficient_identity'] = vector_identity(
                    ['response_0', 'response_1'], coefficients)
                diagnostics['coefficients_sealed_before_commit'] = True
                write_json(output / 'SCORING.json', diagnostics)
                del gradients, direction, fast
                scorer.close()
                del scorer
                original_step = self.backend.optimizer.step
                def committed_step():
                    reservation = None if self.adam_ledger is None else self.adam_ledger.reserve(
                        self.point3_arm, window=window)
                    result = original_step()
                    if self.adam_ledger is not None:
                        self.adam_ledger.complete(reservation)
                    return result
                self.backend.optimizer.step = committed_step
                try:
                    result = train_window(self, pair, enabled=True, responses=responses,
                        snapshots=snapshots, coefficients=coefficients, work=work,
                        pass_index=pass_index)
                finally:
                    self.backend.optimizer.step = original_step
                # Compare before advancing to the next actual window; anchors
                # are observed from normal RNG sampling, never forced to match C.
                compare_schedule(result['rows'], self.retained_C_records[len(rows):len(rows)+2])
                work.value['native_adam_steps'] = 1
                work.save()
                diagnostics.update(total_window_s=time.perf_counter()-started,
                                   native=result['window'],
                                   native_preclip_norm=self.window_records[-1]['gradient_norm'],
                                   native_clipping_active=self.window_records[-1]['gradient_norm'] > 1.)
                write_json(output / 'RESULT.json', diagnostics)
                result['rows'][-1]['training_s'] += diagnostics['score_s']
                rows.extend(result['rows'])
                del responses, snapshots, coefficients
        require(len(rows) == len(self.response_records) == 256
                and len(self.window_records) == 128, 'Point3 committed work does not reconcile')
        return rows

    def _checkpoint_state(self, event):
        require(self.point3_reference_work['windows'] == 128
                and self.point3_reference_work['response_forwards'] == 256
                and self.point3_reference_work['response_backwards'] == 256
                and self.point3_reference_work['adam_steps'] == 0,
                'Point3 no-step C reference work does not reconcile')
        event['point3_reference_work'] = deepcopy(self.point3_reference_work)
        event['objective_work']['reference_operations'] = 256
        event['objective_work']['reference_operation_unit'] = 'native_C_response_forward_backward'
        event.update(method=self.point3_arm, dataset='OpenCodeInstruct',
                     **{CONTRACT_KEY: production_contract(self.point3_arm, self.point3_binding)})
        validate_event(event, arm=self.point3_arm, binding=self.point3_binding, require_completed=False)
        state = SpecForgeEventTrainer._checkpoint_state(self, event)
        validate_boundary(state, trainer=self)
        return state

    def train_event(self, event_id, feature_paths, passes=2):
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes)
        validate_event(result, arm=self.point3_arm, binding=self.point3_binding,
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


def sparse_diagnostics(scorer, fast):
    """Small allocation counters from existing frozen predictions and scores."""
    result = []
    for snapshot, assigned, scores in zip(scorer.snapshots, fast['allocations'], fast['scores']):
        roles = snapshot['roles']
        beta, prefix = roles['beta'], roles['prefix']
        centered = []
        for row_beta, row_scores, row_prefix in zip(beta.reshape(-1, 16), scores.reshape(-1, 16),
                                                   prefix.reshape(-1, 16)):
            if bool(row_prefix.any()):
                weights, values = row_beta[row_prefix], row_scores[row_prefix]
                # Match the allocator's stable centering, including tiny
                # contrasts atop a common offset. This is a readout only.
                offsets = values - values.min()
                centered.append(offsets - (weights*offsets).sum()/weights.sum())
        centers = torch.cat(centered) if centered else scores.new_empty(0)
        result.append(dict(prefix_positions=int(prefix.sum()),
            centered_score_min=float(centers.min()) if centers.numel() else None,
            centered_score_max=float(centers.max()) if centers.numel() else None,
            zero_centered_positions=int((centers == 0).sum()),
            allocation_mass_moved_from_beta=float((assigned-beta).abs().sum()/2),
            allocation_mass_moved_from_C=float((assigned-snapshot['old_C_coefficients']).abs().sum()/2),
            predicted_score_alignment=float(((assigned-beta)*scores*prefix).sum())))
    return result


def run_event(plan, arm, step_ledger=None):
    """One independent restored event; caller owns admission and card clock."""
    from .allocation_campaign import shared_features
    from .point3_launch import Point3StepLedger
    from .point3_readout import evaluate_witness_sets
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    require(arm in ARMS and plan['production_admitted'], 'Point3 event not admitted')
    qualification = json.loads(Path(plan['qualification_result']).read_text())
    require(qualification['status'] == 'completed'
            and qualification['native_reference']['status'] == 'passed'
            and qualification['qualification']['status'] == 'passed',
            'Complete native Point3 qualification is required')
    production = json.loads(Path(plan['production_plan']).read_text())
    paths = shared_features(production)
    status_path = Path(plan['retained_C_status'])
    require(identity(status_path) == plan['retained_C_status_identity'], 'Retained C receipt changed')
    retained = json.loads(status_path.read_text())
    require(retained['status'] == 'completed', 'Retained C event incomplete')
    expected = retained['training']['records']
    base = Path(plan['root']) / 'arms' / arm
    base.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', arm=arm, event_id='event_002')
    write_json(base / 'STATUS.json', status)
    trainer = None
    started = time.perf_counter()
    try:
        write_json(base / 'RUNTIME_COMPATIBILITY.json', pin_retained_graph_runtime(base))
        trainer = Point3Trainer(plan['configs'][arm], base / 'trainer', arm=arm,
            binding=plan['binding'], parent_plan=plan['parent_plan'], retained_C_records=expected,
            adam_ledger=step_ledger or Point3StepLedger(plan['step_ledger'], plan['contract_identity']))
        status['startup'] = trainer.discarded_startup(paths, base / 'startup')
        status['training'] = trainer.train_event('event_002', paths, passes=2)
        checkpoint = status['training']['checkpoint_path']
        status['boundary'] = trainer.verify_boundary(checkpoint, base / 'FINAL_STATE_CHECK.json')
        status['schedule'] = compare_schedule(status['training']['records'], expected)
        witness = Path(production['root'])/'witness'
        witness_work = WorkReceipt(base/'WITNESS_WORK.json')
        status['witness'] = evaluate_witness_sets(trainer,
            json.loads((witness/'BINDINGS.json').read_text()),
            json.loads((witness/'parent.json').read_text()), endpoint=arm, work=witness_work)
        write_json(base/'WITNESS.json', status['witness'])
        status['witness'] = dict(path=str(base/'WITNESS.json'), response_forwards=16)
        status.update(status='completed', checkpoint=checkpoint, recovery=str(trainer.latest),
            checkpoint_identity=identity(Path(checkpoint)/'model.safetensors'),
            recovery_identity=identity(trainer.latest), global_step=256, presentations=512,
            completed_event=2)
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status['wall_s'] = time.perf_counter()-started
        if torch.cuda.is_initialized():
            status.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(base / 'STATUS.json', status)
        if trainer is not None:
            trainer.close()
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    arguments = parser.parse_args()
    run_event(json.loads(Path(arguments.plan).read_text()), arguments.arm)
