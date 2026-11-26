"""Honest BASE/TAIL warm continuation using the ordinary native training loop."""
from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import time
import traceback

import torch

from experiments.full_k_gsm8k import training as native
from experiments.frontier_preservation_pilot.fork import validate_parent_snapshot, scientific_state
from experiments.ok_coding.storage import identity
from experiments.ok_coding.training import _export_equal
from experiments.repair_conditional_kl_pilot import training as parent_training
from online_feedback.kplus import install_kplus
from online_feedback.prefix_allocation import install_prefix_allocation
from online_feedback.repair_conditional_kl import install_repair_conditional_kl
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer

from .configuration import SCHEMA, PARENT_BINDING, digest, experiment_contract, validate_config


def compare_schedule(rows, expected):
    fields = ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after', 'optimizer_stepped')
    if len(rows) != len(expected):
        raise ValueError('Native presentation count changed')
    for index, (actual, reference) in enumerate(zip(rows, expected)):
        if any(actual[name] != reference[name] for name in fields):
            raise ValueError(f'Native schedule differs at presentation {index}')
    return dict(status='passed', presentations=len(rows), forced_anchors=False, reseeded=False)


def validate_event(event, trainer, *, completed=False):
    expected = dict(method=trainer.arm, dataset='OpenCodeInstruct', event_id='event_002',
                    objective='osd_first_error', passes=2, response_presentations=256,
                    optimizer_steps=128, start_step=128, end_step=256, reduction='response_mean')
    if any(event.get(key) != value for key, value in expected.items()):
        raise ValueError('New warm event identity or accounting changed')
    if event[SCHEMA] != experiment_contract(trainer.arm, trainer.binding):
        raise ValueError('New event lost its explicit continuation contract')
    if completed and event.get('status') != 'completed':
        raise ValueError('Cannot validate an incomplete event as completed')
    compare_schedule(event['records'], trainer.retained_event['records'])
    for row in event['records']:
        observed = row['response_observation']
        if (observed['record_id'] != row['record_id'] or observed['sampled_anchors'] != row['sampled_anchors']
                or 'repair_pair_auxiliary' not in observed['ratio_metrics']):
            raise ValueError('Native PAIR or response identity is missing')
        parent_training.validate_allocation_observation(observed)
        values = [observed['loss'], *observed['sum_metrics'].values()]
        values.extend(v for pair in observed['ratio_metrics'].values() for v in pair)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Nonfinite native response observation')
        if (row['learning_rate'] != (3e-4 if row['optimizer_stepped'] else None)):
            raise ValueError('Native constant learning rate changed')
    return dict(status='passed', arm=trainer.arm, presentations=256, adam_steps=128)


def validate_boundary(state, trainer):
    """Validate this contract directly, without relabelling it as a foreign arm."""
    if (state['config_text'] != trainer.config_text or state['algorithm_contract'] != trainer.contract
            or (state['global_step'], state['presentations'], len(state['events'])) != (256, 512, 2)
            or (state['reduction'], state['accumulation_steps'], state['horizon']) != ('response_mean', 2, 7424)
            or digest(state['events'][:1]) != trainer.parent_validation['parent_history_digest']):
        raise ValueError('Continuation endpoint or immutable parent history changed')
    validate_event(state['events'][1], trainer)
    saved = state['backend']['optimizer']
    if native.differences(saved['recipe'], trainer.backend.optimizer.recipe):
        raise ValueError('Native optimizer recipe changed')
    options = saved['optimizer_state_dict']
    group = options['param_groups'][0]
    named = list(trainer.bundle.draft_model.named_parameters())
    if len(named) != 58 or len(group['params']) != 58 or len(options['state']) != 58:
        raise ValueError('Incomplete native draft or Adam state')
    for (name, parameter), identifier in zip(named, group['params']):
        weight = state['draft_state_dict'][name]
        moments = options['state'][identifier]
        if weight.shape != parameter.shape or weight.dtype != torch.bfloat16:
            raise ValueError('Native parameter binding changed: ' + name)
        if float(moments['step']) != 256 or set(moments) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Native Adam step or moments changed: ' + name)
        for tensor in (weight, moments['exp_avg'], moments['exp_avg_sq']):
            if tensor.dtype != torch.bfloat16 or tensor.shape != parameter.shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Nonfinite or incompatible native tensor: ' + name)
    return dict(status='passed', arm=trainer.arm, immutable_parent_history=True,
                counters=[256, 512, 2], native_adam_entries=58)


class WarmContinuationTrainer(native.FullKTrainer):
    """Reuse native observations and loop; retain a separate new arm identity."""

    def __init__(self, config_path, output_root, *, arm, binding, parent_plan,
                 retained_event, qualification_only=False, adam_ledger=None):
        parent = json.loads(Path(parent_plan).read_text())
        self.arm, self.binding = arm, binding
        self.parent_plan = parent
        self.parent_config_text = Path(parent['configs']['PAIR']).read_text()
        self.original_config_text = Path(parent['original_config']).read_text()
        self.parent_recovery = Path(parent['root']) / 'arms/PAIR/trainer/training_state.pt'
        self.retained_event = retained_event
        self.qualification_only = qualification_only
        self.adam_ledger = adam_ledger
        self.completed_event = 1
        self.expected_cuda_rng_states = 1
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        experiment_contract(arm, binding)
        SpecForgeEventTrainer.__init__(self, config_path, output_root,
            resume_state=self.parent_recovery, objective='osd_first_error', reduction='response_mean', ranking_lambda=0.)
        # First authenticate and import the ORIGINAL parent, then declare the new child.
        self.contract[SCHEMA] = experiment_contract(arm, binding)
        if arm != 'BASE':
            from online_feedback.continuation_repair import install_continuation_repair
            install_continuation_repair(self.bundle.model, enabled=arm == 'TAIL')
        self._install_counted_step()
        self._install_counted_backward()
        write_json(self.output_root / 'CONTINUATION_BINDING.json', dict(status='bound',
            contract=self.contract[SCHEMA], parent_validation=self.parent_validation,
            parent_history_unchanged=True))

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        validate_config(self.config_text, self.parent_config_text)

    def _build_runtime(self):
        import rich.style  # Preserve the qualified initialization before native seeding.
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        install_prefix_allocation(self.bundle.model, relative_cap=.25)
        install_repair_conditional_kl(self.bundle.model, enabled=True)
        self.contract[parent_training.CONTRACT_KEY] = parent_training.experiment_contract('PAIR', PARENT_BINDING, 1)
        self.parent_algorithm_contract = deepcopy(self.contract)
        self._install_observations()

    def _restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        self.parent_validation = validate_parent_snapshot(state, parent_identity=identity(path),
            parent_config_text=self.parent_config_text, original_config_text=self.original_config_text,
            expected_contract=self.parent_algorithm_contract, expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=1)
        parent_training.PilotTrainer._load_state(self, state)
        mismatch = native.differences(scientific_state(native.live_state(self)), scientific_state(state))
        if mismatch:
            raise ValueError('Exact PAIR1 restoration failed: ' + repr(mismatch[:8]))
        if self.core.accumulation_remainder or any(p.grad is not None for p in self.bundle.draft_model.parameters()):
            raise ValueError('Restored parent has nonempty accumulation')
        write_json(self.output_root / 'PARENT_RESTORATION.json', dict(status='passed',
            exact_weights_optimizer_rng_counters_history=True, parent_validation=self.parent_validation))

    def _install_counted_step(self):
        ordinary_step = self.backend.optimizer.step
        def counted_step():
            token = self.adam_ledger.reserve(self.arm) if self.adam_ledger is not None else None
            result = ordinary_step()
            if self.adam_ledger is not None:
                self.adam_ledger.complete(token)
            return result
        self.backend.optimizer.step = counted_step

    def _install_counted_backward(self):
        native_backward = self.backend.backward

        def counted_backward(*args, **kwargs):
            self.work['backwards_attempted'] = self.work.get('backwards_attempted', 0) + 1
            result = native_backward(*args, **kwargs)
            self.work['backwards_completed'] = self.work.get('backwards_completed', 0) + 1
            return result

        self.backend.backward = counted_backward

    def _append_observation(self, name, value):
        if name == 'RESPONSE_OBSERVATIONS.jsonl':
            value['prefix_allocation'] = deepcopy(self.bundle.model.prefix_allocation.last_summary)
            extension = getattr(self.bundle.model, 'continuation_repair', None)
            if extension is not None:
                value['continuation_repair'] = {key: float(item.detach().item())
                                                for key, item in extension.last_summary.items()}
        if name == 'WINDOW_OBSERVATIONS.jsonl':
            value['clipped'] = value['gradient_norm'] > 1.
        return native.FullKTrainer._append_observation(self, name, value)

    def _prepare_event(self, event_id, feature_paths, passes):
        if (self.qualification_only or event_id != 'event_002' or passes != 2 or len(feature_paths) != 128
                or (self.global_step, self.presentations, len(self.events)) != (128, 256, 1)):
            raise ValueError('Only the complete declared warm event is authorized')
        self._active_event = event_id
        self.response_records.clear()
        self.window_records.clear()
        self.work.clear()
        features = SpecForgeEventTrainer._prepare_event(self, event_id, feature_paths, passes)
        previous = {r['record_id'] for r in self.events[0]['features']}
        if previous.intersection(r['record_id'] for r in features):
            raise ValueError('Event two reuses a parent training response')
        return features

    def _checkpoint_state(self, event):
        event.update(method=self.arm, dataset='OpenCodeInstruct', **{SCHEMA: self.contract[SCHEMA]})
        validate_event(event, self)
        state = SpecForgeEventTrainer._checkpoint_state(self, event)
        validate_boundary(state, self)
        return state

    def train_event(self, event_id, feature_paths, passes=2):
        result = SpecForgeEventTrainer.train_event(self, event_id, feature_paths, passes,
                                                 presentation_schedule=self.retained_event)
        validate_event(result, self, completed=True)
        self.completed_event = 2
        return result

    def verify_boundary(self, export_path):
        with preserve_rng():
            state = torch.load(self.latest, map_location='cpu', weights_only=False, mmap=True)
            result = validate_boundary(state, self)
            _export_equal(state, export_path)
            result.update(live=native.verify_live(self, state), export_exact=True)
        return result


def response_activity(observations):
    """Count each original response once, never an accumulated metrics row."""
    summaries = [row['continuation_repair'] for row in observations if 'continuation_repair' in row]
    return dict(original_responses=len(observations), continuation_summaries=len(summaries),
                active_response_count=sum(int(row['continuation_active_response']) for row in summaries),
                continuation_changed_high_positions=sum(int(row['continuation_changed_high_positions'])
                                                        for row in summaries))


def train(plan, arm='TAIL'):
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from .launch import AdamLedger, boundary
    qualification = json.loads(Path(plan['qualification_result']).read_text())
    if (arm != 'TAIL' or qualification['status'] != 'passed' or qualification['binding'] != plan['binding']
            or not qualification['base_disabled_exact'] or not qualification['retained_BASE_reusable']):
        raise ValueError('Only the qualified fixed TAIL continuation is admitted')
    root = Path(plan['root']) / 'arms' / arm
    root.mkdir(parents=True, exist_ok=False)
    boundary(plan, remaining_write_bytes=9_000_000_000)
    progress = dict(status='running', arm=arm, execution_commit=plan['execution_commit'])
    trainer = None
    started = time.perf_counter()
    try:
        pin_retained_graph_runtime(root)
        event = json.loads(Path(plan['retained_event']).read_text())
        loading = time.perf_counter()
        trainer = WarmContinuationTrainer(plan['configs'][arm], root/'trainer', arm=arm,
            binding=plan['binding'], parent_plan=plan['parent_plan'], retained_event=event,
            adam_ledger=AdamLedger(Path(plan['root'])/'ADAM_LEDGER.json', phase='training', arm=arm))
        progress['loading_s'] = time.perf_counter() - loading
        features = [r['path'] for r in json.loads(Path(plan['feature_manifest']).read_text())]
        progress['startup'] = parent_training.discarded_startup(trainer, features, root/'startup')
        result = trainer.train_event('event_002', features)
        progress['training'] = result
        progress['boundary'] = trainer.verify_boundary(result['export_path'])
        progress.update(status='completed', checkpoint=result['export_path'], recovery=str(trainer.latest),
                        global_step=trainer.global_step, presentations=trainer.presentations, completed_event=2,
                        checkpoint_identity=identity(Path(result['export_path'])/'model.safetensors'),
                        recovery_identity=identity(trainer.latest), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        progress['distinct_from_retained_BASE'] = progress['checkpoint_identity'] != plan['base_export_identity']
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if trainer is not None:
                progress['actual_work'] = dict(trainer.work)
                progress['activity'] = response_activity(trainer.response_records)
                if progress['status'] != 'completed':
                    trainer.failed = True
                try:
                    trainer.close()
                except BaseException as error:
                    already_failed = progress['status'] == 'failed'
                    progress.update(status='failed', close_error=repr(error))
                    if not already_failed:
                        raise
        finally:
            progress['wall_s'] = time.perf_counter() - started
            write_json(root/'STATUS.json', progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--phase', choices=['qualification', 'training'], required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if args.phase == 'qualification':
        from .qualification import qualify
        qualify(plan)
    else:
        train(plan)


if __name__ == '__main__':
    main()
