"""Per-turn controller over the qualified, source-pinned native serving path.

No GPU/worker startup on import. Dataset-only metadata and CPU tests can import
this file. The published pilot must be reviewed and GPU-qualified before use.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_per_turn_followup_native', HERE.parent / 'dflash_two_update_followup/native.py')
followup = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = followup
_spec.loader.exec_module(followup)
base = followup.base
# Private module only. All historical module globals and source bytes are intact.
base.OUTPUT_CAP = 1024
require, digest, sha, write_json = base.require, base.object_hash, base.sha, base.write_json
training = base._load(HERE / 'training.py', '_per_turn_training')
_original_worker_command = followup._original_worker_command


def source_pins():
    paths = list(HERE.glob('*.py')) + [HERE.parent / 'dflash_two_update_followup' / name for name in
                                     ('native.py', 'author_training.py', 'author_sources.json')]
    return {str(path.relative_to(HERE.parent)): sha(path) for path in sorted(paths) if not path.name.startswith('test_')}


def check_sources(config):
    require(config.get('per_turn_source_sha256') == source_pins(), 'Reviewed per-turn source changed')
    followup.check_followup_sources(config)


def storage_bound(model_bytes, remaining_diagnostic_cases=3):
    # Worst-case three disjoint selected cases (pre+post weights+Adam, lower-LR
    # weights), rolling two states, diagnostic restore/export staging and safety.
    # Teacher features remain transient; compact token/anchor records are kept.
    require(type(remaining_diagnostic_cases) is int and 0 <= remaining_diagnostic_cases <= 3,
            'Remaining diagnostic case budget must be between zero and three')
    # Eleven model-sized units cover rolling weights + two Adam moments,
    # incoming export/state overlap and safety. Each new retained case adds
    # pre/post weights+moments (six units) plus its low-LR weights (one).
    # Cases already on disk are charged by free-space measurement, not twice.
    return (11 + 7 * remaining_diagnostic_cases) * model_bytes + 512 * 2**20


def worker_command(config):
    command = _original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Native command contract changed')
    command[-1] = str(HERE / 'native.py')
    return command


base.worker_command = worker_command
base.episode_storage_bound = storage_bound


def label_valid(label):
    return label == 'F' or bool(re.fullmatch(r'[DL]([1-9]|1[0-9]|2[0-3])', label))


def admit_checkpoint(config, checkpoint):
    base.verify_checkpoint(checkpoint)
    label = checkpoint['checkpoint_id']
    require(label_valid(label), 'Unknown per-turn checkpoint')
    if label == 'F':
        require(checkpoint == config['released'], 'Released checkpoint differs')
    else:
        require(Path(checkpoint['path']) == Path(config['episode_out']) / 'checkpoints' / label,
                'Checkpoint outside this episode or wrong logical identity')
    return label


class NativeBackend(base.NativeBackend):
    def start_episode(self, episode_id):
        pins = {'per_turn_source_sha256': source_pins(),
                'followup_source_sha256': followup.followup_sources(),
                'author_sources_sha256': sha(followup.HERE / 'author_sources.json')}
        for key, value in pins.items():
            require(key not in self.config or self.config[key] == value, 'Configured source pin changed: ' + key)
        self.config.update(pins)
        check_sources(self.config)
        self.updates = {}
        models = json.loads(base.checked_binding(self.config['model_bindings']).read_text())
        remaining = self.config.get('remaining_diagnostic_cases', 3)
        model_bytes = sum(v['bytes'] for name, v in models['released_draft']['checkpoint_files'].items()
                          if name.endswith('.safetensors'))
        needed = storage_bound(model_bytes, remaining)
        maximum = self.config['episode_storage_budget_bytes']
        require(needed <= maximum, 'Planned peak exceeds configured episode storage maximum')
        self.config.update(episode_storage_budget_bytes=needed, episode_storage_maximum_bytes=maximum,
                           remaining_diagnostic_cases=remaining)
        base.episode_storage_bound = lambda size: storage_bound(size, remaining)
        target = next(v for v in models['targets'] if v['target_id'] == 'T0')
        framing = base._load(HERE / 'framing.py', '_per_turn_framing')
        pinned_tokenizer = framing.load_pinned_tokenizer(models.get('tokenizer_path', target['model_path']))
        result = super().start_episode(episode_id)
        self.tokenizer = pinned_tokenizer
        return result

    def encode(self, messages):
        require(self.episode_id is not None, 'Start episode before encoding')
        framing = base._load(HERE / 'framing.py', '_per_turn_framing')
        return framing.encode_history(self.tokenizer, messages)

    def serve(self, input_ids, checkpoint_id, capture=False, *, turn_index, repeat=None):
        require(capture is False, 'Author-path support uses token metadata, never verification-feature capture')
        require(label_valid(checkpoint_id), 'Unknown per-turn checkpoint')
        result = super().serve(input_ids, checkpoint_id, False, turn_index=turn_index, repeat=repeat)
        require(result['input_ids'] == input_ids and result['input_ids_sha256'] == digest(input_ids),
                'Native observation differs from exact canonical input IDs')
        result['cap_hit'] = len(result['output_ids']) == base.OUTPUT_CAP
        result['feedback_capture_enabled'] = False
        return result

    def finalize_retention(self, *, result_path):
        """Release unselected rolling bulk only after successful close+bound report."""
        require(self.worker is None and self.trainer is None and self.episode_id is None,
                'Final retention requires completed worker shutdown')
        result_path = Path(result_path).resolve()
        evidence_binding = base.binding(result_path)
        evidence = json.loads(base.checked_binding(evidence_binding).read_text())
        require(evidence.get('status') == 'complete' and evidence.get('episode_id') == self.out.name and
                evidence.get('implementation_checks', {}).get('status') == 'passed' and
                evidence.get('cleanup', {}).get('status') == 'passed',
                'Only complete, qualified episodes may retire rolling state')
        cases = evidence.get('diagnostics', [])
        require(all(case.get('status') == 'complete' for case in cases), 'Diagnostic bundle incomplete')
        protected = {case['event'] for case in cases}
        result = self.retire({'F'}, protected, _final=True)
        result['completion_evidence'] = evidence_binding
        write_json(self.out / 'FINAL_RETENTION.json', result)
        return result

    def prepare_support(self, item):
        started = time.perf_counter()
        turn, label = item['turn'], item['checkpoint_id']
        require(1 <= turn <= 23 and label == ('F' if turn == 1 else 'D' + str(turn-1)),
                'Support is not the current main online trajectory')
        require(item['checkpoint_identity_sha256'] == self.checkpoints[label]['checkpoint_identity_sha256'],
                'Support model identity differs')
        record = dict(record_id=f'{self.episode_id}:t{turn:02d}', episode_id=self.episode_id,
                      turn=turn, input_ids=list(item['input_ids']), output_ids=list(item['output_ids']),
                      input_token_count=len(item['input_ids']), output_cap=1024, target_id='T0',
                      target_identity_sha256=self.target['identity_sha256'], capture_checkpoint=label,
                      capture_checkpoint_sha256=item['checkpoint_identity_sha256'],
                      input_ids_sha256=digest(item['input_ids']), output_ids_sha256=digest(item['output_ids']),
                      collection_kind='committed_token_metadata_only', feature_capture=False)
        directory = self.out / 'support'; directory.mkdir(exist_ok=True)
        path = directory / f'turn_{turn:02d}.json'
        require(not path.exists(), 'Support must be serialized once from canonical serving')
        write_json(path, record)
        return {**record, 'metadata_binding': base.binding(path), 'collection_s': time.perf_counter()-started,
                'collection_timing_scope': 'Post-serving token metadata only; teacher replay is charged in training.'}

    def train_event(self, event, responses):
        require(self.episode_id is not None and event == len(self.updates)+1, 'Update order differs')
        startup = 0.
        if self.trainer is None:
            require(event == 1, 'Persistent optimizer begins at event1')
            self.trainer = base._WorkerClient(self.config, 'train', self.out / 'training_worker')
            startup = self.trainer.startup_s
            self.totals['startup_s'] += startup
        result = self.trainer.call('train', {'event': event, 'responses': responses})
        result['trainer_startup_s'] = startup
        result['checkpoint_transfer'] = {'performed': False, 'seconds': 0.,
            'scope': 'Shared local filesystem; device copies are inside export and activation.'}
        checkpoint = result['checkpoint']
        require(checkpoint['checkpoint_id'] == 'D'+str(event), 'Main update checkpoint differs')
        base.verify_checkpoint(checkpoint)
        self.checkpoints[checkpoint['checkpoint_id']] = checkpoint
        self.updates[event] = result
        return result

    def diagnostic_event(self, event, original_update):
        require(event in self.updates and original_update['checkpoint'] == self.updates[event]['checkpoint'],
                'Diagnostic must bind an actual main update')
        result = self.trainer.call('train', {'operation': 'diagnostic', 'event': event,
                                          'original_update': original_update})
        require(result['main_state_restored'] is True and result['same_anchor_exposure'] is True,
                'Diagnostic isolation/anchor identity failed')
        checkpoint = result['checkpoint']; base.verify_checkpoint(checkpoint)
        self.checkpoints[checkpoint['checkpoint_id']] = checkpoint
        return result

    def retire(self, keep_ids, protected_event_ids, *, _final=False):
        """Retire only backend-created bulk after controller finishes comparison.

        Protected events retain both weights and Adam/RNG states. This method is
        never invoked automatically before next-turn comparisons. Compact inputs,
        output/metrics, traces, anchor metadata and update receipts are retained.
        """
        require(len(protected_event_ids) <= 3, 'At most three protected diagnostic cases')
        require(set(protected_event_ids) <= set(self.updates), 'Protected event does not exist')
        keep = set(keep_ids) | {'F'}
        keep_state = set()
        # The newest state is needed by the persistent trainer and crash audit.
        events = set(protected_event_ids)
        if self.updates and not _final:
            events.add(max(self.updates))
        for event in events:
            update = self.updates[event]
            for which in ('pre_state', 'post_state'):
                keep_state.add(update[which]['state']['path'])
                keep.add(update[which]['checkpoint']['checkpoint_id'])
            if 'L'+str(event) in self.checkpoints:
                keep.add('L'+str(event))
        for label in keep:
            if label != 'F' and label.startswith('D'):
                event = int(label[1:])
                if event in self.updates:
                    keep_state.add(self.updates[event]['post_state']['state']['path'])
        free_before = shutil.disk_usage(self.out).free
        removed = []
        for label, checkpoint in list(self.checkpoints.items()):
            if label in keep:
                continue
            path = Path(checkpoint['path'])
            require(path.parent == self.out / 'checkpoints' and path.name == label and not path.is_symlink(),
                    'Refuse retirement outside owned checkpoint directory')
            base.verify_checkpoint(checkpoint)
            shutil.rmtree(path)
            removed.append({'path': str(path), 'reason': 'Completed comparison; not rolling/protected',
                            'lost_capability': 'Replay this checkpoint', 'checkpoint_identity_sha256': checkpoint['checkpoint_identity_sha256']})
            del self.checkpoints[label]
        states = {u[k]['state']['path']: u[k]['state'] for u in self.updates.values() for k in ('pre_state', 'post_state')}
        for path, bound in states.items():
            file = Path(path)
            if path in keep_state or not file.exists():
                continue
            require(file.parent == self.out / 'training_worker' / 'states' and not file.is_symlink(),
                    'Refuse state retirement outside owned state directory')
            base.checked_binding(bound)
            file.unlink()
            removed.append({'path': path, 'reason': 'Completed comparison; no protected restoration',
                            'lost_capability': 'Exact optimizer/RNG continuation at this historical point',
                            'sha256': bound['sha256']})
        receipt = dict(status='passed', keep_ids=sorted(keep), protected_event_ids=sorted(protected_event_ids),
                       removed=removed, free_before_bytes=free_before, free_after_bytes=shutil.disk_usage(self.out).free)
        directory = self.out / 'retirement'; directory.mkdir(exist_ok=True)
        write_json(directory / f'{len(list(directory.glob("*.json"))):04d}.json', receipt)
        return receipt


class ServingService(base._ServingService):
    def __init__(self, config):
        check_sources(config)
        super().__init__(config)
        self.logical_checkpoint = 'F'
        cap_receipt = json.loads((self.out / 'OUTPUT_CAP_COMPATIBILITY.json').read_text())
        cap_receipt['scope'] = 'Per-turn pilot: 1024-or-EOS; historical 256-token campaigns remain separate.'
        write_json(self.out / 'OUTPUT_CAP_COMPATIBILITY.json', cap_receipt)
        write_json(self.out / 'PER_TURN_SERVING.json', {'output_cap': 1024, 'block_size': 10,
            'cache_policy': 'cold request and weight switch; full exact input tokens',
            'physical_trace_alias': 'F for released; U1 for every learned checkpoint; exact hash and logical label disambiguate',
            'feature_capture': False,
            'legacy_trace_domain': 'code_math is an unused inherited trace tag; per-turn records carry actual source and stream.'})

    def publish(self, checkpoint):
        label = admit_checkpoint(self.config, checkpoint)
        physical = 'F' if label == 'F' else 'U1'
        self.bridge._two_update_admitted.add(Path(checkpoint['path']))
        started = time.perf_counter()
        receipt = self.bridge.publish({'model_path': checkpoint['path'], 'reset': True})
        require(receipt['status'] == 'passed' and receipt['success'] and
                receipt['helper_matches_current_layers'] and receipt['helper_norm_matches_current_layers'] and
                receipt['cache_after']['tree_total_size'] == 0, 'Native publication/cache refresh failed')
        require(self.bridge.state()['draft_model_path'] == checkpoint['path'], 'Wrong native weight path')
        self.logical_checkpoint = label
        self.active = {**checkpoint, 'checkpoint_id': physical}
        write_json(self.out / 'ACTIVE_CHECKPOINT.json', self.active)
        write_json(self.out / 'ACTIVE_PER_TURN_CHECKPOINT.json', checkpoint)
        return dict(status='passed', checkpoint_id=label, activation_s=time.perf_counter()-started, native_receipt=receipt)

    def generate(self, body):
        require(body['capture'] is False and body['checkpoint_id'] == self.logical_checkpoint,
                'Capture forbidden or wrong logical checkpoint')
        physical = 'F' if self.logical_checkpoint == 'F' else 'U1'
        result = super().generate({**body, 'checkpoint_id': physical})
        require(result['input_ids_sha256'] == digest(body['input_ids']), 'Observer input hash differs')
        document = json.loads(base.checked_binding(result['trajectory']).read_text())
        observed = document['input_token_ids']
        require(observed == body['input_ids'], 'Exact native observer input tokens differ')
        result.update(input_ids=observed, logical_checkpoint_id=self.logical_checkpoint)
        return result


class TrainerService(followup.TrainerService):
    def __init__(self, config):
        check_sources(config)
        super().__init__(config)
        self.trainer = training.PersistentPerTurnTrainer(self.draft, self.components.embed_tokens,
            self.components.lm_head, self.teacher_features, episode_id=config['episode_id'], target_id='T0',
            target_identity_sha256=config['target']['identity_sha256'],
            released_checkpoint_sha256=config['released']['checkpoint_identity_sha256'], mask_token_id=128002,
            attention_backend='flex_attention', check_frozen=self.check_frozen)
        effective = json.loads((self.out / 'AUTHOR_RUNTIME.json').read_text())
        effective.update(recipe=training.RECIPE, per_turn_source_sha256=config['per_turn_source_sha256'],
                         schedule='One newest response twice; two optimizer steps, no old-response loss')
        write_json(self.out / 'AUTHOR_RUNTIME.json', effective)
        self.states = self.out / 'states'; self.states.mkdir()
        self.updates = {}
        initial = self.trainer.save_state(self.states / 'initial.pt')
        self.current_state = {'state': initial, 'checkpoint': config['released']}
        write_json(self.out / 'PER_TURN_RECIPE.json', {'recipe': training.RECIPE,
            'max_length': base.CONTEXT_LIMIT, 'output_cap': 1024,
            'same_author_sampler_loss': True, 'responses_per_event': 1,
            'recovery': 'Bound states support isolated diagnostics; automatic interrupted-episode resume is not implemented.'})

    def _load_state(self, reference):
        import torch
        from safetensors.torch import load_file
        state = torch.load(base.checked_binding(reference['state']), map_location='cpu', weights_only=False)
        require(training.state_digest(state) == reference['state']['state_identity_sha256'], 'State content differs')
        base.verify_checkpoint(reference['checkpoint'])
        require(state['active_checkpoint_sha256'] == reference['checkpoint']['checkpoint_identity_sha256'],
                'Optimizer state/weights identity mismatch')
        weights = {}
        for name in reference['checkpoint']['files']:
            if name.endswith('.safetensors'):
                weights.update(load_file(str(Path(reference['checkpoint']['path']) / name), device='cpu'))
        self.trainer.restore(state, draft_state=weights)

    def _export_label(self, label):
        # Preserve the qualified export inventory/hash/finite-value checks.
        started = time.perf_counter()
        result = super()._export(label)
        path = Path(result['path']); destination = path.parent / label
        require(not destination.exists(), 'Never replace an existing checkpoint')
        os.rename(path, destination)
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return {**result, 'path': str(destination), 'checkpoint_id': label,
                'native_export_s': result['export_s'], 'export_s': time.perf_counter()-started}

    def _check_responses(self, responses):
        for response in responses:
            path = base.checked_binding(response['metadata_binding'])
            require(path.is_relative_to(Path(self.config['episode_out']) / 'support'), 'Support outside episode')
            saved = json.loads(path.read_text())
            require(all(response.get(k) == v for k, v in saved.items()), 'Support metadata changed')

    def train(self, body):
        if body.get('operation') == 'diagnostic':
            return self.diagnostic(body)
        event, responses = body['event'], body['responses']
        self._check_responses(responses)
        require(event == len(self.updates)+1, 'Persistent chronological update order required')
        pre = self.current_state
        accounting = self.trainer.train_event(event, responses)
        checkpoint = self._export_label('D'+str(event))
        self.trainer.bind_checkpoint(event, checkpoint['checkpoint_identity_sha256'])
        state_bytes = sum(v.numel()*v.element_size() for s in self.trainer.optimizer.state.values()
                          for v in s.values() if hasattr(v, 'numel'))
        require(shutil.disk_usage(self.states).free >= state_bytes+64*2**20+self.config.get('reserve_bytes', 8*2**30),
                'Saving recovery state would cross reserve')
        started = time.perf_counter()
        state = self.trainer.save_state(self.states / f'D{event}.pt')
        state_seconds = time.perf_counter()-started
        self.current_state = {'state': state, 'checkpoint': checkpoint}
        result = dict(status='passed', event=event, checkpoint_id=checkpoint['checkpoint_id'], checkpoint=checkpoint,
                      pre_state=pre, post_state=self.current_state, accounting=accounting,
                      optimizer_steps=accounting['optimizer_steps_total'], responses=responses,
                      training_s=accounting['costs']['training_s'], export_s=checkpoint['export_s'],
                      optimizer_state_save_s=state_seconds,
                      costs={**accounting['costs'], 'export_s': checkpoint['export_s'], 'optimizer_state_save_s': state_seconds})
        self.updates[event] = result
        write_json(self.out / f'UPDATE_{event}.json', result)
        return result

    def diagnostic(self, body):
        event = body['event']; original = self.updates[event]
        require(body['original_update']['checkpoint'] == original['checkpoint'], 'Wrong diagnostic update binding')
        current = self.current_state
        before_rng = training.rng_state()
        before_state = self.trainer.snapshot(include_draft=False)
        before_digest = training.state_digest(before_state)
        before_weights = training.state_digest(self.draft.state_dict())
        # Only latest or a retained selected state is replayable; no silent
        # reconstruction of retired optimizer histories.
        error = None; result = None; restored = False
        started = time.perf_counter()
        try:
            self._load_state(original['pre_state'])
            self.trainer.learning_rate = 1e-4
            self.trainer.optimizer.param_groups[0]['lr'] = 1e-4
            accounting = self.trainer.train_event(event, original['responses'])
            require(accounting['anchor_exposure_sha256'] == original['accounting']['anchor_exposure_sha256'] and
                    accounting['optimizer_steps_this_event'] == original['accounting']['optimizer_steps_this_event'] == 2,
                    'Diagnostic feedback/anchors/exposure changed')
            checkpoint = self._export_label('L'+str(event))
            result = dict(status='passed', event=event, checkpoint=checkpoint, accounting=accounting,
                          same_anchor_exposure=True, diagnostic_lr=1e-4, export_s=checkpoint['export_s'])
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            try:
                self._load_state(current)
                training.restore_rng(before_rng)
                require(training.state_digest(self.trainer.snapshot(include_draft=False)) == before_digest,
                        'Main optimizer/counters/anchor/global RNG state changed during diagnostic')
                require(training.state_digest(self.draft.state_dict()) == before_weights,
                        'Main draft tensors changed during diagnostic')
                restored = True
            finally:
                receipt = dict(status='passed' if error is None and restored else 'failed', event=event,
                               main_state_restored=restored, main_state_before_sha256=before_digest,
                               main_weights_before_sha256=before_weights,
                               error=error, wall_s=time.perf_counter()-started)
                write_json(self.out / f'DIAGNOSTIC_{event}_ISOLATION.json', receipt)
        result.update(main_state_restored=True, isolation=receipt,
                      diagnostic_wall_s=time.perf_counter()-started)
        write_json(self.out / f'DIAGNOSTIC_{event}.json', result)
        return result


def worker_main():
    check_sources(base._worker_config())
    base._ServingService = ServingService
    base._TrainerService = TrainerService
    base.worker_main()


if __name__ == '__main__':
    require(sys.argv[1:] == ['--worker'], 'Use the controlled experiment CLI')
    worker_main()
