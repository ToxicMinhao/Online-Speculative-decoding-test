"""Two independent first updates over the qualified per-turn native path.

Every branch gets a new trainer process, released weights, a fresh AdamW and
seed 17. No optimizer is shared or serialized: this experiment has no subsequent
update. Retained weights, inputs, anchors and accounting support all replays.
Historical scientific modules are loaded privately and never edited.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    '_first_update_per_turn_native', HERE.parent / 'dflash_per_turn/native.py')
pilot = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pilot
_spec.loader.exec_module(pilot)
base, followup, training = pilot.base, pilot.followup, pilot.training
require, digest, sha, write_json = pilot.require, pilot.digest, pilot.sha, pilot.write_json
BRANCH_LR = {'D1': 3e-4, 'L1': 1e-4}


def source_pins():
    return {str(p.relative_to(HERE.parent)): sha(p) for p in sorted(HERE.glob('*.py'))
            if not p.name.startswith('test_')}


def check_sources(config):
    require(config.get('first_update_source_sha256') == source_pins(),
            'Reviewed first-update source changed')
    pilot.check_sources(config)


def storage_bound(model_bytes):
    """Two exports, one conservative export-overlap unit, 512 MiB records.

    Adam and teacher replay features stay in process memory and are never written.
    The normal 8 GiB free-space reserve is separate and remains enforced.
    Existing prior-episode checkpoints are already charged against free space.
    """
    require(type(model_bytes) is int and model_bytes > 0, 'Positive model size required')
    return 3 * model_bytes + 512 * 2**20


def worker_command(config):
    command = pilot._original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Worker command changed')
    command[-1] = str(HERE / 'native.py')
    return command


# These are private imported module objects; historical file bytes are untouched.
base.worker_command = worker_command
base.episode_storage_bound = storage_bound


def exposure_projection(accounting):
    """Exact learning inputs/weights, excluding measured losses and phase timers."""
    fields = ('event', 'episode_id', 'arm_name', 'author_commit', 'response_audits',
              'capture_checkpoint_sha256', 'support_turns', 'optimizer_steps_this_event',
              'optimizer_steps_total', 'response_presentations_this_event',
              'response_presentations_total', 'eligible_position_presentations_this_event',
              'eligible_block_presentations_this_event', 'response_order', 'anchor_exposure_sha256')
    result = {key: accounting[key] for key in fields}
    result['recipe'] = {key: value for key, value in accounting['recipe'].items()
                        if key != 'learning_rate'}
    result['batches'] = []
    for batch in accounting['batches']:
        result['batches'].append({
            'pass_index': batch['pass_index'],
            'optimizer_steps_after': batch['optimizer_steps_after'],
            'presentations': [{key: value for key, value in p.items()
                               if key not in ('loss', 'phase_seconds')}
                              for p in batch['presentations']]})
    return result


def verify_exposure(accounting, original):
    require(accounting['event'] == original['event'] == 1 and
            accounting['optimizer_steps_this_event'] == original['optimizer_steps_this_event'] == 2 and
            accounting['optimizer_steps_total'] == original['optimizer_steps_total'] == 2 and
            accounting['response_presentations_this_event'] == original['response_presentations_this_event'] == 2,
            'First-update schedule differs from original two passes/two steps')
    actual, expected = exposure_projection(accounting), exposure_projection(original)
    require(digest(actual) == digest(expected), 'Original anchors or supervised exposure differ')
    return dict(status='passed', original_exposure_sha256=digest(expected),
                reconstructed_exposure_sha256=digest(actual),
                anchor_exposure_sha256=accounting['anchor_exposure_sha256'],
                same_anchors_and_supervised_positions=True, same_recipe_except_learning_rate=True,
                optimizer_steps=2, response_presentations=2)


class NativeBackend(pilot.NativeBackend):
    def start_episode(self, episode_id):
        pins = {'first_update_source_sha256': source_pins(),
                'per_turn_source_sha256': pilot.source_pins(),
                'followup_source_sha256': followup.followup_sources(),
                'author_sources_sha256': sha(followup.HERE / 'author_sources.json')}
        for key, value in pins.items():
            require(key not in self.config or self.config[key] == value, 'Configured source pin changed: ' + key)
        self.config.update(pins)
        check_sources(self.config)
        models = json.loads(base.checked_binding(self.config['model_bindings']).read_text())
        model_bytes = sum(v['bytes'] for name, v in models['released_draft']['checkpoint_files'].items()
                          if name.endswith('.safetensors'))
        needed = storage_bound(model_bytes)
        maximum = self.config['episode_storage_budget_bytes']
        require(needed <= maximum, 'Planned peak exceeds configured storage maximum')
        self.config.update(episode_storage_budget_bytes=needed, episode_storage_maximum_bytes=maximum)
        self.branches = {}
        target = next(item for item in models['targets'] if item['target_id'] == 'T0')
        framing = base._load(pilot.HERE / 'framing.py', '_first_update_framing')
        pinned_tokenizer = framing.load_pinned_tokenizer(models.get('tokenizer_path', target['model_path']))
        # Bypass only the historical pilot's three-diagnostic retention budget.
        # All base runtime/model/environment/free-space admission checks remain.
        result = base.NativeBackend.start_episode(self, episode_id)
        self.tokenizer = pinned_tokenizer
        return result

    def serve(self, input_ids, checkpoint_id, capture=False, *, turn_index, repeat=None):
        require(checkpoint_id in ('F', 'D1', 'L1') and turn_index in (1, 2),
                'This investigation has three checkpoints and two fixed inputs only')
        return super().serve(input_ids, checkpoint_id, capture, turn_index=turn_index, repeat=repeat)

    def train_branch(self, label, learning_rate, support, original_update):
        require(self.episode_id is not None and self.trainer is None, 'No trainer may carry over between branches')
        require(label in BRANCH_LR and learning_rate == BRANCH_LR[label] and label not in self.branches,
                'Branch label/rate changed or already executed')
        require(support['turn'] == 1 and support['capture_checkpoint'] == 'F' and
                support['capture_checkpoint_sha256'] == self.checkpoints['F']['checkpoint_identity_sha256'],
                'Only the saved released-draft first response may train')
        cfg = {**self.config, 'first_update_branch': {'checkpoint_id': label, 'learning_rate': learning_rate}}
        started = time.perf_counter()
        client = base._WorkerClient(cfg, 'train', self.out / ('training_' + label))
        self.trainer = client
        self.totals['startup_s'] += client.startup_s
        try:
            result = client.call('train', {'responses': [support], 'original_update': original_update})
            result['trainer_startup_s'] = client.startup_s
            result['checkpoint_transfer'] = {'performed': False, 'seconds': 0.,
                'scope': 'Shared local filesystem; device copies are charged in export and activation.'}
            checkpoint = result['checkpoint']
            require(checkpoint['checkpoint_id'] == label, 'Branch checkpoint label differs')
            base.verify_checkpoint(checkpoint)
            verify_exposure(result['accounting'], original_update['accounting'])
        finally:
            shutdown_started = time.perf_counter()
            client.close()
            shutdown_s = time.perf_counter() - shutdown_started
            self.totals['shutdown_s'] += shutdown_s
            self.trainer = None
        result.update(trainer_shutdown_s=shutdown_s, branch_wall_s=time.perf_counter()-started)
        self.checkpoints[label] = checkpoint
        self.branches[label] = result
        write_json(self.out / ('BRANCH_' + label + '.json'), result)
        return result


class ServingService(pilot.ServingService):
    def __init__(self, config):
        check_sources(config)
        super().__init__(config)


class TrainerService(pilot.TrainerService):
    def __init__(self, config):
        check_sources(config)
        branch = config['first_update_branch']
        require(branch['checkpoint_id'] in BRANCH_LR and
                branch['learning_rate'] == BRANCH_LR[branch['checkpoint_id']], 'Unsupported independent branch')
        # Qualified loader verifies every released draft tensor. Bypass only the
        # pilot's persistent trajectory/diagnostic-state serialization.
        followup.TrainerService.__init__(self, config)
        self.trainer = training.PersistentPerTurnTrainer(self.draft, self.components.embed_tokens,
            self.components.lm_head, self.teacher_features, episode_id=config['episode_id'], target_id='T0',
            target_identity_sha256=config['target']['identity_sha256'],
            released_checkpoint_sha256=config['released']['checkpoint_identity_sha256'], mask_token_id=128002,
            attention_backend='flex_attention', check_frozen=self.check_frozen)
        self.trainer.learning_rate = branch['learning_rate']
        self.trainer.optimizer.param_groups[0]['lr'] = branch['learning_rate']
        self.trainer._check_owner()
        require(not self.trainer.optimizer.state and self.trainer.optimizer_steps == 0 and
                self.trainer.completed_events == 0 and not self.trainer.seen_record_ids and
                self.trainer.active_checkpoint_sha256 == config['released']['checkpoint_identity_sha256'],
                'Branch must start with released weights and fresh Adam')
        self.used = False
        self.initial = dict(status='passed', released_checkpoint_sha256=config['released']['checkpoint_identity_sha256'],
            optimizer_empty=True, optimizer_steps=0, completed_events=0, seed=17,
            learning_rate=branch['learning_rate'], trainer_pid=__import__('os').getpid(),
            anchor_rng_sha256=training.state_digest(self.trainer.generator.get_state()),
            global_rng_sha256=training.state_digest(training.rng_state()),
            branch_isolation='New trainer process; no weights, Adam or RNG loaded from another branch')
        write_json(self.out / 'INITIAL_BRANCH.json', self.initial)
        effective = json.loads((self.out / 'AUTHOR_RUNTIME.json').read_text())
        effective.update(recipe={**training.RECIPE, 'learning_rate': branch['learning_rate']},
                         first_update_source_sha256=config['first_update_source_sha256'],
                         schedule='Saved response1 twice; exactly two optimizer steps; fresh branch')
        write_json(self.out / 'AUTHOR_RUNTIME.json', effective)

    def train(self, body):
        require(not self.used, 'Each branch executes exactly one update')
        responses = body['responses']
        self._check_responses(responses)
        require(len(responses) == 1 and responses[0]['turn'] == 1, 'First response only')
        original = body['original_update']['accounting']
        audit = training.audit_response(responses[0])
        require(digest([audit]) == digest(original['response_audits']), 'Original input/output or response mask differs')
        require(original['recipe']['learning_rate'] == 3e-4 and original['optimizer_steps_total'] == 2 and
                original['event'] == 1 and original['capture_checkpoint_sha256'] ==
                self.config['released']['checkpoint_identity_sha256'], 'Wrong original first-update reference')
        self.used = True
        accounting = self.trainer.train_event(1, responses)
        exposure_check = verify_exposure(accounting, original)
        label = self.config['first_update_branch']['checkpoint_id']
        checkpoint = self._export_label(label)
        self.trainer.bind_checkpoint(1, checkpoint['checkpoint_identity_sha256'])
        self.check_frozen()
        result = dict(status='passed', event=1, checkpoint_id=label, checkpoint=checkpoint,
            learning_rate=self.trainer.learning_rate, initial_state=self.initial,
            fresh_initialization_verified=True, original_exposure_match=True,
            optimizer_steps=2, accounting=accounting, exposure_check=exposure_check, responses=responses,
            training_s=accounting['costs']['training_s'], export_s=checkpoint['export_s'],
            optimizer_state_save_s=0., optimizer_state_retained=False,
            retention_scope='Final checkpoint, complete anchor/mask accounting and initial-state proof retained; '
                            'no later optimizer continuation is part of this isolated first-update experiment.',
            costs={**accounting['costs'], 'export_s': checkpoint['export_s'], 'optimizer_state_save_s': 0.})
        write_json(self.out / 'UPDATE_1.json', result)
        return result


def worker_main():
    check_sources(base._worker_config())
    base._ServingService = ServingService
    base._TrainerService = TrainerService
    base.worker_main()


if __name__ == '__main__':
    require(sys.argv[1:] == ['--worker'], 'Use the controlled experiment CLI')
    worker_main()
