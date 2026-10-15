"""Follow-up adapters over the unchanged, source-pinned two-update runtime.

No model import or process launch occurs on import. The original serving bridge,
KV/cache policy, output observer and publication transaction remain common to
all checkpoints. Only the training service changes. Native qualification is
pending review; CPU tests do not establish GPU compatibility or capacity.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_followup_base_native', HERE.parent / 'dflash_two_update/native.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
require, sha, digest, write_json = base.require, base.sha, base.object_hash, base.write_json
AUTHOR_COMMIT = '507b7aba5fbccaa99a400f161e4f2794ba6a388c'
PHYSICAL_IDS = {'F': 'F', 'C1': 'U1', 'C2': 'U2', 'A1': 'U1', 'A2': 'U2'}


def followup_sources():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}


def check_followup_sources(config):
    require(config.get('followup_source_sha256') == followup_sources(), 'Follow-up source changed')
    pins = json.loads((HERE / 'author_sources.json').read_text())
    require(pins['commit'] == AUTHOR_COMMIT, 'Author revision changed')
    author = Path(config['author_root']).resolve()
    root = Path(config['workspace_root']).resolve()
    require(author.is_relative_to(root), 'Author source must be under the approved project')
    for name, expected in pins['source_sha256'].items():
        require(sha(author / name) == expected, 'Pinned author source differs: ' + name)
    require(config.get('author_sources_sha256') == sha(HERE / 'author_sources.json'), 'Author pin manifest changed')
    return pins


_original_worker_command = base.worker_command


def worker_command(config):
    command = _original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Worker command interface changed')
    command[-1] = str(HERE / 'native.py')
    return command


# The base module is loaded under a private name; the historical entry point and
# source files are not edited. Child configuration still verifies both sources.
base.worker_command = worker_command


class NativeBackend(base.NativeBackend):
    def start_episode(self, episode_id, *, reference):
        require(reference['episode_id'] == episode_id, 'External checkpoint episode differs')
        external = {}
        for event, update in enumerate(reference['updates'], 1):
            checkpoint = copy.deepcopy(update['checkpoint'])
            checkpoint['checkpoint_id'] = 'C' + str(event)
            base.verify_checkpoint(checkpoint)
            external[checkpoint['checkpoint_id']] = checkpoint
        reviewed = {'followup_source_sha256': followup_sources(),
                    'author_sources_sha256': sha(HERE / 'author_sources.json')}
        for key, actual in reviewed.items():
            require(key not in self.config or self.config[key] == actual, 'Configured reviewed source differs: ' + key)
        self.config.update(external_checkpoints=external, **reviewed)
        check_followup_sources(self.config)
        result = super().start_episode(episode_id)
        require(self.checkpoints['F']['checkpoint_identity_sha256'] ==
                reference['canonical'][0]['checkpoint_identity_sha256'], 'Frozen initializer changed')
        self.checkpoints.update(external)
        return result

    def serve(self, input_ids, checkpoint_id, capture=False, *, turn_index, repeat=None):
        require(capture is False, 'Follow-up benchmarking and token collection forbid feature capture')
        require(checkpoint_id in PHYSICAL_IDS, 'Unknown follow-up checkpoint')
        return super().serve(input_ids, checkpoint_id, capture=False, turn_index=turn_index, repeat=repeat)

    def prepare_support(self, item):
        started = time.perf_counter()
        turn, label = item['turn'], item['checkpoint_id']
        require(9 <= turn <= 24 and label == ('F' if turn <= 16 else 'A1'), 'Support is not new canonical feedback')
        require(item['checkpoint_identity_sha256'] == self.checkpoints[label]['checkpoint_identity_sha256'],
                'Support weight identity differs')
        record = dict(record_id=f'{self.episode_id}:t{turn:02d}', episode_id=self.episode_id,
                      turn=turn, support_turn=turn, input_ids=list(item['input_ids']), output_ids=list(item['output_ids']),
                      input_token_count=len(item['input_ids']), output_cap=base.OUTPUT_CAP,
                      target_id='T0', target_identity_sha256=self.target['identity_sha256'],
                      capture_checkpoint=label, capture_checkpoint_sha256=item['checkpoint_identity_sha256'],
                      input_ids_sha256=digest(item['input_ids']), output_ids_sha256=digest(item['output_ids']),
                      collection_kind='committed_token_metadata_only', feature_capture=False)
        directory = self.out / 'support'; directory.mkdir(exist_ok=True)
        path = directory / ('turn_%02d.json' % turn)
        write_json(path, record)
        return {**record, 'metadata_binding': base.binding(path),
                'collection_s': time.perf_counter() - started,
                'collection_timing_scope': 'After serving; token metadata serialization only, no feature capture or teacher replay.'}

    def train_event(self, event, responses):
        require(self.episode_id is not None and event in (1, 2), 'Invalid author update')
        startup = 0.
        if self.trainer is None:
            require(event == 1, 'Persistent author trainer must begin at event1')
            self.trainer = base._WorkerClient(self.config, 'train', self.out / 'training_worker')
            self.totals['startup_s'] += self.trainer.startup_s
            startup = self.trainer.startup_s
        result = self.trainer.call('train', {'event': event, 'responses': responses})
        result['trainer_startup_s'] = startup
        result['checkpoint_transfer'] = {'performed': False, 'seconds': 0.,
            'scope': 'Shared local checkpoint path; device copies charged inside export/publication.'}
        checkpoint = result['checkpoint']
        require(checkpoint['checkpoint_id'] == 'A' + str(event), 'Author checkpoint label differs')
        base.verify_checkpoint(checkpoint)
        self.checkpoints[checkpoint['checkpoint_id']] = checkpoint
        return result


def admit_checkpoint(config, checkpoint):
    """Read-only old weights need an exact manifest binding; new ones stay local."""
    base.verify_checkpoint(checkpoint)
    label = checkpoint['checkpoint_id']
    require(label in PHYSICAL_IDS, 'Unknown follow-up checkpoint')
    if label == 'F':
        require(checkpoint == config['released'], 'Frozen checkpoint differs')
    elif label in ('C1', 'C2'):
        require(checkpoint == config['external_checkpoints'][label], 'Unbound retained checkpoint')
    else:
        expected = Path(config['episode_out']) / 'checkpoints' / ('U' + label[-1])
        require(Path(checkpoint['path']) == expected, 'Author checkpoint belongs to another episode')
    return label


class ServingService(base._ServingService):
    def __init__(self, config):
        check_followup_sources(config)
        super().__init__(config)
        self.logical_checkpoint = 'F'

    def publish(self, checkpoint):
        label = admit_checkpoint(self.config, checkpoint)
        self.bridge._two_update_admitted.add(Path(checkpoint['path']))
        started = time.perf_counter()
        receipt = self.bridge.publish({'model_path': checkpoint['path'], 'reset': True})
        require(receipt['status'] == 'passed' and receipt['success'] and
                receipt['helper_matches_current_layers'] and receipt['helper_norm_matches_current_layers'] and
                receipt['cache_after']['tree_total_size'] == 0, 'Native publication/cache refresh failed')
        require(self.bridge.state()['draft_model_path'] == checkpoint['path'], 'Active weight path differs')
        # Existing trace version IDs are numeric F/U1/U2. Logical arm is recorded
        # separately; exact content hash distinguishes C1 from A1 and C2 from A2.
        self.logical_checkpoint = label
        self.active = {**checkpoint, 'checkpoint_id': PHYSICAL_IDS[label]}
        write_json(self.out / 'ACTIVE_CHECKPOINT.json', self.active)
        write_json(self.out / 'ACTIVE_FOLLOWUP_CHECKPOINT.json', checkpoint)
        return dict(status='passed', checkpoint_id=label,
                    activation_s=time.perf_counter() - started, native_receipt=receipt)

    def generate(self, body):
        require(body['capture'] is False and body['checkpoint_id'] == self.logical_checkpoint,
                'Wrong logical checkpoint or feedback capture enabled')
        result = super().generate({**body, 'checkpoint_id': PHYSICAL_IDS[self.logical_checkpoint]})
        require(result['feedback'] is None and result['capture_costs'] is None, 'Timing capture is not off')
        result['logical_checkpoint_id'] = self.logical_checkpoint
        return result


class TrainerService(base._TrainerService):
    def __init__(self, config):
        check_followup_sources(config)
        self.config, self.out = config, Path(config['worker_out'])
        directory = Path(config['workspace_root']) / 'research/target_variant_specialization_001/training_tools'
        sys.path.insert(0, str(directory))
        import native as qualified_native
        require(Path(qualified_native.__file__).resolve() == directory / 'native.py', 'Training runtime collision')
        self.native = qualified_native
        self.runtime = qualified_native.load_runtime(17, base.OUTPUT_CAP)
        require(Path(qualified_native.CORE.AUTHOR).resolve() == Path(config['author_root']).resolve(),
                'Runtime imported a different author tree')
        args = copy.copy(self.runtime.default_args)
        require(config['released']['path'] == str(qualified_native.CORE.DRAFT) and
                config['target']['model_path'] == str(qualified_native.CORE.TARGET), 'Training model pair changed')
        args.target_model_path = config['target']['model_path']
        args.initial_draft_path = args.draft_config_path = config['released']['path']
        args.max_length = base.CONTEXT_LIMIT
        args.sglang_context_length = base.CONTEXT_LIMIT
        args.learning_rate = .0003
        require(args.target_model_backend == 'sglang' and args.sglang_attention_backend == 'fa3',
                'Keep the qualified SGLang teacher backend')
        self.teacher = self.runtime.setup.load_target_model(args)
        import torch
        self.teacher_model = self.teacher.model_runner.model
        self.teacher_model.eval().requires_grad_(False)
        self.components = self.runtime.setup.load_target_embeddings(args).eval().requires_grad_(False)
        self.draft, self.model_config, _, mask = self.runtime.setup.load_draft_stack(args, self.teacher)
        require(mask == 128002 and self.draft.block_size == 10 and self.draft.target_layer_ids == [1, 8, 15, 22, 29],
                'Released DFlash architecture changed')
        parameters = list(self.draft.parameters())
        require(len(parameters) == 58 and sum(p.numel() for p in parameters) == 1048626432 and
                all(p.dtype == torch.bfloat16 and p.requires_grad for p in parameters), 'Full-weight BF16 inventory differs')
        require(float(getattr(self.model_config, 'attention_dropout', 0.)) == 0. and
                not any(getattr(m, 'gradient_checkpointing', False) for m in self.draft.modules()),
                'Backbone checkpointing/dropout differs from matched controls')
        qualified_native.CORE.verify_checkpoint(self.draft, config['released']['path'])
        self.frozen_versions = self._frozen_versions()
        training = base._load(HERE / 'author_training.py', '_followup_author_training')
        self.trainer = training.PersistentAuthorOSDTrainer(self.draft, self.components.embed_tokens,
            self.components.lm_head, self.teacher_features, episode_id=config['episode_id'], target_id='T0',
            target_identity_sha256=config['target']['identity_sha256'],
            released_checkpoint_sha256=config['released']['checkpoint_identity_sha256'], mask_token_id=mask,
            attention_backend='flex_attention', check_frozen=self.check_frozen)
        write_json(self.out / 'AUTHOR_RUNTIME.json', {'args': vars(args), 'recipe': training.RECIPE,
            'source_sha256': config['followup_source_sha256'], 'author_sources': check_followup_sources(config),
            'teacher_context_limit': args.sglang_context_length, 'serving_input_limit': base.INPUT_LIMIT,
            'padding': '9 masked tail slots; never replay artificial padding through teacher',
            'status': 'native_execution_requires_qualification'})

    def _frozen_versions(self):
        return tuple((owner, name, id(p), p._version) for owner, module in
                     (('teacher', self.teacher_model), ('head_embedding', self.components))
                     for name, p in module.named_parameters())

    def check_frozen(self):
        require(self._frozen_versions() == self.frozen_versions and
                all(not p.requires_grad and p.grad is None for m in (self.teacher_model, self.components)
                    for p in m.parameters()), 'Fixed target weights changed')

    def teacher_features(self, ids):
        import torch
        with torch.no_grad():
            result = self.teacher.generate_dflash_data(ids, torch.ones_like(ids), torch.zeros_like(ids, dtype=torch.float32))
        final = result.teacher_hidden_states
        require(final is not None and final.shape == (*ids.shape, self.components.lm_head.in_features),
                'Native normalized final-layer teacher features missing')
        require(result.hidden_states.shape[:2] == ids.shape and result.hidden_states.shape[-1] == 5 * final.shape[-1],
                'Native selected layer features missing or shortened')
        return result.hidden_states.detach(), final.detach()

    def train(self, body):
        event, responses = body['event'], body['responses']
        for response in responses:
            bound = base.checked_binding(response['metadata_binding'])
            require(bound.is_relative_to(Path(self.config['episode_out']) / 'support'), 'Support outside current episode')
            saved = json.loads(bound.read_text())
            require(all(response.get(k) == v for k, v in saved.items()), 'Token feedback metadata changed')
        accounting = self.trainer.train_event(event, responses)
        checkpoint = self._export(event)  # Same export, file hash and storage gate as existing updater.
        checkpoint['checkpoint_id'] = 'A' + str(event)
        self.trainer.bind_checkpoint(event, checkpoint['checkpoint_identity_sha256'])
        state_path = Path(checkpoint['path']).parent / ('A%d_training_state.pt' % event)
        optimizer_bytes = sum(v.numel() * v.element_size() for s in self.trainer.optimizer.state.values()
                              for v in s.values() if hasattr(v, 'numel'))
        require(shutil.disk_usage(state_path.parent).free >= optimizer_bytes + 64 * 2**20 +
                self.config.get('reserve_bytes', 8 * 2**30), 'Optimizer state would cross storage reserve')
        started = time.perf_counter()
        state = self.trainer.save_checkpoint_state(state_path, include_draft=False)
        state_s = time.perf_counter() - started
        result = dict(status='passed', checkpoint_id=checkpoint['checkpoint_id'], checkpoint=checkpoint,
                      checkpoint_path=checkpoint['path'], optimizer_steps=accounting['optimizer_steps_total'],
                      training_s=accounting['costs']['training_s'], accounting=accounting,
                      training_state=state, export_s=checkpoint['export_s'], optimizer_state_save_s=state_s,
                      costs={**accounting['costs'], 'export_s': checkpoint['export_s'], 'optimizer_state_save_s': state_s})
        write_json(self.out / ('UPDATE_%d.json' % event), result)
        return result


def worker_main():
    check_followup_sources(base._worker_config())
    base._ServingService = ServingService
    base._TrainerService = TrainerService
    base.worker_main()


if __name__ == '__main__':
    require(sys.argv[1:] == ['--worker'], 'Use run.py for controlled planning/execution')
    worker_main()
