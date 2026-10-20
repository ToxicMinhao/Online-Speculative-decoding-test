"""Fixed-pool orchestration over the unchanged qualified native OSD runtime.

Import is CPU-only. Training is synchronous and its entire process group must
exit before held-out timing. All physical serving checkpoints use the original
F/U1/U2 admission, publication and cold-cache paths.
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
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from experiments.dflash_per_turn import framing
from experiments.dflash_baseline_learning_curve import training

_spec = importlib.util.spec_from_file_location('_learning_curve_followup_native',
    HERE.parent / 'dflash_two_update_followup/native.py')
followup = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = followup
_spec.loader.exec_module(followup)
base = followup.base
base.OUTPUT_CAP = 1024
require, sha, digest, write_json = base.require, base.sha, base.object_hash, base.write_json
PHYSICAL = {'F': 'F', 'S2': 'U1', 'S8': 'U2'}
WARMUP_TEXT = 'Reply with the word ready.'


def source_pins():
    paths = []
    for package in ('dflash_baseline_learning_curve', 'dflash_two_update_followup',
                    'dflash_two_update', 'dflash_per_turn'):
        paths.extend((HERE.parent / package).glob('*.py'))
        paths.extend((HERE.parent / package).glob('*.json'))
        paths.extend((HERE.parent / package).glob('requirements*.txt'))
    return {str(p.relative_to(HERE.parent)): sha(p) for p in sorted(paths)
            if not p.name.startswith('test_')}


def check_sources(config):
    require(config.get('learning_curve_source_sha256') == source_pins(), 'Reviewed learning-curve source changed')
    followup.check_followup_sources(config)


def group_storage_bound(model_bytes):
    # Two weight exports (2M), two optimizer-only states (4M), one staging
    # allowance (M). The qualified BF16 Adam states use the weight dtype.
    # Compact traces, tokens, anchor/source-loss metadata: separate 512 MiB.
    require(type(model_bytes) is int and model_bytes > 0, 'Invalid model storage inventory')
    return 7 * model_bytes + 512 * 2**20


def campaign_storage_bound(model_bytes, groups=6):
    require(groups == 6, 'Exactly six groups are planned')
    return groups * group_storage_bound(model_bytes)


def worker_command(config):
    command = followup._original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Native worker interface changed')
    command[-1] = str(HERE / 'native.py')
    return command


base.worker_command = worker_command
base.episode_storage_bound = group_storage_bound


def process_group_alive(pid):
    """Detect surviving teacher/scheduler children as well as their parent."""
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def normalize_rounds(document, result, input_ids):
    output = result['output_ids']
    require(document['input_token_ids'] == input_ids and document['final_output_token_ids'] == output,
            'Native trace exact inputs/outputs differ')
    require(document['reconciliation']['status'] == 'passed', 'Native trace did not reconcile')
    emitted, rows = [], []
    initial = 0
    for event in document['events']:
        tokens = event.get('committed_token_ids', [])
        if event['kind'] != 'verification':
            require(not rows or not tokens, 'Non-verification emission after drafting began')
            initial += len(tokens)
            emitted.extend(tokens)
            continue
        start = event['output_offset']
        require(start == len(emitted), 'Verification offsets are not contiguous')
        proposals = event['draft_proposed_token_ids']
        rejection = event['rejection_boundary_proposal_index']
        prefix = len(proposals) if rejection is None else rejection
        accepted = event['committed_provenance'].count('accepted_draft')
        rows.append(dict(round_index=len(rows), output_offset=start,
            absolute_token_offset=len(input_ids)+start, accepted_prefix_length=prefix,
            accepted_draft_tokens=accepted, committed_tokens=len(tokens),
            committed_token_ids=list(tokens), proposed_tokens=len(proposals),
            first_rejection_offset=rejection, stop_clipped=bool(event.get('stop_clipped', False))))
        emitted.extend(tokens)
    require(emitted == output, 'Native emitted sequence differs')
    require(len(rows) == result['verification_calls'] and
            sum(r['accepted_draft_tokens'] for r in rows) == result['accepted_draft_tokens'] and
            sum(r['committed_tokens'] for r in rows) == result['verification_committed_tokens'],
            'Native trajectory/count discrepancy')
    return rows, initial


class NativeBackend(base.NativeBackend):
    def start_group(self, group):
        require(self.episode_id is None, 'Close prior group before reset')
        pins = dict(learning_curve_source_sha256=source_pins(),
                    followup_source_sha256=followup.followup_sources(),
                    author_sources_sha256=sha(followup.HERE / 'author_sources.json'))
        for key, value in pins.items():
            require(key not in self.config or self.config[key] == value, 'Source pin changed: '+key)
        self.config.update(pins)
        check_sources(self.config)
        self.group = copy.deepcopy(group)
        self.training_started = False
        self.training_closed = False
        self.retired_trainer_pid = None
        result = super().start_episode(group['group_id'])
        self.tokenizer = framing.load_pinned_tokenizer(self.config['tokenizer_path'])
        self.logical_checkpoints = {'F': self.checkpoints['F']}
        # Re-encode selected wording rather than trusting copied prepared IDs.
        for role in ('feedback', 'evaluation'):
            for item in group[role]:
                require(framing.encode_history(self.tokenizer, [{'role': 'user', 'content': item['text']}]) ==
                        item['input_ids'], 'Prepared tokenizer/text binding differs')
        return {**result, "checkpoints":dict(self.logical_checkpoints), "fresh_optimizer":True,
                "seed":17, "released_reset":True}

    def publish(self, checkpoint_id):
        # base.serve dispatches the physical label back through this method.
        label = PHYSICAL.get(checkpoint_id, checkpoint_id)
        require(label in ('F', 'U1', 'U2'), 'Unknown learning-curve checkpoint')
        result = super().publish(label)
        logical = next(k for k,v in PHYSICAL.items() if v == label)
        return {**result, "checkpoint_id":logical, "native_checkpoint_id":label}

    def assert_training_quiescent(self):
        require(self.trainer is None and self.training_closed, 'Training worker must be stopped before evaluation')
        alive = self.retired_trainer_pid is not None and process_group_alive(self.retired_trainer_pid)
        require(not alive, 'Background trainer/teacher child survives shutdown')
        return dict(status='passed', training_quiescent=True, trainer_processes=[],
                    terminated_process_group=self.retired_trainer_pid)

    def stop_trainer(self):
        require(self.training_started and self.trainer is not None, 'Complete training before closing its worker')
        worker = self.trainer
        pid = worker.process.pid
        started = time.perf_counter()
        worker.close()
        # WorkerClient waits for its parent; explicitly also wait for its owned
        # child process group. Do not allow benchmarks to overlap orphan cleanup.
        deadline = time.monotonic()+30
        while process_group_alive(pid) and time.monotonic() < deadline:
            time.sleep(.1)
        require(worker.process.poll() is not None and not process_group_alive(pid),
                'Training process group did not terminate; timing is blocked')
        self.trainer = None
        self.retired_trainer_pid, self.training_closed = pid, True
        self.totals['shutdown_s'] += time.perf_counter()-started
        receipt = {**self.assert_training_quiescent(), 'shutdown_s': time.perf_counter()-started}
        write_json(self.out / 'TRAINING_QUIESCENT.json', receipt)
        return receipt

    def serve(self, input_ids, checkpoint_id, role, request_id, capture=False):
        require(capture is False and role in ('feedback', 'evaluation', 'warmup'), 'Capture/role contract differs')
        if role in ('evaluation', 'warmup'):
            self.assert_training_quiescent()
        else:
            require(not self.training_started and checkpoint_id == 'F', 'All feedback must precede any training')
        if role == 'warmup':
            ordinal = 1
        else:
            matches = [i for i, item in enumerate(self.group[role], 1) if item['request_id'] == request_id]
            require(len(matches) == 1, 'Unknown group request')
            ordinal = matches[0] + (4 if role == 'evaluation' else 0)
            require(input_ids == self.group[role][matches[0]-1]['input_ids'], 'Request input IDs changed')
        result = super().serve(input_ids, PHYSICAL[checkpoint_id], False, turn_index=ordinal)
        trace = json.loads(base.checked_binding(result['trajectory']).read_text())
        rounds, initial = normalize_rounds(trace, result, input_ids)
        calls = result['verification_calls']
        result.update(input_ids=list(input_ids), checkpoint_id=checkpoint_id, role=role,
            request_id=request_id, output_tokens=len(result['output_ids']), context_tokens=len(input_ids),
            initial_output_tokens=initial, rounds=rounds, capture=False,
            feedback_capture_enabled=False, cap_hit=len(result['output_ids']) == 1024,
            committed_progress_per_call=result['verification_committed_tokens']/calls if calls else None,
            emitted_tokens_per_verification_call=len(result['output_ids'])/calls if calls else None,
            timing_scope='Complete native request: prefill, drafting, verification; no teacher replay, training, publication or cache flush.')
        return result

    def warmup(self, checkpoint_id):
        ids = framing.encode_history(self.tokenizer, [{'role':'user', 'content':WARMUP_TEXT}])
        row = self.serve(ids, checkpoint_id, 'warmup', 'warmup:'+checkpoint_id)
        row['excluded_from_core_generations_and_gains'] = True
        return row

    def prepare_support(self, canonical):
        require(len(canonical) == 4 and not self.training_started, 'Four frozen feedback responses required')
        directory = self.out / 'support'; directory.mkdir()
        records = []
        for turn, item in enumerate(canonical, 1):
            require(item['checkpoint_id'] == 'F' and item['input_ids'] == self.group['feedback'][turn-1]['input_ids'] and
                    item['checkpoint_identity_sha256'] == self.checkpoints['F']['checkpoint_identity_sha256'],
                    'Feedback did not come from frozen checkpoint on its own prompt')
            started = time.perf_counter()
            record = dict(record_id=item['request_id'], episode_id=self.episode_id, turn=turn,
                input_ids=list(item['input_ids']), output_ids=list(item['output_ids']),
                input_token_count=len(item['input_ids']), output_cap=1024, target_id='T0',
                target_identity_sha256=self.target['identity_sha256'], capture_checkpoint='F',
                capture_checkpoint_sha256=item['checkpoint_identity_sha256'],
                input_ids_sha256=digest(item['input_ids']), output_ids_sha256=digest(item['output_ids']),
                collection_kind='frozen_committed_token_metadata_only', feature_capture=False)
            path = directory / f'response_{turn}.json'; write_json(path, record)
            records.append({**record, 'metadata_binding':base.binding(path),
                'collection_s':time.perf_counter()-started})
        return records

    def train_pool(self, responses):
        require(not self.training_started and self.trainer is None, 'Fixed-pool trajectory executes once')
        self.training_started = True
        self.trainer = base._WorkerClient(self.config, 'train', self.out / 'training_worker')
        self.totals['startup_s'] += self.trainer.startup_s
        result = self.trainer.call('train', {'responses':responses})
        result['trainer_startup_s'] = self.trainer.startup_s
        for label in ('S2', 'S8'):
            checkpoint = result['checkpoints'][label]
            require(checkpoint['checkpoint_id'] == PHYSICAL[label], 'Export step/version mapping differs')
            base.verify_checkpoint(checkpoint)
            self.checkpoints[checkpoint['checkpoint_id']] = checkpoint
            self.logical_checkpoints[label] = checkpoint
        write_json(self.out / 'TRAINING_RESULT.json', result)
        return result

    def finish_group(self):
        return super().finish_episode()


class ServingService(base._ServingService):
    def __init__(self, config):
        check_sources(config)
        super().__init__(config)
        write_json(self.out / 'LEARNING_CURVE_SERVING_CONTRACT.json', dict(
            output_cap=1024, block_size=10, context_limit=32768, prefill_chunk_size=8192,
            cache_policy='cold before each independent request and checkpoint publication',
            inherited_output_cap_scope_note='The inherited 256-token prose is historical; effective kwargs/cap are1024.'))

    def generate(self, body):
        require(body['capture'] is False, 'Fixed-pool learning uses committed tokens, not serving feature capture')
        result = super().generate(body)
        require(result['feedback'] is None and result['capture_costs'] is None, 'Feedback capture was active')
        return result


class TrainerService(followup.TrainerService):
    def __init__(self, config):
        check_sources(config)
        # Keep the qualified compiled-backward buffer policy; ordinary training
        # uses a single backward per response and no agreement-filter gradients.
        import torch
        import torch._functorch.config as compiler
        compiler.donated_buffer = False
        super().__init__(config)
        self.trainer = training.FixedFeedbackTrainer(self.draft, self.components.embed_tokens,
            self.components.lm_head, self.teacher_features, episode_id=config['episode_id'], target_id='T0',
            target_identity_sha256=config['target']['identity_sha256'],
            released_checkpoint_sha256=config['released']['checkpoint_identity_sha256'], mask_token_id=128002,
            attention_backend='flex_attention', check_frozen=self.check_frozen,
            artifact_dir=self.out / 'accounting')
        effective = json.loads((self.out / 'AUTHOR_RUNTIME.json').read_text())
        effective['loader_args'] = effective.pop('args')
        effective.update(loader_args_scope='Model/teacher loading only; inherited buffer scheduling fields are unused. Effective training follows recipe and schedule below.',
            recipe=training.RECIPE,
            schedule='Four fixed responses, four saved-order passes, accumulation2, checkpoints0/2/8',
            donated_buffer=False, training_seed=17, source_diagnostics='Separate fixed anchors; training RNG restored',
            status='native_execution_requires_first_group_correctness_validation')
        write_json(self.out / 'AUTHOR_RUNTIME.json', effective)
        self.completed = False

    def train(self, body):
        import torch
        require(not self.completed and set(body) == {'responses'}, 'Exactly one fixed-pool training trajectory')
        responses = body['responses']
        require(len(responses) == 4, 'Four fixed feedback responses required')
        for response in responses:
            bound = base.checked_binding(response['metadata_binding'])
            require(bound.is_relative_to(Path(self.config['episode_out']) / 'support'), 'Foreign feedback path')
            saved = json.loads(bound.read_text())
            require(all(response.get(k) == value for k,value in saved.items()), 'Feedback metadata changed')
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        sources = [{**self.trainer.source_losses(responses), "checkpoint_id":"F"}]
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        checkpoints, updates = {'F':self.config['released']}, []
        for event, step in enumerate((2,8),1):
            accounting = self.trainer.train_to_step(step, responses)
            checkpoint = self._export(event)
            self.trainer.bind_checkpoint(event, checkpoint['checkpoint_identity_sha256'])
            state_path = Path(checkpoint['path']).parent / f'U{event}_training_state.pt'
            state_bytes = sum(v.numel()*v.element_size() for s in self.trainer.optimizer.state.values()
                              for v in s.values() if hasattr(v,'numel'))
            require(shutil.disk_usage(state_path.parent).free >= state_bytes+64*2**20+self.config.get('reserve_bytes',8*2**30),
                    'Optimizer state would cross storage reserve')
            state_started = time.perf_counter()
            state = self.trainer.save_checkpoint_state(state_path, include_draft=False)
            state_s = time.perf_counter()-state_started
            update = dict(step=step, accounting=accounting, checkpoint=checkpoint, training_state=state,
                          export_s=checkpoint['export_s'], optimizer_state_save_s=state_s)
            write_json(self.out / f'STEP_{step}.json', update)
            updates.append(update); checkpoints['S'+str(step)] = checkpoint
            sources.append({**self.trainer.source_losses(responses), "checkpoint_id":"S"+str(step)})
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        self.completed = True
        costs = dict(teacher_preparation_s=sum(u['accounting']['costs']['teacher_preparation_s'] for u in updates),
            training_s=sum(u['accounting']['costs']['training_s'] for u in updates),
            source_loss_diagnostic_s=sum(s['costs']['source_diagnostic_wall_s'] for s in sources),
            source_diagnostic_teacher_preparation_s=sum(s['costs']['teacher_preparation_s'] for s in sources),
            export_s=sum(u['export_s'] for u in updates),
            optimizer_state_save_s=sum(u['optimizer_state_save_s'] for u in updates),
            worker_trajectory_wall_s=time.perf_counter()-started,
            peak_allocated_gpu_bytes=peak_allocated,
            peak_reserved_gpu_bytes=peak_reserved,
            peak_memory_scope="Training worker: maximum across all source checks and training segments; excludes serving worker",
            checkpoint_transfer_s=0.,
            accounting='training_s contains teacher_preparation_s; source diagnostics/export/persistence remain separate, never double-add.')
        result = dict(status='passed',checkpoints=checkpoints,updates=updates,source_losses=sources,
            accounting=dict(recipe=dict(training.RECIPE),optimizer_steps=self.trainer.optimizer_steps,
                            response_presentations=self.trainer.response_presentations,
                            batches=[b for u in updates for b in u['accounting']['batches']]),costs=costs)
        write_json(self.out / 'RESULT.json', result)
        return result


def worker_main():
    check_sources(base._worker_config())
    base._ServingService = ServingService
    base._TrainerService = TrainerService
    base.worker_main()


if __name__ == '__main__':
    require(sys.argv[1:] == ['--worker'], 'Use the reviewed learning-curve entry point')
    worker_main()
