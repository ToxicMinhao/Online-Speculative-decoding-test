"""Read-only source replay and paired held-out loss on the qualified runtime.

Imports are CPU-only. The historical serving/loss code is not edited. A transport
worker named ``train`` is used solely to select the pinned teacher GPU and its
single-rank environment; its RPC service admits only ``evaluate``, has no
optimizer, and cannot train, export, or publish a diagnostic update.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
_spec = importlib.util.spec_from_file_location('_kl_diagnostic_qualified_curve',
    HERE.parent / 'dflash_baseline_learning_curve/native.py')
curve = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = curve
_spec.loader.exec_module(curve)
base, followup = curve.base, curve.followup
require, sha, digest, write_json = base.require, base.sha, base.object_hash, base.write_json
ORIGINAL_NATIVE_ROOT = Path('/data/sources/minhao/EAGLE_SPEC/research/dflash_baseline_learning_curve_20261020/execution_001/attempt_001/native')
GROUP_BYTES = 2**30
STAGING_BYTES = 256 * 2**20
MIN_RESERVE_BYTES = 8 * 2**30
PHYSICAL = {'F': 'F', 'S8': 'U2'}
STOP_IDS = (128001, 128008, 128009)


def source_pins():
    pins = curve.source_pins()
    for pattern in ('*.py', '*.json', 'requirements*.txt'):
        for path in sorted(HERE.glob(pattern)):
            if not path.name.startswith('test_'):
                pins[str(path.relative_to(HERE.parent))] = sha(path)
    return dict(sorted(pins.items()))


def check_sources(config):
    require(config.get('diagnostic_source_sha256') == source_pins(), 'Reviewed diagnostic source changed')
    # Original controller RUNTIME_CONFIG precedes start_group's derived pins.
    # First validate its complete reviewed inventory, then restore those exact
    # historical derived pins; reject any supplied conflicting value.
    require(config.get('learning_curve_source_sha256') == curve.source_pins(), 'Qualified baseline source changed')
    for key, value in dict(followup_source_sha256=followup.followup_sources(),
                          author_sources_sha256=sha(followup.HERE / 'author_sources.json')).items():
        require(key not in config or config[key] == value, 'Qualified derived source pin changed: ' + key)
        config[key] = value
    curve.check_sources(config)


def storage_preflight(config):
    """All-six compact evidence, staging and reserve; no model/state copy."""
    root = Path(config['output_root']).resolve()
    existing = root
    while not existing.exists():
        existing = existing.parent
    reserve = config.get('reserve_bytes', MIN_RESERVE_BYTES)
    require(type(reserve) is int and reserve >= MIN_RESERVE_BYTES, 'Preserve the 8 GiB storage reserve')
    required = 6 * GROUP_BYTES + STAGING_BYTES + reserve
    free = shutil.disk_usage(existing).free
    require(free >= required, 'Diagnostic compact evidence plus staging and reserve exceed free space')
    return dict(status='passed', free_bytes=free, required_bytes=required,
        group_budget_bytes=GROUP_BYTES, groups=6, staging_bytes=STAGING_BYTES,
        reserve_bytes=reserve, checkpoint_copies=0, optimizer_states=0,
        scope='Six groups of compact tokens/traces/anchor metadata; no feature tensors or exports.')


def worker_command(config):
    command = followup._original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Pinned worker transport changed')
    command[-1] = str(HERE / 'native.py')
    return command


# Private imported bridge only. Historical modules/files remain unchanged.
base.worker_command = worker_command


def admit_checkpoint(config, checkpoint):
    label = next((logical for logical, value in config['diagnostic_checkpoints'].items()
                  if checkpoint == value), None)
    require(label in PHYSICAL, 'Checkpoint is not the exact bound F/own-S8 artifact')
    base.verify_checkpoint(checkpoint)
    require(checkpoint['checkpoint_id'] == PHYSICAL[label], 'Physical checkpoint label differs')
    if label == 'F':
        require(checkpoint == config['released'], 'Released initializer differs')
    else:
        require(Path(checkpoint['path']) == ORIGINAL_NATIVE_ROOT / config['episode_id'] / 'checkpoints/U2',
                'S8 belongs to a different original group or run')
    return label


def close_owned_worker(worker, timeout_s=30., poll_s=.1):
    """Reap this client's group, allowing its scheduler children time to exit.

    WorkerClient owns and signals only the group it launched. This extra wait
    does not signal external processes or silently admit a surviving child.
    """
    require(timeout_s >= 0 and poll_s > 0, 'Invalid owned-worker shutdown bound')
    started = time.perf_counter()
    worker.close()
    deadline = time.monotonic() + timeout_s
    alive = curve.process_group_alive(worker.process.pid)
    while alive and time.monotonic() < deadline:
        time.sleep(min(poll_s, max(0., deadline-time.monotonic())))
        alive = curve.process_group_alive(worker.process.pid)
    require(worker.process.poll() is not None and not alive, 'Owned worker process group survived bounded shutdown')
    return dict(status='passed', terminated_process_group=worker.process.pid,
                shutdown_s=time.perf_counter()-started, child_exit_wait_bound_s=timeout_s)


class NativeBackend(base.NativeBackend):
    """No training API or parent training service is exercised by this backend."""
    def __init__(self, config):
        super().__init__(config)
        self.loss_worker = None
        self.source_seen = set()
        self.loss_seen = set()
        self.loss_phase = False

    def start_group(self, original_group):
        require(self.episode_id is None, 'Close prior diagnostic group first')
        check_sources(self.config)
        base.check_environment(self.config)
        dependencies = base.check_dependencies(self.config['workspace_root'])
        require(self.config['runtime_source_sha256'] == dependencies['runtime_source_sha256'],
                'Exact qualified runtime pins required')
        require(self.config.get('output_cap', 1024) == 1024 and self.config.get('block_size', 10) == 10,
                'Keep block 10 and 1024-or-EOS')
        group = copy.deepcopy(original_group)
        group_id = group['group_id']
        require(Path(group_id).name == group_id and group_id not in ('.', '..'), 'Invalid group identifier')
        require(group['status'] == 'complete' and len(group['feedback']) == 4, 'Completed original group required')
        root = Path(self.config['output_root']).resolve()
        root.mkdir(parents=True, exist_ok=True)
        reserve = self.config.get('reserve_bytes', MIN_RESERVE_BYTES)
        require(reserve >= MIN_RESERVE_BYTES and shutil.disk_usage(root).free >= reserve + GROUP_BYTES + STAGING_BYTES,
                'Group compact evidence plus staging would cross the storage reserve')
        out = root / group_id
        require(not out.exists(), 'Never overwrite prior diagnostic evidence')
        models_path = base.checked_binding(self.config['model_bindings'])
        require(sha(models_path) == dependencies['qualified_model_bindings']['sha256'], 'Qualified model registry differs')
        models = json.loads(models_path.read_text())
        target = next(v for v in models['targets'] if v['target_id'] == 'T0')
        released = models['released_draft']
        frozen = dict(path=released['path'], files=released['checkpoint_files'], checkpoint_id='F',
                      checkpoint_identity_sha256=digest(released['checkpoint_files']))
        require(group['checkpoints']['F'] == frozen, 'Original frozen identity differs from pinned registry')
        checkpoints = {'F': frozen, 'S8': copy.deepcopy(group['checkpoints']['S8'])}
        cfg = {**self.config, 'episode_id': group_id, 'episode_out': str(out), 'target': target,
            'released': frozen, 'diagnostic_checkpoints': checkpoints,
            'registry_path': str(models_path), 'registry_sha256': sha(models_path),
            'tokenizer_path': models.get('tokenizer_path', target['model_path']),
            'max_capture_bytes': self.config.get('max_capture_bytes', base.MAX_CAPTURE_BYTES),
            'effective_limits': {'input_tokens':base.INPUT_LIMIT, 'total_context_tokens':base.CONTEXT_LIMIT,
                'output_tokens':1024, 'proposal_block_tokens':10, 'chunked_prefill_tokens':8192,
                'capture_memory_bound_bytes':base.capture_memory_bound()},
            'public_source_sha256': {name: sha(base.HERE / name)
                                    for name in ('native.py', 'training.py', 'native_dependencies.json')}}
        for checkpoint in checkpoints.values():
            admit_checkpoint(cfg, checkpoint)
        for name, bound in target['checkpoint_files'].items():
            require(Path(name).name == name, 'Target file escapes directory')
            base.checked_binding({'path': str(Path(target['model_path']) / name), **bound})
        require(all(row['checkpoint_identity_sha256'] == frozen['checkpoint_identity_sha256']
                    and row['checkpoint_id'] == 'F' for row in group['feedback']),
                'Original source responses are not released-F feedback')
        out.mkdir()
        self.group, self.config = group, cfg
        self.models, self.target, self.out, self.episode_id = models, target, out, group_id
        self.checkpoints = {PHYSICAL[k]: v for k, v in checkpoints.items()}
        self.source_seen, self.loss_seen, self.loss_phase = set(), set(), False
        write_json(out / 'NATIVE_CONFIG.json', cfg)
        self._ensure_worker('sd')
        return dict(status='passed', group_id=group_id, checkpoints=checkpoints,
            startup_s=self.worker.startup_s, optimizer_created=False, optimizer_loaded=False,
            optimizer_steps=0, warmup_generations=0,
            storage=dict(group_bytes=GROUP_BYTES, staging_bytes=STAGING_BYTES, reserve_bytes=reserve))

    def train_event(self, *args, **kwargs):
        raise ValueError('Read-only diagnostic forbids training')

    def serve_source(self, canonical_feedback_row):
        require(self.episode_id is not None and not self.loss_phase, 'Source replay must precede held-out loss phase')
        original = [r for r in self.group['feedback'] if r['request_id'] == canonical_feedback_row['request_id']]
        require(len(original) == 1 and original[0] == canonical_feedback_row, 'Source row differs from retained evidence')
        row = original[0]
        require(row['request_id'] not in self.source_seen, 'One natural S8 generation per source request')
        self.source_seen.add(row['request_id'])
        ordinal = self.group['feedback'].index(row) + 1
        result = super().serve(list(row['input_ids']), 'U2', False, turn_index=ordinal)
        trace = json.loads(base.checked_binding(result['trajectory']).read_text())
        rounds, initial = curve.normalize_rounds(trace, result, row['input_ids'])
        calls = result['verification_calls']
        result.update(input_ids=list(row['input_ids']), checkpoint_id='S8', request_id=row['request_id'],
            group_id=self.episode_id, domain=self.group.get('domain'), role=row['role'], diagnostic_role='source_replay',
            capture=False, feedback_capture_enabled=False,
            output_tokens=len(result['output_ids']), context_tokens=len(row['input_ids']),
            rounds=rounds, initial_output_tokens=initial,
            cap_hit=len(result['output_ids']) == 1024, eos_hit=result['output_ids'][-1] in STOP_IDS,
            committed_progress_per_call=result['verification_committed_tokens']/calls if calls else None,
            emitted_tokens_per_verification_call=len(result['output_ids'])/calls if calls else None,
            timing_scope='Fresh S8 diagnostic only; historical F versus fresh S8 is not a latency comparison.')
        return result

    def _close_serving(self):
        if self.worker is not None:
            worker = self.worker
            started = time.perf_counter()
            try:
                self.serving_shutdown = close_owned_worker(worker)
            finally:
                self.totals['shutdown_s'] += time.perf_counter() - started
            self.worker = None

    def evaluate_heldout(self, records, checkpoint_id):
        require(self.episode_id is not None and checkpoint_id in PHYSICAL, 'Only F/S8 held-out loss is allowed')
        require(checkpoint_id not in self.loss_seen, 'Each checkpoint evaluates four held-out responses once')
        expected = {r['request_id']: r for r in self.group['scores'] if r['checkpoint_id'] == 'F'}
        require(len(records) == len(expected) == 4 and {r['request_id'] for r in records} == set(expected),
                'Use exactly this group\'s four held-out canonical responses')
        for record in records:
            original = expected[record['request_id']]
            require(record['role'] == 'evaluation' and record['group_id'] == self.episode_id and
                    record['input_ids'] == original['input_ids'] and record['output_ids'] == original['output_ids'],
                    'Held-out input/output identity or evaluation role changed')
        self._close_serving()
        self.loss_phase = True
        if self.loss_worker is None:
            self.loss_worker = base._WorkerClient(self.config, 'train', self.out / 'read_only_loss_worker')
            self.totals['startup_s'] += self.loss_worker.startup_s
        self.loss_seen.add(checkpoint_id)
        try:
            return self.loss_worker.call('evaluate', {'records': records, 'checkpoint_id': checkpoint_id})
        except Exception as error:
            path = self.out / 'read_only_loss_worker' / ('HELDOUT_'+checkpoint_id+'_PARTIAL.json')
            if path.is_file():
                error.diagnostic_partial = json.loads(path.read_text())
                error.diagnostic_partial_binding = base.binding(path)
            raise

    def finish_group(self):
        errors, receipts = [], []
        for name in ('worker', 'loss_worker'):
            worker = getattr(self, name)
            if worker is None:
                continue
            started = time.perf_counter()
            try:
                receipts.append(dict(worker=name, **close_owned_worker(worker)))
            except Exception as error:
                errors.append(dict(worker=name, error=repr(error)))
            finally:
                self.totals['shutdown_s'] += time.perf_counter()-started
                setattr(self, name, None)
        receipt = super().finish_episode()
        receipt.update(optimizer_steps=0, optimizer_created=False, optimizer_loaded=False,
                       checkpoint_exports=0, warmup_generations=0, worker_shutdown=receipts,
                       worker_errors=errors)
        if hasattr(self, 'serving_shutdown'):
            receipt['source_serving_shutdown'] = self.serving_shutdown
        if errors:
            receipt['status'] = 'failed'
        if hasattr(self, 'out'):
            write_json(self.out / 'DIAGNOSTIC_FINISH.json', receipt)
        require(not errors, 'Diagnostic owned worker shutdown failed')
        return receipt


class ServingService(base._ServingService):
    def __init__(self, config):
        check_sources(config)
        super().__init__(config)
        write_json(self.out / 'DIAGNOSTIC_SERVING_CONTRACT.json', dict(output_cap=1024, block_size=10,
            context_limit=32768, prefill_chunk_size=8192, feedback_capture=False,
            cache_policy='Cold every request and checkpoint switch; original native publication transaction.',
            optimizer_steps=0, warmups=0))

    def publish(self, checkpoint):
        label = admit_checkpoint(self.config, checkpoint)
        self.bridge._two_update_admitted.add(Path(checkpoint['path']))
        started = time.perf_counter()
        receipt = self.bridge.publish({'model_path': checkpoint['path'], 'reset': True})
        require(receipt['status'] == 'passed' and receipt['success'] and
                receipt['helper_matches_current_layers'] and receipt['helper_norm_matches_current_layers'] and
                receipt['cache_after']['tree_total_size'] == 0 and
                self.bridge.state()['draft_model_path'] == checkpoint['path'], 'Native publication/cache invalidation failed')
        self.active = checkpoint
        write_json(self.out / 'ACTIVE_CHECKPOINT.json', checkpoint)
        return dict(status='passed', checkpoint_id=checkpoint['checkpoint_id'], logical_checkpoint_id=label,
                    activation_s=time.perf_counter()-started, native_receipt=receipt)

    def generate(self, body):
        require(body['capture'] is False and body['checkpoint_id'] == 'U2', 'Only capture-disabled S8 source generation')
        result = super().generate(body)
        require(result['feedback'] is None and result['capture_costs'] is None, 'Feedback capture was active')
        return result


class ReadOnlyLossService:
    """Load only teacher/embeddings/draft; never construct a trainer or optimizer."""
    def __init__(self, config):
        check_sources(config)
        self.config, self.out = config, Path(config['worker_out'])
        directory = Path(config['workspace_root']) / 'research/target_variant_specialization_001/training_tools'
        sys.path.insert(0, str(directory))
        import native as qualified_native
        require(Path(qualified_native.__file__).resolve() == directory / 'native.py', 'Qualified loader import collision')
        self.native = qualified_native
        self.runtime = qualified_native.load_runtime(17, 1024)
        require(Path(qualified_native.CORE.AUTHOR).resolve() == Path(config['author_root']).resolve(), 'Author tree changed')
        args = copy.copy(self.runtime.default_args)
        require(config['released']['path'] == str(qualified_native.CORE.DRAFT) and
                config['target']['model_path'] == str(qualified_native.CORE.TARGET), 'Pinned model pair changed')
        args.target_model_path = config['target']['model_path']
        args.initial_draft_path = args.draft_config_path = config['released']['path']
        args.max_length = args.sglang_context_length = 32768
        require(args.target_model_backend == 'sglang' and args.sglang_attention_backend == 'fa3', 'Teacher backend changed')
        self.teacher = self.runtime.setup.load_target_model(args)
        self.teacher_model = self.teacher.model_runner.model.eval().requires_grad_(False)
        self.components = self.runtime.setup.load_target_embeddings(args).eval().requires_grad_(False)
        self.draft, self.model_config, _, mask = self.runtime.setup.load_draft_stack(args, self.teacher)
        import torch
        require(mask == 128002 and self.draft.block_size == 10 and self.draft.target_layer_ids == [1,8,15,22,29],
                'Qualified draft architecture changed')
        parameters = list(self.draft.parameters())
        require(len(parameters) == 58 and sum(p.numel() for p in parameters) == 1048626432 and
                all(p.dtype == torch.bfloat16 for p in parameters), 'BF16 parameter inventory differs')
        require(float(getattr(self.model_config, 'attention_dropout', 0.)) == 0. and
                not any(getattr(m, 'gradient_checkpointing', False) for m in self.draft.modules()), 'Draft evaluation configuration differs')
        self.draft.eval().requires_grad_(False)
        qualified_native.CORE.verify_checkpoint(self.draft, config['released']['path'])
        self.active_checkpoint = 'F'
        self.completed = set()
        self.fixed_versions = self._fixed_versions()
        write_json(self.out / 'READ_ONLY_RUNTIME.json', dict(loader_args=vars(args), mode='eval/no_grad',
            optimizer_created=False, optimizer_loaded=False, optimizer_steps=0,
            worker_transport_kind='train selects teacher GPU only; RPC is evaluate-only',
            schedule='Four fixed held-out responses per F/S8 checkpoint; independent diagnostic anchor RNG.',
            author_sources=followup.check_followup_sources(config), diagnostic_source_sha256=config['diagnostic_source_sha256']))

    def _fixed_versions(self):
        return tuple((owner, name, id(p), p._version) for owner,module in
            (('teacher', self.teacher_model), ('head_embedding', self.components)) for name,p in module.named_parameters())

    def _assert_read_only(self):
        require(self.fixed_versions == self._fixed_versions(), 'Frozen target/head tensor changed')
        require(all(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters())
                    for m in (self.draft, self.teacher_model, self.components)), 'Read-only eval/frozen contract failed')

    def teacher_features(self, ids):
        import torch
        with torch.no_grad():
            result = self.teacher.generate_dflash_data(ids, torch.ones_like(ids), torch.zeros_like(ids, dtype=torch.float32))
        final = result.teacher_hidden_states
        require(final is not None and final.shape == (*ids.shape, self.components.lm_head.in_features) and
                result.hidden_states.shape[:2] == ids.shape and result.hidden_states.shape[-1] == 5 * final.shape[-1],
                'Native teacher features missing, shifted or shortened')
        return result.hidden_states.detach(), final.detach()

    def _load_checkpoint(self, label):
        started = time.perf_counter()
        checkpoint = self.config['diagnostic_checkpoints'][label]
        admit_checkpoint(self.config, checkpoint)
        if self.active_checkpoint != label:
            import torch
            from safetensors.torch import load_file
            files = [name for name in checkpoint['files'] if name.endswith('.safetensors')]
            require(len(files) == 1, 'Expected the pinned single-file DFlash checkpoint')
            state = load_file(str(Path(checkpoint['path']) / files[0]), device='cpu')
            with torch.no_grad():
                self.draft.load_state_dict(state, strict=True)
            del state
            self.native.CORE.verify_checkpoint(self.draft, checkpoint['path'])
            self.draft.eval().requires_grad_(False)
            self.active_checkpoint = label
        self._assert_read_only()
        return time.perf_counter()-started

    def evaluate(self, body):
        require(set(body) == {'records', 'checkpoint_id'} and body['checkpoint_id'] in PHYSICAL,
                'Only F/S8 held-out diagnostic evaluation is supported')
        label = body['checkpoint_id']
        require(label not in self.completed, 'Checkpoint loss evaluations cannot repeat')
        self.completed.add(label)
        loading_s = self._load_checkpoint(label)
        import torch
        from experiments.dflash_kl_native_diagnostic.loss import evaluate_responses
        torch.cuda.reset_peak_memory_stats()
        versions = tuple((name,id(p),p._version) for name,p in self.draft.named_parameters())
        started = time.perf_counter()
        result = None
        try:
            with torch.no_grad():
                result = evaluate_responses(self.draft, self.components.embed_tokens, self.components.lm_head,
                    self.teacher_features, body['records'], checkpoint_id=label,
                    checkpoint_identity_sha256=self.config['diagnostic_checkpoints'][label]['checkpoint_identity_sha256'])
            self._assert_read_only()
            require(versions == tuple((name,id(p),p._version) for name,p in self.draft.named_parameters()),
                    'Diagnostic evaluation mutated draft weights')
        except BaseException as error:
            from experiments.dflash_baseline_learning_curve.experiment import preserve_invalid_numbers
            partial = getattr(error, 'diagnostic_partial', dict(status='failed', checkpoint_id=label,
                response_losses=(result or {}).get('response_losses', []), error_type=type(error).__name__,
                error=str(error), optimizer_steps=0))
            partial.setdefault('costs', {}).update(checkpoint_loading_s=loading_s,
                loss_service_wall_s=time.perf_counter()-started,
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())
            partial = preserve_invalid_numbers(partial)
            error.diagnostic_partial = partial
            write_json(self.out / ('HELDOUT_'+label+'_PARTIAL.json'), partial)
            raise
        result.setdefault('costs', {}).update(checkpoint_loading_s=loading_s,
            loss_service_wall_s=time.perf_counter()-started,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(), optimizer_steps=0,
            scope='Read-only held-out diagnostic; preparation may be nested in loss-service wall time.')
        write_json(self.out / ('HELDOUT_'+label+'.json'), result)
        return result

    def close(self):
        self.runtime.destroy_distributed()


def worker_main():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import traceback
    config = base._worker_config()
    check_sources(config)
    base.check_dependencies(config['workspace_root'])
    allocation = os.environ.pop('CUDA_VISIBLE_DEVICES')
    try:
        base.check_environment(config)
    finally:
        os.environ['CUDA_VISIBLE_DEVICES'] = allocation
    require(config['kind'] in ('sd', 'train'), 'Only source-serving/read-only-loss workers are permitted')
    service = ReadOnlyLossService(config) if config['kind'] == 'train' else ServingService(config)
    poisoned = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        def log_message(self, *_):
            pass
        def do_POST(self):
            try:
                require(self.headers.get('X-Two-Update-Token') == config['rpc_token'], 'Foreign diagnostic client')
                length = int(self.headers.get('Content-Length', '0'))
                require(0 < length <= 64 * 2**20, 'Invalid RPC payload length')
                body, action = json.loads(self.rfile.read(length)), self.path.lstrip('/')
                require(not poisoned, 'Diagnostic worker poisoned; no automatic retry')
                if action == 'ready':
                    value = dict(status='passed', pid=os.getpid())
                else:
                    allowed = {'evaluate'} if config['kind'] == 'train' else {'generate','publish','flush'}
                    require(action in allowed, 'Diagnostic forbids this operation')
                    value = getattr(service, action)(body)
                status = 200
            except Exception as error:
                poisoned.append(repr(error))
                value = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
                if hasattr(error, 'diagnostic_partial'):
                    value['diagnostic_partial'] = error.diagnostic_partial
                write_json(Path(config['worker_out']) / ('FAILURE_%d.json' % len(poisoned)), value)
                status = 500
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)

    server = HTTPServer(('127.0.0.1', config['port']), Handler)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    require(sys.argv[1:] == ['--worker'], 'Use the reviewed diagnostic entry point')
    worker_main()
