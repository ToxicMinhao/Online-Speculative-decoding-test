"""Explicit child-only F/own-S8 natural generation on the pinned native runtime.

Import and construction are CPU-only. The private historical bridge supplies
normal generation, cold cache reset and helper refresh. Historical source replay
and training APIs/guards are untouched. This module has no trainer/loss/export
service. Native GPU compatibility remains unqualified until the bounded run.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap
import time

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
_spec = importlib.util.spec_from_file_location('_related_instance_qualified_diagnostic',
    HERE.parent / 'dflash_kl_native_diagnostic/native.py')
kl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kl
_spec.loader.exec_module(kl)
curve, base, followup = kl.curve, kl.base, kl.followup
require, sha, digest, write_json = base.require, base.sha, base.object_hash, base.write_json
from experiments.dflash_baseline_learning_curve.experiment import (
    digest as payload_digest, preserve_invalid_numbers)
from experiments.dflash_per_turn import framing
from experiments.dflash_related_instance_transfer.protocol import SERVING_CONTRACT
from experiments.dflash_related_instance_transfer.storage import (
    MIN_RESERVE_BYTES, GROUP_LOG_METADATA_BYTES, runtime_payload, preflight_paths, serialized_bytes)
PHYSICAL = {'F': 'F', 'S8': 'U2'}
STOP_IDS = kl.STOP_IDS
ORIGINAL_NATIVE_ROOT = kl.ORIGINAL_NATIVE_ROOT
close_owned_worker = kl.close_owned_worker


def source_pins():
    pins = kl.source_pins()
    # The evidence loader uses the qualified PR10 context loader from this
    # package; bind that actual dependency as well as native runtime sources.
    for path in sorted((HERE.parent / 'dflash_lower_lr_control').glob('*')):
        if path.is_file() and path.suffix in ('.py', '.json') and not path.name.startswith('test_'):
            pins[str(path.relative_to(HERE.parent))] = sha(path)
    for pattern in ('*.py', '*.json', 'requirements*.txt'):
        for path in sorted(HERE.glob(pattern)):
            if not path.name.startswith('test_'):
                pins[str(path.relative_to(HERE.parent))] = sha(path)
    return dict(sorted(pins.items()))


def check_sources(config):
    require(config.get('related_instance_source_sha256') == source_pins(), 'Reviewed related-instance source changed')
    require('diagnostic_source_sha256' not in config or config['diagnostic_source_sha256'] == kl.source_pins(),
            'Qualified diagnostic source changed')
    config['diagnostic_source_sha256'] = kl.source_pins()
    kl.check_sources(config)


def storage_preflight(config):
    require('storage_budget' in config and 'controller_root' in config, 'Bind actual artifact storage destinations')
    require(config['storage_budget']['runtime_payload_sha256'] == payload_digest(runtime_payload(config)),
            'Storage plan belongs to different runtime configuration')
    return preflight_paths(config['storage_budget'], config['output_root'], config['controller_root'],
        config.get('reserve_bytes', MIN_RESERVE_BYTES))


def worker_command(config):
    command = followup._original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Qualified worker transport changed')
    command[-1] = str(HERE / 'native.py')
    return command


# Only this privately loaded module changes transport. Public modules and files
# retain their historical service factories, source guards and admission logic.
base.worker_command = worker_command


def admit_checkpoint(config, checkpoint):
    label = next((name for name, bound in config['related_instance_checkpoints'].items()
                  if checkpoint == bound), None)
    require(label in PHYSICAL, 'Checkpoint is not exact bound F/own-S8')
    require(checkpoint['checkpoint_id'] == PHYSICAL[label], 'Physical checkpoint label differs')
    if label == 'F':
        require(checkpoint == config['released'], 'Released F checkpoint differs')
    else:
        require(Path(checkpoint['path']) == ORIGINAL_NATIVE_ROOT / config['episode_id'] / 'checkpoints/U2',
                'S8 belongs to a different original group or run')
    base.verify_checkpoint(checkpoint)
    return label


def _validate_children(group):
    require(group.get('group_payload_sha256') == payload_digest({k: v for k, v in group.items()
        if k != 'group_payload_sha256'}), 'Sealed group payload changed')
    children = group.get('children')
    require(isinstance(children, list) and len(children) == 2 and
            len({c['request_id'] for c in children}) == 2, 'Exactly two distinct children required')
    parents = {r['request_id'] for r in group['feedback']}
    old_ids = parents | {r['request_id'] for r in group['evaluation_requests']}
    for child in children:
        require(child.get('role') == 'related_instance' and child.get('group_id') == group['group_id'] and
                child.get('domain') == group['domain'] and child.get('parent_request_id') in parents and
                child.get('request_id') not in old_ids, 'New request role/group/parent/ID isolation failed')
        require(isinstance(child.get('text'), str) and child['text'] and
                child.get('input_ids_sha256') == digest(child.get('input_ids')), 'Child text/token binding changed')
        require(not any(k in child for k in ('history', 'messages', 'output_ids', 'answer', 'parent_answer')),
                'Child generation must contain only its new prompt, no answer/history')
        framing.check_input(child['input_ids'])
    return children


def _finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite native measurement')
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


class NativeBackend(base.NativeBackend):
    def __init__(self, config):
        super().__init__(config)
        self.initial_config = copy.deepcopy(config)
        self.attempted = []
        self.returned = []
        self.poisoned = False

    def start_group(self, original_group):
        require(self.episode_id is None, 'Close previous related-instance group first')
        self.config = copy.deepcopy(self.initial_config)
        storage = storage_preflight(self.config)
        check_sources(self.config)
        require(self.config.get('serving_contract') == SERVING_CONTRACT, 'Full pinned serving contract changed')
        base.check_environment(self.config)
        dependencies = base.check_dependencies(self.config['workspace_root'])
        require(self.config['runtime_source_sha256'] == dependencies['runtime_source_sha256'], 'Qualified runtime changed')
        require(self.config.get('output_cap', 1024) == 1024 and self.config.get('block_size', 10) == 10,
                'Keep block10 and 1024-or-EOS')
        group = copy.deepcopy(original_group)
        gid = group['group_id']
        require(Path(gid).name == gid and gid not in ('.', '..'), 'Invalid original group ID')
        require(group['status'] == 'complete' and len(group['feedback']) == 4, 'Complete original group required')
        children = _validate_children(group)
        require(self.config['storage_budget']['group_payload_sha256'].get(gid) == group['group_payload_sha256'],
                'Storage budget belongs to different child/evidence payload')
        models_path = base.checked_binding(self.config['model_bindings'])
        require(sha(models_path) == dependencies['qualified_model_bindings']['sha256'], 'Qualified model registry changed')
        models = json.loads(models_path.read_text())
        target = next(v for v in models['targets'] if v['target_id'] == 'T0')
        released = models['released_draft']
        frozen = dict(path=released['path'], files=released['checkpoint_files'], checkpoint_id='F',
            checkpoint_identity_sha256=digest(released['checkpoint_files']))
        require(group['checkpoints']['F'] == frozen, 'Original F identity differs from qualified registry')
        checkpoints = dict(F=frozen, S8=copy.deepcopy(group['checkpoints']['S8']))
        root = Path(self.config['output_root']).resolve()
        out = root / gid
        require(not out.exists(), 'Never overwrite a related-instance attempt')
        cfg = {**self.config, 'episode_id': gid, 'episode_out': str(out), 'target': target, 'released': frozen,
            'related_instance_checkpoints': checkpoints, 'original_group': group,
            'registry_path': str(models_path), 'registry_sha256': sha(models_path),
            'tokenizer_path': models.get('tokenizer_path', target['model_path']),
            'max_capture_bytes': self.config.get('max_capture_bytes', base.MAX_CAPTURE_BYTES),
            'effective_limits': dict(input_tokens=31744, total_context_tokens=32768, output_tokens=1024,
                effective_prompt_tokens=31735, proposal_reserve_tokens=9, proposal_block_tokens=10,
                chunked_prefill_tokens=8192, capture_memory_bound_bytes=base.capture_memory_bound()),
            'public_source_sha256': {name: sha(base.HERE / name)
                for name in ('native.py', 'training.py', 'native_dependencies.json')}}
        for checkpoint in checkpoints.values():
            admit_checkpoint(cfg, checkpoint)
        for name, bound in target['checkpoint_files'].items():
            require(Path(name).name == name, 'Target filename escapes directory')
            base.checked_binding(dict(bound, path=str(Path(target['model_path']) / name)))
        # Re-render exactly one user turn from the qualified, hash-checked
        # tokenizer. Parent text/answer never enters this encoding call.
        tokenizer = framing.load_pinned_tokenizer(cfg['tokenizer_path'])
        for child in children:
            require(framing.encode_history(tokenizer, [{'role': 'user', 'content': child['text']}]) == child['input_ids'],
                    'Child tokens are not its standalone qualified user request')
        out.mkdir(parents=True)
        self.config, self.group, self.models, self.target = cfg, group, models, target
        self.out, self.episode_id = out, gid
        self.checkpoints = {PHYSICAL[k]: v for k, v in checkpoints.items()}
        self.attempted, self.returned, self.poisoned = [], [], False
        self.totals = {'startup_s': 0., 'shutdown_s': 0., 'activation_s': 0., 'native_activation_s': 0.}
        write_json(out / 'NATIVE_CONFIG.json', cfg)
        write_json(out / 'STORAGE_PREFLIGHT.json', storage)
        # Start lazily on first intended request; there are no explicit
        # generation warmups. Normal runtime initialization remains charged.
        return dict(status='passed', group_id=gid, checkpoints=checkpoints, optimizer_created=False,
            optimizer_loaded=False, optimizer_steps=0, response_kl_evaluations=0, checkpoint_exports=0,
            warmup_generations=0, model_worker_started=False, storage=storage,
            initialization_policy='Qualified Engine initialization is unchanged; no explicit generation warmups.')

    def _ensure_worker(self, kind):
        require(kind == 'sd' and self.trainer is None, 'Only native speculative serving worker allowed')
        return super()._ensure_worker(kind)

    def serve(self, *args, **kwargs):
        raise ValueError('Use explicit sealed new-request serve_request; historical replay is forbidden')

    def serve_request(self, child, label):
        require(self.episode_id is not None and not self.poisoned, 'Active unpoisoned group required')
        children = self.group['children']
        schedule = [(arm, row) for arm in ('F', 'S8') for row in children]
        ordinal = len(self.attempted)
        require(ordinal < len(schedule) and (label, child) == schedule[ordinal],
                'Follow sealed F child0/child1, own S8 child0/child1 order; no repeats')
        _validate_children(self.group)
        require(self.trainer is None, 'Read-only backend cannot own a trainer')
        # Register before RPC so exceptions cannot permit a retry.
        self.attempted.append(dict(checkpoint_id=label, request_id=child['request_id']))
        result = None
        raw_path = self.out / ('RAW_NATIVE_%02d.json' % (ordinal + 1))
        try:
            result = base.NativeBackend.serve(self, list(child['input_ids']), PHYSICAL[label], False,
                turn_index=ordinal + 1)
            self.returned.append(dict(self.attempted[-1]))
            # Preserve before all finite/count/trace/identity validation.
            from experiments.dflash_baseline_learning_curve.experiment import preserve_invalid_numbers
            raw_path = self.out / ('RAW_NATIVE_%02d.json' % (ordinal + 1))
            write_json(raw_path, preserve_invalid_numbers(result))
            _finite(result)
            require(result.get('input_ids_sha256') == child['input_ids_sha256'], 'Native child input token hash differs')
            require(result['checkpoint_identity_sha256'] == self.group['checkpoints'][label]['checkpoint_identity_sha256'],
                    'Serving checkpoint differs from sealed own-group identity')
            require(result.get('feedback') is None and result.get('capture_costs') is None,
                    'Feedback capture must remain disabled')
            require(serialized_bytes(result) <= self.config['storage_budget']['request_artifact_bound_bytes'][child['request_id']],
                    'Request receipt exceeded storage envelope')
            trace_path = base.checked_binding(result['trajectory'])
            require(trace_path.resolve().is_relative_to(self.out.resolve()), 'Native trace escapes this group')
            trace = json.loads(trace_path.read_text())
            require(trace_path.stat().st_size <= self.config['storage_budget']['request_artifact_bound_bytes'][child['request_id']],
                    'Request trace exceeded storage envelope')
            rounds, initial = curve.normalize_rounds(trace, result, child['input_ids'])
            calls = result['verification_calls']
            result.update(input_ids=list(child['input_ids']), input_ids_sha256=child['input_ids_sha256'],
                checkpoint_id=label, physical_checkpoint_id=PHYSICAL[label], request_id=child['request_id'],
                parent_request_id=child['parent_request_id'], group_id=self.episode_id, domain=self.group['domain'],
                role='related_instance', population='related_instance', serving_contract=dict(SERVING_CONTRACT),
                capture=False, feedback_capture_enabled=False, output_tokens=len(result['output_ids']),
                context_tokens=len(child['input_ids']), rounds=rounds, initial_output_tokens=initial,
                cap_hit=len(result['output_ids']) == 1024, eos_hit=result['output_ids'][-1] in STOP_IDS,
                committed_progress_per_call=result['verification_committed_tokens'] / calls if calls else None,
                emitted_tokens_per_verification_call=len(result['output_ids']) / calls if calls else None,
                raw_native_receipt=base.binding(raw_path),
                prefill_accounting=dict(input_tokens=len(child['input_ids']), initial_emitted_tokens=initial,
                    isolated_duration_s=None, duration_available=False,
                    timing_scope='Prefill is included in native_generation_s and request_s; qualified runtime exposes no standalone prefill duration.'),
                initialization_policy='Normal qualified runtime initialization charged separately; zero explicit generation warmups.',
                timing_scope='Fresh single fixed-order natural request; not evidence of serving speedup.')
            self._check_artifact_capacity()
            return result
        except BaseException as error:
            self.poisoned = True
            if result is not None:
                error.native_partial = dict(raw_receipt=result, stage='returned_native_validation',
                    generation_returned=True, request_id=child['request_id'], checkpoint_id=label,
                    raw_binding=base.binding(raw_path) if raw_path.is_file() else None)
            elif self.worker is not None:
                # A worker can persist the model reply, then fail its own
                # reconciliation before RPC returns. Preserve that known
                # returned-invalid work instead of calling it unknown.
                try:
                    worker_out = Path(self.worker.out)
                    response_path = worker_out / ('RAW_RESPONSE_%06d.json' % (ordinal + 1))
                    declaration_path = worker_out / 'FEATURE_PENDING.json'
                    if response_path.is_file() and declaration_path.is_file():
                        declaration = json.loads(declaration_path.read_text())
                        require(declaration.get('record_id') == child['request_id'] and
                                declaration.get('draft_version') == PHYSICAL[label] and
                                declaration.get('capture') is False, 'Partial native response identity differs')
                        returned = json.loads(response_path.read_text())
                        self.returned.append(dict(self.attempted[-1]))
                        error.native_partial = dict(raw_receipt=returned, stage='worker_native_validation',
                            generation_returned=True, request_id=child['request_id'], checkpoint_id=label,
                            raw_binding=base.binding(response_path), declaration_binding=base.binding(declaration_path))
                except Exception as recovery_error:
                    # Malformed or unbound partial files cannot prove return.
                    error.native_partial_recovery_error = repr(recovery_error)
            raise

    def _check_artifact_capacity(self):
        budget = self.config['storage_budget']
        require(shutil.disk_usage(self.out).free >= self.config.get('reserve_bytes', MIN_RESERVE_BYTES),
                'Related-instance writes crossed storage reserve')
        logs = list(self.out.rglob('*.log'))
        require(sum(p.stat().st_size for p in logs) <= GROUP_LOG_METADATA_BYTES,
                'Native logs exceeded the declared group storage allowance')
        used = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        gid = self.episode_id
        bound = (budget['native_config_copies'] * budget['serialized_group_config_bytes'][gid] +
            2 * budget['native_result_copies'] * sum(budget['request_artifact_bound_bytes'][c['request_id']]
                for c in self.group['children']) + GROUP_LOG_METADATA_BYTES)
        require(used <= bound, 'Native group artifacts exceeded conservative write envelope')
        return dict(used_bytes=used, bound_bytes=bound)

    def train_event(self, *args, **kwargs):
        raise ValueError('Related-instance transfer forbids training')

    def start_episode(self, *args, **kwargs):
        raise ValueError('Use only the explicit related-instance group path')

    def finish_group(self):
        errors, shutdown = [], []
        for name in ('worker', 'trainer'):
            worker = getattr(self, name)
            if worker is None:
                continue
            try:
                receipt = close_owned_worker(worker)
                shutdown.append(dict(worker=name, **receipt))
                self.totals['shutdown_s'] += receipt['shutdown_s']
                if name == 'trainer':
                    errors.append('Unexpected trainer existed in read-only backend')
            except Exception as error:
                errors.append(dict(worker=name, error=repr(error)))
            finally:
                setattr(self, name, None)
        unchanged = False
        if self.episode_id is not None:
            try:
                for checkpoint in self.config['related_instance_checkpoints'].values():
                    admit_checkpoint(self.config, checkpoint)
                for name, bound in self.target['checkpoint_files'].items():
                    base.checked_binding(dict(bound, path=str(Path(self.target['model_path']) / name)))
                require(sha(self.config['registry_path']) == self.config['registry_sha256'], 'Model registry changed')
                for name, expected in framing.TOKENIZER_SHA256.items():
                    require(sha(Path(self.config['tokenizer_path']) / name) == expected, 'Tokenizer file changed')
                unchanged = True
            except Exception as error:
                errors.append(dict(check='checkpoint/model/tokenizer_mutation', error=repr(error)))
        result = dict(status='failed' if errors else 'passed', group_id=self.episode_id, **self.totals,
            checkpoint_weights_unchanged=unchanged, training_quiescent=self.trainer is None,
            optimizer_created=False, optimizer_loaded=False, optimizer_steps=0, training_presentations=0,
            response_kl_evaluations=0, checkpoint_exports=0, warmup_generations=0,
            native_attempted=len(self.attempted), native_returned=len(self.returned),
            native_unknown=len(self.attempted) - len(self.returned), native_unattempted=4 - len(self.attempted),
            worker_shutdown=shutdown, errors=errors)
        if self.episode_id is not None:
            result['free_after_bytes'] = shutil.disk_usage(self.out).free
            write_json(self.out / 'NATIVE_FINISH.json', result)
        self.episode_id = None
        if errors:
            error = ValueError('Related-instance checkpoint mutation or owned-worker cleanup failed')
            error.native_finish = result
            raise error
        return result


def _new_request_generate_method():
    """Change identity/raw persistence only; preserve generation/reconciliation."""
    original = base._ServingService.generate
    source = textwrap.dedent(inspect.getsource(original))
    replacements = {
        "rid = self.config['episode_id'] + ':t' + str(body['turn']).zfill(2)":
            "rid = self.current_child['request_id']",
        "'turn': body['turn'], 'domain': 'code_math', 'role': ROLE, 'target_id': 'T0'":
            "'turn': body['turn'], 'domain': self.current_child['domain'], 'role': 'related_instance', 'target_id': 'T0'",
        "write_json(self.out / ('RAW_RESPONSE_%06d.json' % self.serial), value)":
            "write_json(self.out / ('RAW_RESPONSE_%06d.json' % self.serial), _related_preserve_invalid_numbers(value))",
    }
    for before, after in replacements.items():
        require(source.count(before) == 1, 'Qualified new-request trace declaration changed')
        source = source.replace(before, after)
    scope = {}
    namespace = dict(base.__dict__, _related_preserve_invalid_numbers=preserve_invalid_numbers)
    exec(compile(source, original.__code__.co_filename, 'exec'), namespace, scope)
    return scope['generate']


class ServingService(base._ServingService):
    native_generate = _new_request_generate_method()

    def __init__(self, config):
        check_sources(config)
        require(config['kind'] == 'sd' and config.get('serving_contract') == SERVING_CONTRACT,
                'Only pinned speculative serving is allowed')
        _validate_children(config['original_group'])
        self.attempted_children = 0
        self.poisoned = False
        super().__init__(config)
        write_json(self.out / 'RELATED_INSTANCE_SERVING_CONTRACT.json', dict(SERVING_CONTRACT,
            output_dtype='bfloat16', normal_runtime_initialization_unchanged=True,
            explicit_generation_warmups=0, response_kl_evaluations=0, optimizer_steps=0,
            checkpoint_exports=0, tokenizer_path=config['tokenizer_path'],
            runtime_source_sha256=config['runtime_source_sha256'], native_gpu_qualification='pending'))

    def publish(self, checkpoint):
        label = admit_checkpoint(self.config, checkpoint)
        self.bridge._two_update_admitted.add(Path(checkpoint['path']))
        started = time.perf_counter()
        receipt = self.bridge.publish({'model_path': checkpoint['path'], 'reset': True})
        require(receipt['status'] == 'passed' and receipt['success'] and receipt['helper_matches_current_layers'] and
                receipt['helper_norm_matches_current_layers'] and receipt['cache_after']['tree_total_size'] == 0 and
                self.bridge.state()['draft_model_path'] == checkpoint['path'], 'Native publication/cache/helper refresh failed')
        self.active = checkpoint
        write_json(self.out / 'ACTIVE_CHECKPOINT.json', checkpoint)
        return dict(status='passed', checkpoint_id=checkpoint['checkpoint_id'], logical_checkpoint_id=label,
            activation_s=time.perf_counter() - started, native_receipt=receipt)

    def generate(self, body):
        require(not self.poisoned and body.get('capture') is False and body.get('repeat') is None,
                'Only one capture-disabled natural generation per sealed child/arm')
        schedule = [(PHYSICAL[label], child) for label in ('F', 'S8')
                    for child in self.config['original_group']['children']]
        require(self.attempted_children < len(schedule), 'No extra generation or warmup allowed')
        physical, child = schedule[self.attempted_children]
        require(body.get('checkpoint_id') == physical and body.get('turn') == self.attempted_children + 1 and
                body.get('input_ids') == child['input_ids'], 'Worker input/order differs from sealed new request')
        require(self.active == self.config['related_instance_checkpoints']['F' if physical == 'F' else 'S8'],
                'Loaded native checkpoint identity differs')
        self.current_child = child
        self.attempted_children += 1
        try:
            result = self.native_generate(body)
            _finite(result)
            require(result['feedback'] is None and result['capture_costs'] is None, 'Feedback capture was active')
            return result
        except BaseException:
            self.poisoned = True
            raise


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
    require(config['kind'] == 'sd', 'No trainer/loss/AR role in related-instance worker')
    service = ServingService(config)
    poisoned = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        def log_message(self, *_):
            pass
        def do_POST(self):
            try:
                require(self.headers.get('X-Two-Update-Token') == config['rpc_token'], 'Foreign native client')
                length = int(self.headers.get('Content-Length', '0'))
                require(0 < length <= 64 * 2**20, 'Invalid native RPC payload length')
                body, action = json.loads(self.rfile.read(length)), self.path.lstrip('/')
                require(not poisoned, 'Worker poisoned; no automatic retries')
                if action == 'ready':
                    value = dict(status='passed', pid=os.getpid())
                else:
                    require(action in ('generate', 'publish', 'flush'), 'Read-only new-request operation required')
                    value = getattr(service, action)(body)
                status = 200
            except Exception as error:
                poisoned.append(repr(error))
                value = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
                write_json(Path(config['worker_out']) / ('FAILURE_%d.json' % len(poisoned)), value)
                status = 500
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
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
    require(sys.argv[1:] == ['--worker'], 'Use explicit related-instance controller entry point')
    worker_main()
