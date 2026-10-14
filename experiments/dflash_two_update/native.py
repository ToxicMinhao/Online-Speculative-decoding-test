"""Explicit bridge to the installed, hash-pinned W8/D202 native dependencies.

Importing this module does not load a model or start a worker. The public tree
does not vendor the private qualified runtime. ``native_dependencies.json``
lists the exact source installation required by this bridge. New integration
behavior (U1 feature capture, continuous optimizer, and cold publication of a
learned checkpoint) requires GPU qualification; CPU tests are not that proof.

NativeBackend accepts an explicit local configuration, never an SSH address.
SD and target-only AR occupy the serving GPU sequentially. A separate training
worker retains the same draft/Adam objects through both updates. Full request
inputs are saved by the controller and sent as IDs; assistant history uses the
emitted IDs without a decode/encode round trip.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
OUTPUT_CAP, INPUT_LIMIT, CONTEXT_LIMIT, BLOCK_SIZE = 512, 16384, 17408, 10
USER_HEADER = '<|start_header_id|>user<|end_header_id|>\n\n'
ASSISTANT_HEADER = '<|start_header_id|>assistant<|end_header_id|>\n\n'
EOT = '<|eot_id|>'
ROLE = 'TWO_UPDATE_SUPPORT'
REQUIRED_RUNTIME = {
    'speculative/dflash_worker.py', 'speculative/dflash_info.py',
    'layers/logits_processor.py', 'model_executor/cuda_graph_runner.py',
    'models/llama.py', 'model_executor/forward_batch_info.py',
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(4 * 2**20), b''):
            result.update(value)
    return result.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def binding(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked_binding(value):
    path = Path(value['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'Ordinary absolute bound file required')
    require(sha(path) == value['sha256'] and
            ('bytes' not in value or path.stat().st_size == value['bytes']), 'Bound artifact changed')
    return path


def check_dependencies(workspace_root):
    root = Path(workspace_root).resolve()
    manifest = json.loads((HERE / 'native_dependencies.json').read_text())
    for relative, expected in manifest['source_sha256'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Dependency path escapes workspace')
        actual = root / path
        require(actual.is_file() and actual.resolve().is_relative_to(root) and sha(actual) == expected,
                'Qualified dependency missing or changed: ' + relative)
    return manifest


def check_input(ids):
    require(isinstance(ids, list) and 0 < len(ids) <= INPUT_LIMIT and
            all(type(x) is int and 0 <= x < 128256 for x in ids),
            'Exact complete input must fit 16384 tokens; truncation is prohibited')
    require(len(ids) + OUTPUT_CAP <= CONTEXT_LIMIT, 'Complete history exceeds native context')


def encode_history(tokenizer, messages):
    """Original answer IDs plus checked Llama framing; preserve prompt whitespace."""
    require(isinstance(messages, list) and messages, 'Conversation is empty')
    cursor = 1 if messages[0]['role'] == 'system' else 0
    require(cursor < len(messages) and messages[cursor]['role'] == 'user', 'First request must be user')
    require((len(messages) - cursor) % 2 == 1, 'Conversation must end in a new user request')
    eot = tokenizer.encode(EOT, add_special_tokens=False)
    require(len(eot) == 1, 'Qualified EOT framing changed')
    first = messages[:cursor + 1]
    ids = list(tokenizer.apply_chat_template(
        [{'role': m['role'], 'content': m['content']} for m in first],
        tokenize=True, add_generation_prompt=True, return_dict=False))
    first_segment = tokenizer.encode(USER_HEADER + messages[cursor]['content'] + EOT + ASSISTANT_HEADER,
                                     add_special_tokens=False)
    require(ids[-len(first_segment):] == first_segment,
            'Tokenizer template strips/changes original prompt text; no silent rewrite')
    for index in range(cursor + 1, len(messages), 2):
        answer, question = messages[index:index + 2]
        require(answer['role'] == 'assistant' and question['role'] == 'user', 'Conversation roles differ')
        output = answer.get('output_ids')
        require(isinstance(output, list) and output and
                all(type(x) is int and 0 <= x < 128256 for x in output),
                'Original assistant output_ids required; retokenized answers are forbidden')
        ids.extend(output)
        if output[-1] != eot[0]:
            ids.extend(eot)
        user = question['content']
        segment = tokenizer.encode(USER_HEADER + user + EOT + ASSISTANT_HEADER, add_special_tokens=False)
        # Guard framing per exact prompt, including leading/trailing whitespace.
        template = tokenizer.apply_chat_template([{'role': 'user', 'content': user}],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        require(list(template)[-len(segment):] == segment,
                'Tokenizer template strips/changes original prompt text; no silent rewrite')
        ids.extend(segment)
    check_input(ids)
    return ids


def _load(path, name, replacements=()):
    """Exact source transformations in isolated child processes; originals untouched."""
    source = Path(path).read_text()
    edits = []
    for before, after in replacements:
        require(source.count(before) == 1, 'Compatibility edit no longer matches exactly: ' + before)
        source = source.replace(before, after)
        edits.append({'before': before, 'after': after})
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    module.two_update_compatibility = {'source_sha256': sha(path),
        'effective_sha256': hashlib.sha256(source.encode()).hexdigest(), 'edits': edits}
    return module


def capture_modules(root):
    """Retain every native chronology/eligibility check, admit active draft metadata."""
    directory = Path(root) / 'research/fast_w8_001/data/collector'
    records = _load(directory / 'records.py', 'records', [
        ("ALLOWED_ROLES = {'Q', 'FAST_W8_SUPPORT'}", "ALLOWED_ROLES = {'TWO_UPDATE_SUPPORT'}"),
        ("document['draft_version'] == 'released_F'", "document['draft_version'] in ('F', 'U1', 'U2')"),
    ])
    original_validate = records.validate

    def validate(document, allowed_roles=records.ALLOWED_ROLES):
        original_validate(document, allowed_roles)
        require(document['draft_version'] == ('F' if 9 <= document['turn'] <= 16 else 'U1') and
                9 <= document['turn'] <= 24, 'Support must come from the chronologically serving draft')
        require(document['capture_checkpoint_sha256'] == document['draft_identity_sha256'] and
                document['input_token_count'] == len(document['input_ids']) and
                document['record_id'] == document['episode_id'] + ':t' + str(document['turn']).zfill(2),
                'Response ownership/capture identity differs')
        for block in document['blocks']:
            expected = ownership_fields(document, block)
            require(all(block.get(key) == value for key, value in expected.items()),
                    'Saved native response ownership metadata differs')
        return document

    records.validate = validate
    cache = _load(directory / 'cache.py', 'cache', [
        ("declaration['role'] in ('Q', 'FAST_W8_SUPPORT')", "declaration['role'] == 'TWO_UPDATE_SUPPORT'"),
        ("draft_version='released_F'", "draft_version=self.declaration['draft_version']"),
        ('        validate(doc)', '        add_response_ownership(doc)\n        validate(doc)'),
    ])
    cache.add_response_ownership = add_response_ownership
    return records, cache


def ownership_fields(response, block):
    owned = [slot for slot, eligible in enumerate(block['eligible_mask']) if eligible]
    require(all(block['positions'][slot] >= response['input_token_count'] for slot in owned),
            'Supervision leaked into earlier history/user prefix')
    return {'response_record_id': response['record_id'], 'response_turn': response['turn'],
        'input_token_count': response['input_token_count'],
        'capture_checkpoint_sha256': response['capture_checkpoint_sha256'],
        'target_id': response['target_id'], 'target_identity_sha256': response['target_identity_sha256'],
        'output_cap': response['output_cap'],
        'slot_owner_record_ids': [response['record_id'] if slot in owned else None for slot in range(BLOCK_SIZE)],
        'owned_eligible_slots': owned}


def add_response_ownership(response):
    for block in response['blocks']:
        block.update(ownership_fields(response, block))


class ReplayRecords:
    """Add explicit current-response ownership to unchanged W8 tensor iteration."""
    def __init__(self, records):
        self.records = records

    def iter_blocks(self, response):
        for block in self.records.iter_blocks(response):
            expected = ownership_fields(response, block)
            require(all(block[key] == value for key, value in expected.items()), 'Replay ownership changed')
            yield block


def verify_checkpoint(checkpoint):
    path = Path(checkpoint['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_dir(), 'Checkpoint must be an ordinary local directory')
    files = checkpoint['files']
    require('config.json' in files and any(n.endswith('.safetensors') for n in files), 'Incomplete native checkpoint')
    require(object_hash(files) == checkpoint['checkpoint_identity_sha256'], 'Checkpoint identity differs')
    for name, item in files.items():
        require(Path(name).name == name, 'Checkpoint filename escapes directory')
        actual = path / name
        require(actual.is_file() and actual.stat().st_size == item['bytes'] and sha(actual) == item['sha256'],
                'Checkpoint bytes changed: ' + name)
    return checkpoint


def _check_gpu_idle(device):
    uuid = subprocess.check_output(['nvidia-smi', '-i', device, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True)
    require(not any(line.split(',')[0].strip() == uuid for line in active.splitlines() if line.strip()),
            'Assigned GPU already has active work')
    return {'gpu_uuid': uuid, 'active_jobs_checked': True}


def worker_command(config):
    """Source the approved helper in every worker shell, without shell interpolation."""
    return ['bash', '-c', 'set -euo pipefail\nsource "$1"\nsource "$2"\n'
            'conda activate "$3"\nsource "$1"\ncd "$4"\nexec "$5" -u -B "$6" --worker',
            'two-update-worker', str(Path(config['workspace_root']) / 'setup/a800/a800_env.sh'),
            config['conda_bootstrap'], config['conda_prefix'], config['workspace_root'],
            config['python_executable'], str(Path(__file__).resolve())]


def check_environment(config):
    require(os.environ.get('HF_ENDPOINT') == 'https://hf-mirror.com', 'Source approved environment helper before imports')
    root = Path(os.environ['A800_STORAGE_ROOT'])
    project = Path(os.environ['A800_PROJECT_ROOT'])
    require(root.is_absolute() and root.resolve() == root and project == Path(config['workspace_root']).resolve(),
            'Supplied workspace differs from approved helper installation')
    for variable in ('XDG_CACHE_HOME', 'HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'HF_MODULES_CACHE',
                     'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                     'CUDA_CACHE_PATH', 'SGLANG_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE', 'TVM_FFI_CACHE_DIR',
                     'TMPDIR', 'TMP', 'TEMP'):
        require(Path(os.environ[variable]).resolve().is_relative_to(root), 'Cache/temp escapes approved root: ' + variable)
    require(Path(config['output_root']).resolve().is_relative_to(root), 'Runtime output escapes approved storage')
    prefix = Path(config['conda_prefix']).resolve()
    require(prefix.is_relative_to(Path(os.environ['CONDA_ENVS_PATH']).resolve()) and
            Path(config['python_executable']).resolve().is_relative_to(prefix), 'Select the explicit qualified Conda Python')
    require(Path(config['conda_bootstrap']).is_file(), 'Existing Conda activation script required')
    requested = {config['serving_cuda_visible_devices'], config['training_cuda_visible_devices']}
    require(requested == {'0', '1'}, 'Explicit separate qualified A800 devices required')
    assigned = os.environ.get('CUDA_VISIBLE_DEVICES')
    require(not assigned or requested <= set(assigned.split(',')), 'Requested GPUs escape inherited allocation')
    require(not {'HOME', 'home', 'CODEX_HOME', 'CUDA_VISIBLE_DEVICES'} & set(config.get('worker_env', {})),
            'Worker overrides may not redirect home/storage or assigned devices')


def episode_storage_bound(model_bytes):
    # Retain sixteen feature records, two full exports and both Adam moment
    # states. No rolling deletion is assumed. Reserve is additional to this peak.
    response = (INPUT_LIMIT + OUTPUT_CAP - 1) * 20480 * 2 + (OUTPUT_CAP - 1) * 10 * 4096 * 2
    return 16 * response + 6 * model_bytes + 512 * 2**20


class _WorkerClient:
    def __init__(self, config, kind, out):
        self.config, self.kind, self.out = config, kind, Path(out)
        self.process = self.log = None
        self.out.mkdir(parents=True, exist_ok=False)
        gpu = config['training_cuda_visible_devices' if kind == 'train' else 'serving_cuda_visible_devices']
        occupancy = _check_gpu_idle(gpu)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        self.url = 'http://127.0.0.1:' + str(port)
        self.token = os.urandom(24).hex()
        worker_config = {**config, 'kind': kind, 'worker_out': str(self.out), 'port': port,
                         'rpc_token': self.token, 'gpu_preflight': occupancy}
        path = self.out / 'WORKER_CONFIG.json'; write_json(path, worker_config)
        env = {**os.environ, **config.get('worker_env', {}), 'CUDA_VISIBLE_DEVICES': gpu,
            'TWO_UPDATE_WORKER_CONFIG': str(path), 'TWO_UPDATE_WORKER_CONFIG_SHA256': sha(path),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'SGLANG_ENABLE_SPEC_V2': '0',
            'SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN': '1', 'PYTHONUNBUFFERED': '1'}
        if kind == 'train':
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); master_port = sock.getsockname()[1]
            env.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(master_port), RANK='0',
                       WORLD_SIZE='1', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1')
        # Private installation environment/toolchain settings remain user supplied.
        command = worker_command(config)
        self.log = (self.out / 'worker.log').open('xb')
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            deadline = time.monotonic() + config.get('startup_timeout_s', 1800)
            while time.monotonic() < deadline:
                require(self.process.poll() is None, 'Native worker exited during startup; see ' + str(self.out / 'worker.log'))
                try:
                    ready = self.call('ready', {}, timeout=2)
                    require(ready['pid'] == self.process.pid, 'Foreign listener; worker PID mismatch')
                    self.startup_s = time.perf_counter() - started
                    return
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(.25)
            raise TimeoutError('Native startup deadline exceeded')
        except BaseException:
            self.close()
            raise

    def call(self, action, value, timeout=7200):
        request = urllib.request.Request(self.url + '/' + action,
            data=json.dumps(value, allow_nan=False).encode(),
            headers={'Content-Type': 'application/json', 'X-Two-Update-Token': self.token}, method='POST')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors='replace')
            raise RuntimeError('Native worker failed; raw failure retained: ' + detail) from error
        require(result.get('status') == 'passed', 'Native worker returned failed response')
        return result

    def close(self):
        if self.process is not None:
            # Child schedulers may remain after the bridge parent has crashed.
            # The process group was created exclusively by this client.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        if self.log is not None:
            self.log.close()


class NativeBackend:
    """Controller interface. Construction is metadata-only; start_episode runs native code."""
    def __init__(self, config):
        self.config = dict(config)
        self.worker = self.trainer = None
        self.episode_id = None
        self.serial = 0
        self.totals = {'startup_s': 0., 'shutdown_s': 0., 'activation_s': 0., 'native_activation_s': 0.}

    def start_episode(self, episode_id, config=None):
        require(self.episode_id is None, 'Finish prior episode before resetting draft/optimizer')
        if config:
            self.config.update(config)
        cfg = self.config
        check_environment(cfg)
        dependencies = check_dependencies(cfg['workspace_root'])
        require(cfg['serving_cuda_visible_devices'] != cfg['training_cuda_visible_devices'],
                'Resident trainer requires a separate assigned GPU')
        require(cfg.get('output_cap', OUTPUT_CAP) == OUTPUT_CAP and cfg.get('block_size', BLOCK_SIZE) == BLOCK_SIZE,
                'Qualified decoding geometry is fixed')
        require(cfg['runtime_source_sha256'] == dependencies['runtime_source_sha256'] and
                REQUIRED_RUNTIME <= set(cfg['runtime_source_sha256']), 'Exact qualified collector runtime pins required')
        reserve = cfg.get('reserve_bytes', 8 * 2**30)
        require(reserve >= 8 * 2**30 and cfg['episode_storage_budget_bytes'] > 0, 'Explicit complete peak storage budget required')
        out = Path(cfg['output_root']).resolve() / episode_id
        require(Path(episode_id).name == episode_id and episode_id not in ('.', '..'), 'Invalid episode directory name')
        out.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(out.parent).free >= reserve + cfg['episode_storage_budget_bytes'],
                'Complete episode peak writes plus reserve exceed free space')
        out.mkdir(exist_ok=False)
        models_path = checked_binding(cfg['model_bindings'])
        require(sha(models_path) == dependencies['qualified_model_bindings']['sha256'],
                'Use the exact qualified model-binding artifact')
        models = json.loads(models_path.read_text())
        target = next(value for value in models['targets'] if value['target_id'] == 'T0')
        released = models['released_draft']
        model_bytes = sum(value['bytes'] for name, value in released['checkpoint_files'].items() if name.endswith('.safetensors'))
        require(cfg['episode_storage_budget_bytes'] >= episode_storage_bound(model_bytes),
                'Peak allowance must retain 16 complete capture records, U1/U2 and both optimizer states')
        frozen = {'path': released['path'], 'files': released['checkpoint_files'],
                  'checkpoint_identity_sha256': object_hash(released['checkpoint_files']), 'checkpoint_id': 'F'}
        verify_checkpoint(frozen)
        for name, item in target['checkpoint_files'].items():
            require(Path(name).name == name, 'Target filename escapes directory')
            checked_binding({'path': str(Path(target['model_path']) / name), **item})
        self.models, self.target, self.checkpoints = models, target, {'F': frozen}
        self.out, self.episode_id = out, episode_id
        self.config = {**cfg, 'episode_id': episode_id, 'episode_out': str(out), 'target': target,
            'released': frozen, 'registry_path': str(models_path), 'registry_sha256': sha(models_path),
            'tokenizer_path': models.get('tokenizer_path', target['model_path']),
            'public_source_sha256': {name: sha(HERE / name) for name in ('native.py', 'training.py', 'native_dependencies.json')}}
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_path'], local_files_only=True)
        require(len(self.tokenizer) == 128256, 'Qualified tokenizer vocabulary changed')
        write_json(out / 'NATIVE_CONFIG.json', self.config)
        self._ensure_worker('sd')
        return {'status': 'passed', 'episode_id': episode_id, 'checkpoint_id': 'F',
                'startup_s': self.worker.startup_s}

    def _ensure_worker(self, kind):
        if self.worker is not None and self.worker.kind == kind:
            return 0.
        if self.worker is not None:
            before = time.perf_counter(); self.worker.close()
            self.totals['shutdown_s'] += time.perf_counter() - before
            self.worker = None
        self.serial += 1
        self.worker = _WorkerClient(self.config, kind, self.out / ('worker_%03d_%s' % (self.serial, kind)))
        self.totals['startup_s'] += self.worker.startup_s
        self.active_checkpoint = 'F' if kind == 'sd' else 'AR'
        return self.worker.startup_s

    def encode(self, messages):
        require(self.episode_id is not None, 'Start episode before encoding')
        return encode_history(self.tokenizer, messages)

    def publish(self, checkpoint):
        if isinstance(checkpoint, str):
            checkpoint = self.checkpoints[checkpoint]
        require(checkpoint['checkpoint_id'] in self.checkpoints and
                checkpoint == self.checkpoints[checkpoint['checkpoint_id']], 'Unregistered learned checkpoint')
        self._ensure_worker('sd')
        if self.active_checkpoint == checkpoint['checkpoint_id']:
            return {'status': 'passed', 'activation_s': 0., 'native_activation_s': 0.,
                    'checkpoint_id': self.active_checkpoint}
        # Validation precedes the worker's internal timer. Charge the entire
        # synchronous call; the narrower native timer is a nested diagnostic.
        started = time.perf_counter()
        result = self.worker.call('publish', checkpoint)
        activation_s = time.perf_counter() - started
        result = {**result, 'native_activation_s': result['activation_s'], 'activation_s': activation_s}
        self.active_checkpoint = checkpoint['checkpoint_id']
        self.totals['activation_s'] += result['activation_s']
        self.totals['native_activation_s'] += result['native_activation_s']
        return result

    def serve(self, input_ids, checkpoint_id, capture=False, *, turn_index, repeat=None):
        check_input(input_ids)
        require(type(turn_index) is int and 1 <= turn_index <= 32, 'Invalid episode turn')
        require(checkpoint_id == 'AR' or checkpoint_id in self.checkpoints, 'Checkpoint not available')
        if capture:
            require(repeat is None and 9 <= turn_index <= 24 and
                    checkpoint_id == ('F' if turn_index <= 16 else 'U1'),
                    'Feedback must be fresh canonical support from the serving draft')
        startup = self._ensure_worker('ar' if checkpoint_id == 'AR' else 'sd')
        activation = native_activation = 0.
        if checkpoint_id != 'AR':
            publication = self.publish(checkpoint_id)
            activation, native_activation = publication['activation_s'], publication['native_activation_s']
        reset = self.worker.call('flush', {})
        started = time.perf_counter()
        result = self.worker.call('generate', {'input_ids': input_ids, 'checkpoint_id': checkpoint_id,
            'capture': bool(capture), 'turn': turn_index, 'repeat': repeat})
        result['request_s'] = time.perf_counter() - started
        result.update(startup_s=startup, activation_s=activation, native_activation_s=native_activation,
                      cache_maintenance_s=reset['seconds'])
        return result

    def train_event(self, event, responses):
        require(self.episode_id is not None and event in (1, 2), 'Invalid update event')
        startup = 0.
        if self.trainer is None:
            require(event == 1, 'Optimizer must start at update 1')
            self.trainer = _WorkerClient(self.config, 'train', self.out / 'training_worker')
            self.totals['startup_s'] += self.trainer.startup_s
            startup = self.trainer.startup_s
        result = self.trainer.call('train', {'event': event, 'responses': responses})
        result['trainer_startup_s'] = startup
        # Both workers read the same local checkpoint path. Device copies are
        # inside export/publication; there is no separate checkpoint transfer.
        result['checkpoint_transfer'] = {'performed': False, 'seconds': 0.,
            'scope': 'Shared local filesystem; device copies remain inside export and activation.'}
        checkpoint = result['checkpoint']
        require(checkpoint['checkpoint_id'] == 'U' + str(event), 'Training export version differs')
        verify_checkpoint(checkpoint)
        self.checkpoints[checkpoint['checkpoint_id']] = checkpoint
        return result

    def finish_episode(self):
        errors = []
        for worker in (self.worker, self.trainer):
            if worker is not None:
                try:
                    before = time.perf_counter(); worker.close()
                    self.totals['shutdown_s'] += time.perf_counter() - before
                except Exception as error:
                    errors.append(repr(error))
        self.worker = self.trainer = None
        receipt = {'status': 'failed' if errors else 'passed', 'episode_id': self.episode_id,
                   **self.totals, 'errors': errors}
        if self.episode_id is not None:
            receipt['free_after_bytes'] = shutil.disk_usage(self.out).free
            write_json(self.out / 'NATIVE_FINISH.json', receipt)
        self.episode_id = None
        self.totals = {'startup_s': 0., 'shutdown_s': 0., 'activation_s': 0., 'native_activation_s': 0.}
        require(not errors, 'Native shutdown failed: ' + str(errors))
        return receipt


def _worker_config():
    path = Path(os.environ['TWO_UPDATE_WORKER_CONFIG'])
    require(sha(path) == os.environ['TWO_UPDATE_WORKER_CONFIG_SHA256'], 'Worker config changed')
    config = json.loads(path.read_text())
    for name, expected in config['public_source_sha256'].items():
        require(Path(name).name == name and sha(HERE / name) == expected, 'Public bridge source changed')
    return config


def _serving_entry(config):
    root = Path(config['workspace_root'])
    directory = root / 'research/target_variant_specialization_001/native_tools'
    sys.path.insert(0, str(directory))
    os.environ.update(TV_TARGET_REGISTRY=config['registry_path'],
        TV_TARGET_REGISTRY_SHA256=config['registry_sha256'], TV_TARGET_ID='T0',
        TV_TARGET_IDENTITY_SHA256=config['target']['identity_sha256'], TV_OUTPUT_CAP=str(OUTPUT_CAP),
        D181_SERVING_OUT=config['worker_out'])
    import tv_d202_serving as entry
    require(Path(entry.inherited.__file__).resolve() == root /
        'research/multiturn_online_setting_20261008/online_driver_d181/serving.py', 'Native serving import collision')
    return entry


def _sd_scheduler(*args, **kwargs):
    config = _worker_config()
    entry = _serving_entry(config)
    records, cache = capture_modules(config['workspace_root'])
    native_trace = entry.RequestTrace

    class CapturingTrace(native_trace):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            pending = json.loads((Path(config['worker_out']) / 'FEATURE_PENDING.json').read_text())
            active = json.loads((Path(config['worker_out']) / 'ACTIVE_CHECKPOINT.json').read_text())
            require(pending['draft_identity_sha256'] == active['checkpoint_identity_sha256'] and
                    pending['draft_version'] == active['checkpoint_id'] and
                    pending['prompt_id'] == self.context['conversation_id'], 'Capture is bound to a stale/different draft')
            self.feature_capture = cache.ResponseCapture(pending, self.input_ids,
                max_bytes=config.get('max_capture_bytes', 2 * 2**30)) if pending['capture'] else None

        def finish(self, output_ids, meta):
            value = super().finish(output_ids, meta)
            if self.feature_capture is not None:
                destination = Path(config['worker_out']) / 'records' / ('request_%06d' % self.context['request_index'])
                value['verification_record'] = self.feature_capture.finish(output_ids, value, destination=destination)
            return value

    entry.RequestTrace = CapturingTrace
    directory = Path(config['workspace_root']) / 'research/fast_w8_001/data/collector'
    hooks = _load(directory / 'native_hooks.py', 'native_hooks')
    write_json(Path(config['worker_out']) / 'CAPTURE_COMPATIBILITY.json', {
        'records': records.two_update_compatibility, 'cache': cache.two_update_compatibility,
        'hooks': hooks.two_update_compatibility, 'gpu_qualification': 'required',
        'active_draft_binding': 'Each fresh response is checked against the completed native publication receipt.'})
    # Only hook installation is reused: its old prepare() assumes fixed-F capture.
    hooks.install(entry)
    return entry.serving_scheduler(*args, **kwargs)


def _ar_scheduler(*args, **kwargs):
    return _serving_entry(_worker_config()).ar_serving_scheduler(*args, **kwargs)


class _ServingService:
    def __init__(self, config):
        self.config, self.out = config, Path(config['worker_out'])
        self.entry = entry = _serving_entry(config)
        native, _ = entry.inherited.dependencies()
        require(config['target']['model_path'] == native.TARGET and
                config['released']['path'] == native.RELEASED, 'Qualified model paths/revisions differ')
        import common
        runtime = common.runtime_bindings()
        collector_config = {'runtime_source_sha256': config['runtime_source_sha256']}
        write_json(self.out / 'COLLECTOR_CONFIG.json', collector_config)
        os.environ['D213_COLLECTOR_CONFIG'] = str(self.out / 'COLLECTOR_CONFIG.json')
        os.environ['D213_COLLECTOR_CONFIG_SHA256'] = sha(self.out / 'COLLECTOR_CONFIG.json')
        from sglang.srt.entrypoints.engine import Engine
        is_ar = config['kind'] == 'ar'
        Engine.run_scheduler_process_func = staticmethod(_ar_scheduler if is_ar else _sd_scheduler)
        kwargs = native.engine_kwargs(config['released']['path'])
        if is_ar:
            for name in ('speculative_draft_model_path', 'speculative_algorithm', 'speculative_num_draft_tokens'):
                kwargs.pop(name)
        write_json(self.out / 'ENGINE_KWARGS.json', kwargs)
        self.engine = Engine(**kwargs)
        if is_ar:
            self.bridge = entry.ARBridge(self.engine, self.out)
        else:
            self.engine.collective_rpc('d181_install', capture=False)
            self.bridge = entry.inherited.Bridge(self.engine, self.out, False, config['released']['path'])
            # A cold request policy needs no prefix replay. Generalize only the
            # existing reset checkpoint allowlist, retaining the actual native
            # idle/flush/load/helper-refresh transaction and failure poisoning.
            import inspect
            import textwrap
            original = entry._original_bridge.publish
            source = textwrap.dedent(inspect.getsource(original))
            before = "not reset or path == Path(native.RELEASED).resolve()"
            after = 'not reset or path in self._two_update_admitted'
            require(source.count(before) == 1, 'Native cold-publication guard changed')
            edited = source.replace(before, after)
            scope = {}; exec(compile(edited, original.__code__.co_filename, 'exec'), entry.inherited.__dict__, scope)
            self.bridge.publish = scope['publish'].__get__(self.bridge)
            self.bridge._two_update_admitted = {Path(config['released']['path'])}
            write_json(self.out / 'PUBLICATION_COMPATIBILITY.json', {
                'before': before, 'after': after, 'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'effective_sha256': hashlib.sha256(edited.encode()).hexdigest(),
                'gpu_qualification': 'required', 'policy': 'flush every request and before every weight switch'})
        self.active = config['released']
        self.serial = 0
        write_json(self.out / 'ACTIVE_CHECKPOINT.json', self.active)
        write_json(self.out / 'RUNTIME_SOURCES.json', runtime)
        self.records, _ = capture_modules(config['workspace_root'])

    def flush(self, _):
        result = self.bridge.flush()
        require(result['success'] and result['state']['live_slots'] == result['state']['valid_slot_count'] == 0 and
                result['state']['cache']['tree_total_size'] == 0, 'Cold cache policy failed')
        return result

    def publish(self, checkpoint):
        require(self.config['kind'] == 'sd', 'AR engine has no draft to publish')
        verify_checkpoint(checkpoint)
        require(checkpoint['checkpoint_id'] in ('F', 'U1', 'U2'), 'Unknown checkpoint version')
        if checkpoint['checkpoint_id'] != 'F':
            require(Path(checkpoint['path']).is_relative_to(Path(self.config['episode_out']) / 'checkpoints'),
                    'Learned checkpoint belongs to another episode')
        else:
            require(checkpoint == self.config['released'], 'Frozen checkpoint differs')
        self.bridge._two_update_admitted.add(Path(checkpoint['path']))
        started = time.perf_counter()
        receipt = self.bridge.publish({'model_path': checkpoint['path'], 'reset': True})
        require(receipt['status'] == 'passed' and receipt['success'] and
                receipt['helper_matches_current_layers'] and receipt['helper_norm_matches_current_layers'] and
                receipt['cache_after']['tree_total_size'] == 0, 'Publication/helper invalidation failed')
        actual = self.bridge.state()
        require(actual['draft_model_path'] == checkpoint['path'], 'Active native draft path differs')
        self.active = checkpoint
        write_json(self.out / 'ACTIVE_CHECKPOINT.json', checkpoint)
        return {'status': 'passed', 'activation_s': time.perf_counter() - started,
                'checkpoint_id': checkpoint['checkpoint_id'], 'native_receipt': receipt}

    def generate(self, body):
        ids = body['input_ids']; check_input(ids)
        capture = body['capture']; is_ar = self.config['kind'] == 'ar'
        require(not is_ar or not capture, 'AR cannot emit draft feedback')
        require(body['checkpoint_id'] == ('AR' if is_ar else self.active['checkpoint_id']), 'Requested/loaded draft differ')
        self.serial += 1
        rid = self.config['episode_id'] + ':t' + str(body['turn']).zfill(2)
        declaration = {'record_id': rid, 'prompt_id': rid, 'episode_id': self.config['episode_id'],
            'turn': body['turn'], 'domain': 'code_math', 'role': ROLE, 'target_id': 'T0',
            'target_identity_sha256': self.config['target']['identity_sha256'],
            'draft_identity_sha256': self.active['checkpoint_identity_sha256'],
            'capture_checkpoint_sha256': self.active['checkpoint_identity_sha256'],
            'draft_version': self.active['checkpoint_id'], 'input_token_count': len(ids), 'capture': capture,
            'source_sha256': {**json.loads((HERE / 'native_dependencies.json').read_text())['source_sha256'],
                **{'public/' + k: v for k, v in self.config['public_source_sha256'].items()}},
            'target_head_norm_binding': {'target_identity_sha256': self.config['target']['identity_sha256'],
                'checkpoint_files': self.config['target']['checkpoint_files'], 'head': 'lm_head.weight',
                'norm': 'model.norm.weight', 'teacher_boundary': 'native_final_normalized_prehead'},
            'numerical_mode': {'dtype': 'bfloat16', 'attention_backend': 'fa3', 'greedy': True,
                               'block_size': BLOCK_SIZE, 'generation_seed': 42}}
        write_json(self.out / 'FEATURE_PENDING.json', declaration)
        context = {'request_index': self.serial, 'conversation_id': rid, 'turn_index': body['turn'],
                   'kind': 'timed', 'draft_version': {'F': 0, 'U1': 1, 'U2': 2}[self.active['checkpoint_id']]}
        value = self.bridge.generate({'input_ids': ids, 'sampling_params': {'temperature': 0, 'top_p': 1,
            'top_k': 1, 'max_new_tokens': OUTPUT_CAP, 'ignore_eos': False}, 'stream': False, 'trace_context': context})
        write_json(self.out / ('RAW_RESPONSE_%06d.json' % self.serial), value)
        trace = value['_trajectory']; path = self.out / trace['relative_path']
        require(path.resolve().is_relative_to(self.out) and sha(path) == trace['sha256'], 'Native trace binding changed')
        document = json.loads(path.read_text()); output = value.get('output_ids', value.get('output_token_ids'))
        require(isinstance(output, list) and 0 < len(output) <= OUTPUT_CAP and
                all(type(x) is int and x >= 0 for x in output), 'Native output IDs absent')
        meta = value['meta_info']
        require(meta['cached_tokens'] == 0, 'Serving reused a prefix cache')
        reconciliation = document['reconciliation']
        require(reconciliation['input_ids_sha256'] == object_hash(ids) and
                reconciliation['output_ids_sha256'] == object_hash(output), 'Native exact-token reconciliation failed')
        feedback = None
        if capture:
            record = document['verification_record']
            response = self.records.load_response(record['path'], declared=record, allowed_roles={ROLE},
                target_identity=self.config['target']['identity_sha256'])
            require(response['input_ids'] == ids and response['output_ids'] == output and
                    response['capture_checkpoint_sha256'] == self.active['checkpoint_identity_sha256'],
                    'Capture differs from current native response')
            feedback = {k: v for k, v in response.items() if k != '_tensor_path'}
            feedback['record'] = {k: record[k] for k in ('path', 'bytes', 'sha256')}
        calls = 0 if is_ar else meta['spec_verify_ct']
        require(is_ar or calls == reconciliation['verification_rounds'], 'Native verification counts differ')
        return {'status': 'passed', 'output_ids': output, 'text': value['text'],
            'input_ids_sha256': object_hash(ids),
            'accepted_draft_tokens': 0 if is_ar else reconciliation['actual_accepted_draft_tokens'],
            'verification_calls': calls, 'native_generation_s': trace['native_generation_s'],
            'verification_committed_tokens': 0 if is_ar else sum(len(event['committed_token_ids'])
                for event in document['events'] if event['kind'] == 'verification'),
            'finish_reason': meta['finish_reason'], 'feedback': feedback, 'trajectory': binding(path),
            'capture_costs': None if feedback is None else feedback['costs'],
            'capture_timing_scope': 'Raw capture sub-timers overlap; do not sum append_capture_s and transfer_s. NativeReplay phases are recorded by the trainer.',
            'checkpoint_identity_sha256': None if is_ar else self.active['checkpoint_identity_sha256'],
            'timing_scope': 'Common native token observer; optional capture overhead included in request time; capture transfers overlap sub-timers.'}

    def close(self):
        self.engine.shutdown()


class _TrainerService:
    def __init__(self, config):
        self.config, self.out = config, Path(config['worker_out'])
        root = Path(config['workspace_root'])
        directory = root / 'research/target_variant_specialization_001/training_tools'
        sys.path.insert(0, str(directory))
        import native as qualified_native
        require(Path(qualified_native.__file__).resolve() == directory / 'native.py', 'Training native import collision')
        self.native = qualified_native
        self.runtime = qualified_native.load_runtime(17, OUTPUT_CAP)
        args = copy.copy(self.runtime.default_args)
        require(config['released']['path'] == str(qualified_native.CORE.DRAFT) and
                config['target']['model_path'] == str(qualified_native.CORE.TARGET), 'Qualified training models differ')
        args.target_model_path = config['target']['model_path']
        args.initial_draft_path = args.draft_config_path = config['released']['path']
        args.learning_rate = .0003
        self.components = self.runtime.setup.load_target_embeddings(args).requires_grad_(False)
        target_stub = SimpleNamespace(set_capture_layers=lambda layers:
            require(layers == [1, 8, 15, 22, 29], 'Feature layers differ'))
        draft, self.model_config, _, mask = self.runtime.setup.load_draft_stack(args, target_stub)
        require(mask == 128002 and draft.block_size == 10 and draft.target_layer_ids == [1, 8, 15, 22, 29],
                'Qualified draft architecture changed')
        import torch
        parameters = list(draft.parameters())
        require(len(parameters) == 58 and sum(value.numel() for value in parameters) == 1048626432 and
                all(value.requires_grad and value.dtype == torch.bfloat16 for value in parameters) and
                float(getattr(self.model_config, 'attention_dropout', 0.)) == 0. and
                not any(getattr(module, 'gradient_checkpointing', False) for module in draft.modules()),
                'Full BF16/native deterministic training inventory differs')
        qualified_native.CORE.verify_checkpoint(draft, config['released']['path'])
        self.records, _ = capture_modules(root)
        sys.path.insert(0, str(root / 'research/fast_w8_001/training'))
        replay = _load(root / 'research/fast_w8_001/training/replay.py', 'two_update_w8_replay')
        self.backend = replay.NativeReplay(draft, {'T0': self.components}, ReplayRecords(self.records),
            identities={'T0': config['target']['identity_sha256']})
        self.backend.activate('T0')
        training = _load(HERE / 'training.py', 'two_update_persistent_training')
        self.trainer = training.PersistentFullKLTrainer(draft, self.backend, episode_id=config['episode_id'],
            target_id='T0', target_identity_sha256=config['target']['identity_sha256'],
            released_checkpoint_sha256=config['released']['checkpoint_identity_sha256'], seed=17)
        self.draft = draft
        self.frozen = qualified_native.CORE.state_fingerprints(self.components)

    def train(self, body):
        event = body['event']; responses = []
        for advertised in body['responses']:
            response = self.records.load_response(advertised['record']['path'], declared=advertised['record'],
                allowed_roles={ROLE}, target_identity=self.config['target']['identity_sha256'])
            require(all(response[k] == advertised[k] for k in (
                'record_id', 'turn', 'episode_id', 'input_ids_sha256', 'output_ids_sha256', 'capture_checkpoint_sha256')),
                'Controller support differs from bound native record')
            responses.append(response)
        require(self.native.CORE.state_fingerprints(self.components) == self.frozen, 'Frozen target components changed')
        accounting = self.trainer.train_event(event, responses)
        require(self.native.CORE.state_fingerprints(self.components) == self.frozen, 'Training mutated target components')
        checkpoint = self._export(event)
        self.trainer.bind_checkpoint(event, checkpoint['checkpoint_identity_sha256'])
        state_path = Path(checkpoint['path']).parent / ('U' + str(event) + '_training_state.pt')
        state_start = time.perf_counter()
        optimizer_bytes = sum(value.numel() * value.element_size()
            for values in self.trainer.optimizer.state.values() for value in values.values()
            if hasattr(value, 'numel'))
        require(shutil.disk_usage(state_path.parent).free >= optimizer_bytes + 64 * 2**20 +
                self.config.get('reserve_bytes', 8 * 2**30), 'Optimizer state would cross storage reserve')
        state = self.trainer.save_checkpoint_state(state_path, include_draft=False)
        state_s = time.perf_counter() - state_start
        result = {'status': 'passed', 'checkpoint_id': checkpoint['checkpoint_id'],
            'checkpoint_path': checkpoint['path'], 'checkpoint': checkpoint,
            'optimizer_steps': accounting['optimizer_steps_total'], 'training_s': accounting['costs']['training_s'],
            'accounting': accounting, 'training_state': state, 'export_s': checkpoint['export_s'],
            'optimizer_state_save_s': state_s, 'costs': {**accounting['costs'],
                'export_s': checkpoint['export_s'], 'optimizer_state_save_s': state_s}}
        write_json(self.out / ('UPDATE_%d.json' % event), result)
        return result

    def _export(self, event):
        import torch
        from safetensors.torch import save_file
        started = time.perf_counter()
        parent = Path(self.config['episode_out']) / 'checkpoints'; parent.mkdir(exist_ok=True)
        path = parent / ('U' + str(event)); staging = parent / ('.U' + str(event) + '.staging')
        require(not path.exists() and not staging.exists(), 'Preserve existing checkpoint attempt')
        state = {name: value.detach().cpu().contiguous() for name, value in self.draft.state_dict().items()}
        require(len(state) == 58 and all(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
                for value in state.values()), 'Full-weight draft export inventory/precision differs')
        estimate = sum(value.numel() * value.element_size() for value in state.values()) + 64 * 2**20
        require(shutil.disk_usage(parent).free >= estimate + self.config.get('reserve_bytes', 8 * 2**30),
                'Serving export would cross storage reserve')
        staging.mkdir(); save_file(state, str(staging / 'model.safetensors'))
        (staging / 'config.json').write_text(self.model_config.to_json_string())
        (staging / 'version.txt').write_text('1')
        for entry in staging.iterdir():
            with entry.open('rb') as stream:
                os.fsync(stream.fileno())
        self.native.CORE.verify_checkpoint(self.draft, staging)
        files = {entry.name: {'bytes': entry.stat().st_size, 'sha256': sha(entry)} for entry in sorted(staging.iterdir())}
        os.rename(staging, path)
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return {'path': str(path), 'files': files, 'checkpoint_identity_sha256': object_hash(files),
                'checkpoint_id': 'U' + str(event), 'export_s': time.perf_counter() - started}

    def close(self):
        self.runtime.destroy_distributed()


def worker_main():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import traceback
    config = _worker_config()
    check_dependencies(config['workspace_root'])
    # Allocation has already been narrowed to this child, so do not reinterpret
    # it as the controller's original two-device scheduler allocation.
    child_allocation = os.environ.pop('CUDA_VISIBLE_DEVICES')
    try:
        check_environment(config)
    finally:
        os.environ['CUDA_VISIBLE_DEVICES'] = child_allocation
    service = _TrainerService(config) if config['kind'] == 'train' else _ServingService(config)
    poisoned = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                require(self.headers.get('X-Two-Update-Token') == config['rpc_token'], 'Foreign worker client')
                length = int(self.headers.get('Content-Length', '0'))
                require(0 < length <= 64 * 2**20, 'Invalid RPC body size')
                body = json.loads(self.rfile.read(length)); action = self.path.lstrip('/')
                require(not poisoned, 'Worker poisoned by prior failure; no automatic retry')
                if action == 'ready':
                    value = {'status': 'passed', 'pid': os.getpid()}
                else:
                    allowed = {'train'} if config['kind'] == 'train' else {'generate', 'publish', 'flush'}
                    require(action in allowed, 'Unknown worker action')
                    value = getattr(service, action)(body)
                status = 200
            except Exception as error:
                poisoned.append(repr(error))
                value = {'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()}
                write_json(Path(config['worker_out']) / ('FAILURE_%d.json' % len(poisoned)), value)
                status = 500
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

    server = HTTPServer(('127.0.0.1', config['port']), Handler)

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close(); service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.worker, 'Use the experiment controller; this module never launches implicitly')
    worker_main()
