"""Read-only matched-prefix backend on the exact PR11 serving runtime.

Importing this module only inspects Python source. Model loading is confined to
``start`` and the worker admits one-block probes only, never training/generation.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
_spec = importlib.util.spec_from_file_location('_matched_prefix_qualified_l8',
    HERE.parent / 'dflash_lower_lr_control/native.py')
qualified = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = qualified
_spec.loader.exec_module(qualified)
base, kl = qualified.base, qualified.kl
require, sha, digest, write_json = base.require, base.sha, base.object_hash, base.write_json
ARMS = ('F', 'S8', 'L8')
MIN_RESERVE_BYTES = 8 * 2**30
GROUP_BYTES, STAGING_BYTES = 64 * 2**20, 64 * 2**20


def source_pins():
    pins = qualified.source_pins()
    # The old probe is provenance, never imported (it has a different runtime).
    predecessor = HERE.parent / 'dflash_osd_attribution/native_probe.py'
    pins[str(predecessor.relative_to(HERE.parent))] = sha(predecessor)
    for pattern in ('*.py', '*.json', 'requirements*.txt'):
        for path in sorted(HERE.glob(pattern)):
            if not path.name.startswith('test_'):
                pins[str(path.relative_to(HERE.parent))] = sha(path)
    return dict(sorted(pins.items()))


def check_sources(config):
    require(config.get('matched_prefix_source_sha256') == source_pins(), 'Reviewed matched-prefix source changed')
    qualified.check_sources(config)


def storage_preflight(config):
    """Budget all compact records and retrieval staging; no model/tensor copies."""
    root = Path(config['output_root']).resolve()
    while not root.exists():
        root = root.parent
    reserve = config.get('reserve_bytes', MIN_RESERVE_BYTES)
    require(type(reserve) is int and reserve >= MIN_RESERVE_BYTES, 'Preserve the 8 GiB storage reserve')
    writes = 6 * GROUP_BYTES + STAGING_BYTES
    free = shutil.disk_usage(root).free
    require(free >= writes + reserve, 'Compact diagnostics, retrieval staging and reserve exceed free space')
    return dict(status='passed', free_bytes=free, required_bytes=writes + reserve,
        planned_write_bytes=writes, groups=6, group_budget_bytes=GROUP_BYTES,
        staging_bytes=STAGING_BYTES, reserve_bytes=reserve, checkpoint_copies=0,
        optimizer_states=0, feature_tensor_exports=0,
        scope='At most 720 compact block/position receipts plus configuration, logs and retrieval staging')


def worker_command(config):
    command = qualified.followup._original_worker_command(config)
    require(command[-1] == str(base.HERE / 'native.py'), 'Qualified transport changed')
    command[-1] = str(HERE / 'native_probe.py')
    return command


def admit_checkpoint(config, checkpoint):
    label = next((arm for arm, value in config['matched_prefix_checkpoints'].items() if value == checkpoint), None)
    require(label in ARMS, 'Checkpoint differs from exact group F/S8/L8 binding')
    require(checkpoint['checkpoint_id'] == {'F':'F', 'S8':'U2', 'L8':'L8'}[label],
            'Physical checkpoint label changed')
    if label == 'F':
        require(checkpoint == config['released'], 'Released initializer differs')
    elif label == 'S8':
        require(Path(checkpoint['path']) == kl.ORIGINAL_NATIVE_ROOT / config['episode_id'] / 'checkpoints/U2',
                'S8 belongs to a different original group')
    else:
        require(checkpoint.get('group_id') == config['episode_id'] and checkpoint.get('optimizer_steps') == 8 and
                checkpoint.get('learning_rate') == 3e-5 and
                Path(checkpoint['path']).parts[-3:] == (config['episode_id'], 'checkpoints', 'L8'),
                'L8 group, path or retained learning trajectory changed')
    base.verify_checkpoint(checkpoint)
    return label


class NativeBackend:
    """Metadata-only construction; start one exact group and close its owned worker."""
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.worker = None
        self.group = None
        self.used = set()
        self.costs = dict(loading_s=0., activation_s=0., cache_maintenance_s=0.,
            probe_wall_s=0., shutdown_s=0., attempted_probes=0, native_draft_block_evaluations=0,
            native_head_calls=0, target_prefill_chunk_count=0, head_inspection_s=0.)

    def start(self, group):
        from .native_probe import validate_plan
        require(self.group is None and self.worker is None, 'Close the previous native group first')
        check_sources(self.config)
        base.check_environment(self.config)
        deps = base.check_dependencies(self.config['workspace_root'])
        require(self.config['runtime_source_sha256'] == deps['runtime_source_sha256'], 'Qualified runtime changed')
        require(self.config.get('output_cap', 1024) == base.OUTPUT_CAP == 1024 and
                self.config.get('block_size', 10) == base.BLOCK_SIZE == 10 and base.CONTEXT_LIMIT == 32768,
                'Keep the original BF16 block10/context32768 runtime')
        group = copy.deepcopy(group)
        group_id = group['group_id']
        require(Path(group_id).name == group_id and group_id not in ('.', '..'), 'Invalid group identifier')
        require(set(group['checkpoints']) == set(ARMS), 'Exact F/S8/L8 group checkpoints required')
        root = Path(self.config['output_root']).resolve()
        root.mkdir(parents=True, exist_ok=True)
        out = root / group_id
        require(not out.exists(), 'Never overwrite a native group attempt')
        reserve = self.config.get('reserve_bytes', MIN_RESERVE_BYTES)
        require(type(reserve) is int and reserve >= MIN_RESERVE_BYTES and
                shutil.disk_usage(root).free >= GROUP_BYTES + STAGING_BYTES + reserve,
                'Group evidence and staging would cross the storage reserve')
        registry = base.checked_binding(self.config['model_bindings'])
        require(sha(registry) == deps['qualified_model_bindings']['sha256'], 'Qualified model registry changed')
        models = json.loads(registry.read_text())
        target = next(row for row in models['targets'] if row['target_id'] == 'T0')
        released = models['released_draft']
        frozen = dict(path=released['path'], files=released['checkpoint_files'], checkpoint_id='F',
                      checkpoint_identity_sha256=digest(released['checkpoint_files']))
        require(group['checkpoints']['F'] == frozen, 'Released checkpoint differs from registry')
        retained = self.config['matched_prefix_native_runtime']
        require(retained['target'] == target and retained['released'] == frozen and
                retained['tokenizer_path'] == models.get('tokenizer_path',target['model_path']) and
                retained['effective_limits']['total_context_tokens'] == 32768 and
                retained['effective_limits']['chunked_prefill_tokens'] == 8192,
                'Original PR11 target/tokenizer/context configuration changed')
        cfg = {**self.config, 'episode_id':group_id, 'episode_out':str(out), 'target':target,
            'released':frozen, 'matched_prefix_checkpoints':group['checkpoints'],
            'matched_prefix_probe_plan':group['anchors'], 'registry_path':str(registry),
            'registry_sha256':sha(registry), 'tokenizer_path':models.get('tokenizer_path',target['model_path']),
            'public_source_sha256':{name:sha(base.HERE / name) for name in ('native.py','training.py','native_dependencies.json')}}
        for checkpoint in group['checkpoints'].values():
            admit_checkpoint(cfg, checkpoint)
        for name, bound in target['checkpoint_files'].items():
            require(Path(name).name == name, 'Target file escapes directory')
            base.checked_binding({'path':str(Path(target['model_path']) / name), **bound})
        validate_plan(cfg)
        out.mkdir()
        self.group, self.config, self.out = group, cfg, out
        write_json(out / 'NATIVE_CONFIG.json', cfg)
        if group['anchors']:
            previous = base.worker_command
            base.worker_command = worker_command
            started = time.perf_counter()
            try:
                self.worker = base._WorkerClient(cfg, 'sd', out / 'probe_worker')
            finally:
                base.worker_command = previous
                self.costs['loading_s'] += time.perf_counter() - started
        return dict(status='passed', group_id=group_id, checkpoints=group['checkpoints'],
            costs=dict(self.costs), optimizer_created=False, optimizer_loaded=False,
            optimizer_steps=0, warmup_generations=0, native_worker_started=self.worker is not None,
            qualification='pending_native_probes' if group['anchors'] else 'insufficient_no_selected_locations')

    def probe(self, anchor, arm):
        require(self.group is not None and self.worker is not None and arm in ARMS, 'No active declared native probe')
        planned = next((a for a in self.group['anchors'] if a['anchor_id'] == anchor['anchor_id']), None)
        require(planned == anchor, 'Anchor differs from sealed native group')
        key = (anchor['anchor_id'], arm)
        require(key not in self.used, 'No repeated probe or automatic retry')
        self.used.add(key)
        self.costs['attempted_probes'] += 1
        started = time.perf_counter()
        value = None
        try:
            value = self.worker.call('probe', dict(anchor_id=anchor['anchor_id'], arm=arm,
                checkpoint=anchor['checkpoints'][arm]))
            return value
        except BaseException as error:
            # The transport raises on HTTP500. Recover the scheduler's durable
            # partial row, including zero-forward seed failures, before rethrow.
            path = self.out / 'probe_worker/blocks' / (digest(list(key)) + '.json')
            if path.exists():
                value = json.loads(path.read_text())
                error.diagnostic_partial = value
                error.partial_binding = base.binding(path)
            raise
        finally:
            self.costs['probe_wall_s'] += time.perf_counter() - started
            if value:
                for field in ('native_draft_block_evaluations','native_head_calls'):
                    count = value.get(field)
                    if type(count) is int:
                        self.costs[field] += count
                self.costs['target_prefill_chunk_count'] += len(value.get('target_prefill_chunks', []))
                for field in ('activation_s','cache_maintenance_s','head_inspection_s'):
                    self.costs[field] += value.get('costs', {}).get(field, 0.)

    def close(self):
        if self.worker is not None:
            started = time.perf_counter()
            try:
                kl.close_owned_worker(self.worker)
            finally:
                self.costs['shutdown_s'] += time.perf_counter() - started
            self.worker = None
        value = dict(status='passed', group_id=self.group['group_id'] if self.group else None, costs=dict(self.costs))
        if self.group is not None:
            value['free_after_bytes'] = shutil.disk_usage(self.out).free
            write_json(self.out / 'NATIVE_FINISH.json', value)
        self.group = None
        return value
