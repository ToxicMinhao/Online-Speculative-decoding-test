"""Host interface for delayed, fixed-input native probe scoring on one A800.

Training finishes and releases the card before this evaluator starts. Exported
P/N/U/T weights represent the exact original window states; probe outcomes never
enter selection. One native worker retains compact CPU snapshots for all starts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

from online_feedback.storage import write_json
from experiments.lost_repair.run import block_observation
from experiments.longer_run_headroom.pairs import neutral_parity
from experiments.verification_transition.native_worker import NativeTransitionBackend


WINDOWS = tuple(range(8, 129, 8))
STATES = ('P', 'N', 'U', 'T')


def read(path):
    return json.loads(Path(path).read_text())


def probe_key(row):
    return row['request_id'], row['canonical_start'], row['round_ordinal']


def score_observation(record, key, observed, *, window, state, checkpoint):
    """Score native accepted proposals, including matching EOS but never seed."""
    if observed['status'] != 'completed' or len(observed['blocks']) != 1:
        raise ValueError('Probe requires one completed native block')
    block = observed['blocks'][0]
    original = record['metadata']['trace'][key['round_ordinal']]
    fields = ('output_start', 'anchor_token', 'positions')
    if (any(block[name] != original[name] for name in fields)
            or original['output_start'] != key['canonical_start']
            or record['record_id'] != key['request_id']):
        raise ValueError('Probe uses a different canonical prefix or position')
    start = block['output_start'] + 1
    end = start + block['committed_progress']
    expected_finished = end == len(record['response_ids'])
    if (block['committed_tokens'] != record['response_ids'][start:end]
            or observed['response_ids'] != record['response_ids'][:end]
            or not observed['output_matches_canonical_prefix'] or not observed['cap_unchanged']
            or observed['original_cap'] != record['max_new_tokens']
            or observed['native_finished'] != expected_finished or block['finished'] != expected_finished
            or observed['reconstruction']['status'] != 'passed'):
        raise ValueError('Probe canonical output, cap, termination or cache validity failed')
    score = block_observation(block, record)
    a, valid = score['accepted'], score['valid_proposals']
    if a < valid and block['draft_tokens'][a] == record['response_ids'][start+a]:
        raise ValueError('Native rejection does not identify the first canonical mismatch')
    return dict(key, window=window, state=state, checkpoint=str(checkpoint), accepted_prefix=a,
        valid_proposals=valid, first_error=a + 1 if a < valid else None,
        matched_eos=any(token in record['stop_token_ids'] for token in block['draft_tokens'][:a]),
        censored_at_cap=(a == valid and record['finish_reason'] == 'length' and start + valid == len(record['response_ids'])),
        valid=True, reused_from=None, diagnostic_only=True)


def resource_estimate(plan, keys=None):
    """Conservative cache estimate from pinned shapes; report actual sizes too."""
    target = read(Path(plan['target_path']) / 'config.json')
    draft = read(Path(plan['checkpoints']['K6']['export']) / 'config.json')
    def kv_bytes(config):
        heads = config.get('num_key_value_heads', config['num_attention_heads'])
        dimension = config.get('head_dim', config['hidden_size'] // config['num_attention_heads'])
        return 2 * 2 * config['num_hidden_layers'] * heads * dimension
    kv_per_token = kv_bytes(target) + kv_bytes(draft)
    hidden_per_token = 2 * target['hidden_size'] * len(draft['dflash_config']['target_layer_ids'])
    groups = defaultdict(list)
    for row in read(plan['probe_keys']) if keys is None else keys:
        groups[row['request_id']].append(row)
    retained, temporary = 0, 0
    rounds = 0
    for rid, keys in groups.items():
        parent = read(plan['parent_records'][rid])
        lengths = [len(parent['prompt_ids']) + row['canonical_start'] for row in keys]
        longest = max(lengths)
        retained += longest * (kv_per_token + hidden_per_token)
        temporary = max(temporary, sum(lengths) * kv_per_token)
        rounds += len(parent['metadata']['trace'])
    return dict(requests=len(groups), starts=sum(map(len, groups.values())),
        saved_reconstruction_rounds=rounds, reconstruction_target_prefills=len(groups),
        neutral_rebuild_blocks=sum(map(len, groups.values())),
        estimated_compacted_cache_and_ingestion_bytes=retained,
        conservative_precompaction_peak_bytes=retained + temporary,
        host_reserve_bytes=plan.get('host_reserve_bytes', 32 * 1024**3),
        estimate_scope='BF16 KV plus committed target ingestions; native runtime/metadata extra',
        complete_response_generations=0)


class ProbeBackend(NativeTransitionBackend):
    def __init__(self, target_path, draft_path, *, worker_python, output_root,
                 budget_connection, serving_gpu='0', host_reserve_bytes=32 * 1024**3):
        from online_feedback.sglang_backend import SGLangDFlashBackend
        variable = 'DFLASH_TRANSITION_BUDGET'
        previous = os.environ.get(variable)
        os.environ[variable] = json.dumps(budget_connection)
        try:
            self.backend = SGLangDFlashBackend(target_path, draft_path, worker_python=worker_python,
                output_root=output_root, serving_gpu=serving_gpu, context_limit=4096, block_size=16,
                worker_module='experiments.threatened_repair_replay.native_probe')
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
        self.loading_s = self.backend.loading_s
        self.serial = 0
        self.published_checkpoint = None
        self.host_reserve_bytes = host_reserve_bytes

    def select_checkpoint(self, checkpoint):
        # Engine initialization has no publication receipt; explicitly publish
        # K6 once even though the constructor already loaded those weights.
        if self.published_checkpoint != str(checkpoint):
            self.backend.publish(checkpoint)
            self.published_checkpoint = str(checkpoint)

    def reconstruct(self, record, keys, checkpoint):
        if record['stop_token_ids'] != self.backend.stop_tokens:
            raise ValueError('Retained EOS set differs from native runtime')
        self.select_checkpoint(checkpoint)
        return self._rpc('transition_reconstruct_saved', record=record, keys=keys,
                         checkpoint=str(checkpoint), host_reserve_bytes=self.host_reserve_bytes)

    def one_block(self, snapshot, checkpoint):
        self.select_checkpoint(checkpoint)
        return self._rpc('transition_one_block', handle=snapshot['handle'], checkpoint=str(checkpoint))


def validate_exports(exports):
    lookup = {(row['window'], row['state']): row for row in exports}
    if len(lookup) != len(exports) or set(lookup) != {(w, s) for w in WINDOWS for s in STATES}:
        raise ValueError('Require one exact exported-state binding for every window and P/N/U/T')
    for (window, state), row in lookup.items():
        reused = row.get('reused_from')
        if reused and (reused not in STATES or reused == state or
                       row['checkpoint'] != lookup[(window, reused)]['checkpoint']):
            raise ValueError('Reused probe state must bind the same exported weights')
    for window in WINDOWS:
        ordered_states(lookup, window)
    return lookup


def ordered_states(exports, window):
    """Explicit aliases may name either U or T; reject cycles, never duplicate work."""
    remaining, order = list(STATES), []
    while remaining:
        ready = [state for state in remaining
                 if not exports[(window, state)].get('reused_from')
                 or exports[(window, state)]['reused_from'] in order]
        if not ready:
            raise ValueError('Cyclic exported-state reuse bindings')
        state = ready[0]
        order.append(state)
        remaining.remove(state)
    return order


def evaluate_exports(backend, exports, keys, parents, snapshots, root, budget, receipt, outcomes):
    """Persist each attempted row, including failures within a partial state."""
    def append(row):
        outcomes.append(row)
        with (root / 'PROBE_OUTCOMES.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        receipt['conceptual_probe_rows'] = len(outcomes)
        write_json(root / 'STATUS.json', receipt)

    for window in sorted({key[0] for key in exports}):
        measured = {}
        for state in ordered_states(exports, window):
            export = exports[(window, state)]
            reused = export.get('reused_from')
            if reused:
                rows = [dict(row, state=state, reused_from=reused) for row in measured[reused]]
                receipt['reused_probe_rows'] += len(rows)
                for row in rows:
                    append(row)
            else:
                rows = []
                for key in keys:
                    budget.check_time()
                    path = root / f'window_{window:03d}' / state / key['request_id'] / f"{key['round_ordinal']:04d}.json"
                    receipt['active_probe'] = dict(key, window=window, state=state)
                    receipt['probe_attempts'] += 1
                    write_json(root / 'STATUS.json', receipt)
                    try:
                        observed = backend.one_block(snapshots[probe_key(key)], export['checkpoint'])
                        receipt['physical_probe_blocks'] += 1
                        write_json(path, observed)
                        row = score_observation(parents[key['request_id']], key, observed,
                            window=window, state=state, checkpoint=export['checkpoint'])
                    except BaseException as error:
                        row = dict(key, window=window, state=state, checkpoint=export['checkpoint'],
                            valid=False, error=repr(error), execution_id=str(path.relative_to(root)))
                        receipt['invalid_probe_rows'] += 1
                        append(row)
                        raise
                    row['execution_id'] = str(path.relative_to(root))
                    rows.append(row)
                    receipt['active_probe'] = None
                    append(row)
            measured[state] = rows


def sealed_probe_keys(plan):
    keys = read(plan['probe_keys'])
    counts = Counter(row['request_id'] for row in keys)
    if len(keys) != 256 or len(counts) != 64 or set(counts.values()) != {4} or len({probe_key(k) for k in keys}) != 256:
        raise ValueError('Require the unchanged complete 64-request/four-start panel')
    return keys


def host_memory_available():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def check_host_admission(estimate):
    available = host_memory_available()
    if available < estimate['conservative_precompaction_peak_bytes'] + estimate['host_reserve_bytes']:
        raise MemoryError('Sealed probe cache estimate exceeds available host memory and reserve')
    return available


def qualify(plan, output, budget_connection, *, serving_gpu='0'):
    """One disposable worker checks the first sealed parent and its four starts.

    Reconstruct the complete retained native path, then rebuild each sealed start
    at K6. These checks create neither new answers nor optimizer updates. All
    native startup, target, proposal, context and loading work uses the campaign
    budget service; the worker is closed before training can start.
    """
    from experiments.verification_transition.budget import connect_budget
    from experiments.verification_transition.native import run_counted

    keys = sealed_probe_keys(plan)
    rid = plan['probe_requests'][0]['request_id']
    subset = [row for row in keys if row['request_id'] == rid]
    if len(subset) != 4:
        raise ValueError('First sealed request must have its original four starts')
    record = read(plan['parent_records'][rid])
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    estimate = resource_estimate(plan, subset)
    write_json(root / 'RESOURCE_ESTIMATE.json', estimate)
    budget = connect_budget(budget_connection)
    before = budget.summary()
    started = time.perf_counter()
    backend = None
    receipt = dict(status='running', qualification_only=True, request_id=rid,
        planned_requests=1, planned_neutral_blocks=4, planned_saved_rounds=len(record['metadata']['trace']),
        requests_reconstructed=0, neutral_blocks_attempted=0, neutral_blocks_completed=0,
        neutral_blocks_valid=0, new_generations=0, adam_steps=0, worker_closed=False,
        host_reserve_bytes=estimate['host_reserve_bytes'], budget_before=before, checks=[])
    write_json(root / 'RESULT.json', receipt)
    try:
        receipt['host_available_before_bytes'] = check_host_admission(estimate)
        checkpoint = plan['checkpoints']['K6']['export']
        backend = run_counted(budget, 'model_load', lambda: ProbeBackend(plan['target_path'], checkpoint,
            worker_python=plan['worker_python'], output_root=root / 'native',
            budget_connection=budget_connection, serving_gpu=serving_gpu,
            host_reserve_bytes=estimate['host_reserve_bytes']),
            metadata=dict(operation='disposable_probe_qualification_engine'))
        receipt['loading_s'] = backend.loading_s
        reconstructed = backend.reconstruct(record, subset, checkpoint)
        write_json(root / 'RECONSTRUCTION.json', reconstructed)
        if reconstructed['status'] != 'passed':
            raise ValueError('Saved-parent reconstruction did not pass')
        receipt['requests_reconstructed'] = 1
        receipt['saved_rounds_replayed'] = reconstructed['rounds_replayed']
        for key in subset:
            budget.check_time()
            receipt['active_start'] = key
            receipt['neutral_blocks_attempted'] += 1
            write_json(root / 'RESULT.json', receipt)
            snapshot = reconstructed['snapshots'][str(key['round_ordinal'])]
            observed = backend.one_block(snapshot, checkpoint)
            receipt['neutral_blocks_completed'] += 1
            path = root / f"{key['round_ordinal']:04d}_NEUTRAL.json"
            write_json(path, observed)
            score_observation(record, key, observed, window=0, state='K6', checkpoint=checkpoint)
            parity = neutral_parity(record['metadata']['trace'][key['round_ordinal']], observed['blocks'][0])
            write_json(path.with_suffix('.parity.json'), parity)
            receipt['checks'].append(dict(key, parity=parity))
            if parity['status'] != 'passed':
                raise ValueError('Rebuilt K6 block differs from all fifteen retained proposals/commits')
            receipt['neutral_blocks_valid'] += 1
            receipt['active_start'] = None
            write_json(root / 'RESULT.json', receipt)
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
                receipt['worker_closed'] = True
        except BaseException as error:
            receipt.update(status='failed', cleanup_error=repr(error))
            raise
        finally:
            after = budget.summary()
            receipt['budget_after'] = after
            receipt['work_delta'] = {kind: {
                name: after[kind].get(name, 0) - before[kind].get(name, 0)
                for name in sorted(set(before[kind]) | set(after[kind]))}
                for kind in ('attempted', 'completed', 'failed', 'unknown')}
            receipt.update(wall_s=time.perf_counter() - started,
                unattempted_neutral_blocks=4 - receipt['neutral_blocks_attempted'])
            write_json(root / 'RESULT.json', receipt)
    return receipt


def run(plan, export_rows, output, budget_connection, *, serving_gpu='0'):
    """Evaluate sealed exports after training; all raw observations precede checks.

    export_rows contains all 64 {window, state, checkpoint, reused_from?} entries.
    The caller authenticates exports and owns overall GPU admission/time limits.
    This function uses the same live budget service for startup and native work.
    """
    from experiments.verification_transition.budget import connect_budget
    from experiments.verification_transition.native import run_counted

    exports = validate_exports(export_rows)
    keys = sealed_probe_keys(plan)
    counts = Counter(row['request_id'] for row in keys)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    estimate = resource_estimate(plan)
    write_json(root / 'RESOURCE_ESTIMATE.json', estimate)
    check_host_admission(estimate)
    budget = connect_budget(budget_connection)
    backend = None
    outcomes, snapshots, parents = [], {}, {}
    started = time.perf_counter()
    receipt = dict(status='running', phase='saved_path_reconstruction', new_generations=0,
        delayed_scoring='exact window exports; held-out outcomes unavailable during training/selection',
        requests_reconstructed=0, neutral_blocks=0, physical_probe_blocks=0,
        conceptual_probe_rows=0, reused_probe_rows=0, planned_probe_rows=16384,
        probe_attempts=0, invalid_probe_rows=0, active_probe=None)
    write_json(root / 'STATUS.json', receipt)
    try:
        checkpoint = plan['checkpoints']['K6']['export']
        backend = run_counted(budget, 'model_load', lambda: ProbeBackend(plan['target_path'], checkpoint,
            worker_python=plan['worker_python'], output_root=root / 'native',
            budget_connection=budget_connection, serving_gpu=serving_gpu,
            host_reserve_bytes=estimate['host_reserve_bytes']),
            metadata=dict(operation='probe_engine_initialization'))
        for rid in sorted(counts):
            record = read(plan['parent_records'][rid])
            parents[rid] = record
            subset = [row for row in keys if row['request_id'] == rid]
            reconstructed = backend.reconstruct(record, subset, checkpoint)
            write_json(root / 'parents' / rid / 'RECONSTRUCTION.json', reconstructed)
            for key in subset:
                snapshot = reconstructed['snapshots'][str(key['round_ordinal'])]
                snapshots[probe_key(key)] = snapshot
                observed = backend.one_block(snapshot, checkpoint)
                path = root / 'parents' / rid / f"{key['round_ordinal']:04d}_NEUTRAL.json"
                write_json(path, observed)
                score_observation(record, key, observed, window=0, state='K6', checkpoint=checkpoint)
                parity = neutral_parity(record['metadata']['trace'][key['round_ordinal']], observed['blocks'][0])
                write_json(path.with_suffix('.parity.json'), parity)
                if parity['status'] != 'passed':
                    raise ValueError('Rebuilt K6 block differs from all fifteen retained proposals/commits')
                receipt['neutral_blocks'] += 1
            receipt['requests_reconstructed'] += 1
            write_json(root / 'STATUS.json', receipt)
        receipt['phase'] = 'exact_export_probe_scoring'
        evaluate_exports(backend, exports, keys, parents, snapshots, root, budget, receipt, outcomes)
        receipt.update(status='completed', phase='complete', loading_s=backend.loading_s)
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(root / 'PROBE_OUTCOMES.json', outcomes)
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            receipt.update(status='failed', cleanup_error=repr(error))
            raise
        finally:
            receipt.update(wall_s=time.perf_counter() - started,
                unattempted_conceptual_rows=16384 - len(outcomes))
            write_json(root / 'STATUS.json', receipt)
    return dict(receipt, rows=outcomes)
