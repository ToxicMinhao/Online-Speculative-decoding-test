"""Two-phase native diagnostic: seal all K4/K5 evidence before loading K6.

One live engine retains compact, immutable parent states between phases. This
avoids regenerating histories after matching and preserves original cache
construction. There is no training or timed serving endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from online_feedback.storage import write_json
from experiments.verification_transition.data import identity, read
from experiments.verification_transition.budget import WorkBudget, BudgetService, BudgetExceeded, connect_budget
from experiments.verification_transition.run import active_owned_processes, stop_owned_groups
from experiments.longer_run_headroom.pairs import neutral_parity
from experiments.lost_repair.run import block_observation, counted

GPU_SECONDS = 10800


class MarginBudget(WorkBudget):
    def check_time(self):
        if self.gpu_seconds >= self.gpu_seconds_limit:
            self._write(dict(kind='budget_stop', reason='gpu_time', gpu_seconds=self.gpu_seconds))
            raise BudgetExceeded('Cumulative occupied GPU time reached the three-hour limit')


def seal_pre_k6(payload, output):
    """Persist the full comparison population before any outcome-model load."""
    path = Path(output) / 'PRE_K6.json'
    if path.exists():
        raise FileExistsError('The pre-outcome comparison seal is immutable')
    write_json(path, payload)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    receipt = dict(status='sealed', files={path.name: identity(path)},
                   sealed_at_unix_s=time.time(), k6_loaded=False)
    write_json(Path(output)/'PRE_K6_SEAL.json', receipt)
    return receipt


def check_seal(output, receipt):
    for name, expected in receipt['files'].items():
        if identity(Path(output)/name) != expected:
            raise ValueError('Pre-K6 labels, margins or pairs changed after sealing')


def validate_parent(record, canonical, archived):
    """Archived uninstrumented evidence is the parity reference, not a new run."""
    fields = ('prompt_ids', 'response_ids', 'finish_reason', 'max_new_tokens', 'stop_token_ids')
    differences = [name for name in fields if record[name] != canonical[name]]
    rows = []
    if archived is not None:
        differences.extend('archive:'+name for name in fields if record[name] != archived[name])
        expected, observed = archived['metadata']['trace'], record['metadata']['trace']
        if len(expected) != len(observed):
            differences.append('archive:verification_calls')
        rows = [neutral_parity(a, b) for a, b in zip(expected, observed)]
        if any(row['status'] != 'passed' for row in rows):
            differences.append('archive:native_blocks')
    return dict(status='passed' if not differences else 'failed', differences=differences,
                archived_trace_available=archived is not None, block_parity=rows)


def validate_commits(block, record):
    start = block['output_start'] + 1
    committed = block['committed_tokens']
    if (committed != record['response_ids'][start:start+len(committed)]
            or block['committed_progress'] != len(committed)):
        raise ValueError('Native committed tokens/progress differ from canonical continuation')


def native_worker(plan, output, connection):
    from .native_worker import MarginBackend
    from .analysis import build_pre_k6, analyze_k6

    budget = connect_budget(read(connection))
    budget.register_process(os.getpid(), os.getpgrp())
    starts, outcomes, states, captures = [], [], {}, {}
    backend = None
    phase, completed = 'pre_k6', 0
    began = time.perf_counter()

    def persist(status, **extra):
        write_json(output/'STARTS.json', starts)
        write_json(output/'OUTCOMES.json', outcomes)
        write_json(output/'STATUS.json', dict(status=status, phase=phase,
            requests_completed=completed, requests_planned=len(plan['requests']),
            starts=len(starts), k6_starts_completed=len(outcomes), **extra))

    try:
        backend = counted(budget, 'model_load', lambda: MarginBackend(
            plan['target_path'], plan['checkpoints']['K5']['export'],
            worker_python=plan['worker_python'], output_root=output/'native',
            budget_connection=read(connection)))
        for item in plan['requests']:
            source_id = item['source_id']
            current = output/'requests'/item['request_id']
            current.mkdir(parents=True)
            record, captured = counted(budget, 'parent_walk',
                lambda: backend.capture(item, plan['checkpoints']['K5']['export'], current),
                source_id=source_id)
            write_json(current/'PARENT.json', record)
            canonical = read(item['canonical_path'])
            archived = read(item['archived_trace_path']) if item.get('archived_trace_path') else None
            parity = validate_parent(record, canonical, archived)
            write_json(current/'PARENT_PARITY.json', parity)
            if parity['status'] != 'passed':
                raise ValueError('Instrumented K5 walk differs from its canonical/native reference')
            trace = record['metadata']['trace']
            if len(captured) != len(trace):
                raise ValueError('Incomplete parent-start capture, including opening start')
            captures[source_id] = dict(record=record, item=item, folder=current)
            request_rows = []
            for ordinal, parent in enumerate(trace):
                observation = block_observation(parent, record)
                row = dict(source_id=source_id, request_id=item['request_id'],
                    start_offset=parent['output_start'], round_ordinal=ordinal+1,
                    a5=observation['accepted'], valid_proposal_count=observation['valid_proposals'],
                    terminal=observation['terminal'], stop_clipped=observation['stop_clipped'],
                    valid=False, margins=[])
                request_rows.append(row)
                states[(source_id, row['start_offset'])] = captured[str(ordinal)]
            starts.extend(request_rows)
            persist('running', active_request=source_id)
            compact = backend.compact_request(item['request_id'])
            write_json(current/'COMPACT_STATES.json', compact)
            for arm in ('K5', 'K4'):
                for ordinal, parent in enumerate(trace):
                    row = request_rows[ordinal]
                    observed = counted(budget, 'neutral_block' if arm == 'K5' else 'historical_block',
                        lambda: backend.one_block(captured[str(ordinal)], plan['checkpoints'][arm]['export']),
                        source_id=source_id, arm=arm, start_offset=row['start_offset'])
                    write_json(current/f'{ordinal:04d}_{arm}.json', observed)
                    block = observed['blocks'][0]
                    validate_commits(block, record)
                    if (block['output_start'] != parent['output_start']
                            or block['anchor_token'] != parent['anchor_token']
                            or block['positions'] != parent['positions']):
                        raise ValueError('Replay differs in canonical parent conditioning')
                    measurement = block_observation(block, record)
                    if arm == 'K5':
                        parity = neutral_parity(parent, block)
                        write_json(current/f'{ordinal:04d}_PARITY.json', parity)
                        if parity['status'] != 'passed':
                            raise ValueError('Neutral K5 proposals, acceptance, commits or clipping differ')
                    else:
                        row['a4'] = measurement['accepted']
                        row['k4_first_error'] = measurement['genuine_rejection']
                        row['valid'] = True
            margin_receipt = backend.request_margins(source_id)
            write_json(current/'MARGINS.json', margin_receipt)
            for row in request_rows:
                row['margins'] = margin_receipt['starts'][str(row['start_offset'])]
            completed += 1
            persist('running')
        qa = backend.margin_qa()
        write_json(output/'MARGIN_QA.json', qa)
        if qa['status'] != 'passed':
            raise ValueError('Native full-vocabulary margin QA failed')
        comparison = build_pre_k6(starts)
        seal = seal_pre_k6(comparison, output)
        phase = 'k6_outcomes'
        persist('running', pair_seal=seal)
        # Only this point permits publication of outcome checkpoint K6.
        for key in comparison['required_k6_starts']:
            source_id, offset = key['source_id'], key['start_offset']
            check_seal(output, seal)
            saved = captures[source_id]
            observed = counted(budget, 'outcome_block',
                lambda: backend.one_block(states[(source_id, offset)], plan['checkpoints']['K6']['export']),
                source_id=source_id, start_offset=offset, pre_k6_sha256=seal['files']['PRE_K6.json']['sha256'])
            write_json(saved['folder']/f'K6_{offset:05d}.json', observed)
            block = observed['blocks'][0]
            validate_commits(block, saved['record'])
            parent = next(row for row in saved['record']['metadata']['trace'] if row['output_start'] == offset)
            if (block['output_start'], block['anchor_token'], block['positions']) != (
                    parent['output_start'], parent['anchor_token'], parent['positions']):
                raise ValueError('K6 replay differs in parent conditioning')
            score = block_observation(block, saved['record'])
            canonical = saved['record']['response_ids'][offset+1:offset+1+score['valid_proposals']]
            outcomes.append(dict(source_id=source_id, start_offset=offset, valid=True,
                a6=score['accepted'], valid_proposal_count=score['valid_proposals'],
                k6_first_error=score['genuine_rejection'],
                direct_correct={str(j): proposed == target for j, (proposed, target) in
                                enumerate(zip(block['draft_tokens'], canonical), 1)}))
            persist('running')
        check_seal(output, seal)
        result = analyze_k6(comparison, outcomes)
        write_json(output/'ANALYSIS.json', result)
        persist('completed')
        write_json(output/'NATIVE_RESULT.json', dict(status='completed', wall_s=time.perf_counter()-began,
            work=budget.summary(), pre_k6_seal=seal, qa=qa))
    except BaseException as error:
        persist('failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()


def preflight(plan, output):
    if not 0 < len(plan['requests']) <= 192:
        raise ValueError('Require the sealed eligible panel of at most192 requests')
    if len({row['source_id'] for row in plan['requests']}) != len(plan['requests']):
        raise ValueError('Duplicate selected dataset request identity')
    if plan['gpu_seconds_limit'] != GPU_SECONDS:
        raise ValueError('Three aggregate occupied GPU-hour ceiling must remain fixed')
    if not output.resolve().is_relative_to('/data/sources/minhao/runs'):
        raise ValueError('Run root outside approved storage')
    source = Path(__file__).resolve().parents[2]
    sources = read(plan['source_manifest_path'])
    if sources['commit'] != plan['execution_commit']:
        raise ValueError('Execution commit differs from source seal')
    for name, expected in sources['files'].items():
        if identity(source/name) != expected:
            raise ValueError('Published source differs: '+name)
    for name, expected in sources['symlinks'].items():
        if str((source/name).readlink()) != expected:
            raise ValueError('Published source symlink differs: '+name)
    files = dict(plan['pinned_files'])
    files.update({str(Path(c['export'])/'model.safetensors'): c['weights_identity']
                  for c in plan['checkpoints'].values()})
    for path, expected in files.items():
        if identity(path) != expected:
            raise ValueError('Pinned input/checkpoint changed: '+path)
    runtime = Path(plan['worker_python']).parent.parent/'lib/python3.12/site-packages/sglang/srt'
    for name, expected in plan['runtime_contract']['native_runtime_sha256'].items():
        actual = identity(runtime/name)
        if actual['sha256'] != expected:
            raise ValueError('Pinned native runtime changed: '+name)
        files[str(runtime/name)] = actual
    python = plan['worker_python']
    environment = dict(os.environ, CONDA_PREFIX=str(Path(python).parent.parent))
    raw = subprocess.check_output([python, '-c',
        'import json,platform,torch,transformers; print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)))'],
        text=True, env=environment)
    versions = json.loads(raw.strip().splitlines()[-1])
    if versions != plan['versions']['serving']:
        raise ValueError('Serving environment versions changed')
    free = shutil.disk_usage(output.parent).free
    if free < plan['storage']['required_free_bytes']:
        raise RuntimeError('Planned peak artifact writes plus reserve unavailable')
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < plan['host_memory']['required_available_bytes']:
        raise RuntimeError('Compact native-state retention plus host reserve unavailable')
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    if jobs.strip():
        raise RuntimeError('Native diagnostic admission requires idle GPUs')
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                         '--format=csv,noheader,nounits'], text=True).splitlines()
    gpu_inventory = [dict(index=int(fields[0]), uuid=fields[1].strip(), free_mib=int(fields[2]))
                     for fields in (line.split(',') for line in gpu_lines)]
    selected = next(row for row in gpu_inventory if row['index'] == plan['physical_gpu_index'])
    if selected['uuid'] != plan['physical_gpu_uuid']:
        raise ValueError('Selected physical GPU identity changed')
    return dict(status='passed', execution_commit=sources['commit'], source_files=len(sources['files']),
                checked_files=files, versions=versions, free_bytes=free, host_available_bytes=available,
                storage=plan['storage'], host_memory=plan['host_memory'], gpu_inventory=gpu_inventory)


def supervise(plan_path, output):
    plan = read(plan_path)
    if output.exists():
        raise FileExistsError('Preserve prior evidence; no automatic retry')
    admission = preflight(plan, output)
    output.mkdir(parents=True)
    write_json(output/'PLAN.json', plan)
    write_json(output/'PREFLIGHT.json', admission)
    count = len(plan['requests'])
    budget = MarginBudget(output/'WORK.jsonl', gpu_seconds=GPU_SECONDS, limits=dict(
        parent_walk=count, neutral_block=count*2048, historical_block=count*2048,
        outcome_block=count*2048, draft_block=count*2048*4+1024))
    service = BudgetService(budget)
    connection = output/'BUDGET_CONNECTION.json'
    write_json(connection, service.connection)
    connection.chmod(0o600)
    proxy = connect_budget(service.connection)
    child, failure = None, None
    began = time.perf_counter()
    proxy.start_gpu(1)
    try:
        command = [sys.executable, '-u', '-m', 'experiments.repair_history_margin.run',
                   '--plan', str(plan_path), '--output', str(output),
                   '--native-worker', '--budget-connection', str(connection)]
        with (output/'native.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while child.poll() is None:
                    proxy.check_time()
                    write_json(output/'BUDGET.json', proxy.summary())
                    time.sleep(2)
            except BaseException:
                write_json(output/'CLEANUP.json', stop_owned_groups(proxy, child))
                raise
        groups = {row['pgid'] for row in proxy.processes()}
        if active_owned_processes(groups):
            write_json(output/'CLEANUP.json', stop_owned_groups(proxy, child))
            if child.returncode == 0:
                raise RuntimeError('Owned native worker survived phase shutdown')
        if child.returncode:
            raise RuntimeError(f'Native phase failed with exit {child.returncode}; evidence preserved')
        for path, expected in admission['checked_files'].items():
            if identity(path) != expected:
                raise ValueError('Frozen input/checkpoint changed during diagnostic: '+path)
        sources = read(plan['source_manifest_path'])
        source = Path(__file__).resolve().parents[2]
        if any(identity(source/name) != expected for name, expected in sources['files'].items()):
            raise ValueError('Scientific source changed during execution')
        write_json(output/'POSTFLIGHT.json', dict(status='passed', inputs_and_checkpoints_unchanged=True,
                                                source_unchanged=True))
    except BaseException as error:
        failure = repr(error)
        status = read(output/'STATUS.json') if (output/'STATUS.json').exists() else {}
        write_json(output/'STATUS.json', dict(status, status='failed', error=failure))
    finally:
        groups = {row['pgid'] for row in proxy.processes()}
        if active_owned_processes(groups):
            write_json(output/'FINAL_CLEANUP.json', stop_owned_groups(proxy, child))
        proxy.end_gpu()
        write_json(output/'BUDGET.json', proxy.summary())
        write_json(output/'RESULT.json', dict(status='failed' if failure else 'completed', error=failure,
            execution_commit=plan['execution_commit'], wall_s=time.perf_counter()-began,
            work=proxy.summary(), optimizer_steps=0, backward_calls=0, timed_endpoints=0,
            physical_gpu_index=plan['physical_gpu_index'], physical_gpu_uuid=plan['physical_gpu_uuid']))
        service.close()
    from .report import report
    report(plan, output)
    return 1 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-worker', action='store_true')
    parser.add_argument('--budget-connection', type=Path)
    args = parser.parse_args()
    if args.native_worker:
        native_worker(read(args.plan), args.output, args.budget_connection)
        return 0
    return supervise(args.plan, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
