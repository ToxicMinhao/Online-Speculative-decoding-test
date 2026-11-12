"""Bounded frozen census, then scheduled read-only native training exposure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import traceback

from online_feedback.storage import write_json
from experiments.verification_transition.data import identity, read
from experiments.verification_transition.budget import WorkBudget, BudgetService, connect_budget
from experiments.longer_run_headroom.pairs import neutral_parity
from experiments.longer_run_headroom.census import classify, validate_walk
from experiments.verification_transition.run import active_owned_processes, stop_owned_groups

GPU_SECONDS = 4 * 3600


def counted(budget, kind, function, **metadata):
    ticket = budget.attempt(kind, **metadata)
    try:
        result = function()
    except BaseException as error:
        budget.fail(ticket, repr(error))
        raise
    budget.complete(ticket)
    return result


def block_observation(block, record):
    start = block['output_start']
    labels = record['response_ids'][start + 1:]
    scored = classify(block['draft_tokens'], labels, block['anchor_token'], record['stop_token_ids'])
    accepted = block['accepted_draft_tokens']
    if accepted > scored['valid_slots'] or block['draft_tokens'][:accepted] != labels[:accepted]:
        raise ValueError('Native accepted prefix differs from the saved canonical response')
    genuine = scored['j'] is not None and scored['j'] == accepted + 1
    return dict(accepted=accepted, valid_proposals=scored['valid_slots'],
                genuine_rejection=genuine, first_canonical_mismatch=scored['j'],
                terminal=block['finished'], stop_clipped=block['stop_clipped'])


def native_worker(plan, output, connection):
    from .native_worker import CensusBackend
    budget = connect_budget(read(connection))
    budget.register_process(os.getpid(), os.getpgrp())
    folder = output / 'census'
    folder.mkdir()
    rows, features = [], []
    backend = None
    completed = 0
    began = time.perf_counter()

    def persist(status, **extra):
        write_json(folder / 'CENSUS.json', rows)
        write_json(folder / 'FEATURES.json', dict(feature_paths=features))
        write_json(output / 'STATUS.json', dict(status=status, requests_completed=completed,
            requests_planned=128, census_complete=completed == 128,
            training_complete=False, natural_starts_captured=len(rows), **extra))

    try:
        backend = counted(budget, 'model_load', lambda: CensusBackend(
            plan['target_path'], plan['checkpoints']['K6']['export'],
            worker_python=plan['worker_python'], output_root=output/'native',
            budget_connection=read(connection)))
        for item in plan['requests']:
            request_id = item['request_id']
            current = folder / request_id
            current.mkdir()
            record, states = counted(budget, 'feedback_capture',
                lambda: backend.capture(item, plan['checkpoints']['K6']['export'], current),
                request_id=request_id)
            validate_walk(record, dict(request=item, canonical=record, request_id=request_id))
            if len(states) != len(record['metadata']['trace']):
                raise ValueError('Captured state count differs from all natural starts')
            offset = len(rows)
            for ordinal, block in enumerate(record['metadata']['trace']):
                observation = block_observation(block, record)
                rows.append(dict(request_id=request_id, canonical_start=block['output_start'],
                    round_ordinal=ordinal + 1, a4=None, a5=None, a6=observation['accepted'],
                    genuine_k6_rejection=observation['genuine_rejection'],
                    valid_proposals=observation['valid_proposals'],
                    terminal=observation['terminal'], stop_clipped=observation['stop_clipped'],
                    parity_valid=False))
            persist('census_running', active_request=request_id)
            # Prepare from this capture's committed rows. No answers regenerated.
            preparation = counted(budget, 'teacher_preparation',
                lambda: backend.backend.prepare_feedback(record, current/'features.pt'),
                request_id=request_id)
            write_json(current/'PREPARATION.json', preparation)
            if preparation['status'] != 'prepared':
                raise ValueError('Scheduled response has no eligible native training anchors')
            features.append(preparation['feature_path'])
            for arm in ('K6', 'K4', 'K5'):
                for ordinal, parent in enumerate(record['metadata']['trace']):
                    row = rows[offset + ordinal]
                    if str(ordinal) not in states:
                        raise ValueError('Missing natural-start snapshot, including opening start')
                    observed = counted(budget, 'neutral_block' if arm == 'K6' else 'historical_block',
                        lambda: backend.one_block(states[str(ordinal)], plan['checkpoints'][arm]['export']),
                        request_id=request_id, arm=arm, round_ordinal=ordinal + 1)
                    write_json(current/f'{ordinal:04d}_{arm}.json', observed)
                    block = observed['blocks'][0]
                    if (block['output_start'] != parent['output_start']
                            or block['anchor_token'] != parent['anchor_token']
                            or block['positions'] != parent['positions']):
                        raise ValueError('Historical block differs in native conditioning/start')
                    measurement = block_observation(block, record)
                    if arm == 'K6':
                        parity = neutral_parity(parent, block)
                        write_json(current/f'{ordinal:04d}_PARITY.json', parity)
                        row['parity_valid'] = parity['status'] == 'passed'
                        if not row['parity_valid']:
                            raise ValueError('Neutral K+6 proposals/acceptance/commits differ')
                    else:
                        row['a' + arm[-1]] = measurement['accepted']
                        row['genuine_k' + arm[-1] + '_rejection'] = measurement['genuine_rejection']
                persist('census_running', active_request=request_id, completed_arm=arm)
            backend.release_request()
            completed += 1
            persist('census_running')
        persist('census_completed')
        write_json(folder/'RESULT.json', dict(status='completed', requests=completed,
            natural_starts=len(rows), wall_s=time.perf_counter()-began, work=budget.summary()))
    except BaseException as error:
        persist('failed', failure_phase='census', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()


def preflight(plan, output):
    if len(plan['requests']) != 128 or len({r['request_id'] for r in plan['requests']}) != 128:
        raise ValueError('Require exactly the original event007 batch')
    if not output.resolve().is_relative_to('/data/sources/minhao/runs'):
        raise ValueError('Run root is outside approved storage')
    source = Path(__file__).resolve().parents[2]
    sources = read(plan['source_manifest_path'])
    for name, expected in sources['files'].items():
        if identity(source/name) != expected:
            raise ValueError('Published source differs: ' + name)
    for name, expected in sources['symlinks'].items():
        if str((source/name).readlink()) != expected:
            raise ValueError('Published source symlink differs: ' + name)
    files = dict(plan['pinned_files'])
    files.update({str(Path(c['export'])/'model.safetensors'): c['weights_identity']
                  for c in plan['checkpoints'].values()})
    for path, expected in files.items():
        if identity(path) != expected:
            raise ValueError('Pinned input/checkpoint changed: ' + path)
    runtime_root = Path(plan['worker_python']).parent.parent / 'lib' / 'python3.12' / 'site-packages/sglang/srt'
    for name, expected in plan['runtime_contract']['native_runtime_sha256'].items():
        path = runtime_root/name
        actual = identity(path)
        if actual['sha256'] != expected:
            raise ValueError('Pinned installed native runtime changed: ' + str(path))
        files[str(path)] = actual
    versions = {}
    for name, python in [('serving', plan['worker_python']), ('training', plan['training_python'])]:
        environment = dict(os.environ)
        environment['CONDA_PREFIX'] = str(Path(python).parent.parent)
        probe = subprocess.check_output([python, '-c',
            'import json,platform,torch,transformers; print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)))'],
            text=True, env=environment)
        versions[name] = json.loads(probe.strip().splitlines()[-1])
        if versions[name] != plan['versions'][name]:
            raise ValueError('Pinned environment versions changed: ' + name)
    free = shutil.disk_usage(output.parent).free
    if free < plan['storage']['required_free_bytes']:
        raise RuntimeError('Peak artifact allowance plus reserve is unavailable')
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    if jobs.strip():
        raise RuntimeError('Selected diagnostic requires idle GPUs at admission')
    return dict(status='passed', execution_commit=sources['commit'], source_files=len(sources['files']),
                checked_files=files, versions=versions, free_bytes=free, storage=plan['storage'])


def supervise(plan_path, output):
    plan = read(plan_path)
    if output.exists():
        raise FileExistsError('Preserve previous evidence; no automatic retry')
    admission = preflight(plan, output)
    output.mkdir(parents=True)
    write_json(output/'PLAN.json', plan)
    write_json(output/'PREFLIGHT.json', admission)
    budget = WorkBudget(output/'WORK.jsonl', gpu_seconds=GPU_SECONDS, limits=dict(
        feedback_capture=128, teacher_preparation=128, training_forward=256,
        neutral_block=128*2048, historical_block=2*128*2048, draft_block=1100000))
    service = BudgetService(budget)
    connection = output/'BUDGET_CONNECTION.json'
    write_json(connection, service.connection)
    connection.chmod(0o600)
    proxy = connect_budget(service.connection)
    child = None
    phases, failure = {}, None
    began = time.perf_counter()
    proxy.start_gpu(1)
    try:
        commands = [
            ('census', [sys.executable, '-u', '-m', 'experiments.lost_repair.run',
                       '--plan', str(plan_path), '--output', str(output),
                       '--native-worker', '--budget-connection', str(connection)]),
            ('training', [plan['training_python'], '-u', '-m', 'experiments.lost_repair.training_probe',
                          '--features', str(output/'census/FEATURES.json'),
                          '--config', plan['config_path'], '--recovery', plan['recovery_path'],
                          '--output', str(output/'training'), '--budget-connection', str(connection)]),
        ]
        for name, command in commands:
            start = time.perf_counter()
            environment = dict(os.environ)
            environment['CONDA_PREFIX'] = str(Path(command[0]).parent.parent)
            environment['PATH'] = str(Path(command[0]).parent) + os.pathsep + environment['PATH']
            with (output/f'{name}.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=environment, start_new_session=True)
                try:
                    while child.poll() is None:
                        proxy.check_time()
                        write_json(output/'BUDGET.json', proxy.summary())
                        time.sleep(2)
                except BaseException:
                    write_json(output/f'{name}_CLEANUP.json', stop_owned_groups(proxy, child))
                    raise
            phases[name] = dict(wall_s=time.perf_counter()-start, exit_code=child.returncode)
            groups = {row['pgid'] for row in proxy.processes()}
            if active_owned_processes(groups):
                write_json(output/f'{name}_CLEANUP.json', stop_owned_groups(proxy, child))
                if child.returncode == 0:
                    raise RuntimeError('Owned worker survived successful phase shutdown')
            if child.returncode:
                raise RuntimeError(f'{name} failed with exit {child.returncode}; preserve log and partial evidence')
        for path, expected in admission['checked_files'].items():
            if identity(path) != expected:
                raise ValueError('Checkpoint/input changed during frozen diagnostic: ' + path)
        sources = read(plan['source_manifest_path'])
        source = Path(__file__).resolve().parents[2]
        if any(identity(source/name) != expected for name, expected in sources['files'].items()):
            raise ValueError('Scientific source changed during execution')
        write_json(output/'POSTFLIGHT.json', dict(status='passed', inputs_and_checkpoints_unchanged=True, source_unchanged=True))
        write_json(output/'STATUS.json', dict(status='completed', requests_completed=128,
            census_complete=True, training_complete=True, optimizer_steps=0, backward_calls=0))
    except BaseException as error:
        failure = repr(error)
        status = read(output/'STATUS.json') if (output/'STATUS.json').exists() else {}
        write_json(output/'STATUS.json', dict(status, status='failed', error=failure))
    finally:
        groups = {row['pgid'] for row in proxy.processes()}
        if active_owned_processes(groups):
            write_json(output/'FINAL_CLEANUP.json', stop_owned_groups(proxy, child))
        proxy.end_gpu()
        work = proxy.summary()
        write_json(output/'BUDGET.json', work)
        write_json(output/'RESULT.json', dict(status='failed' if failure else 'completed',
            error=failure, execution_commit=plan['execution_commit'],
            wall_s=time.perf_counter()-began, phase_costs=phases, work=work,
            optimizer_steps=0, backward_calls=0, timed_endpoints=0))
        service.close()
    from .report import report
    report(plan, output)
    return 1 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--native-worker', action='store_true')
    parser.add_argument('--budget-connection', type=Path)
    args = parser.parse_args()
    if args.native_worker:
        native_worker(read(args.plan), args.output, args.budget_connection)
        return 0
    return supervise(args.plan, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
