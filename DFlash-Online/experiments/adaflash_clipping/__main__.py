"""Run the approved six disposable steps; never export a serving checkpoint."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.osd_components.prepare import verify_files, verify_native_runtime


def preflight(source, output, execution_commit, source_manifest):
    import torch
    import transformers
    from experiments.ok_coding.training import _export_equal
    from experiments.kw_v1.preflight import hardware_admission

    sealed = source / 'sealed'
    source_pin = json.loads(Path(source_manifest).read_text())
    if source_pin['execution_commit'] != execution_commit:
        raise RuntimeError('Source manifest and requested execution commit differ')
    code_root = Path(__file__).resolve().parents[2]
    verify_files(code_root, source_pin['files'])
    original = json.loads((sealed / 'PLAN.json').read_text())
    initial = source / 'campaign_001/arms/A/trainer/initial_state.pt'
    # Six reconstructed responses, temporary tensor serialization, compiler
    # caches and compact diagnostics. No optimizer/checkpoint exports are made.
    budget = dict(additional_writes_bytes=16 * 2**30, reserve_bytes=8 * 2**30)
    free = shutil.disk_usage(output).free
    if free < sum(budget.values()):
        raise RuntimeError('Insufficient capacity for bounded diagnostic and reserve')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    if available < 80 * 2**30:
        raise RuntimeError('Need 80 GiB available host memory for paired tensor comparisons')
    result = dict(status='running', execution_commit=execution_commit,
                  storage=dict(budget, free_before_bytes=free), host_available_bytes=available,
                  hardware=hardware_admission())
    pins = json.loads((sealed / 'MODEL_PIN.json').read_text())
    for name in ('target', 'draft'):
        verify_files(original[name], pins[name]['files'], exact_files=True)
    result['model_pins_exact'] = True
    result['runtime'] = verify_native_runtime(original['worker_python'],
        json.loads((sealed / 'RUNTIME_PIN.json').read_text()))
    actual = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)
    if actual != json.loads((sealed / 'TRAINING_PIN.json').read_text())['environment']:
        raise RuntimeError('Training runtime differs from the qualified A58 runtime')
    result['training_runtime'] = actual
    state = torch.load(initial, map_location='cpu', weights_only=False)
    if Path(original['configs']['A']).read_text() != state['config_text']:
        raise RuntimeError('Original configuration differs from retained initial state')
    if (state['global_step'] != 0 or state['presentations'] != 0 or state['events']
            or state['backend']['optimizer']['optimizer_state_dict']['state']):
        raise RuntimeError('Retained initialization is not released/fresh/empty')
    _export_equal(state, original['draft'])
    result['initial_state'] = dict(path=str(initial), identity=identity(initial),
                                  released_weights_exact=True, empty_adam=True, counters_zero=True)
    result['status'] = 'passed'
    write_json(output / 'PREFLIGHT.json', result)
    plan = dict(execution_commit=execution_commit, source=str(source),
        config_path=original['configs']['A'], initial_state_path=str(initial),
        reconstruction=dict(target_path=original['target'], draft_path=original['draft'],
                            worker_python=original['worker_python'], serving_gpu='0'),
        budget=dict(independent_batches=3, branches_per_batch=2, optimizer_steps=6,
                    training_presentations=12, post_step_response_forwards=12,
                    startup_response_forwards=2, startup_backwards=2,
                    complete_response_generations=0, serving_exports=0),
        sampling='first six pass-0 presentation records; exact retained anchors, no resampling',
        interpretation='local source-fit and update sensitivity, not endpoint or serving performance')
    write_json(output / 'PLAN.json', plan)
    return plan, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execution-commit', required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    args = parser.parse_args()
    def terminate(_signal, _frame):
        raise TimeoutError('Bounded runner received termination; preserve partial work and close owned runtimes')
    signal.signal(signal.SIGTERM, terminate)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = dict(status='running', execution_commit=args.execution_commit)
    try:
        plan, admission = preflight(args.source, args.output, args.execution_commit, args.source_manifest)
        print('Preflight passed; reconstructing the six sealed responses.', flush=True)
        from .prepare import prepare
        prepared = prepare(plan['reconstruction'], args.source / 'campaign_001', args.output / 'preparation')
        print('Reconstruction passed; starting six independent one-step branches.', flush=True)
        from .run import run
        result = run(plan['config_path'], plan['initial_state_path'], prepared, args.output / 'diagnostic')
        if result['status'] != 'passed':
            raise RuntimeError('Diagnostic did not pass its execution checks')
        unchanged = identity(plan['initial_state_path']) == admission['initial_state']['identity']
        if not unchanged:
            raise RuntimeError('Retained initial artifact changed')
        free = shutil.disk_usage(args.output).free
        if free < admission['storage']['reserve_bytes']:
            raise RuntimeError('Storage reserve violated')
        status.update(status='passed', initial_artifact_unchanged=True, free_after_bytes=free,
                      report=str(args.output / 'diagnostic/RESULT.json'))
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status['wall_s'] = time.perf_counter() - started
        write_json(args.output / 'STATUS.json', status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
