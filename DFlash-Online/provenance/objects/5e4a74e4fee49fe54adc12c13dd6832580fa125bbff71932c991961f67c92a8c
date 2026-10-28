"""One sealed, fixed-checkpoint O/K panel; no training or feedback path."""

import argparse
import json
from pathlib import Path
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.fresh_confirmation.evaluate import evaluate


def validate_plan(plan):
    requests = plan['requests']
    if len(requests) != 64 or list(plan['checkpoints']) != ['O', 'K']:
        raise ValueError('Require the sealed 64-request, O/K-only comparison')
    if plan['selection_seed'] != 20261027 or plan['planned_generations'] != 384:
        raise ValueError('The declared selection seed and execution budget are fixed')
    if plan['freshness_scope'] != 'current_one_event_O_K_study':
        raise ValueError('Require the explicitly approved freshness scope')
    for request in requests:
        if request['max_new_tokens'] != 2048 or len(request['prompt_ids']) + 2048 > 4096:
            raise ValueError('Keep the full prompt and fixed cap within context4096')
        if 'response_ids' in request or 'finish_reason' in request:
            raise ValueError('The new panel must not inherit a historical canonical answer')


def run(root):
    from online_feedback.sglang_backend import SGLangDFlashBackend

    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    validate_plan(plan)
    output = root / 'campaign_001'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    status = dict(status='running', phase='loading_native', planned_generations=384,
                  training_steps=0, feedback_collections=0, extra_generations=0)
    costs, backend = {}, None
    write_json(output / 'STATUS.json', status)
    try:
        backend = SGLangDFlashBackend(
            plan['target'], plan['released_draft'], worker_python=plan['worker_python'],
            output_root=output / 'native', serving_gpu='0', context_limit=4096, block_size=16)
        costs['backend_loading_s'] = backend.loading_s
        status['phase'] = 'fresh_two_arm_evaluation'
        write_json(output / 'STATUS.json', status)
        status['evaluation'] = evaluate(backend, plan['requests'], plan['checkpoints'],
                                        output / 'evaluation', qualification_requests=8)
        status.update(status='completed', phase='finished')
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        cleanup_error = None
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                status['cleanup_error'] = repr(error)
                if status['status'] != 'failed':
                    status.update(status='failed', error_type=type(error).__name__, error=str(error))
                    cleanup_error = error
        costs['campaign_wall_s'] = time.perf_counter() - started
        costs['accounting'] = (
            'Request latency excludes checkpoint loading/switching. Three timing repeats are '
            'benchmark work; calls use one validated trajectory per request. Native initialization '
            'is separate and performs no extra generation warmups. No training or feedback.')
        write_json(output / 'COSTS.json', costs)
        write_json(output / 'STATUS.json', status)
        if cleanup_error is not None:
            raise cleanup_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    run(args.root)
