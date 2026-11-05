"""Seven disposable qualification windows, three events, 768 native replays."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import traceback
from types import SimpleNamespace

from online_feedback.storage import write_json
from experiments.osd_one_event.protocol import validate_manifest
from experiments.prefix_ranking.protocol import load_ordinary_counts
from experiments.first_error.campaign import OBJECTIVES, evaluate, require_qualification, train_arm
import qualify_first_error


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_scope(manifest, ordinary):
    """Seal the whole event/panel before any model construction or updates."""
    validate_manifest(manifest)
    expected = dict(status='completed', objective='osd', passes=2,
                    optimizer_steps=128, response_presentations=256, start_step=0, end_step=128)
    if any(ordinary.get(key) != value for key, value in expected.items()):
        raise ValueError('Retained O must be the completed original one-event control')
    features, records = ordinary['features'], ordinary['records']
    identities = [row['record_id'] for row in features]
    if (len(features) != 128 or len(set(identities)) != 128 or len(records) != 256
            or len({row['path'] for row in features}) != 128
            or set(identities) != {row['request_id'] for row in manifest['train_requests']}):
        raise ValueError('Require the exact128 distinct retained feedback identities and256 presentations')
    for pass_index in range(2):
        rows = records[pass_index * 128:(pass_index + 1) * 128]
        if (any(row['pass'] != pass_index or len(row['sampled_anchors']) != 1 for row in rows)
                or Counter(row['record_id'] for row in rows) != Counter(identities)):
            raise ValueError('Every pass must contain each feedback response once with retained anchors')
    for index, row in enumerate(records):
        if (row['optimizer_stepped'] != bool(index % 2)
                or row['optimizer_step_after'] != (index + 1) // 2):
            raise ValueError('Retain the original two-response accumulation and128 steps')


def validate_qualification(result, args):
    require_qualification(result)
    for path in (args.config, args.ordinary_receipt, args.pinned_initial,
                 args.controlled_runner, args.diagnostic_runner, qualify_first_error.__file__):
        if result.get('input_sha256', {}).get(str(path)) != digest(path):
            raise ValueError('GPU qualification input binding differs: ' + str(path))
    expected = dict(response_presentations=14, temporary_optimizer_step_completed=7,
                    backward_completed=14, current_training_forward_completed=14)
    if (result.get('generations') != 0 or result.get('main_optimizer_steps') != 0
            or any(result.get('work_counts', {}).get(key) != count for key, count in expected.items())):
        raise ValueError('Qualification work does not match the declared seven discarded windows')
    counts = result['work_counts']
    for key, count in expected.items():
        if key.endswith('_completed') and counts.get(key.replace('_completed', '_attempted')) != count:
            raise ValueError('Qualification includes incomplete or extra attempts: ' + key)
    if any(counts.get(key, 0) != 0 for key in (
            'reference_training_forward_attempted', 'reference_training_forward_completed',
            'reference_draft_forwards_observed')):
        raise ValueError('C/K/A qualification must not use a reference model')


def update_work(progress, root):
    """Include completed and interrupted events, without treating unknown work as zero."""
    states = {}
    for arm in OBJECTIVES:
        path = Path(root) / f'{arm}_CLOSED.json'
        if path.is_file():
            states[arm] = json.loads(path.read_text())
    progress['arm_counters'] = states
    progress['main_optimizer_steps'] = sum(row['global_step'] for row in states.values())
    progress['main_response_presentations'] = sum(row['presentations'] for row in states.values())
    phase = progress.get('phase', '')
    active = phase.removeprefix('training_') if phase.startswith('training_') else None
    unknown = active if active in OBJECTIVES and active not in states else None
    progress['unknown_arm_counters'] = unknown
    progress['main_steps_remaining'] = None if unknown else 384 - progress['main_optimizer_steps']


def run(args):
    from online_feedback.qualify_sglang import check_recipe
    from online_feedback.recipe import load_event_config
    from online_feedback.sglang_backend import SGLangDFlashBackend

    started = time.perf_counter()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text())
    ordinary = json.loads(Path(args.ordinary_receipt).read_text())
    validate_scope(manifest, ordinary)
    config, _ = load_event_config(args.config)
    check_recipe(args.config)
    draft, target = config.model.draft_checkpoint_path, config.model.target_model_path
    counts = load_ordinary_counts(args.ordinary_measurements, manifest['eval_requests'])
    shutil.copy2(args.config, root / 'config.yaml')
    write_json(root / 'MANIFEST.json', manifest)
    progress = dict(status='running', phase='gpu_qualification', main_optimizer_steps=0,
                    main_response_presentations=0, main_steps_remaining=384,
                    new_feedback_generations=0, new_teacher_preparation_forwards=0,
                    planned_evaluation_generations=768, automatic_retry=False)
    costs, backend = {}, None
    write_json(root / 'STATUS.json', progress)
    try:
        qualification_started = time.perf_counter()
        try:
            qualification = qualify_first_error.run_qualification(
                args.config, args.ordinary_receipt, args.pinned_initial, root / 'qualification',
                controlled_runner=args.controlled_runner, diagnostic_runner=args.diagnostic_runner)
        finally:
            costs['qualification_wall_s'] = time.perf_counter() - qualification_started
        costs['qualification_driver_wall_s'] = qualification['wall_s']
        validate_qualification(qualification, args)
        progress['qualification_status'] = 'passed'
        startup = SimpleNamespace(prepare=lambda trainer, destination:
            qualify_first_error.prepare_main(trainer, args.pinned_initial, destination),
            save_failure=qualify_first_error.save_failure)
        print('Seven-window qualification passed; starting independent C/K/A events.', flush=True)
        checkpoints = {'O': ordinary['checkpoint_path']}
        for arm in OBJECTIVES:
            progress['phase'] = 'training_' + arm
            write_json(root / 'STATUS.json', progress)
            checkpoints[arm], costs[arm] = train_arm(args.config, ordinary, root, arm, startup=startup)
            update_work(progress, root)
            if progress['arm_counters'][arm].get('closed') is not True:
                raise RuntimeError('Training runtime did not close: ' + arm)
            write_json(root / 'COSTS.json', costs)
        if progress['main_optimizer_steps'] != 384 or progress['main_response_presentations'] != 768:
            raise RuntimeError('The three main events did not finish their exact budget')
        progress['phase'] = 'loading_native'
        write_json(root / 'STATUS.json', progress)
        backend = SGLangDFlashBackend(target, draft, worker_python=args.worker_python,
            output_root=root / 'native', serving_gpu=args.serving_gpu, context_limit=4096, block_size=16)
        costs['backend_loading_s'] = backend.loading_s
        progress['phase'] = 'capture_disabled_evaluation'
        write_json(root / 'STATUS.json', progress)
        print('All training closed; beginning768 O/C/K/A capture-disabled replays.', flush=True)
        progress['evaluation'] = evaluate(backend, manifest['eval_requests'], checkpoints,
                                         root / 'evaluation', ordinary_counts=counts)
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            backend.close()
        update_work(progress, root)
        costs['actual_campaign_wall_s'] = time.perf_counter() - started
        costs['free_bytes_after'] = shutil.disk_usage(root).free
        costs['accounting'] = (
            'Each event_wall_s includes training, selection, teacher-head work when needed, persistence '
            'and export; do not add nested timers twice. Loading and fresh-state checks are separate. '
            'Qualification is discarded diagnostic work and supplies no serving checkpoint. Reused '
            'feedback/features have zero new collection/preparation cost, not zero historical cost. '
            'Request latency excludes training, loading and activation. Benchmark repeats/switches '
            'are diagnostic work, not deployment costs. No concurrent training or timed capture.')
        write_json(root / 'COSTS.json', costs)
        write_json(root / 'STATUS.json', progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'ordinary-receipt', 'ordinary-measurements', 'worker-python',
                 'output', 'controlled-runner', 'diagnostic-runner', 'pinned-initial'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--serving-gpu', default='0')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
