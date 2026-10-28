"""One bounded R qualification/event, then fresh F/O/P/R checkpoint replays."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.prefix_ranking.run import check_training, exposure
from experiments.prefix_ranking.evaluate import evaluate
from experiments.prefix_ranking.protocol import load_ordinary_counts
import qualify_r

ARMS = ('F', 'O', 'P', 'R')
COMPARISONS = (('R', 'O'), ('R', 'P'), ('R', 'F'), ('O', 'F'), ('P', 'O'))
DIAGNOSTIC_SUMS = (
    'kl_loss_numerator', 'auxiliary_loss_numerator', 'loss_denominator',
    'protected_active_positions', 'protected_positions',
    'reference_correct_positions', 'reference_correct_current_incorrect_positions',
)


def diagnostic_totals(receipt):
    """Each completed accumulation window counts its responses exactly once."""
    rows = [row for row in receipt['records'] if row['optimizer_stepped']]
    sums = {key: sum(row['metrics'][key] for row in rows) for key in DIAGNOSTIC_SUMS}
    ratios = {}
    for label, numerator, denominator in (
        ('protected_activation_rate', 'protected_active_positions', 'protected_positions'),
        ('reference_correct_to_incorrect_rate', 'reference_correct_current_incorrect_positions',
         'reference_correct_positions'),
        ('pooled_weighted_forward_kl', 'kl_loss_numerator', 'loss_denominator'),
        ('pooled_weighted_auxiliary_loss', 'auxiliary_loss_numerator', 'loss_denominator'),
    ):
        ratios[label] = sums[numerator] / sums[denominator] if sums[denominator] else None
    return {'boundary_rows': len(rows), 'sums': sums, 'ratios': ratios,
            'note': 'Rates count sampled occurrences; KL/auxiliary ratios pool weighted denominators. Optimization remains response mean.'}


def train_r(config, ordinary, root, *, pinned_initial):
    import torch
    from online_feedback.training import SpecForgeEventTrainer

    root = Path(root)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = SpecForgeEventTrainer(config, root / 'R', objective='osd_reference_margin',
                                   ranking_lambda=1.0, reduction='response_mean')
    loading_s = time.perf_counter() - started
    try:
        initialization = qualify_r.prepare_main(trainer, pinned_initial, root / 'R/initialization')
        initial = trainer.status()
        if (initial['global_step'] or initial['presentations'] or initial['completed_events']
                or initial['accumulation_remainder']):
            raise RuntimeError('R must start from released weights and fresh optimizer/counters')
        write_json(root / 'R_INITIAL.json', {**initial, 'controlled_initialization': initialization})
        started = time.perf_counter()
        receipt = trainer.train_event('event_001', [Path(row['path']) for row in ordinary['features']],
                                      passes=2, presentation_schedule=ordinary)
        event_wall_s = time.perf_counter() - started
        check_training(receipt, trainer.status(), 'osd_reference_margin')
        costs = {
            'loading_s': loading_s, 'initialization_control_s': initialization['wall_s'],
            'event_wall_s': event_wall_s, **receipt['costs'],
            'objective_work': receipt['objective_work'], 'exposure': exposure(receipt),
            'diagnostics': diagnostic_totals(receipt),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'teacher_preparation_s': 0.0, 'new_feedback_generations': 0,
            'reference_cost_is_nested_in_event_wall': True,
        }
        write_json(root / 'R_COSTS.json', costs)
        return receipt['checkpoint_path'], costs
    except BaseException as error:
        trainer.failed = True
        failure = {'error': repr(error), 'traceback': traceback.format_exc(), 'status': trainer.status()}
        try:
            qualify_r.save_failure(trainer, root / 'R/failure_state.pt')
        except BaseException as capture_error:
            failure['capture_error'] = repr(capture_error)
        write_json(root / 'R_FAILURE.json', failure)
        raise
    finally:
        trainer.close()
        write_json(root / 'R_CLOSED.json', trainer.status())


def run(args):
    from online_feedback.qualify_sglang import check_recipe
    from online_feedback.recipe import load_event_config
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.osd_one_event.protocol import validate_manifest

    started = time.perf_counter()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    ordinary = json.loads(Path(args.ordinary_receipt).read_text())
    previous_p = json.loads(Path(args.prefix_receipt).read_text())
    config, _ = load_event_config(args.config)
    check_recipe(args.config)
    draft, target = config.model.draft_checkpoint_path, config.model.target_model_path
    shutil.copy2(args.config, root / 'config.yaml')
    write_json(root / 'MANIFEST.json', manifest)
    counts = load_ordinary_counts(args.ordinary_measurements, manifest['eval_requests'])
    progress = {'status': 'running', 'phase': 'gpu_qualification', 'main_optimizer_steps': 0,
                'main_response_presentations': 0, 'new_feedback_generations': 0,
                'new_teacher_preparation_forwards': 0, 'automatic_retry': False}
    costs, backend = {}, None
    write_json(root / 'STATUS.json', progress)
    try:
        qualification = qualify_r.run_qualification(args.config, args.ordinary_receipt,
            args.pinned_initial, root / 'qualification',
            controlled_runner=args.controlled_runner, diagnostic_runner=args.diagnostic_runner)
        if qualification['status'] != 'passed':
            raise RuntimeError('R GPU qualification did not pass')
        progress['qualification'] = qualification
        costs['qualification_wall_s'] = qualification['wall_s']
        progress['phase'] = 'training_R'
        write_json(root / 'STATUS.json', progress)
        print('R correctness qualification passed. Starting one fresh 128-step R event.', flush=True)
        checkpoint, costs['R'] = train_r(args.config, ordinary, root, pinned_initial=args.pinned_initial)
        progress['main_optimizer_steps'], progress['main_response_presentations'] = 128, 256
        write_json(root / 'COSTS.json', costs)
        progress['phase'] = 'loading_native'
        write_json(root / 'STATUS.json', progress)
        checkpoints = {'F': str(draft), 'O': ordinary['checkpoint_path'],
                       'P': previous_p['checkpoint_path'], 'R': checkpoint}
        backend = SGLangDFlashBackend(target, draft, worker_python=args.worker_python,
            output_root=root / 'native', serving_gpu=args.serving_gpu, context_limit=4096, block_size=16)
        costs['backend_loading_s'] = backend.loading_s
        progress['phase'] = 'capture_disabled_evaluation'
        write_json(root / 'STATUS.json', progress)
        print('All training closed. Starting 768 fresh F/O/P/R replays; no transfers during timing.', flush=True)
        progress['evaluation'] = evaluate(backend, manifest['eval_requests'], checkpoints,
            root / 'evaluation', ordinary_counts=counts, arms=ARMS, comparisons=COMPARISONS)
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
        if (root / 'R_CLOSED.json').is_file():
            state = json.loads((root / 'R_CLOSED.json').read_text())
            progress['main_optimizer_steps'] = state['global_step']
            progress['main_response_presentations'] = state['presentations']
        raise
    finally:
        if backend is not None:
            backend.close()
        costs['actual_campaign_wall_s'] = time.perf_counter() - started
        costs['free_bytes_after'] = shutil.disk_usage(root).free
        costs['accounting'] = (
            'R event_wall_s includes reference, gradients, checkpoint persistence and export; '
            'do not add nested timers twice. Loading and fresh-state validation are separate. '
            'All four qualification trials, including the discarded process warmup, are diagnostic costs. '
            'No extra warmup or diagnostic weights enter main training. Reused feedback has zero new '
            'collection/preparation cost, not zero historical cost. Fresh evaluation latency excludes '
            'training/loading/activation; benchmark-only switches and repeats are not deployment costs.')
        write_json(root / 'COSTS.json', costs)
        write_json(root / 'STATUS.json', progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'ordinary-receipt', 'ordinary-measurements', 'prefix-receipt',
                 'worker-python', 'output', 'controlled-runner', 'diagnostic-runner', 'pinned-initial'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--serving-gpu', default='0')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
