"""Three ordinary first-response backwards in one runtime with identical RNG.

This preserves numerical settings and performs no optimizer step or generation.
Compilation/reducer state may warm between repeats; exact RNG control does not by
itself identify which internal implementation causes a difference.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from experiments.prefix_ranking.qualify import RankingObserver, WorkLog, override, run_window
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer
from specforge.training.controller import TrainerCore
from runner import compare_states, named_gradients, parameter_layout, runtime_flags, scientific_state


def reset_window(trainer):
    """Discard only temporary gradients/counters; no model or Adam update occurs."""
    trainer.backend.optimizer.zero_grad()
    trainer.core = TrainerCore(trainer.strategy, trainer.backend, accumulation_steps=trainer.accumulation)
    trainer.global_step = 0
    trainer.presentations = 0
    trainer._anchor_observations.clear()


def run_backward_control(config, ordinary_receipt_path, output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    ordinary = json.loads(Path(ordinary_receipt_path).read_text())
    if len(ordinary['features']) != 128 or len(ordinary['records']) != 256:
        raise ValueError('Use the retained complete 128-response event')
    summary = dict(status='running', plan=dict(forwards=3, backwards=3, optimizer_steps=0, generations=0),
                   repeats=[], config=str(config), ordinary_receipt=str(ordinary_receipt_path))
    log = WorkLog(root)
    log.counts['optimizer_steps_attempted'] = 0
    write_json(root / 'PLAN.json', summary)
    trainer = None
    started = time.perf_counter()
    try:
        trainer = SpecForgeEventTrainer(config, root / 'trainer', objective='osd', ranking_lambda=0.)
        summary['loading_s'] = time.perf_counter() - started
        summary['runtime'] = runtime_flags()
        initial = scientific_state(trainer)
        save_torch(root / 'initial.pt', initial)
        initial_rng = initial['rng']
        torch.cuda.reset_peak_memory_stats()
        original_backward = trainer.backend.backward

        def backward(*args, **kwargs):
            with log.work('backward'):
                return original_backward(*args, **kwargs)

        def forbidden_step(*args, **kwargs):
            log.counts['optimizer_steps_attempted'] += 1
            raise AssertionError('Backward-only control reached an optimizer step')

        for index in range(3):
            reset_window(trainer)
            restore_rng(initial_rng)
            assert_state_equal(capture_rng(), initial_rng, f'repeat_{index}_initial_rng')
            assert_state_equal(trainer.backend.optimizer.state_dict(), initial['optimizer'], 'unchanged_initial_optimizer')
            log.trial, log.presentation = f'repeat_{index}', 0
            model = trainer.bundle.model
            model.begin_event()
            observer = RankingObserver(model, log)
            window = None
            try:
                with observer.installed(), override(trainer.backend, 'backward', backward), \
                     override(trainer.backend.optimizer, 'step', forbidden_step):
                    window = run_window(trainer, ordinary, log)
                    row = next(window)
                    if row['optimizer_stepped'] or trainer.global_step != 0 or trainer.presentations != 1:
                        raise AssertionError('Unexpected first-response accounting')
            finally:
                if window is not None:
                    window.close()
            path = root / f'backward_{index}.pt'
            save_torch(path, dict(gradients=named_gradients(trainer), rng=capture_rng()))
            repeat = dict(index=index, record=row, gradient_path=str(path),
                          parameter_layout=parameter_layout(trainer), objective_work=model.end_event())
            assert_state_equal(trainer.bundle.draft_model.state_dict(), initial['draft'], 'weights_unchanged')
            assert_state_equal(trainer.backend.optimizer.state_dict(), initial['optimizer'], 'adam_unchanged')
            repeat['weights_and_adam_unchanged'] = True
            summary['repeats'].append(repeat)
            write_json(root / 'PROGRESS.json', summary)
            del observer

        # Persist every draw/gradient before comparing; no failure is hidden by
        # abandoning a later comparison after the first numerical disagreement.
        summary['comparisons'] = {}
        for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
            left = torch.load(root / f'backward_{left_index}.pt', map_location='cpu', weights_only=False, mmap=True)
            right = torch.load(root / f'backward_{right_index}.pt', map_location='cpu', weights_only=False, mmap=True)
            result = compare_states(left, right)
            label = f'repeat_{left_index}_vs_{right_index}'
            write_json(root / f'{label}.json', result)
            summary['comparisons'][label] = dict(exact_equal=result['exact_equal'], totals=result['totals'])
            del left, right
        reset_window(trainer)
        restore_rng(initial_rng)
        assert_state_equal(scientific_state(trainer), initial, 'restored_initial_state')
        summary['final_state_restored_exactly'] = True
        assert_state_equal(log.counts['current_training_forward_completed'], 3, 'forward_budget')
        assert_state_equal(log.counts['backward_completed'], 3, 'backward_budget')
        assert_state_equal(log.counts['optimizer_steps_attempted'], 0, 'no_optimizer_steps')
        summary['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        summary['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
        summary['status'] = 'completed'
        return summary
    except BaseException as error:
        summary['status'] = 'failed'
        summary['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        summary['wall_s'] = time.perf_counter() - started
        summary['work_counts'] = dict(log.counts)
        write_json(root / 'RESULT.json', summary)
        if trainer is not None:
            # No partial accumulated gradients or counters survive into any run.
            reset_window(trainer)
            trainer.failed = summary['status'] != 'completed'
            trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ordinary-receipt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run_backward_control(args.config, args.ordinary_receipt, args.output)


if __name__ == '__main__':
    main()
