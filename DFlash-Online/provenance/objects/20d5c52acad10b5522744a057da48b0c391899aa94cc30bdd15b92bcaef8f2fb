"""Optional real-model integration check; never launched by importing the loop.

Four serving requests supply two update events. Three final replays check the
updated, frozen and reloaded drafts on an identical held-out input. The fixture
is infrastructure verification, not a scientific performance experiment.
"""

import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import traceback

from .controller import OnlineController
from .native import NativeDFlashBackend
from .rng import capture_rng
from .storage import write_json
from .training import SpecForgeEventTrainer


PROMPTS = [
    'Write a Python function that returns the sum of the even integers in a list. Include one short example.',
    'Explain why this Python function is incorrect, and provide a corrected version:\n\ndef double_all(xs):\n    for x in xs:\n        x = 2 * x\n    return xs\n',
    'A box contains 18 red and 12 blue marbles. Five red marbles are removed. How many marbles remain? Explain briefly.',
    'Write a Python function that counts each character in a string. Include a short example with repeated characters.',
    'Write a Python function that returns all positive even integers from a list while preserving their order. Include one example.',
]


def assert_state_equal(actual, expected, path='state'):
    """Exact comparison for qualification only; no hashing on the serving path."""
    import numpy as np
    import torch

    if isinstance(expected, torch.Tensor):
        equal = (actual.dtype == expected.dtype and actual.shape == expected.shape
                 and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    elif isinstance(expected, np.ndarray):
        equal = actual.dtype == expected.dtype and np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f'{path}: state keys differ')
        for name, value in expected.items():
            assert_state_equal(actual[name], value, f'{path}.{name}')
        return
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f'{path}: sequence structure differs')
        for index, value in enumerate(expected):
            assert_state_equal(actual[index], value, f'{path}[{index}]')
        return
    else:
        equal = type(actual) is type(expected) and actual == expected
    if not equal:
        raise AssertionError(f'{path}: restored value differs')


def check_publication(backend, trainer, event_id):
    expected = trainer.bundle.draft_model.state_dict()
    assert_state_equal(backend.draft.state_dict(), expected, 'published_draft')
    return {'event_id': event_id, 'tensors': len(expected), 'all_equal': True}


def check_restart(controller, config_path, trainer_root):
    """Release the old trainer, restore its saved state, then compare contents."""
    import torch

    checkpoint = trainer_root / 'training_state.pt'
    # Compare the live state with what was saved, as well as with the restored
    # runtime. Otherwise a faulty save could reproduce itself and pass.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    check_saved_training_state(controller.trainer, saved)
    before = controller.trainer.status()
    objective = controller.trainer.objective
    reduction = controller.trainer.reduction
    controller.trainer.close()
    controller.trainer = None
    gc.collect()
    torch.cuda.empty_cache()

    restored = SpecForgeEventTrainer(
        config_path, trainer_root, resume_state=checkpoint,
        objective=objective, reduction=reduction,
    )
    controller.trainer = restored
    check_saved_training_state(restored, saved)
    assert_state_equal(restored.status(), before, 'training_counters')
    return {'draft_equal': True, 'optimizer_equal': True, 'rng_equal': True, 'counters_equal': True}


def check_saved_training_state(trainer, saved):
    """Qualification-only direct comparison, without another full state copy."""
    assert_state_equal(trainer.bundle.draft_model.state_dict(), saved['draft_state_dict'], 'draft')
    assert_state_equal(trainer.backend.optimizer.state_dict(), saved['backend']['optimizer'], 'optimizer')
    assert_state_equal(capture_rng(), saved['rng'], 'rng')
    assert_state_equal(trainer.global_step, saved['global_step'], 'global_step')
    assert_state_equal(trainer.presentations, saved['presentations'], 'presentations')


def admit_storage(root):
    # Two full states can coexist during atomic replacement, plus both exports,
    # feedback/staging and the project's required 8 GiB free-space reserve.
    required = 2 * 14_681_000_000 + 2 * 2_100_000_000 + 2 * 2**30 + 8 * 2**30
    free = shutil.disk_usage(root).free
    write_json(root / 'STORAGE_ADMISSION.json', {
        'free_bytes': free, 'required_bytes': required, 'reserve_bytes': 8 * 2**30,
    })
    if free < required:
        raise ValueError('Insufficient storage for checkpoint overlap and reserve')


def run(args):
    import torch
    from accelerate.utils import set_seed

    root = Path(args.output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    admit_storage(root)
    set_seed(17)
    torch.cuda.reset_peak_memory_stats()
    write_json(root / 'PLAN.json', {
        'kind': 'infrastructure qualification only',
        'serving_generations': 4, 'evaluation_generations': 3,
        'optimizer_steps': 4, 'events': 2, 'response_presentations': 8,
        'max_new_tokens': 128, 'block_size': 16, 'context_limit': 2048,
        'seed': 17, 'prompts': PROMPTS, 'config_path': str(args.config),
        'target_path': str(args.target), 'draft_path': str(args.draft),
        'training_objective': args.objective,
        'training_reduction': 'response_mean',
    })
    shutil.copyfile(args.config, root / 'config.yaml')
    started = time.perf_counter()
    controller = None
    summary = {'status': 'running', 'serving': [], 'events': [], 'replays': [], 'publication_checks': []}

    try:
        backend = NativeDFlashBackend(args.target, args.draft, context_limit=2048, block_size=16)
        inputs = [backend.render([{'role': 'user', 'content': text}]) for text in PROMPTS]
        write_json(root / 'INPUTS.json', {'prompts': PROMPTS, 'input_ids': inputs})
        trainer_root = root / 'trainer'
        trainer = SpecForgeEventTrainer(args.config, trainer_root, objective=args.objective)
        controller = OnlineController(backend, trainer, root / 'loop', buffer_capacity=2)
        summary['model_loading_s'] = backend.loading_s

        for index, prompt_ids in enumerate(inputs[:4], start=1):
            record = controller.serve(
                prompt_ids, request_id=f'qualification_{index}', session_id='foundation',
                request_index=index, max_new_tokens=128,
            )
            summary['serving'].append(record)
            if index % 2 == 0:
                event_id = f'event_{index // 2}'
                event = controller.update(event_id, count=2, passes=2)
                summary['events'].append(event)
                summary['publication_checks'].append(check_publication(backend, controller.trainer, event_id))
                if index == 2:
                    summary['restart_check'] = check_restart(controller, args.config, trainer_root)
            write_json(root / 'PROGRESS.json', summary)

        reference = backend.generate(
            inputs[4], max_new_tokens=128, request_id='heldout_updated',
            session_id='evaluation', request_index=1, purpose='evaluation',
            save_to=root / 'loop/evaluation/heldout_updated.json',
        )
        summary['replays'].append({'label': 'updated', 'record': reference, 'output_equal': True})
        summary['frozen_activation'] = backend.publish(args.draft)
        summary['replays'].append(controller.evaluate(
            inputs[4], canonical_response_ids=reference['response_ids'],
            label='heldout_frozen', max_new_tokens=128,
        ))
        checkpoint = summary['events'][-1]['training']['checkpoint_path']
        summary['updated_activation'] = backend.publish(checkpoint)
        reloaded = controller.evaluate(
            inputs[4], canonical_response_ids=reference['response_ids'],
            label='heldout_reloaded', max_new_tokens=128,
        )
        summary['replays'].append(reloaded)
        assert_state_equal(reloaded['record']['metadata']['verification_calls'],
                           reference['metadata']['verification_calls'], 'reloaded_verification_calls')
        status = controller.trainer.status()
        assert_state_equal(status['global_step'], 4, 'optimizer_steps')
        assert_state_equal(status['presentations'], 8, 'response_presentations')
        assert_state_equal(controller.buffer.pending(), [], 'pending_feedback')
        summary['trainer_status'] = status
        summary['status'] = 'qualified'
    except BaseException as error:
        # One outer boundary preserves incomplete evidence, including Ctrl-C.
        summary['status'] = 'failed'
        summary['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        raise
    finally:
        summary['wall_s'] = time.perf_counter() - started
        summary['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        summary['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
        summary['free_bytes_after'] = shutil.disk_usage(root).free
        write_json(root / 'RESULT.json', summary)
        if controller is not None and controller.trainer is not None:
            controller.trainer.close()
        print(json.dumps({'status': summary['status'], 'output': str(root)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--objective', choices=('upstream', 'hard_ce', 'projected', 'osd'),
                        default='upstream', help='Training loss; other settings come from --config.')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
