"""Retain three one-step diagnostic trials before comparing numerical state.

This does not repair the failed exact qualification or launch U/P training.
Existing DFlash-Online and its original full-pool/anchor schedule remain intact.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.prefix_ranking.qualify import RankingObserver, WorkLog, cpu_copy, override, run_window
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

TRIALS = (("O1", "osd"), ("O2", "osd"), ("U0", "osd_uniform"))
GRADIENT_STAGES = ("after_backward_0", "after_backward_1", "pre_optimizer")


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def tensor_difference(left, right, *, chunk_elements=262144, sample_limit=8):
    """Bounded-memory FP64 discrepancy statistics; no tolerance is introduced."""
    result = dict(left_shape=list(left.shape), right_shape=list(right.shape),
                  left_dtype=str(left.dtype), right_dtype=str(right.dtype))
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(result, comparable=False, exact_equal=False)
    left, right = left.detach().cpu().reshape(-1), right.detach().cpu().reshape(-1)
    unequal = finite_pairs = nonfinite_left = nonfinite_right = 0
    squared_error = squared_left = squared_right = maximum = 0.
    samples = []
    for start in range(0, left.numel(), chunk_elements):
        raw_a = left[start:start + chunk_elements]
        raw_b = right[start:start + chunk_elements]
        different = raw_a != raw_b
        a, b = raw_a.double(), raw_b.double()
        unequal += int(different.sum())
        a_finite, b_finite = torch.isfinite(a), torch.isfinite(b)
        valid = a_finite & b_finite
        finite_pairs += int(valid.sum())
        nonfinite_left += int((~a_finite).sum())
        nonfinite_right += int((~b_finite).sum())
        error = a[valid] - b[valid]
        squared_error += float((error * error).sum())
        squared_left += float((a[valid] * a[valid]).sum())
        squared_right += float((b[valid] * b[valid]).sum())
        if error.numel():
            maximum = max(maximum, float(error.abs().max()))
        if len(samples) < sample_limit:
            for offset in different.nonzero().flatten()[:sample_limit - len(samples)].tolist():
                samples.append(dict(flat_index=start + offset, left=json_number(a[offset]),
                                    right=json_number(b[offset]), error=json_number(a[offset] - b[offset])))
    return dict(result, comparable=True, exact_equal=unequal == 0, coordinates=left.numel(),
                unequal_count=unequal, unequal_fraction=unequal / max(1, left.numel()),
                finite_pairs=finite_pairs, nonfinite_left=nonfinite_left, nonfinite_right=nonfinite_right,
                finite_max_abs_error=maximum, finite_discrepancy_l2=math.sqrt(squared_error),
                finite_left_l2=math.sqrt(squared_left), finite_right_l2=math.sqrt(squared_right), samples=samples)


def flatten_state(value, prefix=""):
    """Keep named tensors and scalar metadata separately, including RNG state."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value.copy())
    if isinstance(value, torch.Tensor):
        return {prefix: value}, {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        items = enumerate(value)
    else:
        return {}, {prefix: value}
    tensors, metadata = {}, {}
    for key, child in items:
        child_tensors, child_metadata = flatten_state(child, f"{prefix}.{key}" if prefix else str(key))
        tensors.update(child_tensors)
        metadata.update(child_metadata)
    return tensors, metadata


def compare_states(left, right):
    left_tensors, left_meta = flatten_state(left)
    right_tensors, right_meta = flatten_state(right)
    missing_left = sorted(set(right_tensors) - set(left_tensors))
    missing_right = sorted(set(left_tensors) - set(right_tensors))
    rows = {name: tensor_difference(left_tensors[name], right_tensors[name])
            for name in sorted(set(left_tensors) & set(right_tensors))}
    metadata_differences = []
    for name in sorted(set(left_meta) | set(right_meta)):
        a, b = left_meta.get(name), right_meta.get(name)
        if name not in left_meta or name not in right_meta or type(a) is not type(b) or a != b:
            metadata_differences.append(dict(name=name, left=repr(a), right=repr(b)))
    equal = not missing_left and not missing_right and not metadata_differences and all(
        row['exact_equal'] for row in rows.values())
    comparable = [row for row in rows.values() if row['comparable']]
    return dict(exact_equal=equal, missing_left=missing_left, missing_right=missing_right,
                metadata_differences=metadata_differences, tensors=rows,
                totals=dict(tensors=len(rows), unequal_tensors=sum(not row['exact_equal'] for row in rows.values()),
                            coordinates=sum(row['coordinates'] for row in comparable),
                            unequal_coordinates=sum(row['unequal_count'] for row in comparable),
                            finite_discrepancy_l2=math.sqrt(sum(row['finite_discrepancy_l2'] ** 2 for row in comparable)),
                            finite_max_abs_error=max((row['finite_max_abs_error'] for row in comparable), default=0.)))


def scientific_state(trainer):
    return dict(draft=cpu_copy(trainer.bundle.draft_model.state_dict()),
                optimizer=cpu_copy(trainer.backend.optimizer.state_dict()), rng=capture_rng(),
                global_step=trainer.global_step, presentations=trainer.presentations,
                accumulation_remainder=trainer.core.accumulation_remainder,
                training_modes={name: module.training for name, module in trainer.bundle.model.named_modules()})


def runtime_flags():
    return dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                matmul_bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                device=torch.cuda.get_device_name() if torch.cuda.is_initialized() else 'CPU fixture')


def parameter_layout(trainer):
    return [dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                 parameter_stride=list(parameter.stride()), gradient_is_none=parameter.grad is None,
                 gradient_stride=None if parameter.grad is None else list(parameter.grad.stride()),
                 gradient_dtype=None if parameter.grad is None else str(parameter.grad.dtype))
            for name, parameter in trainer.bundle.draft_model.named_parameters()]


def named_gradients(trainer):
    return {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in trainer.bundle.draft_model.named_parameters()}


@contextmanager
def capture_gradients(trainer, root, log):
    """Observe completed backwards and normalized, pre-clipping Adam input."""
    original_backward = trainer.backend.backward
    original_step = trainer.backend.optimizer.step
    captures = []

    def persist(stage):
        started = time.perf_counter()
        path = root / f"{stage}.pt"
        save_torch(path, dict(gradients=named_gradients(trainer), rng=capture_rng()))
        row = dict(stage=stage, path=str(path), bytes=path.stat().st_size,
                   persistence_s=time.perf_counter() - started, parameter_layout=parameter_layout(trainer))
        captures.append(row)
        log.record('gradient_snapshot', **row)

    def backward(*args, **kwargs):
        with log.work('backward'):
            result = original_backward(*args, **kwargs)
        persist(f"after_backward_{log.presentation}")
        return result

    def step(*args, **kwargs):
        persist('pre_optimizer')
        with log.work('temporary_optimizer_step'):
            return original_step(*args, **kwargs)

    with override(trainer.backend, 'backward', backward), override(trainer.backend.optimizer, 'step', step):
        yield captures


def export_diagnostic(trainer, trial, root):
    """Use the existing restart schema and exporter; never publish into old O."""
    event = dict(event_id='diagnostic_one_window', diagnostic_only=True,
                 partial_feedback_event=True, response_presentations=2, optimizer_steps=1,
                 start_step=0, end_step=1, records=trial['records'])
    started = time.perf_counter()
    save_torch(trainer.pending_state, trainer._checkpoint_state(event))
    trial['checkpoint_save_s'] = time.perf_counter() - started
    started = time.perf_counter()
    destination = root / 'export'
    trainer._export(destination)
    trial['export_s'] = time.perf_counter() - started
    os.replace(trainer.pending_state, trainer.latest)
    return dict(checkpoint=str(trainer.latest), export=str(destination))


def compare_retained_trials(root, trials):
    """All three trials already exist on disk; inspect every stage, never fail fast."""
    results = {}
    for baseline, other in (('O1', 'O2'), ('O1', 'U0'), ('O2', 'U0')):
        pair = baseline + '_vs_' + other
        comparisons = {}
        for stage in ('initial', *GRADIENT_STAGES, 'final'):
            left = torch.load(root / baseline / f'{stage}.pt', map_location='cpu', weights_only=False, mmap=True)
            right = torch.load(root / other / f'{stage}.pt', map_location='cpu', weights_only=False, mmap=True)
            comparisons[stage] = compare_states(left, right)
            del left, right
            write_json(root / f'{pair}.json', comparisons)
        # Runtime is intentionally not part of the numerical equality test.
        def numerical_rows(label):
            trial = next(row for row in trials if row['label'] == label)
            return [{key: value for key, value in row.items() if key != 'training_s'}
                    for row in trial['records']]
        comparisons['metrics'] = compare_states(
            {'rows': numerical_rows(baseline)}, {'rows': numerical_rows(other)})
        write_json(root / f'{pair}.json', comparisons)
        results[pair] = {stage: dict(exact_equal=value['exact_equal'], totals=value.get('totals'))
                         for stage, value in comparisons.items()}
    return results


def run_trials(config, ordinary_receipt_path, output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    ordinary = json.loads(Path(ordinary_receipt_path).read_text())
    if len(ordinary['features']) != 128 or len(ordinary['records']) != 256:
        raise ValueError('Use the complete retained 128-response O event')
    summary = dict(status='running', config=str(config), ordinary_receipt=str(ordinary_receipt_path),
                   plan=dict(trials=3, presentations=6, optimizer_steps=3, generations=0),
                   process_order=['O1', 'O2', 'U0'],
                   compilation_note='O1 is first in this process; O2 and U0 may reuse process-wide compiled kernels. Ordinary repeat disagreement does not alone identify a kernel cause.',
                   gradient_semantics=dict(after_backward='Accumulated gradients, before TrainerCore normalization',
                                           pre_optimizer='Response-normalized gradients, before clipping and AdamW'),
                   trials=[])
    write_json(root / 'PLAN.json', summary)
    log = WorkLog(root)
    trainer = None
    started = time.perf_counter()
    try:
        for label, objective in TRIALS:
            trial_root = root / label
            log.trial, log.presentation = label, None
            trial = dict(label=label, objective=objective, ranking_lambda=0., status='running', records=[])
            summary['trials'].append(trial)
            load_started = time.perf_counter()
            trainer = SpecForgeEventTrainer(config, trial_root / 'trainer', objective=objective, ranking_lambda=0.)
            trial['loading_s'] = time.perf_counter() - load_started
            trial['runtime'] = runtime_flags()
            trial['initial_parameter_layout'] = parameter_layout(trainer)
            trial['optimizer_recipe'] = trainer.backend.optimizer.recipe
            trial['seed'] = trainer.cfg.training.seed
            assert_state_equal(trial['seed'], 17, 'diagnostic_seed')
            torch.cuda.reset_peak_memory_stats()
            save_torch(trial_root / 'initial.pt', scientific_state(trainer))
            model = trainer.bundle.model
            rng = capture_rng()
            model.begin_event()
            assert_state_equal(capture_rng(), rng, 'begin_event_rng')
            if model.reference is not None or model.ranking_enabled:
                raise AssertionError('Lambda-zero diagnostic entered the active ranking/reference path')
            observer = RankingObserver(model, log)
            with observer.installed(), capture_gradients(trainer, trial_root, log) as captures:
                for row in run_window(trainer, ordinary, log):
                    trial['records'].append(row)
                    write_json(root / 'PROGRESS.json', summary)
            trial['gradient_snapshots'] = captures
            trial['objective_work'] = model.end_event()
            save_torch(trial_root / 'final.pt', scientific_state(trainer))
            trial['artifacts'] = export_diagnostic(trainer, trial, trial_root)
            trial['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            trial['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            trial['status'] = 'retained'
            trainer.close()
            trainer = None
            del model, observer, captures
            gc.collect()
            torch.cuda.empty_cache()
            write_json(root / 'PROGRESS.json', summary)
        assert_state_equal(log.counts['temporary_optimizer_step_completed'], 3, 'diagnostic_step_budget')
        assert_state_equal(log.counts['current_training_forward_completed'], 6, 'diagnostic_forward_budget')
        summary['comparisons'] = compare_retained_trials(root, summary['trials'])
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
            trainer.failed = True
            trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ordinary-receipt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run_trials(args.config, args.ordinary_receipt, args.output)


if __name__ == '__main__':
    main()
