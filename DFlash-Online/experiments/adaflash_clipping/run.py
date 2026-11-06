"""Three fixed batches, two isolated clipping rules, one native Adam step each.

Full gradients and displacements live only in host memory until paired comparison.
The durable outputs contain scalar statistics and compact block predictions. No
updated model is published, and these source-fit probes are not serving results.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import traceback

import torch

from experiments.adaflash_clipping.loss import ClippingComparisonLoss, MODES
from experiments.ok_continuation.resume import differences
from experiments.prefix_guard.qualify import observe_forwards, pin_retained_graph_runtime
from experiments.prefix_ranking.qualify import WorkLog, override
from online_feedback.prefix_guard import prefix_changes, preserved_probe_runtime
from online_feedback.ranking import fixed_anchors
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer, ranking_eos_token_ids


PLAN = dict(batches=3, branches_per_batch=2, optimizer_steps=6,
            training_presentations=12, post_response_forwards=12,
            core_response_forwards=24, core_backwards=12,
            startup_response_forwards=2, startup_backwards=2,
            startup_optimizer_steps=0, generations=0, exports=0)


def serializable(value):
    """Keep nonfinite evidence readable while writing strict JSON."""
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


class ObservedLoss:
    """Capture native proposals once; backward recomputation adds no samples."""
    def __init__(self, mode, chunk_blocks):
        self.loss = ClippingComparisonLoss(mode, chunk_blocks=chunk_blocks)
        self.collecting = False
        self.responses = []

    def prepare_response(self, weights):
        self.loss.prepare_response(weights)
        self.collecting = True
        self.chunks = []

    def chunk_terms(self, logits, teacher, labels, weights):
        if self.collecting:
            self.chunks.append(tuple(tensor.detach().cpu().clone()
                for tensor in (logits.argmax(-1), labels, weights)))
        return self.loss.chunk_terms(logits, teacher, labels, weights)

    def response_metrics(self, terms, denominator):
        sums, ratios = self.loss.response_metrics(terms, denominator)
        self.collecting = False
        predictions, labels, weights = (torch.cat(items, dim=1)
                                       for items in zip(*self.chunks))
        self.responses.append(dict(predictions=predictions, labels=labels, weights=weights,
            metrics={key: value.detach().cpu().clone() for key, value in sums.items()}))
        self.chunks = []
        return sums, ratios


def paired_tensor_statistics(left, right):
    """FP64 reductions of native gradients or FP32 realized displacements."""
    rows = []
    totals = dict(coordinates=0, different=0, left_nonzero=0, right_nonzero=0,
                  left_squared=0., right_squared=0., difference_squared=0., dot=0.)
    if left.keys() != right.keys():
        raise ValueError("Paired trainable parameter names differ")
    for name, a in left.items():
        b = right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Paired parameter layout differs: {name}")
        row = dict(name=name, shape=list(a.shape), dtype=str(a.dtype), **dict.fromkeys(totals, 0))
        x, y = a.reshape(-1), b.reshape(-1)
        for start in range(0, x.numel(), 262144):
            u, v = x[start:start + 262144].double(), y[start:start + 262144].double()
            if not bool(torch.isfinite(u).all() & torch.isfinite(v).all()):
                raise ValueError(f"Nonfinite gradient/displacement in {name}")
            row['coordinates'] += u.numel()
            row['different'] += int((u != v).sum())
            row['left_nonzero'] += int((u != 0).sum())
            row['right_nonzero'] += int((v != 0).sum())
            row['left_squared'] += float(u.square().sum())
            row['right_squared'] += float(v.square().sum())
            row['difference_squared'] += float((u - v).square().sum())
            row['dot'] += float((u * v).sum())
        for key in totals:
            totals[key] += row[key]
        rows.append(row)
    for row in [totals, *rows]:
        row['left_norm'] = math.sqrt(row['left_squared'])
        row['right_norm'] = math.sqrt(row['right_squared'])
        row['difference_norm'] = math.sqrt(row['difference_squared'])
        norm_product = row['left_norm'] * row['right_norm']
        row['cosine'] = row['dot'] / norm_product if norm_product else None
        row['relative_difference_norm'] = row['difference_norm'] / row['left_norm'] if row['left_norm'] else None
        row['different_fraction'] = row['different'] / row['coordinates'] if row['coordinates'] else None
    return dict(left='paper_entry', right='repo_position', global_statistics=totals, parameters=rows)


def initial_check(trainer, initial, path, *, load):
    """Restore the saved fresh state, then check native bits and empty buffers."""
    if load:
        trainer.bundle.draft_model.load_state_dict(initial['draft_state_dict'], strict=True)
        trainer.backend.load_state_dict(initial['backend'])
    trainer.backend.optimizer.zero_grad()
    restore_rng(initial['rng'])
    model = trainer.bundle.model
    model.train()
    for name in ('_cached_block_mask', '_cached_seq_len', '_cached_bsz'):
        if hasattr(model, name):
            setattr(model, name, None)
    trainer.global_step = trainer.presentations = trainer.core._micro = 0
    trainer.events = []
    trainer.core._ratio_totals.clear()
    trainer.core._sum_totals.clear()
    trainer._anchor_observations.clear()
    failures = differences(dict(trainer.bundle.draft_model.state_dict()), initial['draft_state_dict'])
    failures += differences(trainer.backend.optimizer.state_dict(), initial['backend']['optimizer'], 'optimizer')
    failures += differences(capture_rng(), initial['rng'], 'rng')
    receipt = dict(exact_equal=not failures, differing_paths=failures,
                   global_step=0, presentations=0, empty_accumulation=True,
                   fresh_adam=not trainer.backend.optimizer.optimizer.state,
                   gradients_cleared=all(p.grad is None for p in trainer.backend.optimizer.model_params))
    write_json(path, receipt)
    if failures or not receipt['fresh_adam'] or not receipt['gradients_cleared']:
        raise AssertionError(f"Initial restoration failed: {path}")
    return receipt


@contextmanager
def retained_anchors(model, record):
    """Intentionally replay saved anchors; do not claim a natural RNG redraw."""
    draw, = record['sampled_anchors']
    device = next(model.draft_model.parameters()).device
    anchors = torch.tensor(draw['positions'], dtype=torch.long, device=device)
    keep = torch.tensor(draw['keep_mask'], dtype=torch.bool, device=device)
    with fixed_anchors(model, anchors, keep):
        yield


def observed_backward(trainer, log, kind):
    original = trainer.backend.backward
    def backward(*args, **kwargs):
        with log.work(kind):
            return original(*args, **kwargs)
    return override(trainer.backend, 'backward', backward)


def run_startup(trainer, initial, pair, log, root):
    """Warm the approved compiler path without TrainerCore or an Adam step."""
    from specforge.training.strategies.base import StepContext
    trainer.bundle.model.point_loss_extension = ObservedLoss('paper_entry', trainer.bundle.model.objective_chunk_blocks)
    trainer.bundle.model.begin_event()
    with observe_forwards(trainer, log), observed_backward(trainer, log, 'startup_backward'):
        for ordinal, (path, record) in enumerate(pair):
            raw = torch.load(path, map_location='cpu', weights_only=True)
            with retained_anchors(trainer.bundle.model, record), log.work('startup_response_forward'):
                output = trainer.strategy.forward_loss(trainer._batch(raw), StepContext(
                    global_step=0, total_steps=trainer.horizon, collect_detailed_metrics=True))
            trainer.backend.backward(output.loss / trainer.accumulation, is_boundary=ordinal == 1)
            del output, raw
    # No weights/Adam are loaded here: equality proves startup did not update them.
    initial_check(trainer, initial, root / 'STARTUP_RESTORATION.json', load=False)


def run_branch(trainer, initial, pair, mode, batch, eos_ids, log, root):
    """One isolated two-response step, then two detached source-fit forwards."""
    from specforge.training.strategies.base import StepContext
    label = f'batch_{batch:02d}_{mode}'
    log.trial = label
    destination = root / label
    destination.mkdir()
    started = time.perf_counter()
    restoration = initial_check(trainer, initial, destination / 'INITIAL_STATE.json', load=True)
    observer = ObservedLoss(mode, trainer.bundle.model.objective_chunk_blocks)
    trainer.bundle.model.point_loss_extension = observer
    trainer.bundle.model.begin_event()
    gradients = {}
    native_step = trainer.backend.optimizer.step
    named = {name: value for name, value in trainer.bundle.draft_model.named_parameters() if value.requires_grad}

    def capture_then_step():
        for name, value in named.items():
            if value.grad is None or not bool(torch.isfinite(value.grad).all()):
                write_json(destination / 'GRADIENT_FAILURE.json', dict(parameter=name,
                    missing=value.grad is None, nonfinite_coordinates=None if value.grad is None
                    else int((~torch.isfinite(value.grad)).sum())))
                raise ValueError(f'Missing/nonfinite pre-clip mixed gradient: {name}')
            gradients[name] = value.grad.detach().cpu().clone()
        with log.work('adam_step'):
            return native_step()

    rows = []
    with observe_forwards(trainer, log), observed_backward(trainer, log, 'core_backward'), \
            override(trainer.backend.optimizer, 'step', capture_then_step):
        for ordinal, (path, record) in enumerate(pair):
            log.presentation = ordinal
            raw = torch.load(path, map_location='cpu', weights_only=True)
            with retained_anchors(trainer.bundle.model, record), log.work('training_response_forward'):
                result = trainer.core.train_step(trainer._batch(raw), StepContext(
                    global_step=trainer.global_step, total_steps=trainer.horizon, collect_detailed_metrics=True))
            metrics = result.materialize_metrics()
            rows.append(dict(record_id=record['record_id'], metrics=metrics,
                             metrics_scope='accumulation_window' if result.optimizer_stepped else 'presentation'))
            write_json(destination / 'TRAINING.json', serializable(rows))
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError('Nonfinite native training metric')
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
            del result, raw
        before = observer.responses[:]
        training_wall_s = time.perf_counter() - started
        rng_before_probe = capture_rng()
        probes_started = time.perf_counter()
        with preserved_probe_runtime(trainer.bundle.model), torch.no_grad():
            for path, record in pair:
                raw = torch.load(path, map_location='cpu', weights_only=True)
                with retained_anchors(trainer.bundle.model, record), log.work('post_response_forward'):
                    output = trainer.strategy.forward_loss(trainer._batch(raw), StepContext(
                        global_step=1, total_steps=trainer.horizon, collect_detailed_metrics=True))
                probe_loss = float(output.loss.detach())
                if not math.isfinite(probe_loss) or any(
                        not math.isfinite(float(value)) for value in output.sum_metrics.values()):
                    write_json(destination / 'PROBE_FAILURE.json', dict(record_id=record['record_id'],
                        loss=serializable(probe_loss), metrics=serializable(output.sum_metrics)))
                    raise ValueError('Nonfinite post-step source-fit loss')
                del output, raw
        if differences(rng_before_probe, capture_rng()):
            raise AssertionError('Source-fit scoring changed training RNG')
    after = observer.responses[2:]
    if len(before) != 2 or len(after) != 2 or trainer.global_step != 1 or trainer.core.accumulation_remainder:
        raise AssertionError('Disposable branch accounting differs')
    adam_steps = {float(state['step']) for state in trainer.backend.optimizer.optimizer.state.values()}
    if adam_steps != {1.}:
        raise AssertionError('Native Adam counters do not equal one')
    displacements = {name: value.detach().cpu().float() - initial['draft_state_dict'][name].float()
                     for name, value in named.items()}
    prefixes = []
    for (_, record), old, new in zip(pair, before, after):
        if differences(old['labels'], new['labels']) or differences(old['weights'], new['weights']):
            raise AssertionError('Post-step labels/position weights changed')
        prefixes.append(dict(record_id=record['record_id'], **prefix_changes(
            old['predictions'], new['predictions'], old['labels'], old['weights'], eos_ids),
            available_position_occurrences=int((old['weights'] > 0).sum())))
    receipt = dict(status='completed', batch=batch, mode=mode, restoration=restoration,
        contract=observer.loss.contract, source_fit=prefixes,
        responses_before=before, responses_after=after, training_rows=rows,
        adam_steps=1, presentations=2, source_fit_forwards=2, core_accumulation_remainder=0,
        anchor_policy='exact_retained_draw_forced_no_resampling',
        post_probe_rng_preserved=True, wall_s=time.perf_counter() - started,
        instrumented_training_and_restoration_s=training_wall_s,
        post_source_fit_s=time.perf_counter()-probes_started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
        peak_memory_scope='this_branch_including_runtime_construction')
    write_json(destination / 'RESULT.json', serializable(receipt))
    return receipt, gradients, displacements, capture_rng()


def run(config_path, initial_state_path, prepared_json_or_dict, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    log = WorkLog(root)
    summary = dict(status='running', plan=PLAN, branches=[], comparisons=[],
                   initial_state=str(initial_state_path), config=str(config_path),
                   interpretation='same-response source fitting; no transfer or serving claim')
    write_json(root / 'PLAN.json', summary)
    trainer = None
    started = time.perf_counter()
    try:
        prepared = (json.loads(Path(prepared_json_or_dict).read_text())
                    if isinstance(prepared_json_or_dict, (str, Path)) else prepared_json_or_dict)
        paths, records = prepared['feature_paths'], prepared['records']
        if len(paths) != 6 or len(records) != 6 or len({r['record_id'] for r in records}) != 6:
            raise ValueError('Require exactly the six fixed, distinct retained presentations')
        initial = torch.load(initial_state_path, map_location='cpu', weights_only=False, mmap=True)
        if initial['global_step'] or initial['presentations'] or initial['events'] or initial['backend']['optimizer']['optimizer_state_dict']['state']:
            raise ValueError('Require the retained released-weight/fresh-Adam initial state')
        pin_retained_graph_runtime(root)
        trainer = SpecForgeEventTrainer(config_path, root / 'disposable_runtime',
                                       objective='osd', reduction='response_mean')
        if trainer.accumulation != 2 or trainer.config_text != initial['config_text']:
            raise ValueError('Initial configuration or accumulation differs')
        trainer._prepare_event('diagnostic_validation', paths, 1)
        for path, record in zip(paths, records):
            raw = torch.load(path, map_location='cpu', weights_only=True)
            if raw['record_id'] != record['record_id'] or record['pass'] != 0:
                raise ValueError('Feature identity differs from retained first-pass presentation')
            del raw
        with preserve_rng():
            eos_ids = ranking_eos_token_ids(trainer.cfg.model.target_model_path)
        initial_check(trainer, initial, root / 'FRESH_INITIAL_STATE.json', load=True)
        pairs = [list(zip(paths[start:start+2], records[start:start+2])) for start in (0, 2, 4)]
        log.trial = 'startup'
        startup_started = time.perf_counter()
        run_startup(trainer, initial, pairs[0], log, root)
        summary['startup_wall_s'] = time.perf_counter()-startup_started
        # Discard DDP reducer/bucket startup state as in the qualified procedure.
        # Restoring tensors alone would not make two reused live runtimes equal.
        trainer.close()
        trainer = None
        for batch, pair in enumerate(pairs, 1):
            outcomes = []
            for mode in MODES:
                if torch.cuda.is_initialized():
                    torch.cuda.reset_peak_memory_stats()
                construction_started = time.perf_counter()
                trainer = SpecForgeEventTrainer(config_path,
                    root / f'batch_{batch:02d}_{mode}_runtime',
                    objective='osd', reduction='response_mean')
                construction_s = time.perf_counter()-construction_started
                outcome = run_branch(trainer, initial, pair, mode, batch, eos_ids, log, root)
                outcome[0]['runtime_construction_s'] = construction_s
                write_json(root / f'batch_{batch:02d}_{mode}' / 'RESULT.json', serializable(outcome[0]))
                outcomes.append(outcome)
                summary['branches'].append(dict(batch=batch, mode=mode, status='completed'))
                write_json(root / 'PROGRESS.json', serializable(dict(summary, work=dict(log.counts))))
                trainer.close()
                trainer = None
            left, right = outcomes
            # Loss metrics deliberately differ; canonical blocks and predictions must not.
            same_pre = all(not differences({k: a[k] for k in ('predictions','labels','weights')},
                                          {k: b[k] for k in ('predictions','labels','weights')})
                           for a,b in zip(left[0]['responses_before'],right[0]['responses_before']))
            comparison = dict(batch=batch, identical_preupdate_blocks=same_pre,
                identical_final_training_rng=not differences(left[3], right[3]),
                gradients=paired_tensor_statistics(left[1], right[1]),
                displacements=paired_tensor_statistics(left[2], right[2]))
            write_json(root / f'BATCH_{batch:02d}_COMPARISON.json', comparison)
            summary['comparisons'].append(comparison)
            if not same_pre:
                raise AssertionError('Paired initial native predictions/labels/weights differ')
            if not comparison['identical_final_training_rng']:
                raise AssertionError('Paired clipping branches advanced training RNG differently')
            del outcomes, left, right, outcome
        expected = dict(adam_step_completed=6, training_response_forward_completed=12,
                        core_backward_completed=12, post_response_forward_completed=12,
                        startup_response_forward_completed=2, startup_backward_completed=2)
        if any(log.counts[key] != value for key,value in expected.items()):
            raise AssertionError('Bounded work accounting differs')
        summary['status'] = 'passed'
    except BaseException as error:
        if trainer is not None:
            trainer.failed = True
        summary.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), automatic_retry=False)
        raise
    finally:
        summary.update(work=dict(log.counts), wall_s=time.perf_counter()-started,
                       uncompleted_branches=6-len(summary['branches']))
        write_json(root / 'RESULT.json', serializable(summary))
        if trainer is not None:
            trainer.close()
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--initial-state', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.config, args.initial_state, args.prepared, args.output)
