"""At most six disposable source/port Adam steps on fixed retained responses.

The literal source runs first. Nonfinite representative gradients stop before
Adam, preserving the input evidence rather than changing the released loss.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from experiments.adaflash_clipping.run import (
    initial_check, observed_backward, paired_tensor_statistics, retained_anchors,
    run_startup, serializable,
)
from experiments.ok_continuation.resume import differences
from experiments.prefix_guard.qualify import observe_forwards, pin_retained_graph_runtime
from experiments.prefix_ranking.qualify import WorkLog, cpu_copy, override
from online_feedback.adaflash_repo import (
    LiteralSourceAdaFlashOPD, ReleasedAdaFlashOPD, NUMERICAL_TOLERANCES,
    adaflash_repo_contract, literal_source_loss, literal_source_terms, repo_position_terms,
)
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer


BUDGET = dict(maximum_optimizer_steps=6, training_presentations=12,
              core_response_forwards=12, core_backwards=12,
              startup_response_forwards=2, startup_backwards=2,
              startup_optimizer_steps=0, fixture_loss_forwards=6,
              fixture_loss_backwards=6, complete_generations=0, exports=0)
BRANCHES = ('literal_source', 'ported_loss')


def fixed_loss_checks(log, device):
    """Known active/inactive caps and tiny positive probabilities, no Adam."""
    result = []
    cases = (
        ('active_cap', [4., -4., 0.], [-4., 4., 0.]),
        ('inactive_cap', [.1, .2, .3], [.1, .2, .3]),
        ('tiny_positive_probability', [0., -20., -80.], [0., -10., -40.]),
    )
    for name, student, target in cases:
        outputs = []
        teacher = torch.tensor(target, dtype=torch.bfloat16, device=device).reshape(1, 1, 1, -1)
        weights = torch.ones((1, 1, 1), device=device)
        for branch in BRANCHES:
            logits = torch.tensor(student, dtype=torch.bfloat16, device=device).reshape(1, 1, 1, -1).requires_grad_()
            log.trial = f'fixture_{name}_{branch}'
            with log.work('fixture_loss_forward'):
                denominator = weights.sum() + 1e-6
                function = literal_source_terms if branch == 'literal_source' else repo_position_terms
                rows = function(logits, teacher, denominator)
                hard_loss = (rows['ce'] * weights.flatten()).sum() / denominator
                rkl_loss = (rows['clipped_rkl'] * weights.flatten()).sum()
                loss = .8 * hard_loss + (1. - .8) * rkl_loss
            with log.work('fixture_loss_backward'):
                gradient, = torch.autograd.grad(loss, logits)
            if not bool(torch.isfinite(loss) & torch.isfinite(gradient).all()):
                raise ValueError('Literal/port nonfinite in fixed numerical fixture: ' + name)
            outputs.append((loss.detach(), gradient, rows))
        for left, right in zip(outputs[0][:2], outputs[1][:2]):
            torch.testing.assert_close(left, right, rtol=0., atol=0.)
        for key in outputs[0][2]:
            torch.testing.assert_close(outputs[0][2][key], outputs[1][2][key], rtol=0., atol=0.)
        result.append(dict(case=name, status='passed', loss=float(outputs[0][0]),
            clipped=bool((outputs[0][2]['normalized_rkl'] > .01).any()),
            smallest_probability=float(outputs[0][2]['probabilities'].min()),
            dtypes={key: str(value.dtype) for key, value in outputs[0][2].items()}))
    if not result[0]['clipped'] or result[1]['clipped']:
        raise AssertionError('Fixed fixtures did not exercise both clipping branches')
    return result


class QualificationLoss:
    """Record only original forwards; keep at most two zero-probability rows."""
    def __init__(self, branch, destination):
        self.loss = LiteralSourceAdaFlashOPD() if branch == 'literal_source' else ReleasedAdaFlashOPD()
        self.destination = destination
        self.responses = []
        self.collecting = False
        self.zero_rows_saved = False

    def prepare_response(self, weights):
        self.loss.prepare_response(weights)
        self.collecting = True
        self.chunks = []

    def chunk_terms(self, logits, teacher, labels, weights):
        if self.collecting:
            with torch.no_grad():
                q = torch.softmax(logits.reshape(-1, logits.shape[-1]), -1)
                zero_rows = (q == 0).any(-1)
                positive = q[q > 0]
                self.chunks.append(dict(
                    predictions=logits.argmax(-1).cpu(), labels=labels.cpu(), weights=weights.cpu(),
                    zero_probabilities=int((q == 0).sum()),
                    zero_rows=int(zero_rows.sum()),
                    supervised_zero_rows=int((zero_rows & (weights.flatten() > 0)).sum()),
                    smallest_positive_probability=float(positive.min()) if positive.numel() else None,
                    denominator=float(self.loss.denominator), dtype=str(logits.dtype)))
                if zero_rows.any() and not self.zero_rows_saved:
                    indices = zero_rows.nonzero().flatten()[:2]
                    save_torch(self.destination / 'ZERO_PROBABILITY_INPUTS.pt', dict(
                        logits=logits.reshape_as(q)[indices].cpu(),
                        teacher_logits=teacher.reshape_as(q)[indices].cpu(),
                        weights=weights.flatten()[indices].cpu(),
                        full_response_denominator=self.loss.denominator.cpu(),
                        response_ordinal=len(self.responses), chunk_ordinal=len(self.chunks)-1))
                    self.zero_rows_saved = True
        return self.loss.chunk_terms(logits, teacher, labels, weights)

    def reduce_response(self, terms, denominator):
        return self.loss.reduce_response(terms, denominator)

    def response_metrics(self, terms, denominator):
        sums, ratios = self.loss.response_metrics(terms, denominator)
        self.collecting = False
        self.responses.append(dict(metrics=cpu_copy(sums), chunks=self.chunks))
        self.chunks = []
        write_json(self.destination / 'RESPONSES.json', serializable(self.responses))
        if not all(bool(torch.isfinite(value).all()) for value in sums.values()):
            raise ValueError('Nonfinite representative source/port forward; see RESPONSES.json')
        return sums, ratios


def finite_gradient_check(named, destination, presentation):
    failures = []
    for name, parameter in named.items():
        gradient = parameter.grad
        if gradient is None:
            failures.append(dict(parameter=name, missing=True))
        elif not bool(torch.isfinite(gradient).all()):
            bad = ~torch.isfinite(gradient)
            indices = bad.flatten().nonzero().flatten()[:8]
            failures.append(dict(parameter=name, shape=list(gradient.shape), dtype=str(gradient.dtype),
                nonfinite=int(bad.sum()), total=gradient.numel(),
                sample_flat_indices=indices.cpu().tolist(),
                sample_values=serializable(gradient.flatten()[indices])))
    if failures:
        write_json(destination / 'NONFINITE_GRADIENTS.json', dict(
            status='failed', presentation=presentation, parameters=failures,
            optimizer_called=False, action='stop_without_stabilization'))
        raise ValueError('Nonfinite representative gradients; no optimizer step permitted')


def run_branch(trainer, initial, pair, branch, batch, log, root):
    from specforge.training.strategies.base import StepContext
    directory = root / f'batch_{batch:02d}_{branch}'
    directory.mkdir()
    log.trial = directory.name
    restoration = initial_check(trainer, initial, directory / 'INITIAL_STATE.json', load=True)
    observer = QualificationLoss(branch, directory)
    trainer.bundle.model.point_loss_extension = observer
    trainer.bundle.model.begin_event()
    named = {name: p for name, p in trainer.bundle.draft_model.named_parameters() if p.requires_grad}
    gradients = {}
    ordinary_step = trainer.backend.optimizer.step

    def step():
        finite_gradient_check(named, directory, 1)
        gradients.update({name: p.grad.detach().cpu().clone() for name, p in named.items()})
        with log.work('adam_step'):
            return ordinary_step()

    rows = []
    started = time.perf_counter()
    with observe_forwards(trainer, log), observed_backward(trainer, log, 'core_backward'), \
            override(trainer.backend.optimizer, 'step', step):
        for ordinal, (path, record) in enumerate(pair):
            log.presentation = ordinal
            raw = torch.load(path, map_location='cpu', weights_only=True)
            with retained_anchors(trainer.bundle.model, record), log.work('training_response_forward'):
                result = trainer.core.train_step(trainer._batch(raw), StepContext(
                    global_step=trainer.global_step, total_steps=trainer.horizon,
                    collect_detailed_metrics=True))
            metrics = result.materialize_metrics()
            rows.append(dict(record_id=record['record_id'], metrics=metrics,
                             optimizer_stepped=result.optimizer_stepped))
            write_json(directory / 'TRAINING.json', serializable(rows))
            if not result.optimizer_stepped:
                finite_gradient_check(named, directory, ordinal)
            if not all(torch.isfinite(torch.tensor(float(value))) for value in metrics.values()):
                raise ValueError('Nonfinite representative training metrics')
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
            del result, raw
    if trainer.global_step != 1 or trainer.presentations != 2 or trainer.core.accumulation_remainder:
        raise AssertionError('Disposable branch exceeded one step/two presentations')
    displacements = {name: p.detach().cpu().float()-initial['draft_state_dict'][name].float()
                     for name, p in named.items()}
    receipt = dict(status='passed', branch=branch, batch=batch, restoration=restoration,
                   rows=rows, responses=observer.responses, wall_s=time.perf_counter()-started,
                   optimizer_steps=1, presentations=2,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
    write_json(directory / 'RESULT.json', serializable(receipt))
    return receipt, gradients, displacements, cpu_copy(trainer.backend.optimizer.state_dict()), capture_rng()


def run(config, initial_path, prepared_path, output, *, execution_commit=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    log = WorkLog(root)
    summary = dict(status='running', budget=BUDGET, tolerances=NUMERICAL_TOLERANCES,
                   loss_contract=adaflash_repo_contract(), branches=[], comparisons=[],
                   execution_commit=execution_commit)
    write_json(root / 'PLAN.json', summary)
    trainer = None
    started = time.perf_counter()
    try:
        prepared = json.loads(Path(prepared_path).read_text())
        paths, records = prepared['feature_paths'], prepared['records']
        if (prepared['status'] != 'completed' or len(paths) != 6 or len(records) != 6
                or len({record['record_id'] for record in records}) != 6):
            raise ValueError('Require all six retained representative feature sets')
        for path, record in zip(paths, records):
            raw = torch.load(path, map_location='cpu', weights_only=True)
            if raw['record_id'] != record['record_id'] or record['pass'] != 0:
                raise ValueError('Retained feature/presentation identity mismatch')
            del raw
        initial = torch.load(initial_path, map_location='cpu', weights_only=False, mmap=True)
        if (initial['global_step'] or initial['presentations'] or initial['events']
                or initial['backend']['optimizer']['optimizer_state_dict']['state']):
            raise ValueError('Qualification requires the retained released/fresh initial state')
        from experiments.ok_coding.storage import identity
        from experiments.ok_coding.training import _export_equal
        import yaml
        config_data = yaml.safe_load(Path(config).read_text())
        _export_equal(initial, config_data['model']['draft_checkpoint_path'])
        summary['input_identities'] = {str(path): identity(path)
            for path in (initial_path, config, prepared_path, *paths)}
        write_json(root / 'INPUT_IDENTITIES.json', summary['input_identities'])
        pairs = [list(zip(paths[start:start+2], records[start:start+2])) for start in (0, 2, 4)]
        pin_retained_graph_runtime(root)
        summary['fixed_loss_checks'] = fixed_loss_checks(log, 'cuda:0')
        write_json(root / 'FIXED_LOSS_CHECKS.json', summary['fixed_loss_checks'])
        trainer = SpecForgeEventTrainer(config, root / 'startup_runtime', objective='osd', reduction='response_mean')
        if trainer.config_text != initial['config_text'] or trainer.accumulation != 2:
            raise ValueError('Original configuration or accumulation changed')
        initial_check(trainer, initial, root / 'FRESH_INITIAL_STATE.json', load=True)
        log.trial = 'disposable_startup'
        run_startup(trainer, initial, pairs[0], log, root)
        trainer.close()
        trainer = None
        for batch, pair in enumerate(pairs, 1):
            outcomes = []
            for branch in BRANCHES:
                torch.cuda.reset_peak_memory_stats()
                trainer = SpecForgeEventTrainer(config, root / f'batch_{batch:02d}_{branch}_runtime',
                                               objective='osd', reduction='response_mean')
                outcome = run_branch(trainer, initial, pair, branch, batch, log, root)
                outcomes.append(outcome)
                summary['branches'].append(dict(batch=batch, branch=branch, status='passed'))
                write_json(root / 'PROGRESS.json', serializable(summary))
                trainer.close()
                trainer = None
            left, right = outcomes
            comparison = dict(batch=batch,
                gradients=paired_tensor_statistics(left[1], right[1]),
                displacements=paired_tensor_statistics(left[2], right[2]),
                optimizer_differences=differences(left[3], right[3]),
                rng_differences=differences(left[4], right[4]),
                response_differences=differences(left[0]['responses'], right[0]['responses']))
            write_json(root / f'BATCH_{batch:02d}_PARITY.json', comparison)
            summary['comparisons'].append(comparison)
            if (comparison['gradients']['global_statistics']['different']
                    or comparison['displacements']['global_statistics']['different']
                    or any(comparison[key] for key in ('optimizer_differences', 'rng_differences', 'response_differences'))):
                raise AssertionError('Literal-source/ported native update parity failed')
            del outcomes, outcome, left, right
        expected = dict(adam_step_completed=6, training_response_forward_completed=12,
                        core_backward_completed=12, startup_response_forward_completed=2,
                        startup_backward_completed=2)
        if any(log.counts[key] != value for key, value in expected.items()):
            raise AssertionError('Qualification work differs from the fixed budget')
        summary['status'] = 'passed'
    except BaseException as error:
        if trainer is not None:
            trainer.failed = True
        summary.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), automatic_retry=False)
        raise
    finally:
        # Keep the primary failure durable even if runtime cleanup also fails.
        summary.update(work=dict(log.counts), wall_s=time.perf_counter()-started)
        write_json(root / 'RESULT.json', serializable(summary))
        try:
            if trainer is not None:
                trainer.close()
        except BaseException as cleanup_error:
            summary.update(status='failed', cleanup_error=repr(cleanup_error))
            raise
        finally:
            summary.update(work=dict(log.counts), wall_s=time.perf_counter()-started,
                           uncompleted_branches=6-len(summary['branches']),
                           scientific_states_retained=False, main_training_authorized=summary['status']=='passed')
            write_json(root / 'RESULT.json', serializable(summary))
            write_json(root / 'QUALIFICATION.json', serializable(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    plan = json.loads((args.root / 'sealed/PLAN.json').read_text())
    qualification = plan['qualification']
    run(qualification['config'], qualification['initial_state'], qualification['prepared'],
        args.root / 'qualification', execution_commit=plan['execution_source']['commit'])


if __name__ == '__main__':
    main()
