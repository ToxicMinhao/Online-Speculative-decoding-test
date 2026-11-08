"""Zero-Adam isolation of the retained two-anchor auxiliary VJP failure.

The ordinary endpoint is loaded, never recomputed. Head layout comparisons use
the same hidden tensors. Their adjoints then traverse the same draft graph, so
head arithmetic and draft backward repeatability can be distinguished.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
import traceback

import torch

from experiments.ok_coding.storage import identity
from experiments.ok_continuation.resume import differences
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from online_feedback.fsg_gradient_diagnostic import (
    HeadDiagnosticError, REFERENCE, VARIANTS, run_head_diagnostic, tensor_comparison,
)
from online_feedback.functional_geometry import FunctionalGeometry, isolated_geometry_runtime
from online_feedback.functional_geometry_math import finite
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import write_json
from online_feedback.training import validate_feature
from .native_oracles import gradient_comparison, NATIVE_VJP_TOLERANCES, optimizer_versions
from .qualify import cleanup, runtime_state, restore_runtime, scientific_state
from .training import FunctionalGeometryTrainer


BUDGET = dict(response_forward=3, head_backward=6, draft_backward=4,
              optimizer_step=0, feedback_generation=0, teacher_preparation=0,
              serving_generation=0)


class DiagnosticWork:
    """Persist attempts before execution and reject work beyond the fixed cap."""

    def __init__(self, root, limits=None):
        self.budget = dict(BUDGET if limits is None else limits)
        self.path = Path(root) / 'WORK.json'
        self.counts = {}

    def record(self, kind, status):
        key = kind + '_' + status
        count = self.counts.get(key, 0) + 1
        if status == 'attempted' and kind in self.budget and count > self.budget[kind]:
            raise ValueError('Diagnostic work limit exceeded: ' + kind)
        self.counts[key] = count
        write_json(self.path, dict(budget=self.budget, counts=self.counts))

    @contextmanager
    def operation(self, kind):
        self.record(kind, 'attempted')
        try:
            yield
        except BaseException:
            self.record(kind, 'failed')
            raise
        else:
            self.record(kind, 'completed')


def forbid_step(*args, **kwargs):
    raise RuntimeError('Optimizer applications are forbidden in this diagnostic')


def pin_diagnostic_runtime(root):
    """Reuse the qualified pin with labels specific to this zero-update run."""
    receipt = pin_retained_graph_runtime(root)
    receipt.update(applied_before='one retained-origin trainer construction',
                   scope='zero-Adam head and draft VJP diagnosis; no startup backwards')
    write_json(Path(root) / 'RUNTIME_COMPATIBILITY.json', receipt)
    return receipt


def load_response(trainer, feature_path, presentations_path):
    raw = torch.load(feature_path, map_location='cpu', weights_only=False)
    validity = validate_feature(raw, max_length=trainer.max_length)
    first = json.loads(Path(presentations_path).read_text())[0]
    if first['record_id'] != raw['record_id'] or len(first['sampled_anchors']) != 1:
        raise ValueError('Retained response and original anchor identity differ')
    draw = first['sampled_anchors'][0]
    anchors = torch.tensor(draw['positions'], dtype=torch.int64)[:, :2]
    keep = torch.tensor(draw['keep_mask'], dtype=torch.bool)[:, :2]
    if anchors.shape != (1, 2) or not bool(keep.all()):
        raise ValueError('The retained two-anchor fixture is unavailable')
    batch = trainer._batch(raw).tensors
    labels, weights, _ = trainer.bundle.model._block_labels_and_weights(
        batch['input_ids'], batch['loss_mask'], anchors, keep)
    response = dict(batch, anchors=anchors, keep=keep, labels=labels,
                    validity_weights=weights, record_id=raw['record_id'])
    return response, dict(validity, anchors=anchors.tolist(), keep=keep.tolist(),
                          original_anchor_count=len(draw['positions'][0]),
                          selection='first two original anchors; no sampler call')


def parameter_vjp(hidden, adjoint, parameters, work):
    with preserve_rng(), work.operation('draft_backward'):
        values = torch.autograd.grad(hidden, parameters, grad_outputs=adjoint,
                                     retain_graph=True, allow_unused=True)
    for value in values:
        if value is not None:
            finite(value, 'diagnostic parameter VJP')
    return tuple(None if value is None else value.detach() for value in values)


@dataclass
class RetainedGradientCase:
    """One authenticated two-anchor fixture and its unchanged draft graph."""
    hook: FunctionalGeometry
    response: dict
    native_hidden: torch.Tensor
    hidden: torch.Tensor
    parameters: tuple
    names: list
    work: DiagnosticWork
    root: Path
    pre_forward_rng: dict
    pre_forward_runtime: dict


def diagnose_native_case(case):
    """The original BF16 isolation, retained as a separate diagnostic case."""
    hook, response = case.hook, case.response
    native_hidden, hidden = case.native_hidden, case.hidden
    # Transfer graph ownership to this case so the optional fresh-graph branch
    # can release the old graph before restoring its runtime buffers.
    case.native_hidden = case.hidden = None
    parameters, names = case.parameters, case.names
    work, root = case.work, case.root
    pre_forward_rng, pre_forward_runtime = case.pre_forward_rng, case.pre_forward_runtime
    receipt = {}
    weights, proposals, mask, margins = hook._scan_reference(hidden.detach(), response, 1)
    full_weights, full_proposals, full_mask, full_margins = hook._scan_reference(hidden.detach(), response, 0)
    receipt['reference'] = dict(protected_occurrences=int(mask.sum()),
        prefix_mask_exact=torch.equal(mask, full_mask),
        proposals_exact=torch.equal(proposals, full_proposals),
        weights=tensor_comparison(weights, full_weights),
        margins=tensor_comparison(margins, full_margins))
    write_json(root / 'REFERENCE.json', receipt['reference'])
    if not receipt['reference']['prefix_mask_exact'] or not torch.equal(weights, full_weights):
        raise ValueError('Original exact reference-mask/weight agreement did not reproduce')
    try:
        head, tensors = run_head_diagnostic(hook.model.lm_head,
            hidden.detach(), native_hidden, weights, return_tensors=True,
            work_callback=work.record)
    except HeadDiagnosticError as error:
        write_json(root / 'HEAD.json', error.receipt)
        raise
    write_json(root / 'HEAD.json', head)
    del native_hidden
    # Same graph, two observed head adjoints, then the identical reference
    # adjoint again: this separates head changes from draft repeatability.
    chunked = parameter_vjp(hidden, tensors['adjoints'][VARIANTS[0]], parameters, work)
    full = parameter_vjp(hidden, tensors['adjoints'][REFERENCE], parameters, work)
    receipt['parameter_comparisons'] = dict(chunked_vs_full=gradient_comparison(chunked, full, names))
    write_json(root / 'PARAMETERS.json', receipt['parameter_comparisons'])
    del chunked
    repeated = parameter_vjp(hidden, tensors['adjoints'][REFERENCE], parameters, work)
    receipt['parameter_comparisons']['same_graph_repeat'] = gradient_comparison(repeated, full, names)
    receipt['same_graph_repeat_exact'] = not differences(repeated, full)
    del repeated
    # Only investigate a fresh graph if the head-local explanation does
    # not reproduce the failed coordinate gate, or repetition differs.
    unresolved = (receipt['parameter_comparisons']['chunked_vs_full']['status'] == 'passed'
                  or not receipt['same_graph_repeat_exact']
                  or head['comparisons']['initial'][VARIANTS[0]]['adjoint']['exact'])
    receipt['fresh_graph_executed'] = unresolved
    if unresolved:
        saved_hidden = hidden.detach().clone()
        del hidden
        with preserve_rng():
            restore_rng(pre_forward_rng)
            restore_runtime(hook.model, pre_forward_runtime)
            with work.operation('response_forward'):
                replay = hook.forward_hidden(response, 'diagnostic_fresh_graph', grad=True)
            receipt['fresh_graph_hidden'] = tensor_comparison(replay.detach(), saved_hidden)
            repeated = parameter_vjp(replay, tensors['adjoints'][REFERENCE], parameters, work)
        receipt['parameter_comparisons']['fresh_graph_same_adjoint'] = gradient_comparison(repeated, full, names)
        receipt['fresh_graph_same_adjoint_exact'] = not differences(repeated, full)
        del replay, repeated, saved_hidden
    else:
        del hidden
    del full, tensors
    write_json(root / 'PARAMETERS.json', receipt['parameter_comparisons'])
    return receipt


def run(plan_path, *, case_runner=diagnose_native_case, work_limits=None):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'PLAN.json', plan)
    work = DiagnosticWork(root, work_limits)
    started = time.perf_counter()
    receipt = dict(status='running', interpretation='diagnostic, not qualification acceptance',
        execution_commit=plan['execution_commit'], budget=work.budget,
        tolerances=NATIVE_VJP_TOLERANCES, additional_adam_steps=0,
        startup='ordinary retained endpoint loaded; no warmup optimizer or backward')
    trainer = hook = before = None
    failure = None
    try:
        free = shutil.disk_usage(root).free
        required = plan['storage']['maximum_additional_bytes'] + plan['storage']['reserve_bytes']
        if free < required:
            raise ValueError('Insufficient storage for the sealed zero-Adam diagnostic')
        admission = dict(free_bytes=free, required_free_bytes=required, dependencies=[])
        for item in plan['dependencies']:
            actual = identity(item['path'])
            admission['dependencies'].append(dict(path=item['path'], actual=actual,
                                                 matched=actual == item['identity']))
        write_json(root / 'ADMISSION.json', admission)
        if not all(item['matched'] for item in admission['dependencies']):
            raise ValueError('Retained diagnostic dependency changed')
        original = json.loads(Path(plan['qualification_plan']).read_text())
        item = original['datasets']['gsm8k']
        pin_diagnostic_runtime(root)
        if 'minimum_gpu_free_bytes' in plan:
            free_gpu, total_gpu = torch.cuda.mem_get_info()
            receipt['gpu_admission'] = dict(free_bytes=free_gpu, total_bytes=total_gpu,
                required_free_bytes=plan['minimum_gpu_free_bytes'])
            if free_gpu < plan['minimum_gpu_free_bytes']:
                raise ValueError('Insufficient GPU memory for the sealed auxiliary-head test')
        torch.cuda.reset_peak_memory_stats()
        trainer = FunctionalGeometryTrainer(item['configs']['KPLUS'], root / 'trainer',
            arm='KPLUS', dataset='gsm8k', binding=original['binding'],
            original_config_text=Path(original['original_config']).read_text(),
            origin_state=item['origin_state'], origin_export=item['origin_export'])
        # Tripwires cover both optimizer entry points. No training method is called.
        trainer.backend.optimizer.step = forbid_step
        trainer.backend.optimizer.optimizer.step = forbid_step
        baseline_runtime = runtime_state(trainer.bundle.model)
        before = scientific_state(trainer)
        optimizer = trainer.backend.optimizer.optimizer
        versions = optimizer_versions(optimizer)
        hook = FunctionalGeometry(trainer)
        names = json.loads(Path(plan['failed_step']).read_text())['active_parameter_names']
        named = dict(trainer.bundle.draft_model.named_parameters())
        parameters = tuple(named[name] for name in names)
        hook.active_parameters = parameters
        response, receipt['fixture'] = load_response(trainer, plan['feature'], plan['presentations'])
        native = torch.load(plan['native_endpoint'], map_location='cpu', weights_only=False, mmap=True)
        if native['global_step'] != 513 or before['global_step'] != 512:
            raise ValueError('Diagnostic endpoints are not the retained first ordinary step')
        with isolated_geometry_runtime(trainer.bundle.model):
            # Finish all parameter copies before retaining a live pre-state graph.
            trainer.bundle.draft_model.load_state_dict(native['weights'], strict=True)
            with work.operation('response_forward'):
                native_hidden = hook.forward_hidden(response, 'diagnostic_native', grad=False).detach()
            trainer.bundle.draft_model.load_state_dict(before['weights'], strict=True)
            pre_forward_rng = capture_rng()
            pre_forward_runtime = runtime_state(trainer.bundle.model)
            with work.operation('response_forward'):
                hidden = hook.forward_hidden(response, 'diagnostic_prestate', grad=True)
            case = RetainedGradientCase(hook, response, native_hidden, hidden,
                parameters, names, work, root, pre_forward_rng, pre_forward_runtime)
            del hidden, native_hidden
            receipt.update(case_runner(case))
            del case
        receipt['optimizer_objects_and_versions_unchanged'] = versions == optimizer_versions(optimizer)
        if not receipt['optimizer_objects_and_versions_unchanged']:
            raise ValueError('Diagnostic altered optimizer state')
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc())
    finally:
        def finish(action, label):
            nonlocal failure
            try:
                cleanup(action, failure, receipt, label)
            except BaseException as error:
                failure = error
                receipt.update(status='failed', error_type=type(error).__name__, error=str(error))

        if trainer is not None and before is not None:
            # No diagnostic endpoint is published or used for training/serving.
            def restore_and_compare():
                trainer.bundle.draft_model.load_state_dict(before['weights'], strict=True)
                restore_runtime(trainer.bundle.model, baseline_runtime)
                restore_rng(before['rng'])
                restored = scientific_state(trainer)
                mismatch = differences(before, restored)
                receipt['restoration'] = dict(exact=not mismatch, differences=mismatch)
                if mismatch:
                    raise ValueError('Diagnostic state restoration differs')
            finish(restore_and_compare, 'diagnostic state restoration')
        if hook is not None:
            receipt['physical_work'] = dict(work=hook.work, calls=hook.forward_calls)
            finish(hook.close, 'diagnostic hook cleanup')
        if trainer is not None:
            finish(trainer.close, 'diagnostic trainer cleanup')
        receipt.update(work=work.counts, duration_s=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        def postcheck():
            receipt['dependency_postcheck'] = [dict(path=x['path'], matched=identity(x['path']) == x['identity'])
                                              for x in plan['dependencies']]
            if not all(x['matched'] for x in receipt['dependency_postcheck']):
                raise ValueError('Diagnostic changed a retained dependency')
        finish(postcheck, 'diagnostic dependency postcheck')
        write_json(root / 'RESULT.json', receipt)
    if failure is not None:
        raise failure
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    run(args.plan)
