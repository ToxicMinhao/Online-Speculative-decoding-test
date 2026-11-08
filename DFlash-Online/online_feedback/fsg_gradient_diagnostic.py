"""Isolate FSG head arithmetic on the same two retained hidden-state blocks.

This helper performs no draft/backbone forward, parameter backward, or update.
Its three layouts differ only in vocabulary-head chunking/checkpointing. If they
disagree, an optional second comparison supplies one common detached cotangent
to every layout, separating forward arithmetic from the head's hidden adjoint.
"""
from contextlib import contextmanager
import math

import torch
from torch.utils.checkpoint import _StopRecomputationError

from specforge.core.chunking import checkpointed_chunk_reduce

from .functional_geometry_math import geometry_terms, VECTOR_CHUNK_SIZE
from .rng import preserve_rng


# Identical to the sealed native qualification thresholds, not a new gate.
GRADIENT_TOLERANCES = dict(rtol=0.016, atol=1e-5, maximum_relative_l2=0.016)
VARIANTS = ('chunked_checkpoint', 'chunked_no_checkpoint', 'unchunked_no_checkpoint')
REFERENCE = VARIANTS[-1]


class HeadDiagnosticError(ValueError):
    """Carry already-collected compact evidence when the diagnostic stops."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def _number(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
    value = float(value)
    return value if math.isfinite(value) else str(value)


@torch.no_grad()
def tensor_comparison(actual, reference, *, sample_size=8):
    """Summarize all coordinates, retaining only a bounded mismatch sample.

    The same numerical thresholds are also shown for logits/cotangents as
    descriptive checks. Only hidden-adjoint comparisons concern the VJP gate.
    Nonfinite values remain explicit and cannot silently pass a comparison.
    """
    if actual.shape != reference.shape:
        raise ValueError('Head diagnostic tensor shapes differ')
    a, b = actual.detach().reshape(-1), reference.detach().reshape(-1)
    norms = torch.zeros(3, dtype=torch.float64, device=a.device)
    mismatch_count = violation_count = nonfinite_count = 0
    maximum_error = maximum_ratio = 0.
    samples, differences = [], []
    for start in range(0, a.numel(), VECTOR_CHUNK_SIZE):
        # Match the sealed native check's FP32 coordinate comparison exactly;
        # FP64 scalar energies remain FP64 so their differences are not hidden.
        arithmetic = torch.float64 if torch.float64 in (a.dtype, b.dtype) else torch.float32
        aa = a[start:start+VECTOR_CHUNK_SIZE].to(arithmetic)
        bb = b[start:start+VECTOR_CHUNK_SIZE].to(arithmetic)
        finite = torch.isfinite(aa) & torch.isfinite(bb)
        error = aa-bb
        tolerance = GRADIENT_TOLERANCES['atol'] + GRADIENT_TOLERANCES['rtol']*bb.abs()
        different = ~finite | (aa != bb)
        violations = ~finite | (error.abs() > tolerance)
        mismatch_count += int(different.sum())
        violation_count += int(violations.sum())
        nonfinite_count += int((~finite).sum())
        if bool(finite.any()):
            af, bf, ef = aa[finite], bb[finite], error[finite]
            norms += torch.stack((af.double().square().sum(), bf.double().square().sum(), ef.double().square().sum()))
            maximum_error = max(maximum_error, float(ef.abs().max()))
            maximum_ratio = max(maximum_ratio, float((ef.abs()/tolerance[finite]).max()))
        # Retain violations first; ordinary differences are represented by the
        # complete counts/norms rather than crowding out failing coordinates.
        if len(samples) < sample_size and bool(violations.any()):
            for index in violations.nonzero().flatten()[:sample_size-len(samples)].tolist():
                samples.append(dict(index=start+index, actual=_number(aa[index]),
                    reference=_number(bb[index]), error=_number(error[index]),
                    tolerance=_number(tolerance[index])))
        if len(differences) < sample_size and bool(different.any()):
            for index in different.nonzero().flatten()[:sample_size-len(differences)].tolist():
                differences.append(dict(index=start+index, actual=_number(aa[index]),
                    reference=_number(bb[index]), error=_number(error[index]),
                    tolerance=_number(tolerance[index])))
    a_square, b_square, error_square = norms.tolist()
    relative = math.sqrt(error_square/b_square) if b_square else (0. if error_square == 0 else math.inf)
    passed = not nonfinite_count and not violation_count and relative <= GRADIENT_TOLERANCES['maximum_relative_l2']
    return dict(shape=list(actual.shape), actual_dtype=str(actual.dtype), reference_dtype=str(reference.dtype),
        coordinates=a.numel(), exact=mismatch_count == 0, differing_coordinates=mismatch_count,
        violations=violation_count, violation_fraction=violation_count/a.numel() if a.numel() else 0.,
        nonfinite_coordinates=nonfinite_count, maximum_absolute_error=maximum_error,
        maximum_error_to_tolerance=maximum_ratio, actual_norm=math.sqrt(a_square),
        reference_norm=math.sqrt(b_square), discrepancy_norm=math.sqrt(error_square),
        relative_l2=_number(relative), norms_scope='finite coordinates only' if nonfinite_count else 'all coordinates',
        matches_pinned_tolerance=passed, tolerances=dict(GRADIENT_TOLERANCES),
        failing_sample=samples, differing_sample=differences)


def run_head_diagnostic(head, pre_hidden, native_hidden, normalized_weights, *,
                        chunk_blocks=1, include_fixed_cotangent=True,
                        return_tensors=False, work_callback=None):
    """Compare three layouts, plus three fixed-cotangent layouts if needed.

    Inputs are the identical detached BF16 pre/native hidden tensors for at most
    two original anchors, shaped [1, anchors, block, hidden]. Weights already
    contain the complete fixture's fixed response normalization. No renormalizing
    or sampling occurs here. Returned tensors, if requested, are detached hidden
    adjoints and the shared cotangent; callers own their eventual release.
    """
    if (pre_hidden.ndim != 4 or pre_hidden.shape != native_hidden.shape
            or pre_hidden.shape[0] != 1 or not 1 <= pre_hidden.shape[1] <= 2
            or pre_hidden.dtype != torch.bfloat16 or native_hidden.dtype != torch.bfloat16
            or normalized_weights.shape != pre_hidden.shape[:-1] or chunk_blocks != 1):
        raise ValueError('Head diagnostic requires the fixed one/two-block BF16 fixture and chunk size one')
    receipt = dict(status='running', variants=list(VARIANTS), reference=REFERENCE,
        input_shape=list(pre_hidden.shape), weight_sum=float(normalized_weights.double().sum()),
        gradient_tolerances=dict(GRADIENT_TOLERANCES), work={}, trials={}, comparisons={},
        fixed_cotangent_executed=False, interpretation='zero-update diagnostic, not qualification acceptance')
    before_versions = [(id(p), p._version, p.grad) for p in head.parameters()]
    modes = [(m, m.training) for m in head.modules()]
    all_outputs, fixed_outputs = {}, {}
    phase = 'initial'

    @contextmanager
    def work(kind):
        key = kind+'_attempted'; receipt['work'][key] = receipt['work'].get(key, 0)+1
        if work_callback is not None:
            work_callback(kind, 'attempted')
        try:
            yield
        except _StopRecomputationError:
            # Non-reentrant checkpointing stops once needed tensors exist.
            # This is expected partial work, not a failed numerical operation.
            key = kind+'_checkpoint_early_stop'
            receipt['work'][key] = receipt['work'].get(key, 0)+1
            if work_callback is not None:
                work_callback(kind, 'checkpoint_early_stop')
            raise
        except BaseException:
            key = kind+'_failed'; receipt['work'][key] = receipt['work'].get(key, 0)+1
            if work_callback is not None:
                work_callback(kind, 'failed')
            raise
        else:
            key = kind+'_completed'; receipt['work'][key] = receipt['work'].get(key, 0)+1
            if work_callback is not None:
                work_callback(kind, 'completed')

    def project(hidden):
        with work('head_projection'):
            logits = head(hidden)
        if not bool(torch.isfinite(logits).all()):
            receipt['nonfinite'] = dict(phase=phase, quantity='head_logits',
                summary=tensor_comparison(logits, logits))
            raise HeadDiagnosticError('Nonfinite head diagnostic logits', receipt)
        return logits

    def run_variant(name, fixed=None):
        live = pre_hidden.detach().clone().requires_grad_(True)
        native = native_hidden.detach()
        weights = normalized_weights.detach()
        ordinals = torch.arange(live.shape[1], device=live.device).reshape(1, -1)
        captured = {}

        def terms(current, saved_native, fixed_weights, indices, common_cotangent):
            with work('proxy_call'):
                logits = project(current)
                with torch.no_grad():
                    native_logits = project(saved_native)
                    cotangent, energy = geometry_terms(logits, native_logits, fixed_weights)
                    if common_cotangent is not None:
                        cotangent = common_cotangent
                proxy = (logits.float()*cotangent).sum(dtype=torch.float32)
                start = int(indices[0, 0])
                if start not in captured:
                    captured[start] = dict(pre_logits=logits.detach().clone(),
                        native_logits=native_logits.detach().clone(), cotangent=cotangent.detach().clone(),
                        autograd_entry_cotangent=cotangent.detach().to(logits.dtype).clone(),
                        energy=energy.detach().clone(), proxy=proxy.detach().clone())
                return proxy, energy

        chunks = 0 if name == REFERENCE else chunk_blocks
        with torch.enable_grad(), preserve_rng():
            proxy, energy = checkpointed_chunk_reduce(terms, live, native, weights,
                ordinals, fixed, chunk_size=chunks, dim=1, checkpoint=name == VARIANTS[0])
            with work('head_backward'):
                adjoint, = torch.autograd.grad(proxy, live)
        result = {key: torch.cat([captured[i][key] for i in sorted(captured)], dim=1)
                  for key in ('pre_logits', 'native_logits', 'cotangent', 'autograd_entry_cotangent')}
        result.update(adjoint=adjoint.detach(), proxy=proxy.detach().reshape(1), energy=energy.detach().reshape(1))
        receipt['trials'][phase+'/'+name] = dict(chunks=len(captured),
            projection_checkpoint=name == VARIANTS[0], proxy=_number(proxy), energy=_number(energy),
            adjoint_dtype=str(adjoint.dtype), fixed_cotangent=fixed is not None)
        for key, tensor in result.items():
            if not bool(torch.isfinite(tensor).all()):
                summary = tensor_comparison(tensor, tensor)
                receipt['nonfinite'] = dict(phase=phase, variant=name, quantity=key, summary=summary)
                raise HeadDiagnosticError('Nonfinite head diagnostic '+key, receipt)
        return result

    def compare(outputs):
        reference = outputs[REFERENCE]
        return {name: {key: tensor_comparison(output[key], reference[key])
                       for key in reference} for name, output in outputs.items() if name != REFERENCE}

    try:
        with preserve_rng():
            for name in VARIANTS:
                all_outputs[name] = run_variant(name)
            receipt['comparisons']['initial'] = compare(all_outputs)
            receipt['checkpoint_effect'] = {'initial': {
                key: tensor_comparison(all_outputs[VARIANTS[0]][key], all_outputs[VARIANTS[1]][key])
                for key in all_outputs[REFERENCE]}}
            differs = any(not row[key]['exact'] for row in receipt['comparisons']['initial'].values()
                          for key in ('pre_logits', 'native_logits', 'cotangent', 'autograd_entry_cotangent', 'adjoint'))
            if include_fixed_cotangent and differs:
                phase = 'fixed_cotangent'
                common = all_outputs[REFERENCE]['cotangent']
                receipt['fixed_cotangent_executed'] = True
                for name in VARIANTS:
                    fixed_outputs[name] = run_variant(name, common)
                receipt['comparisons']['fixed_cotangent'] = compare(fixed_outputs)
                receipt['checkpoint_effect']['fixed_cotangent'] = {
                    key: tensor_comparison(fixed_outputs[VARIANTS[0]][key], fixed_outputs[VARIANTS[1]][key])
                    for key in fixed_outputs[REFERENCE]}
            receipt.update(status='completed', differences_observed=differs,
                initial_adjoint_gate_passes=all(x['adjoint']['matches_pinned_tolerance']
                    for x in receipt['comparisons']['initial'].values()))
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        if isinstance(error, HeadDiagnosticError):
            raise
        raise HeadDiagnosticError(str(error), receipt) from error
    finally:
        receipt['head_parameters_and_gradients_unchanged'] = all(
            id(p) == identity and p._version == version and p.grad is grad
            for p, (identity, version, grad) in zip(head.parameters(), before_versions))
        receipt['head_modes_unchanged'] = all(m.training == training for m, training in modes)
    if not receipt['head_parameters_and_gradients_unchanged'] or not receipt['head_modes_unchanged']:
        receipt.update(status='failed', error='Head diagnostic mutated model state')
        raise HeadDiagnosticError('Head diagnostic mutated model state', receipt)
    tensors = dict(adjoints={k:v['adjoint'] for k,v in all_outputs.items()},
        fixed_adjoints={k:v['adjoint'] for k,v in fixed_outputs.items()},
        cotangent=all_outputs[REFERENCE]['cotangent'])
    return (receipt, tensors) if return_tensors else receipt
