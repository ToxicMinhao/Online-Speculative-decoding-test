"""Bounded native VJP and expanded-working-set checks; no optimizer updates.

The repeated-anchor fixture tests resource feasibility only. It is never used as
feedback, a production update, or evidence of beneficial adaptation.
"""
from pathlib import Path
import time

import torch

from experiments.ok_continuation.resume import differences
from experiments.prefix_guard.qualify import preserve_failure
from online_feedback.functional_geometry import isolated_geometry_runtime
from online_feedback import functional_geometry_math as geometry
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json
from specforge.core.chunking import checkpointed_chunk_reduce


# BF16 coordinate tolerance matches the existing explicit native-gradient
# convention in prefix_ranking/qualify.py. A global relative-error requirement
# prevents the absolute floor from admitting a wholly inaccurate tiny vector.
NATIVE_VJP_TOLERANCES = dict(rtol=0.016, atol=1e-5, maximum_relative_l2=0.016)
ORACLE_BUDGET = dict(additional_response_forwards=12, additional_backwards=6,
                     optimizer_steps=0, serving_generations=0)


def response_fixture(response, anchors):
    """Take the first two or repeat original anchors, without a sampler call."""
    count = response['anchors'].shape[1]
    if count == 0:
        raise ValueError('Native oracle needs at least one retained anchor')
    index = torch.arange(anchors) % count
    result = dict(response)
    for name in ('anchors', 'keep', 'labels', 'validity_weights'):
        result[name] = response[name].index_select(1, index.to(response[name].device))
    result['qualification_fixture'] = dict(original_anchors=count, fixture_anchors=anchors,
        input_length=response['input_ids'].shape[-1],
        teacher_feature_shape=list(response['hidden_states'].shape),
        construction='original_anchors_in_order_repeated_modulo_count', sampled_new_anchors=False,
        maximum_training_sequence_length_tested=False)
    return result


@torch.no_grad()
def gradient_comparison(left, right, names):
    """Summarize all coordinates and a bounded example before reporting failure."""
    rows, norm_left, norm_right, error_square, violations = [], 0., 0., 0., 0
    for name, actual, expected in zip(names, left, right):
        if actual is None or expected is None:
            equal = actual is None and expected is None
            rows.append(dict(parameter=name, none_pattern_equal=equal, coordinates=0))
            if not equal:
                violations += 1
            continue
        if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
            raise ValueError('Native oracle VJP is not BF16')
        a, b = actual.reshape(-1), expected.reshape(-1)
        sums = torch.zeros(3, dtype=torch.float64, device=a.device)
        bad_count = torch.zeros((), dtype=torch.int64, device=a.device)
        maximum_error = torch.zeros((), dtype=torch.float32, device=a.device)
        sample = []
        for start in range(0, a.numel(), geometry.VECTOR_CHUNK_SIZE):
            aa = a[start:start+geometry.VECTOR_CHUNK_SIZE].float()
            bb = b[start:start+geometry.VECTOR_CHUNK_SIZE].float()
            error = aa-bb
            limit = NATIVE_VJP_TOLERANCES['atol']+NATIVE_VJP_TOLERANCES['rtol']*bb.abs()
            bad = error.abs() > limit
            sums += torch.stack((aa.double().square().sum(), bb.double().square().sum(), error.double().square().sum()))
            bad_count += bad.sum()
            maximum_error = torch.maximum(maximum_error, error.abs().max())
            if len(sample) < 4 and bool(bad.any()):
                for index in bad.nonzero().flatten()[:4-len(sample)].tolist():
                    sample.append(dict(index=start+index, chunked=float(aa[index]), unchunked=float(bb[index]),
                                       error=float(error[index]), tolerance=float(limit[index])))
        al, bl, el = sums.tolist()
        count = int(bad_count)
        norm_left += al; norm_right += bl; error_square += el; violations += count
        rows.append(dict(parameter=name, dtype=str(actual.dtype), shape=list(actual.shape),
            coordinates=a.numel(), violations=count, maximum_absolute_error=float(maximum_error),
            chunked_norm=al**.5, unchunked_norm=bl**.5, discrepancy_norm=el**.5,
            failing_sample=sample, none_pattern_equal=True))
    relative = (error_square/norm_right)**.5 if norm_right else (0. if error_square == 0 else float('inf'))
    passed = violations == 0 and relative <= NATIVE_VJP_TOLERANCES['maximum_relative_l2']
    return dict(status='passed' if passed else 'failed', tolerances=NATIVE_VJP_TOLERANCES,
        violations=violations, relative_l2=relative, discrepancy_norm=error_square**.5,
        chunked_norm=norm_left**.5, unchunked_norm=norm_right**.5, parameters=rows)


def optimizer_versions(optimizer):
    """Version/identity checks complement the root's decoded optimizer equality."""
    return [(id(parameter), name, id(value), value._version, tuple(value.shape))
            for parameter, state in optimizer.state.items()
            for name, value in state.items() if isinstance(value, torch.Tensor)]


@torch.no_grad()
def direct_risk_check(hook, reference, native_hidden):
    """Compare reductions of the same actual BF16 head outputs, not new draws."""
    before = hook.model.lm_head(reference.hidden)
    native = hook.model.lm_head(native_hidden)
    whole = geometry.pooled_risk([geometry.risk_terms(before, native, reference.weights)], before.shape[-1])
    parts = [geometry.risk_terms(before[:, i:i+1], native[:, i:i+1], reference.weights[:, i:i+1])
             for i in range(before.shape[1])]
    chunked = geometry.pooled_risk(parts, before.shape[-1])
    difference = abs(whole.raw-chunked.raw)
    passed = difference <= whole.eta+chunked.eta
    return dict(status='passed' if passed else 'failed', whole=whole.as_dict(), chunked=chunked.as_dict(),
                absolute_difference=difference, same_native_logits=True, response_forwards=0)


def _expanded_reference(hook, response, native_hidden, geometry_hidden):
    """One max-shape pre-state forward followed by two first-order VJPs."""
    hidden = hook.forward_hidden(response, 'oracle_expanded_prestate', grad=True)
    weights, _, mask, _ = hook._scan_reference(hidden.detach(), response, hook.model.objective_chunk_blocks)
    def terms(live_hidden, native, filtered, fixed_weights):
        hook.work['checkpoint_proxy_calls'] += 1
        logits = hook.model.lm_head(live_hidden)
        with torch.no_grad():
            native_logits = hook.model.lm_head(native)
            filtered_logits = hook.model.lm_head(filtered)
            cotangents, covariance = geometry.expanded_terms(logits, native_logits, filtered_logits, fixed_weights)
        proxies = tuple((logits.float()*cotangent).sum(dtype=torch.float32) for cotangent in cotangents)
        return (*proxies, covariance)
    hook.phase = 'oracle_expanded_proxy'
    proxy_native, proxy_filtered, covariance = checkpointed_chunk_reduce(
        terms, hidden, native_hidden, geometry_hidden, weights,
        chunk_size=hook.model.objective_chunk_blocks, dim=1, checkpoint=True)
    gradients = []
    for ordinal, proxy in enumerate((proxy_native, proxy_filtered)):
        hook.phase = 'oracle_expanded_vjp'
        with hook._work('additional_backwards'), hook._timed('oracle_expanded_backward_s'):
            gradient = torch.autograd.grad(proxy, hook.active_parameters, allow_unused=True, retain_graph=ordinal == 0)
        for value in gradient:
            if value is not None:
                geometry.finite(value, 'expanded native VJP')
        gradients.append(tuple(None if value is None else value.detach() for value in gradient))
    return dict(hidden=hidden.detach(), weights=weights.detach(), covariance=covariance.detach(),
                gradients=gradients, protected_occurrences=int(mask.sum()))


def run_oracles(hook, output):
    """Run once inside the FSG committed observer, restoring its exact endpoint."""
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before_work = dict(hook.work)
    before_rng = capture_rng()
    before_modes = tuple(module.training for module in hook.model.modules())
    before_grad = tuple(parameter.grad for parameter in hook.parameters)
    optimizer = hook.trainer.backend.optimizer.optimizer
    versions = optimizer_versions(optimizer)
    with hook._timed('oracle_committed_snapshot_s'):
        committed = tuple(parameter.detach().clone() for parameter in hook.active_parameters)
    hook.work['state_copy_passes'] += 1
    names = [hook.parameter_names[id(parameter)] for parameter in hook.active_parameters]
    small = response_fixture(hook.responses[0], min(2, hook.responses[0]['anchors'].shape[1]))
    resource = [response_fixture(response, 512) for response in hook.responses]
    receipt = dict(status='running', budget=ORACLE_BUDGET, tolerances=NATIVE_VJP_TOLERANCES,
        small_fixture=small['qualification_fixture'], resource_fixtures=[r['qualification_fixture'] for r in resource],
        interpretation='qualification/resource fixtures only; not source-fit or transfer evidence')
    write_json(root/'PLAN.json', receipt)
    failure = None
    try:
        with isolated_geometry_runtime(hook.model):
            # One pre-oracle committed-output observation is reused after the
            # final physical restore. Both are charged response-level forwards.
            committed_hidden = hook.forward_hidden(small, 'oracle_committed_before', grad=False).detach().cpu()
            hook.restore_endpoint(hook.theta_native)
            native_small = hook.forward_hidden(small, 'oracle_small_native', grad=False).detach()
            hook.restore_endpoint(hook.theta0)
            chunked = hook.build_reference(small, native_small, hook.active_parameters, chunk_blocks=1)
            unchunked = hook.build_reference(small, native_small, hook.active_parameters, chunk_blocks=0)
            gradient = gradient_comparison(chunked.gradients, unchunked.gradients, names)
            receipt['native_vjp'] = gradient
            receipt['prefix_mask_exact'] = bool(torch.equal(chunked.prefix_mask, unchunked.prefix_mask))
            receipt['weight_maximum_difference'] = float((chunked.weights-unchunked.weights).abs().max())
            receipt['weights_close'] = bool(torch.allclose(chunked.weights, unchunked.weights,
                rtol=NATIVE_VJP_TOLERANCES['rtol'], atol=NATIVE_VJP_TOLERANCES['atol']))
            receipt['positive_output_energy'] = chunked.c > 0 and unchunked.c > 0
            with hook._timed('oracle_direct_risk_s'):
                receipt['direct_risk'] = direct_risk_check(hook, chunked, native_small)
            write_json(root/'SMALL_ORACLE.json', receipt)
            if (gradient['status'] != 'passed' or not receipt['prefix_mask_exact']
                    or not receipt['weights_close'] or not receipt['positive_output_energy']
                    or receipt['direct_risk']['status'] != 'passed'):
                raise ValueError('Native chunked/unchunked oracle failed; preserved diagnostics')
            # A disposable rank-one candidate defines a second output direction
            # for the resource fixture. Its outcome never affects production.
            sketch = geometry.make_sketch([chunked.gradients], [chunked.c], hook.theta0, hook.theta_native)
            geometry.write_candidate(hook.active_parameters, hook.theta0, hook.theta_native, sketch)
            raw_hidden = [hook.forward_hidden(response, 'oracle_resource_geometry', grad=False).detach()
                          for response in resource]
            production_sigma = sketch.sigma
            del unchunked, chunked, sketch, native_small
            hook.restore_endpoint(hook.theta_native)
            native_hidden = [hook.forward_hidden(response, 'oracle_resource_native', grad=False).detach()
                             for response in resource]
            hook.restore_endpoint(hook.theta0)
            expanded = [_expanded_reference(hook, response, native, filtered)
                        for response, native, filtered in zip(resource, native_hidden, raw_hidden)]
            # Both responses' two native BF16 gradient vectors coexist here.
            native_gradient_bytes = sum(value.numel()*value.element_size()
                for row in expanded for gradient in row['gradients'] for value in gradient if value is not None)
            candidate = geometry.make_expanded_sketch([row['gradients'] for row in expanded],
                [row['covariance'] for row in expanded], hook.theta0, hook.theta_native,
                production_sigma=production_sigma)
            endpoint = geometry.write_candidate(hook.active_parameters, hook.theta0, hook.theta_native, candidate)
            checked_hidden = hook.forward_hidden(resource[0], 'oracle_expanded_candidate', grad=False).detach()
            risk_parts = []
            chunk = hook.model.objective_chunk_blocks
            for start in range(0, checked_hidden.shape[1], chunk):
                stop = start+chunk
                with torch.no_grad():
                    before_logits = hook.model.lm_head(expanded[0]['hidden'][:, start:stop])
                    candidate_logits = hook.model.lm_head(checked_hidden[:, start:stop])
                    risk_parts.append(geometry.risk_terms(before_logits, candidate_logits,
                                                         expanded[0]['weights'][:, start:stop]))
            risk = geometry.pooled_risk(risk_parts, hook.vocab_size)
            receipt['resource_expanded'] = dict(status='passed', response_anchor_counts=[512, 512],
                input_lengths=[response['input_ids'].shape[-1] for response in resource],
                maximum_training_sequence_length_tested=False,
                simultaneous_native_vjp_vectors=4, native_gradient_bytes=native_gradient_bytes,
                protected_occurrences=[row['protected_occurrences'] for row in expanded],
                sketch=candidate.statistics, endpoint=endpoint, first_response_half_weight_risk=risk.as_dict(),
                second_direction_source='disposable_small_anchor_raw_fsg_candidate',
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
            del expanded, candidate, raw_hidden, native_hidden, checked_hidden
            hook.restore_endpoint(committed)
            restored_hidden = hook.forward_hidden(small, 'oracle_committed_restored', grad=False).detach().cpu()
            receipt['restored_hidden_exact'] = bool(torch.equal(committed_hidden, restored_hidden))
            if not receipt['restored_hidden_exact']:
                raise ValueError('Committed endpoint replay changed after native oracles')
        receipt['status'] = 'passed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        def cleanup(action, label):
            nonlocal failure
            if failure is not None:
                preserve_failure(failure, action, receipt, label)
            else:
                try:
                    action()
                except BaseException as error:
                    failure = error
                    receipt.update(status='failed', error_type=type(error).__name__, error=str(error))

        def check_restoration():
            receipt['committed_weights_exact'] = all(torch.equal(parameter, saved)
                                                     for parameter, saved in zip(hook.active_parameters, committed))
            receipt['rng_exact'] = not differences(before_rng, capture_rng())
            receipt['modes_exact'] = before_modes == tuple(module.training for module in hook.model.modules())
            receipt['gradient_pattern_exact'] = all(parameter.grad is saved for parameter, saved in zip(hook.parameters, before_grad))
            receipt['optimizer_tensor_identity_and_versions_exact'] = versions == optimizer_versions(optimizer)

        cleanup(lambda: hook.restore_endpoint(committed), 'oracle final endpoint restoration')
        cleanup(check_restoration, 'oracle final state checks')
        receipt['actual_work'] = {key: hook.work[key]-before_work.get(key, 0) for key in hook.work
                                  if isinstance(hook.work[key], int)}
        receipt['wall_s'] = time.perf_counter()-started
        cleanup(lambda: write_json(root/'RESULT.json', receipt), 'oracle result persistence')
    if failure is not None:
        raise failure
    if not all(receipt[name] for name in ('committed_weights_exact', 'rng_exact', 'modes_exact',
                                          'gradient_pattern_exact', 'optimizer_tensor_identity_and_versions_exact')):
        receipt.update(status='failed', error='Native oracle state restoration failed')
        error = ValueError(receipt['error'])
        preserve_failure(error, lambda: write_json(root/'RESULT.json', receipt), receipt, 'oracle failed-state persistence')
        raise error
    for key, maximum in ORACLE_BUDGET.items():
        actual = receipt['actual_work'].get(key+'_attempted', 0)
        if actual > maximum:
            receipt.update(status='failed', error='Native oracle budget exceeded: '+key)
            error = ValueError(receipt['error'])
            preserve_failure(error, lambda: write_json(root/'RESULT.json', receipt), receipt, 'oracle failed-budget persistence')
            raise error
    return receipt
