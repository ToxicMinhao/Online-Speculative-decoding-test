"""Remaining native numerical and maximum-shape checks, without Adam updates.

The padded repeated-anchor fixture tests resources only. It never supplies
feedback, changes a production checkpoint, or measures adaptation quality.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import time

import torch

from experiments.ok_continuation.resume import differences
from experiments.prefix_guard.qualify import preserve_failure
from online_feedback import functional_geometry_math as geometry
from online_feedback.functional_geometry import isolated_geometry_runtime
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json
from .native_oracles import (
    _expanded_reference, gradient_comparison,
    optimizer_versions, response_fixture, NATIVE_VJP_TOLERANCES,
)


RESOURCE_BUDGET = dict(additional_response_forwards=9, additional_backwards=4,
                       optimizer_steps=0)
SAME_LAYOUT_BUDGET = dict(additional_response_forwards=5, additional_backwards=2,
                          optimizer_steps=0)


def maximum_resource_fixture(response, sequence_length, *, anchors=512):
    """Prepend masked synthetic context, retaining and shifting all bindings.

    Left padding exercises long attended prefixes as well as full context/KV
    allocation. Its zero target features are synthetic and explicitly unsuitable
    for scientific loss or efficacy comparisons.
    """
    original_length = response['input_ids'].shape[1]
    if sequence_length < original_length:
        raise ValueError('Resource context may be extended, never truncated')
    result = response_fixture(response, anchors)
    padding = sequence_length - original_length
    for name in ('input_ids', 'hidden_states', 'loss_mask'):
        original = response[name]
        shape = (original.shape[0], padding, *original.shape[2:])
        prefix = original.new_zeros(shape)
        if name == 'input_ids' and padding:
            prefix.fill_(int(original[0, 0]))
        result[name] = torch.cat((prefix, original), dim=1)
    result['anchors'] = result['anchors'] + padding
    result['qualification_fixture'].update(
        input_length=sequence_length, original_input_length=original_length,
        prepended_masked_positions=padding,
        teacher_feature_shape=list(result['hidden_states'].shape),
        maximum_training_sequence_length_tested=True,
        context_construction='masked left prefix with zero synthetic target features',
        scientific_feedback=False, training_loss_interpretation=False)
    return result


@contextmanager
def restored_check(hook, root, receipt, budget):
    """Restore the actual committed endpoint and preserve the first failure."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before_work = dict(hook.work)
    before_rng = capture_rng()
    before_modes = tuple(module.training for module in hook.model.modules())
    before_grad = tuple((parameter.grad, None if parameter.grad is None else parameter.grad._version)
                        for parameter in hook.parameters)
    optimizer = hook.trainer.backend.optimizer.optimizer
    versions = optimizer_versions(optimizer)
    committed = tuple(parameter.detach().clone() for parameter in hook.active_parameters)
    hook.work['state_copy_passes'] += 1
    receipt.update(status='running', budget=budget)
    write_json(root / 'PLAN.json', receipt)
    failure = None
    try:
        with isolated_geometry_runtime(hook.model):
            yield committed
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

        def state_checks():
            receipt['committed_weights_exact'] = all(torch.equal(parameter, saved)
                for parameter, saved in zip(hook.active_parameters, committed))
            receipt['rng_exact'] = not differences(before_rng, capture_rng())
            receipt['modes_exact'] = before_modes == tuple(module.training for module in hook.model.modules())
            receipt['gradients_unchanged'] = all(parameter.grad is saved and
                (saved is None or saved._version == version)
                for parameter, (saved, version) in zip(hook.parameters, before_grad))
            receipt['optimizer_tensor_identity_and_versions_exact'] = versions == optimizer_versions(optimizer)
            checks = ('committed_weights_exact', 'rng_exact', 'modes_exact',
                      'gradients_unchanged', 'optimizer_tensor_identity_and_versions_exact')
            if not all(receipt[name] for name in checks):
                raise ValueError('Qualification helper did not restore its scientific state')

        cleanup(lambda: hook.restore_endpoint(committed), 'qualification endpoint restore')
        cleanup(state_checks, 'qualification state checks')
        receipt['actual_work'] = {key: value - before_work.get(key, 0)
            for key, value in hook.work.items() if isinstance(value, int)}
        receipt['wall_s'] = time.perf_counter() - started

        def check_budget():
            for kind, maximum in budget.items():
                if receipt['actual_work'].get(kind + '_attempted', 0) > maximum:
                    raise ValueError('Qualification helper exceeded its budget: ' + kind)

        cleanup(check_budget, 'qualification budget checks')
        cleanup(lambda: write_json(root / 'RESULT.json', receipt), 'qualification result persistence')
    if failure is not None:
        raise failure


def run_same_layout_check(hook, output):
    """Compare independent formulas on one complete eight-block tile plus one.

    The original cross-layout failures remain historical evidence. This check
    changes only the explicitly revised numerical oracle, not its tolerances.
    """
    from .same_layout_oracle import same_layout_reference

    if hook.model.objective_chunk_blocks != 8:
        raise ValueError('Native qualification requires the pinned eight-block head layout')
    root = Path(output)
    small = response_fixture(hook.responses[0], 9)
    names = [hook.parameter_names[id(parameter)] for parameter in hook.active_parameters]
    receipt = dict(fixture=small['qualification_fixture'], chunks=[8, 1],
        tolerances=NATIVE_VJP_TOLERANCES, numerical_protocol='independent_same_native_layout',
        historical_cross_layout_gate_passed=False)
    with restored_check(hook, root, receipt, SAME_LAYOUT_BUDGET) as committed:
        committed_hidden = hook.forward_hidden(small, 'same_layout_committed_before', grad=False).detach().cpu()
        hook.restore_endpoint(hook.theta_native)
        native_hidden = hook.forward_hidden(small, 'same_layout_native', grad=False).detach()
        hook.restore_endpoint(hook.theta0)
        production = hook.build_reference(small, native_hidden, hook.active_parameters, chunk_blocks=8)
        live_hidden = hook.forward_hidden(small, 'same_layout_independent_prestate', grad=True)
        receipt['prestate_hidden_exact'] = bool(torch.equal(production.hidden, live_hidden.detach()))
        if not receipt['prestate_hidden_exact']:
            raise ValueError('Same-layout independent graph has different pre-state features')

        def count_backward(kind, status):
            if kind != 'additional_backwards':
                raise ValueError('Unexpected work category from the independent oracle')
            if hook.work_callback is not None:
                hook.work_callback(kind, status)
            hook.work[kind + '_' + status] += 1
            hook._write_work()

        hook.phase = 'same_layout_independent_oracle'
        independent = same_layout_reference(hook.model.lm_head, live_hidden, native_hidden,
            small['labels'], small['validity_weights'], hook.active_parameters,
            eos_token_ids=hook.model.eos_token_ids, chunk_blocks=8, work_callback=count_backward)
        del live_hidden
        gradient = gradient_comparison(production.gradients, independent.gradients, names)
        rtol, atol = NATIVE_VJP_TOLERANCES['rtol'], NATIVE_VJP_TOLERANCES['atol']
        native_risk = geometry.pooled_risk([production.native_risk_terms], hook.vocab_size)
        independent_risk = geometry.pooled_risk([independent.native_risk_terms], hook.vocab_size)
        risk_error = abs(native_risk.raw - independent_risk.raw)
        receipt.update(native_vjp=gradient,
            proposals_exact=bool(torch.equal(production.proposals, independent.proposals)),
            prefix_mask_exact=bool(torch.equal(production.prefix_mask, independent.prefix_mask)),
            weights_exact=bool(torch.equal(production.weights, independent.weights)),
            weights_close=bool(torch.allclose(production.weights, independent.weights, rtol=rtol, atol=atol)),
            weight_maximum_difference=float((production.weights - independent.weights).abs().max()),
            margins_close=bool(torch.allclose(production.margins, independent.margins, rtol=rtol, atol=atol)),
            production_c=production.c, independent_c=independent.c,
            energy_close=math.isclose(production.c, independent.c, rel_tol=rtol, abs_tol=atol),
            positive_output_energy=production.c > 0 and independent.c > 0,
            native_risk=native_risk.as_dict(), independent_risk=independent_risk.as_dict(),
            risk_absolute_difference=risk_error,
            risk_within_existing_deadband=risk_error <= native_risk.eta + independent_risk.eta,
            independent_statistics=independent.statistics)
        write_json(root / 'SAME_LAYOUT_ORACLE.json', receipt)
        required = ('proposals_exact', 'prefix_mask_exact', 'weights_close', 'margins_close',
                    'energy_close', 'positive_output_energy', 'risk_within_existing_deadband')
        if gradient['status'] != 'passed' or not all(receipt[key] for key in required):
            raise ValueError('Independent same-layout native oracle failed; evidence preserved')
        del independent, native_hidden
        hook.restore_endpoint(committed)
        restored = hook.forward_hidden(small, 'same_layout_committed_restored', grad=False).detach().cpu()
        receipt['restored_hidden_exact'] = bool(torch.equal(committed_hidden, restored))
        if not receipt['restored_hidden_exact']:
            raise ValueError('Committed features changed after the independent native oracle')
    return receipt, production


def run_resource_checks(hook, output, seed_reference, *, sequence_length):
    """Exercise 512 anchors and four simultaneous BF16 VJPs once.

    A detached production reference from the numerical oracle supplies a small
    disposable seed direction, just as the original resource check did. None of
    these candidate parameters is retained after the check.
    """
    if sequence_length != hook.trainer.max_length:
        raise ValueError('Resource sequence length must equal the qualified training limit')
    root = Path(output)
    small = response_fixture(hook.responses[0], 9)
    resource = [maximum_resource_fixture(response, sequence_length) for response in hook.responses]
    receipt = dict(resource_fixtures=[row['qualification_fixture'] for row in resource],
        interpretation='synthetic maximum-shape resource check only; not fit or transfer evidence')
    with restored_check(hook, root, receipt, RESOURCE_BUDGET) as committed:
        committed_hidden = hook.forward_hidden(small, 'resource_committed_before', grad=False).detach().cpu()
        seed = geometry.make_sketch([seed_reference.gradients], [seed_reference.c],
                                    hook.theta0, hook.theta_native)
        geometry.write_candidate(hook.active_parameters, hook.theta0, hook.theta_native, seed)
        raw_hidden = [hook.forward_hidden(response, 'resource_seed_candidate', grad=False).detach()
                      for response in resource]
        hook.restore_endpoint(hook.theta_native)
        native_hidden = [hook.forward_hidden(response, 'resource_native', grad=False).detach()
                         for response in resource]
        hook.restore_endpoint(hook.theta0)
        expanded = [_expanded_reference(hook, response, native, raw)
                    for response, native, raw in zip(resource, native_hidden, raw_hidden)]
        gradient_bytes = sum(value.numel() * value.element_size()
            for row in expanded for gradient in row['gradients'] for value in gradient if value is not None)
        candidate = geometry.make_expanded_sketch([row['gradients'] for row in expanded],
            [row['covariance'] for row in expanded], hook.theta0, hook.theta_native,
            production_sigma=seed.sigma)
        endpoint = geometry.write_candidate(hook.active_parameters, hook.theta0, hook.theta_native, candidate)
        checked = hook.forward_hidden(resource[0], 'resource_expanded_candidate', grad=False).detach()
        risk_parts = []
        for start in range(0, checked.shape[1], 8):
            stop = start + 8
            with torch.no_grad():
                before_logits = hook.model.lm_head(expanded[0]['hidden'][:, start:stop])
                after_logits = hook.model.lm_head(checked[:, start:stop])
                risk_parts.append(geometry.risk_terms(before_logits, after_logits,
                                                     expanded[0]['weights'][:, start:stop]))
        risk = geometry.pooled_risk(risk_parts, hook.vocab_size)
        receipt['expanded'] = dict(response_anchor_counts=[512, 512],
            input_lengths=[sequence_length, sequence_length],
            simultaneous_native_vjp_vectors=4, native_gradient_bytes=gradient_bytes,
            protected_occurrences=[row['protected_occurrences'] for row in expanded],
            sketch=candidate.statistics, endpoint=endpoint,
            first_response_half_weight_risk=risk.as_dict(),
            second_direction_source='disposable_numerical_oracle_production_reference',
            maximum_training_sequence_length_tested=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        del expanded, candidate, seed, raw_hidden, native_hidden, checked
        hook.restore_endpoint(committed)
        restored = hook.forward_hidden(small, 'resource_committed_restored', grad=False).detach().cpu()
        receipt['restored_hidden_exact'] = bool(torch.equal(committed_hidden, restored))
        if not receipt['restored_hidden_exact']:
            raise ValueError('Committed prediction features changed after resource check')
    return receipt
