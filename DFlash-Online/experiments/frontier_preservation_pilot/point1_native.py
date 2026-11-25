"""Four restored native updates tracing allocation through optimizer arithmetic.

All large vectors are temporary RAM objects. The parent recovery remains
read-only, and only scalar receipts and frozen small metadata reach disk.
"""

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import time
import traceback

import torch

from experiments.full_k_gsm8k.training import verify_live
from experiments.functional_step_geometry.qualify import runtime_state
from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot.training import PilotTrainer, discarded_startup
from online_feedback.prefix_allocation import bounded_prefix_coefficients
from online_feedback.rng import restore_rng
from online_feedback.storage import write_json
from .allocation_integration import (
    _endpoint_identity, _restore_parent, _science_rows,
    ordinary_control_window, qualification_cached_contract, train_window,
)
from .allocation_qualify import json_diagnostics, vector_identity
from .allocation_scoring import NativeAllocationScorer
from .math import adamw_shadow, optimizer_vjp, require
from .native import WorkReceipt, freeze_two_responses, select_parent
from .repair_native import require_original_window


def cpu_vectors(values):
    return tuple(value.detach().to('cpu', copy=True) for value in values)


def cpu_context(context):
    return {
        key: cpu_vectors(value) if isinstance(value, (tuple, list))
        else value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in context.items()
    }


def ancillary_state(trainer):
    return dict(
        responses=deepcopy(trainer.response_records),
        windows=deepcopy(trainer.window_records),
        anchors=deepcopy(trainer._anchor_observations),
        work=deepcopy(trainer.work), costs=deepcopy(trainer.bundle.model.costs),
        prefix_summary=deepcopy(trainer.bundle.model.prefix_allocation.last_summary),
        active_event=getattr(trainer, '_active_event', None),
    )


def shadow_at_parent(scorer, gradients):
    """Use the inherited continuous shadow, before native clipping or Adam."""
    wrapper = scorer.trainer.backend.optimizer
    optimizer = wrapper.optimizer
    require(len(optimizer.param_groups) == 1 and
            tuple(map(id, optimizer.param_groups[0]['params'])) == tuple(map(id, scorer.parameters)),
            'Native optimizer parameter ordering changed')
    state = [optimizer.state[p] for p in scorer.parameters]
    with scorer._operation('point1_fp32_shadow'):
        return adamw_shadow(
            scorer.parameters, gradients,
            [item['exp_avg'] for item in state],
            [item['exp_avg_sq'] for item in state],
            [item['step'] for item in state], optimizer.param_groups[0],
            max_grad_norm=wrapper.max_grad_norm,
        )


def assessment_cotangent(scorer, leaves, label):
    """Same BF16 functional cast and FP32 accumulation as qualified scoring."""
    accumulated, values = None, []
    with scorer._isolated():
        for index, response in enumerate(scorer.responses):
            represented = {
                name: value.to(parameter.dtype)
                for name, value, parameter in zip(
                    scorer.parameter_names, leaves, scorer.parameters)
            }
            hidden = scorer._blocks(response, represented)
            value = scorer._assessment(hidden, scorer.snapshots[index]['roles'])
            values.append(float(value.detach()))
            with scorer._operation('point1_' + label + '_assessment_backward'):
                current = torch.autograd.grad(value / 2, leaves, allow_unused=False)
            if accumulated is None:
                accumulated = current
            else:
                for total, contribution in zip(accumulated, current):
                    total.add_(contribution)
            del represented, hidden, value, current
    require(all(torch.isfinite(v).all() for v in accumulated), 'Nonfinite cotangent')
    return accumulated, dict(response_assessments=values, window_assessment=sum(values)/2)


@torch.no_grad()
def native_c_coefficients(scorer):
    """Observe the actual native C head/loss path, without an optimizer call."""
    extension = scorer.model.prefix_allocation
    original_chunk = extension.chunk_terms
    captured = []

    def observed(log_probability, token_loss, proposals, labels, weights,
                 selection, eos_token_ids=()):
        from online_feedback.first_error import valid_prefix_mask
        terms = original_chunk(log_probability, token_loss, proposals, labels,
                               weights, selection, eos_token_ids)
        valid = valid_prefix_mask(labels, weights, eos_token_ids)
        prefix = selection.detach() & valid & (proposals.detach() == labels)
        probability = log_probability.detach().gather(
            -1, labels.unsqueeze(-1)).squeeze(-1).float().exp()
        captured.append(bounded_prefix_coefficients(
            probability, weights, valid, prefix, relative_cap=extension.relative_cap))
        return terms

    result = []
    with scorer._isolated():
        extension.chunk_terms = observed
        try:
            for response, snapshot in zip(scorer.responses, scorer.snapshots):
                captured.clear()
                hidden = scorer._blocks(response)
                roles = snapshot['roles']
                for _, part in scorer._slices(hidden):
                    with scorer._operation('point1_native_C_capture_head_chunk'):
                        scorer.model._chunk_loss(
                            hidden[:, part], None, roles['labels'][:, part],
                            roles['beta'][:, part], snapshot['teacher_hidden'][:, part])
                coefficients = torch.cat(captured, dim=1)
                require(torch.equal(coefficients, snapshot['old_C_coefficients']),
                        'Actual native C capture differs from frozen C')
                result.append(coefficients.detach())
                del hidden
        finally:
            extension.chunk_terms = original_chunk
    return result


def seal_scoring(scorer, output):
    """Seal all current allocations and predictions before any native step."""
    from .point1_math import score_contrast
    scorer.freeze_current()
    beta = [item['roles']['beta'] for item in scorer.snapshots]
    control = native_c_coefficients(scorer)
    gradient = scorer.reference_gradient()
    preview, context = shadow_at_parent(scorer, gradient)
    cotangent, assessment = assessment_cotangent(scorer, preview, 'shadow_beta')
    with scorer._operation('point1_optimizer_vjp_shadow_beta'):
        direction = tuple(optimizer_vjp(context, cotangent))
    retained = dict(
        gradients={'BETA': cpu_vectors(gradient)}, shadows={'BETA': cpu_vectors(preview)},
        native={}, context=cpu_context(context), r_beta=cpu_vectors(cotangent),
        w=cpu_vectors(direction),
    )
    del preview, context, cotangent
    fast = scorer.fast_scores(direction)
    allocations = {'BETA': beta, 'C': control, 'F': fast['allocations']}
    predictions = {}
    for reference in ('C', 'BETA'):
        predictions['F-' + reference] = [
            score_contrast(a, b, scores, snapshot['roles']['beta'],
                           snapshot['roles']['prefix'])
            for a, b, scores, snapshot in zip(
                allocations['F'], allocations[reference], fast['scores'], scorer.snapshots)
        ]
    receipt = dict(
        coefficients={name: vector_identity(['response_0', 'response_1'], values)
                      for name, values in allocations.items()},
        reference_gradient=vector_identity(scorer.parameter_names, gradient),
        scores=vector_identity(['response_0', 'response_1'], fast['scores']),
        predictions=predictions, assessment=assessment,
        allocation_diagnostics=json_diagnostics(fast['allocation_diagnostics']),
        direction_policy=fast['direction_policy'],
        direction_identity=vector_identity(scorer.parameter_names, direction),
        effective_direction_identity=vector_identity(scorer.parameter_names, fast['effective_direction']),
        response_denominators=[float(s['roles']['denominator']) for s in scorer.snapshots],
        clipping={key: (float(value) if isinstance(value, torch.Tensor) else value)
                  for key, value in retained['context'].items()
                  if key not in ('gradients', 'adam_diagonal')},
        frozen_before_native_updates=True,
    )
    write_json(output / 'SCORING.json', receipt)
    del gradient, direction, fast
    return allocations, retained, receipt


def run_cases(trainer, scorer, state, runtime, ancillary, features, responses,
              allocations, retained, ledger, work, output):
    """No chained updates; every branch begins and ends at authenticated PAIR1."""
    cases = {}
    for name in ('BETA', 'C', 'F', 'C_REPEAT'):
        _restore_parent(trainer, state, runtime, ancillary)
        verify_live(trainer, state, output / (name + '_INITIAL_STATE.json'))
        trainer._active_event = 'point1_' + name.lower()
        (trainer.events_root / trainer._active_event).mkdir(exist_ok=False)
        case = dict(status='running', adam_attempts=0, adam_completed=0)
        original_step = trainer.backend.optimizer.step

        def observed_step():
            gradients = tuple(p.grad for p in scorer.parameters)
            require(all(g is not None and torch.isfinite(g).all() for g in gradients),
                    'Missing or nonfinite native preclip gradient')
            case['gradient_identity'] = vector_identity(scorer.parameter_names, gradients)
            captured = cpu_vectors(gradients)
            if name == 'BETA':
                case['beta_g0_exact'] = all(torch.equal(a, b) for a, b in zip(
                    captured, retained['gradients']['BETA']))
                require(case['beta_g0_exact'], 'Actual BETA gradient differs from scoring g0')
            if name == 'C_REPEAT':
                require(all(torch.equal(a, b) for a, b in zip(
                    captured, retained['gradients']['C'])), 'Repeated native C gradient changed')
            elif name != 'BETA':
                retained['gradients'][name] = captured
                preview, context = shadow_at_parent(scorer, gradients)
                retained['shadows'][name] = cpu_vectors(preview)
                del preview, context
            del captured
            reservation = ledger.reserve(name)
            case['adam_attempts'] += 1
            write_json(output / (name + '.json'), case)
            work.start('point1_native_adam')
            norm = original_step()
            torch.cuda.synchronize()
            work.finish('point1_native_adam')
            work.value['native_adam_steps'] += 1
            work.save()
            ledger.complete(reservation)
            case['adam_completed'] += 1
            case['native_preclip_norm'] = float(norm)
            return norm

        trainer.backend.optimizer.step = observed_step
        try:
            if name in ('C', 'C_REPEAT'):
                result = ordinary_control_window(trainer, features, work)
            else:
                restore_rng(responses[-1].rng_after_sampling)
                with qualification_cached_contract(trainer):
                    result = train_window(
                        trainer, features, enabled=True, responses=responses,
                        snapshots=scorer.snapshots, coefficients=allocations[name], work=work)
            case.update(status='completed', rows=result['rows'], endpoint=_endpoint_identity(trainer),
                        physical_work_delta={key: value-ancillary['work'].get(key, 0)
                                             for key, value in trainer.work.items()})
            require(case['adam_completed'] == 1, 'Expected exactly one native call')
            require((trainer.global_step, trainer.presentations, len(trainer.events)) == (129,258,1),
                    'Disposable endpoint counters changed')
            expected_anchors = [[dict(positions=r.anchors.cpu().tolist(),
                                     keep_mask=r.keep.cpu().tolist())] for r in responses]
            require([r['sampled_anchors'] for r in result['rows']] == expected_anchors,
                    'Native training did not reuse the retained window')
            if name == 'C_REPEAT':
                require(case['endpoint'] == cases['C']['endpoint']
                        and case['gradient_identity'] == cases['C']['gradient_identity']
                        and _science_rows(case['rows']) == _science_rows(cases['C']['rows']),
                        'Native C repeatability failed')
                case['exact_C_repeat'] = True
            else:
                retained['native'][name] = cpu_vectors(scorer.parameters)
        except BaseException as error:
            case.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            trainer.backend.optimizer.step = original_step
            write_json(output / (name + '.json'), case)
            _restore_parent(trainer, state, runtime, ancillary)
            case['restoration'] = verify_live(trainer, state, output / (name+'_RESTORATION.json'))
            write_json(output / (name + '.json'), case)
        cases[name] = case
    return cases


def native_point_probe(scorer, retained):
    device = scorer.parameters[0].device
    leaves = tuple(v.to(device=device, dtype=torch.float32).requires_grad_()
                   for v in retained['native']['BETA'])
    cotangent, receipt = assessment_cotangent(scorer, leaves, 'native_beta')
    retained['r_native'] = cpu_vectors(cotangent)
    context = device_context(retained['context'], device)
    with scorer._operation('point1_optimizer_vjp_native_beta'):
        retained['w_native'] = cpu_vectors(optimizer_vjp(context, cotangent))
    del leaves, cotangent, context
    return receipt


def device_context(context, device):
    return {
        key: tuple(v.to(device) for v in value) if isinstance(value, (tuple, list))
        else value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in context.items()
    }


def endpoint_readouts(trainer, scorer, retained, contract, work, output):
    from .point1_readout import evaluate_responses, witness_responses
    window_parent = evaluate_responses(trainer, scorer.responses, endpoint='parent', work=work)
    for row, snapshot in zip(window_parent['responses'], scorer.snapshots):
        require([b['proposals'] for b in row['blocks']] == snapshot['proposals'][0].cpu().tolist(),
                'Parent readout differs from frozen scoring proposals')
        for key in ('labels', 'valid', 'prefix', 'repair', 'L', 'e', 'H', 'K', 'm'):
            require([b['origin'][key] for b in row['blocks']] == snapshot['roles'][key][0].cpu().tolist(),
                    'Parent readout differs from frozen scoring roles: ' + key)
    write_json(output / 'window_parent.json', window_parent)
    witness = contract['witness']
    witness_rows = witness['items'] if witness['status'] == 'authenticated' else []
    witnesses = witness_responses(trainer, witness_rows) if witness_rows else []
    witness_parent = None
    if witnesses:
        historical = json.loads(Path(witness['parent_path']).read_text())
        witness_parent = evaluate_responses(
            trainer, witnesses, endpoint='parent', origins=historical['sets']['new_buffer'], work=work)
        write_json(output / 'witness_parent.json', witness_parent)
    device = scorer.parameters[0].device
    results = {'window': {'parent': window_parent}, 'new_buffer': {'parent': witness_parent}}
    for family, source in (('Q', retained['shadows']), ('N', retained['native'])):
        for name in ('BETA', 'C', 'F'):
            parameters = {key: value.to(device=device, dtype=torch.bfloat16)
                          for key, value in zip(scorer.parameter_names, source[name])}
            label = family + '_' + name
            result = evaluate_responses(trainer, scorer.responses, endpoint=label,
                                        origins=window_parent, parameters=parameters, work=work)
            results['window'][label] = result
            write_json(output / ('window_' + label + '.json'), result)
            if family == 'N' and witnesses:
                result = evaluate_responses(trainer, witnesses, endpoint=label,
                    origins=witness_parent, parameters=parameters, work=work)
                results['new_buffer'][label] = result
                write_json(output / ('witness_' + label + '.json'), result)
            del parameters
    return results


def vector_chain(scorer, state, retained, output):
    from .point1_math import DifferenceVectors, optimizer_jvp_view, vector_metrics
    names = scorer.parameter_names
    results = {}
    context = device_context(retained['context'], scorer.parameters[0].device)
    for first, second in (('F','C'), ('F','BETA'), ('C','BETA')):
        key = first + '-' + second
        gradients = DifferenceVectors(retained['gradients'][first], retained['gradients'][second])
        device_gradients = tuple(value.to(scorer.parameters[0].device) for value in gradients)
        with scorer._operation('point1_optimizer_jvp_' + key):
            linear = cpu_vectors(optimizer_jvp_view(context, device_gradients))
        del device_gradients
        finite = DifferenceVectors(retained['shadows'][first], retained['shadows'][second])
        cast = DifferenceVectors(
            tuple(v.to(torch.bfloat16) for v in retained['shadows'][first]),
            tuple(v.to(torch.bfloat16) for v in retained['shadows'][second]))
        native = DifferenceVectors(retained['native'][first], retained['native'][second])
        rows = {}
        for label, reference, candidate in (
            ('linear_to_finite', linear, finite),
            ('finite_to_cast', finite, cast),
            ('cast_to_native', cast, native),
        ):
            rows[label] = vector_metrics(reference, candidate,
                reference_name=label.split('_to_')[0], candidate_name=label.split('_to_')[1],
                cotangent=retained['r_beta'], parameter_names=names)
        rows['native_gradient'] = vector_metrics(gradients, gradients,
            reference_name='native_gradient_difference', candidate_name='same',
            cotangent=retained['w'], parameter_names=names)
        rows['native_point_prediction'] = vector_metrics(gradients, gradients,
            reference_name='native_gradient_difference', candidate_name='same',
            cotangent=retained['w_native'], parameter_names=names)
        results[key] = rows
        write_json(output / 'VECTOR_CHAIN.json', results)
        del cast, rows, gradients, linear, finite, native
    del context
    parent = tuple(state['draft_state_dict'][name] for name, _ in scorer.trainer.bundle.draft_model.named_parameters())
    results['absolute_BETA'] = vector_metrics(
        DifferenceVectors(retained['shadows']['BETA'], parent),
        DifferenceVectors(retained['native']['BETA'], parent),
        reference_name='continuous_BETA_displacement', candidate_name='native_BETA_displacement',
        parameter_names=names)
    for name in ('BETA', 'C', 'F'):
        represented = tuple(value.to(torch.bfloat16) for value in retained['shadows'][name])
        results['cast_native_endpoint_' + name] = vector_metrics(
            represented, retained['native'][name], reference_name='Q_' + name,
            candidate_name='N_' + name, parameter_names=names)
        del represented
    write_json(output / 'VECTOR_CHAIN.json', results)
    return results


def run(contract, output, execution_commit, step_ledger):
    from .point1_launch import Point1StepLedger
    from .point1_prepare import SCHEMA, LIMITS
    contract_path = Path(contract)
    contract = json.loads(contract_path.read_text())
    require(contract['schema'] == SCHEMA and contract['limits'] == LIMITS
            and execution_commit == contract['execution_commit']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == contract['devices']['B'],
            'Point1 source, limits or physical GPU binding changed')
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    work = WorkReceipt(output / 'WORK.json')
    ledger = Point1StepLedger(step_ledger, identity(contract_path))
    result = dict(status='running', execution_commit=execution_commit,
                  contract_identity=identity(contract_path), historical_occupied_card_s=5175.047027264)
    write_json(output / 'RESULT.json', result)
    trainer = state = runtime = ancillary = None
    started = time.perf_counter()
    try:
        plan, recovery, features = select_parent(contract['parent_plan'])
        write_json(output/'RUNTIME_COMPATIBILITY.json', pin_retained_graph_runtime(output))
        trainer = PilotTrainer(
            plan['configs']['PAIR'], output/'trainer', arm='PAIR', binding=plan['binding'],
            original_config_text=Path(plan['original_config']).read_text(), maximum_events=1,
            resume_state=recovery, completed_event=1, qualification_only=True)
        state = torch.load(recovery, map_location='cpu', weights_only=False, mmap=True)
        runtime, ancillary = runtime_state(trainer.bundle.model), ancillary_state(trainer)
        paths = [item['path'] for item in features]
        result['startup'] = discarded_startup(trainer, paths, output/'startup')
        verify_live(trainer, state, output/'AFTER_STARTUP.json')
        torch.cuda.reset_peak_memory_stats()
        responses = freeze_two_responses(trainer, paths)
        original = json.loads((Path(contract['original_probe'])/'WINDOW.json').read_text())
        write_json(output/'WINDOW.json', require_original_window(responses, original))
        scorer = NativeAllocationScorer(trainer, responses, work, collect_diagnostics=True)
        allocations, retained, result['scoring'] = seal_scoring(scorer, output)
        result['scoring_physical_work_delta'] = {
            key: value-ancillary['work'].get(key, 0) for key, value in trainer.work.items()}
        result['cases'] = run_cases(trainer, scorer, state, runtime, ancillary, features,
            responses, allocations, retained, ledger, work, output)
        probe_work_before = deepcopy(trainer.work)
        result['native_point_probe'] = native_point_probe(scorer, retained)
        result['native_point_probe']['physical_work_delta'] = {
            key: value-probe_work_before.get(key, 0) for key, value in trainer.work.items()}
        write_json(output/'NATIVE_POINT_PROBE.json', result['native_point_probe'])
        endpoint_readouts(trainer, scorer, retained, contract, work, output)
        vector_chain(scorer, state, retained, output)
        scorer.close()
        del retained, scorer, allocations, responses
        gc.collect()
        result.update(status='completed', planned_cases_completed=4,
                      new_feedback=0, generations=0, exports=0)
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if trainer is not None and runtime is not None:
                _restore_parent(trainer, state, runtime, ancillary)
                result['final_restoration'] = verify_live(trainer, state, output/'FINAL_RESTORATION.json')
        except BaseException as error:
            result.update(status='failed', restoration_error=repr(error),
                          restoration_traceback=traceback.format_exc())
            raise
        finally:
            result.update(wall_s=time.perf_counter()-started, work=deepcopy(work.value))
            if torch.cuda.is_initialized():
                result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(output/'RESULT.json', result)
            if trainer is not None:
                trainer.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'output', 'execution-commit', 'step-ledger'):
        parser.add_argument('--'+name, required=True)
    run(**vars(parser.parse_args()))
