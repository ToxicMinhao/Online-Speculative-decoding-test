"""Three restored native calls with a C-referenced, independently audited score.

The only changed scientific input to the allocator is its scoring direction.
Native C supplies its actual preclip gradient once. All previews and candidates
use the original PAIR1 state; full parameter/gradient vectors never reach disk.
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
from experiments.ok_continuation.resume import differences
from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot.training import PilotTrainer, discarded_startup
from online_feedback.rng import restore_rng
from online_feedback.storage import write_json
from .allocation_integration import (
    _endpoint_identity, _restore_parent, ordinary_control_window,
    qualification_cached_contract, train_window,
)
from .allocation_qualify import json_diagnostics, vector_identity
from .allocation_scoring import NativeAllocationScorer
from .math import optimizer_vjp, require
from .native import WorkReceipt, freeze_two_responses, select_parent
from .point1_native import (
    ancillary_state, assessment_cotangent, cpu_vectors, native_c_coefficients,
    shadow_at_parent,
)
from .repair_native import require_original_window


CASES = ('C', 'F_BETA', 'F_C')
HISTORICAL_CARD_S = 5510.8580226639515


def json_value(value):
    """Make live tuple options comparable to their authenticated JSON receipt."""
    return json.loads(json.dumps(value))


def scientific_case_identity(case):
    """Retain all recorded science, including observations and complete Adam state.

Only event/directory labels and elapsed times are excluded. In particular,
response metrics, anchors, RNG, moments, options and counters cannot disappear
behind a weights-only equality check.
"""
    rows = deepcopy(case['rows'])
    for row in rows:
        row.pop('training_s', None)
        if 'response_observation' in row:
            row['response_observation'].pop('event_id', None)
    return json_value(dict(
        gradient_identity=case['gradient_identity'], endpoint=case['endpoint'],
        native_preclip_norm=case['native_preclip_norm'], rows=rows,
        adam_completed=case['adam_completed'],
    ))


def authenticate_controls(contract):
    controls = {}
    require(set(contract['historical_controls']) == {'C', 'F_BETA'},
            'Require both authenticated Point1 control cases')
    for name, binding in contract['historical_controls'].items():
        require(identity(binding['path']) == binding['identity'],
                'Historical case identity changed: ' + name)
        case = json.loads(Path(binding['path']).read_text())
        require(case['status'] == 'completed' and case['adam_completed'] == 1,
                'Historical control was not completed: ' + name)
        controls[name] = case
    return controls


def reproduce_control(case, historical):
    expected, actual = scientific_case_identity(historical), scientific_case_identity(case)
    mismatches = differences(actual, expected, 'case')
    receipt = dict(status='passed' if not mismatches else 'failed',
                   complete_recorded_scientific_identity=not mismatches,
                   mismatches=mismatches[:32],
                   excluded_fields=['training_s', 'response_observation.event_id'])
    case['historical_reproduction'] = receipt
    require(not mismatches, 'Historical native control reproduction failed: ' + str(mismatches[:8]))
    return receipt


def verify_parent(trainer, state, runtime, ancillary, path):
    receipt = verify_live(trainer, state)
    extra = differences(runtime_state(trainer.bundle.model), runtime, 'runtime')
    extra += differences(ancillary_state(trainer), ancillary, 'ancillary')
    receipt.update(runtime_exact=not any(p.startswith('runtime') for p in extra),
                   ancillary_exact=not any(p.startswith('ancillary') for p in extra),
                   additional_mismatches=extra[:32])
    if extra:
        receipt['status'] = 'failed'
    write_json(path, receipt)
    require(not extra, 'Parent runtime or ancillary restoration failed: ' + str(extra[:8]))
    return receipt


def restore_parent(trainer, state, runtime, ancillary, path):
    _restore_parent(trainer, state, runtime, ancillary)
    return verify_parent(trainer, state, runtime, ancillary, path)


def _clipping_receipt(context):
    return {key: float(value) if isinstance(value, torch.Tensor) else value
            for key, value in context.items() if key not in ('gradients', 'adam_diagonal')}


def _score_receipt(scorer, *, reference, gradient, direction, fast, assessment,
                   clipping, coefficients, binding=None):
    receipt = dict(
        reference=reference,
        reference_gradient=vector_identity(scorer.parameter_names, gradient),
        coefficients={name: vector_identity(['response_0', 'response_1'], values)
                      for name, values in coefficients.items()},
        coefficient_arrays={name: json_diagnostics(values) for name, values in coefficients.items()},
        scores=vector_identity(['response_0', 'response_1'], fast['scores']),
        score_arrays=json_diagnostics(fast['scores']),
        direction_identity=vector_identity(scorer.parameter_names, direction),
        direction_policy=fast['direction_policy'],
        assessment=assessment, clipping=clipping,
        allocation_diagnostics=json_diagnostics(fast['allocation_diagnostics']),
        response_denominators=[float(s['roles']['denominator']) for s in scorer.snapshots],
        accumulation_factor=0.5, relative_cap=0.25,
        allocation_budget='sum positional beta in each original prefix pool',
        assessment_point='BF16(T32_parent(g_' + reference + '))',
        coefficients_sealed_before_own_native_update=True,
    )
    if 'effective_direction' in fast:
        receipt['effective_direction_identity'] = vector_identity(
            scorer.parameter_names, fast['effective_direction'])
    if binding is not None:
        receipt['reference_binding'] = binding
    return receipt


def seal_beta(scorer, output):
    """Exactly the Point1 beta route, without retaining its unused full vectors."""
    beta = [s['roles']['beta'] for s in scorer.snapshots]
    control = native_c_coefficients(scorer)
    gradient = scorer.reference_gradient()
    preview, context = shadow_at_parent(scorer, gradient)
    cotangent, assessment = assessment_cotangent(scorer, preview, 'shadow_beta')
    with scorer._operation('point2a_optimizer_vjp_shadow_beta'):
        direction = tuple(optimizer_vjp(context, cotangent))
    clipping = _clipping_receipt(context)
    del preview, context, cotangent
    fast = scorer.fast_scores(direction)
    allocations = {'BETA': beta, 'C': control, 'F_BETA': fast['allocations']}
    receipt = _score_receipt(scorer, reference='BETA', gradient=gradient,
        direction=direction, fast=fast, assessment=assessment,
        clipping=clipping, coefficients=allocations)
    write_json(output / 'SCORING_BETA.json', receipt)
    scores = cpu_vectors(fast['scores'])
    del gradient, direction, fast
    return allocations, scores, receipt


def c_direction(scorer, retained, parent_identity, captured_identity):
    """Build J_C and r_C only at the restored parent, using actual preclip g_C."""
    from .point2a_math import require_c_reference
    supplied_identity = vector_identity(scorer.parameter_names, retained['gradients']['C'])
    binding = require_c_reference(reference_name='C', parent_identity=json_value(parent_identity),
        current_identity=json_value(_endpoint_identity(scorer.trainer)),
        captured_gradient_identity=captured_identity, supplied_gradient_identity=supplied_identity)
    device = scorer.parameters[0].device
    gradient = tuple(value.to(device=device) for value in retained['gradients']['C'])
    preview, context = shadow_at_parent(scorer, gradient)
    cotangent, assessment = assessment_cotangent(scorer, preview, 'shadow_C')
    with scorer._operation('point2a_optimizer_vjp_shadow_C'):
        direction = tuple(optimizer_vjp(context, cotangent))
    clipping = _clipping_receipt(context)
    shadow_identity = vector_identity(scorer.parameter_names, preview)
    cotangent_identity = vector_identity(scorer.parameter_names, cotangent)
    del gradient, preview, context, cotangent
    retained['w_C'] = cpu_vectors(direction)
    return direction, dict(binding=binding, assessment=assessment, clipping=clipping,
                           shadow_identity=shadow_identity, cotangent_identity=cotangent_identity)


def run_case(name, trainer, scorer, state, runtime, ancillary, features, responses,
             allocations, retained, ledger, work, output, sealed_identity,
             historical=None):
    """One original native optimizer call, then exact full parent restoration."""
    require(name in CASES, 'Unplanned Point2A case')
    require(vector_identity(['response_0', 'response_1'], allocations[name]) == sealed_identity,
            'Coefficient array changed after sealing: ' + name)
    restore_parent(trainer, state, runtime, ancillary, output / (name + '_INITIAL_STATE.json'))
    trainer._active_event = 'point2a_' + name.lower()
    (trainer.events_root / trainer._active_event).mkdir(exist_ok=False)
    case = dict(name=name, status='running', adam_attempts=0, adam_completed=0,
                coefficient_identity=sealed_identity)
    original_step = trainer.backend.optimizer.step

    def observed_step():
        gradients = tuple(p.grad for p in scorer.parameters)
        require(all(g is not None and torch.isfinite(g).all() for g in gradients),
                'Missing or nonfinite native preclip gradient')
        case['gradient_identity'] = vector_identity(scorer.parameter_names, gradients)
        retained['gradients'][name] = cpu_vectors(gradients)
        # This is the normalized accumulated gradient before clipping. The
        # observer copies it only; the author's original step owns clipping.
        reservation = ledger.reserve(name)
        case['adam_attempts'] += 1
        write_json(output / (name + '.json'), case)
        work.start('point2a_native_adam')
        norm = original_step()
        if scorer.parameters[0].is_cuda:
            torch.cuda.synchronize()
        work.finish('point2a_native_adam')
        work.value['native_adam_steps'] += 1
        work.save()
        ledger.complete(reservation)
        case['adam_completed'] += 1
        case['native_preclip_norm'] = float(norm)
        return norm

    trainer.backend.optimizer.step = observed_step
    try:
        if name == 'C':
            native = ordinary_control_window(trainer, features, work)
        else:
            restore_rng(responses[-1].rng_after_sampling)
            with qualification_cached_contract(trainer):
                native = train_window(trainer, features, enabled=True, responses=responses,
                    snapshots=scorer.snapshots, coefficients=allocations[name], work=work)
        case.update(status='completed', rows=native['rows'], endpoint=_endpoint_identity(trainer),
                    physical_work_delta={key: value-ancillary['work'].get(key, 0)
                                         for key, value in trainer.work.items()})
        require(case['adam_completed'] == 1, 'Expected exactly one native call')
        require((trainer.global_step, trainer.presentations, len(trainer.events)) == (129, 258, 1),
                'Disposable endpoint counters changed')
        expected = [[dict(positions=r.anchors.cpu().tolist(), keep_mask=r.keep.cpu().tolist())]
                    for r in responses]
        require([row['sampled_anchors'] for row in native['rows']] == expected,
                'Native training did not reuse the retained window')
        if historical is not None:
            reproduce_control(case, historical)
        retained['native'][name] = cpu_vectors(scorer.parameters)
    except BaseException as error:
        case.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        trainer.backend.optimizer.step = original_step
        write_json(output / (name + '.json'), case)
        try:
            case['restoration'] = restore_parent(
                trainer, state, runtime, ancillary, output / (name + '_RESTORATION.json'))
        except BaseException as error:
            case.update(status='failed', restoration_error=repr(error),
                        restoration_traceback=traceback.format_exc())
            raise
        finally:
            write_json(output / (name + '.json'), case)
    return case


def execute_cases(trainer, scorer, state, runtime, ancillary, features, responses,
                  ledger, work, contract, controls, parent_identity, output):
    """The fixed schedule is intentionally explicit so a failed audit stops F."""
    from .point2a_math import select_reference_pools, qualify_direction, three_contrasts, score_changes
    from .point2a_readout import endpoint_readouts
    scorer.freeze_current()
    selection = select_reference_pools(scorer.snapshots,
        response_ids=[response.record_id for response in responses])
    write_json(output / 'AUDIT_PLAN.json', selection)
    allocations, scores_beta, beta_receipt = seal_beta(scorer, output)
    beta_work = {key: value-ancillary['work'].get(key, 0)
                 for key, value in trainer.work.items()}
    retained = dict(gradients={}, native={})
    cases = {}
    cases['C'] = run_case('C', trainer, scorer, state, runtime, ancillary, features,
        responses, allocations, retained, ledger, work, output,
        beta_receipt['coefficients']['C'], controls['C'])
    # run_case's finally restores before this function can build the C preview.
    direction, context_receipt = c_direction(
        scorer, retained, parent_identity, cases['C']['gradient_identity'])
    qualification, fast = qualify_direction(
        scorer, direction, output, ledger, work, selection)
    require(qualification['status'] == 'passed', 'New C direction failed qualification')
    # Inspect before restoration: otherwise an accidental optimizer/parameter
    # mutation inside the oracle could be concealed by the following reset.
    after_scoring = json_value(_endpoint_identity(trainer))
    mutation = differences(after_scoring, json_value(parent_identity), 'parent')
    write_json(output / 'SCORING_STATE_PRESERVATION.json', dict(
        status='failed' if mutation else 'passed', mismatches=mutation[:32],
        complete_endpoint_identity=after_scoring, checked_before_restoration=True))
    require(not mutation, 'C scoring modified the parent scientific state: ' + str(mutation[:8]))
    allocations['F_C'] = fast['allocations']
    c_receipt = _score_receipt(scorer, reference='C', gradient=retained['gradients']['C'],
        direction=direction, fast=fast, assessment=context_receipt['assessment'],
        clipping=context_receipt['clipping'], coefficients={'F_C': allocations['F_C']},
        binding=context_receipt['binding'])
    c_receipt.update(shadow_identity=context_receipt['shadow_identity'],
                     cotangent_identity=context_receipt['cotangent_identity'],
                     effective_direction_identity=qualification['direction_metadata']['effective'],
                     qualification_identity=identity(output / 'QUALIFICATION.json'))
    c_receipt['score_changes_from_beta'] = [score_changes(new, old.to(new.device),
        snapshot['roles']['beta'], snapshot['roles']['prefix'])
        for new, old, snapshot in zip(fast['scores'], scores_beta, scorer.snapshots)]
    write_json(output / 'SCORING_C.json', c_receipt)
    scores_c = cpu_vectors(fast['scores'])
    del fast, direction, context_receipt
    c_work = {key: value-ancillary['work'].get(key, 0)
              for key, value in trainer.work.items()}
    # Functional scoring preserves scientific state; observational work is
    # restored before checking all parent runtime and history components.
    restore_parent(trainer, state, runtime, ancillary, output / 'AFTER_QUALIFICATION.json')
    for name in ('F_BETA', 'F_C'):
        receipt = beta_receipt if name == 'F_BETA' else c_receipt
        cases[name] = run_case(name, trainer, scorer, state, runtime, ancillary, features,
            responses, allocations, retained, ledger, work, output,
            receipt['coefficients'][name], controls.get(name))
    contrasts = three_contrasts({name: allocations[name] for name in CASES}, scores_c, scorer.snapshots,
        gradients=retained['gradients'], direction=retained['w_C'])
    write_json(output / 'CONTRASTS.json', contrasts)
    del retained['gradients'], retained['w_C']
    gc.collect()
    readouts = endpoint_readouts(trainer, scorer, retained, contract, work, output)
    if 'contrasts' in readouts:
        for name, value in contrasts.items():
            value['native_readouts'] = {scope: rows.get(name)
                for scope, rows in readouts['contrasts'].items()}
        write_json(output / 'CONTRASTS.json', contrasts)
    return dict(cases=cases, qualification=qualification,
                scoring_beta=beta_receipt, scoring_C=c_receipt,
                scoring_physical_work_delta=dict(beta=beta_work, C=c_work),
                contrasts=contrasts, planned_cases_completed=len(cases))


def run(contract, output, execution_commit, step_ledger):
    from .point2a_launch import Point2AStepLedger
    from .point2a_prepare import SCHEMA, LIMITS
    contract_path = Path(contract)
    contract = json.loads(contract_path.read_text())
    require(contract['schema'] == SCHEMA and contract['limits'] == LIMITS
            and execution_commit == contract['execution_commit']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == contract['devices']['B'],
            'Point2A source, limits or physical GPU binding changed')
    controls = authenticate_controls(contract)
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    work = WorkReceipt(output / 'WORK.json')
    ledger = Point2AStepLedger(step_ledger, identity(contract_path))
    result = dict(status='running', execution_commit=execution_commit,
        contract_identity=identity(contract_path), historical_occupied_card_s=HISTORICAL_CARD_S,
        planned_cases=list(CASES), historical_failed_gates_preserved=True)
    write_json(output / 'RESULT.json', result)
    trainer = state = runtime = ancillary = scorer = None
    started = time.perf_counter()
    try:
        plan, recovery, features = select_parent(contract['parent_plan'])
        write_json(output / 'RUNTIME_COMPATIBILITY.json', pin_retained_graph_runtime(output))
        trainer = PilotTrainer(plan['configs']['PAIR'], output / 'trainer', arm='PAIR',
            binding=plan['binding'], original_config_text=Path(plan['original_config']).read_text(),
            maximum_events=1, resume_state=recovery, completed_event=1, qualification_only=True)
        state = torch.load(recovery, map_location='cpu', weights_only=False, mmap=True)
        runtime, ancillary = runtime_state(trainer.bundle.model), ancillary_state(trainer)
        paths = [item['path'] for item in features]
        result['startup'] = discarded_startup(trainer, paths, output / 'startup')
        restore_parent(trainer, state, runtime, ancillary, output / 'AFTER_STARTUP.json')
        parent_identity = _endpoint_identity(trainer)
        write_json(output / 'PARENT_STATE_IDENTITY.json', parent_identity)
        torch.cuda.reset_peak_memory_stats()
        responses = freeze_two_responses(trainer, paths)
        original = json.loads((Path(contract['original_probe']) / 'WINDOW.json').read_text())
        write_json(output / 'WINDOW.json', require_original_window(responses, original))
        scorer = NativeAllocationScorer(trainer, responses, work, collect_diagnostics=True)
        result.update(execute_cases(trainer, scorer, state, runtime, ancillary, features,
            responses, ledger, work, contract, controls, parent_identity, output))
        scorer.close()
        result.update(status='completed', new_feedback=0, generations=0, exports=0,
                      full_vectors_written=0, native_adam_calls=3)
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if trainer is not None and runtime is not None:
                result['final_restoration'] = restore_parent(
                    trainer, state, runtime, ancillary, output / 'FINAL_RESTORATION.json')
        except BaseException as error:
            result.update(status='failed', restoration_error=repr(error),
                          restoration_traceback=traceback.format_exc())
            raise
        finally:
            result.update(wall_s=time.perf_counter()-started, work=deepcopy(work.value))
            if torch.cuda.is_initialized():
                result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(output / 'RESULT.json', result)
            if trainer is not None:
                trainer.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'output', 'execution-commit', 'step-ledger'):
        parser.add_argument('--' + name, required=True)
    run(**vars(parser.parse_args()))
