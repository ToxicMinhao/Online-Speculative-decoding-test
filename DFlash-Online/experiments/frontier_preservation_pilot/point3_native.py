"""Point3 native C reference and three disposable qualification endpoints.

The production reference performs the same native forward and backward as C,
without entering TrainerCore's commit boundary. The two response gradients are
accumulated in native buffers; the optimizer is never called for this reference.
"""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch

from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
from experiments.ok_continuation.resume import differences
from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot.training import PilotTrainer, discarded_startup
from online_feedback.ranking import fixed_anchors
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import write_json
from .allocation_integration import _context, _endpoint_identity, _read_feature
from .allocation_qualify import vector_identity
from .allocation_scoring import NativeAllocationScorer
from .math import optimizer_vjp, require
from .native import WorkReceipt, freeze_two_responses, rng_digest, select_parent
from .point1_native import (
    ancillary_state, assessment_cotangent, native_c_coefficients, shadow_at_parent,
)
from .point2a_native import (
    _clipping_receipt, _score_receipt, json_value, restore_parent, run_case, verify_parent,
)
from .repair_native import require_original_window

CASES = ('C', 'F_C_SUM', 'F_C_MAX')
HISTORICAL_CARD_S = 5790.93129579185


def native_c_reference_gradient(trainer, features, responses, work=None):
    """Obtain actual preclip g_C with no Adam, sampling, or history commit.

    Anchors have already been sampled once. Calling the original native forward
    avoids the committed-observation wrapper, while backend.backward preserves
    native BF16 AccumulateGrad behavior. Ambient post-sampling RNG, modes,
    caches, observations and cost accumulators are restored before returning.
    """
    require(len(features) == len(responses) == 2
            and trainer.reduction == 'response_mean'
            and trainer.accumulation == 2
            and not trainer.core.accumulation_remainder,
            'Native C reference requires the qualified two-response boundary')
    model = trainer.bundle.model
    parameters = tuple(p for p in model.draft_model.parameters() if p.requires_grad)
    require(all(p.grad is None for p in parameters), 'Reference requires clean gradients')
    trainer.check_installed_arm()
    before_rng, before_runtime = capture_rng(), runtime_state(model)
    before_ancillary = ancillary_state(trainer)
    before_counters = (trainer.global_step, trainer.presentations, trainer.core._micro,
                       len(trainer.events))
    losses, gradients = [], None
    started = time.perf_counter()
    try:
        model.costs = {}
        for index, (feature, response) in enumerate(zip(features, responses)):
            raw = _read_feature(feature)
            require(str(raw['record_id']) == response.record_id,
                    'Reference response differs from frozen window')
            restore_rng(response.rng_after_sampling)
            with fixed_anchors(model, response.anchors, response.keep):
                if work is not None:
                    work.start('point3_native_C_reference_forward')
                output = trainer._native_forward(trainer._batch(raw), _context(trainer))
                if work is not None:
                    work.finish('point3_native_C_reference_forward')
                    work.start('point3_native_C_reference_backward')
                trainer.backend.backward(output.loss / 2, is_boundary=index == 1)
                if work is not None:
                    work.finish('point3_native_C_reference_backward')
                losses.append(float(output.loss.detach()))
                del output, raw
        # TrainerCore applies this same response-mean normalization (2/2) at
        # the native preclip boundary. It does not clip or call the optimizer.
        denominator = parameters[0].new_tensor(2., dtype=torch.float32)
        trainer.core._normalize_gradients(denominator, defer_check=False)
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),
                'Missing or nonfinite native C reference gradient')
        gradients = tuple(p.grad.detach() for p in parameters)
        physical = {key: value - before_ancillary['work'].get(key, 0)
                    for key, value in trainer.work.items()}
    finally:
        trainer.backend.optimizer.zero_grad()
        restore_runtime(model, before_runtime)
        restore_rng(before_rng)
        trainer.response_records[:] = before_ancillary['responses']
        trainer.window_records[:] = before_ancillary['windows']
        trainer._anchor_observations[:] = before_ancillary['anchors']
        trainer.work = before_ancillary['work']
        model.costs = before_ancillary['costs']
        model.prefix_allocation.last_summary = before_ancillary['prefix_summary']
        trainer._active_event = before_ancillary['active_event']
    require(before_counters == (trainer.global_step, trainer.presentations,
                               trainer.core._micro, len(trainer.events)),
            'No-step reference modified training counters')
    return gradients, dict(
        source='actual_native_C_forward_backward', response_losses=losses,
        response_forwards=2, response_backwards=2, native_adam_steps=0,
        committed_presentations=0, sampling_calls=0,
        physical_work=physical, wall_s=time.perf_counter()-started,
        rng_restored=rng_digest(capture_rng()) == rng_digest(before_rng),
        gradients_cleared=all(p.grad is None for p in parameters),
    )


def score_native_C(scorer, gradient):
    """Reuse the Point2A shadow/cotangent route; only scorer.assessment varies."""
    preview, context = shadow_at_parent(scorer, gradient)
    cotangent, assessment = assessment_cotangent(scorer, preview, 'shadow_C')
    with scorer._operation('point3_optimizer_vjp_shadow_C'):
        direction = tuple(optimizer_vjp(context, cotangent))
    receipt = dict(assessment=assessment, clipping=_clipping_receipt(context),
                   assessment_kind=scorer.assessment)
    if scorer.collect_assessment_diagnostics:
        receipt['bottlenecks'] = scorer.assessment_diagnostics[-2:]
    del preview, context, cotangent
    return direction, receipt


def authenticate_controls(contract):
    controls = {}
    require(set(contract['historical_controls']) == {'C', 'F_C_SUM'},
            'Point3 requires authenticated PR62 C and F_C scientific receipts')
    for name, binding in contract['historical_controls'].items():
        require(identity(binding['path']) == binding['identity'],
                'Historical Point3 control changed: ' + name)
        case = json.loads(Path(binding['path']).read_text())
        require(case['status'] == 'completed' and case['adam_completed'] == 1,
                'Historical native control incomplete')
        controls[name] = case
    return controls


def require_historical_coefficients(actual, historical):
    """Endpoint rounding cannot replace exact allocation identity qualification."""
    require(actual == historical['coefficient_identity'],
            'Historical coefficient identity did not reproduce')


def execute_cases(trainer, scorer, state, runtime, ancillary, features, responses,
                  ledger, work, contract, controls, parent_identity, output):
    from .point2a_math import qualify_direction, select_reference_pools
    from .point3_readout import evaluate_responses

    scorer.freeze_current()
    selection = select_reference_pools(scorer.snapshots,
        response_ids=[r.record_id for r in responses])
    write_json(output / 'AUDIT_PLAN.json', selection)
    allocations = {'C': native_c_coefficients(scorer)}
    retained = dict(gradients={}, native={})
    cases = {}
    sealed = vector_identity(['response_0', 'response_1'], allocations['C'])
    if controls['C'] is not None:
        require_historical_coefficients(sealed, controls['C'])
    cases['C'] = run_case('C', trainer, scorer, state, runtime, ancillary, features,
        responses, allocations, retained, ledger, work, output, sealed, controls['C'],
        case_names=CASES, event_prefix='point3')
    gradient, reference = native_c_reference_gradient(trainer, features, responses, work)
    actual_gradient = vector_identity(scorer.parameter_names, gradient)
    reference['gradient_identity'] = actual_gradient
    reference['bitwise_native_preclip_parity'] = actual_gradient == cases['C']['gradient_identity']
    reference['state_after'] = json_value(_endpoint_identity(trainer))
    reference['state_before'] = json_value(parent_identity)
    reference['scientific_state_unchanged'] = reference['state_after'] == reference['state_before']
    reference['status'] = 'passed' if (reference['bitwise_native_preclip_parity']
        and reference['scientific_state_unchanged']) else 'failed'
    write_json(output / 'NATIVE_C_REFERENCE.json', reference)
    require(reference['status'] == 'passed', 'No-step native C reference failed exact parity/isolation')
    # Qualification checks every retained transient state, before any reset can
    # conceal a defect in the no-step reference path.
    verify_parent(trainer, state, runtime, ancillary, output / 'REFERENCE_RESTORATION.json')
    for name, assessment in (('F_C_SUM', 'SUM'), ('F_C_MAX', 'MAX')):
        scorer.assessment = assessment
        direction, preview = score_native_C(scorer, gradient)
        if assessment == 'MAX':
            qualification, fast = qualify_direction(scorer, direction,
                output / 'max_numerics', ledger, work, selection)
            require(qualification['status'] == 'passed', 'MAX token-gradient audit failed')
        else:
            fast = scorer.fast_scores(direction)
        allocations[name] = fast['allocations']
        receipt = _score_receipt(scorer, reference='C', gradient=gradient,
            direction=direction, fast=fast, assessment=preview['assessment'],
            clipping=preview['clipping'], coefficients={name: allocations[name]})
        receipt['assessment_kind'] = assessment
        write_json(output / ('SCORING_' + name + '.json'), receipt)
        if controls.get(name) is not None:
            require_historical_coefficients(receipt['coefficients'][name], controls[name])
        # Reference/preview work is observational. Restore its cost counters
        # before the separately restored native case, retaining physical WORK.
        restore_parent(trainer, state, runtime, ancillary, output / (name + '_SCORING_RESTORED.json'))
        cases[name] = run_case(name, trainer, scorer, state, runtime, ancillary, features,
            responses, allocations, retained, ledger, work, output,
            receipt['coefficients'][name], controls.get(name),
            case_names=CASES, event_prefix='point3')
        del direction, fast
    del gradient, retained['gradients']
    readouts = {}
    for name in ('parent', *CASES):
        parameters = None if name == 'parent' else dict(zip(
            scorer.parameter_names, retained['native'][name]))
        readouts[name] = evaluate_responses(trainer, responses, endpoint=name,
            origins=readouts.get('parent'), parameters=parameters, work=work)
        if name == 'parent':
            for row, snapshot in zip(readouts[name]['responses'], scorer.snapshots, strict=True):
                require([block['proposals'] for block in row['blocks']] ==
                        snapshot['proposals'][0].cpu().tolist(),
                        'Parent readout proposals differ from frozen scoring')
                for key in ('labels', 'valid', 'prefix', 'repair', 'L', 'e', 'H', 'K', 'm'):
                    require([block['origin'][key] for block in row['blocks']] ==
                            snapshot['roles'][key][0].cpu().tolist(),
                            'Parent readout origins differ from frozen scoring: ' + key)
        write_json(output / ('READOUT_' + name + '.json'), readouts[name])
    return dict(cases=cases, native_reference=reference, qualification=qualification,
                readouts=readouts, planned_cases_completed=3)


def evaluate_retained_witnesses(trainer, scorer, contract, output, work):
    """Read parent/C once; both candidate endpoints are read after their events."""
    from .point3_readout import evaluate_witness_sets
    production = json.loads(Path(contract['production_plan']).read_text())
    witness = Path(production['root']) / 'witness'
    bindings = json.loads((witness/'BINDINGS.json').read_text())
    origins = json.loads((witness/'parent.json').read_text())
    results = {}
    results['parent'] = evaluate_witness_sets(trainer, bindings, origins,
                                              endpoint='parent', work=work)
    write_json(output/'WITNESS_parent.json', results['parent'])
    control_path = Path(contract['retained_C_status'])
    require(identity(control_path) == contract['retained_C_status_identity'],
            'Retained C status changed before fixed-origin readout')
    control = json.loads(control_path.read_text())
    require(control['status'] == 'completed'
            and identity(control['recovery']) == control['recovery_identity'],
            'Retained C recovery changed before fixed-origin readout')
    state = torch.load(control['recovery'], map_location='cpu', weights_only=False, mmap=True)
    parameters = {name: state['draft_state_dict'][name.removeprefix('model.draft_model.')]
                  for name in scorer.parameter_names}
    results['C'] = evaluate_witness_sets(trainer, bindings, origins,
        endpoint='C', work=work, parameters=parameters)
    write_json(output/'WITNESS_C.json', results['C'])
    return {name: dict(path=str(output/('WITNESS_'+name+'.json')),
                       response_forwards=16) for name in results}


def run(contract, output, execution_commit, step_ledger):
    """The command is available only for a fully admitted Point3 plan."""
    from .point3_prepare import SCHEMA, LIMITS
    from .point3_launch import Point3StepLedger
    contract_path = Path(contract)
    contract = json.loads(contract_path.read_text())
    require(contract['schema'] == SCHEMA and contract['limits'] == LIMITS
            and contract['production_admitted'] is True
            and execution_commit == contract['execution_commit']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == contract['devices']['B'],
            'Point3 complete-plan admission/source/device binding changed')
    controls = authenticate_controls(contract)
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    work = WorkReceipt(output / 'WORK.json')
    ledger = Point3StepLedger(step_ledger, identity(contract_path))
    result = dict(status='running', execution_commit=execution_commit,
                  contract_identity=identity(contract_path), planned_cases=list(CASES),
                  historical_occupied_card_s=HISTORICAL_CARD_S)
    write_json(output / 'RESULT.json', result)
    trainer = runtime = state = ancillary = None
    started = time.perf_counter()
    try:
        plan, recovery, features = select_parent(contract['parent_plan'])
        write_json(output / 'RUNTIME_COMPATIBILITY.json', pin_retained_graph_runtime(output))
        trainer = PilotTrainer(plan['configs']['PAIR'], output / 'trainer', arm='PAIR',
            binding=plan['binding'], original_config_text=Path(plan['original_config']).read_text(),
            maximum_events=1, resume_state=recovery, completed_event=1, qualification_only=True)
        state = torch.load(recovery, map_location='cpu', weights_only=False, mmap=True)
        runtime, ancillary = runtime_state(trainer.bundle.model), ancillary_state(trainer)
        paths = [row['path'] for row in features]
        result['startup'] = discarded_startup(trainer, paths, output / 'startup')
        restore_parent(trainer, state, runtime, ancillary, output / 'AFTER_STARTUP.json')
        parent_identity = _endpoint_identity(trainer)
        torch.cuda.reset_peak_memory_stats()
        responses = freeze_two_responses(trainer, paths)
        original = json.loads((Path(contract['original_probe']) / 'WINDOW.json').read_text())
        write_json(output / 'WINDOW.json', require_original_window(responses, original))
        scorer = NativeAllocationScorer(trainer, responses, work, collect_diagnostics=True)
        result.update(execute_cases(trainer, scorer, state, runtime, ancillary, features,
            responses, ledger, work, contract, controls, parent_identity, output))
        result['witnesses'] = evaluate_retained_witnesses(trainer, scorer, contract, output, work)
        scorer.close()
        result.update(status='completed', native_adam_calls=3, exports=0,
                      new_feedback=0, generations=0, full_vectors_written=0)
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if trainer is not None and runtime is not None:
                result['final_restoration'] = restore_parent(trainer, state, runtime,
                    ancillary, output / 'FINAL_RESTORATION.json')
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
