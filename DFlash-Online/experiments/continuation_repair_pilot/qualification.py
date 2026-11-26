"""Three disposable native cases; no shadow optimizer or retained tensor copies."""
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import torch

from experiments.frontier_preservation_pilot.allocation_qualify import vector_identity
from experiments.frontier_preservation_pilot.allocation_integration import _endpoint_identity
from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
from experiments.ok_continuation.resume import differences
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.repair_conditional_kl_pilot.training import discarded_startup
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.storage import write_json
from specforge.training.strategies.base import StepContext

from .native import WarmContinuationTrainer


def coefficient_check(payload):
    """Independent literal FP64 coefficient oracle on represented native inputs."""
    beta = payload['beta'].detach().double().cpu()
    high = payload['high'].detach().bool().cpu()
    score = payload['scores'].detach().double().cpu()
    native = payload['native'].detach().double().cpu()
    candidate = payload['candidate'].detach().double().cpu()
    mass = float(beta.sum())
    high_mass = float(beta[high].sum())
    oracle_native = beta.clone()
    c = kappa = 0.
    if mass > 0 and high_mass > 0 and bool((~high).any()):
        fraction = high_mass / mass
        priority = torch.where(high, beta.new_tensor(2.), beta.new_tensor(1.))
        oracle_native = beta * (0.5 + 0.5 * priority / (1 + fraction))
        c = 0.5 * (1 - fraction) / (1 + fraction)
    expected = oracle_native.clone()
    if mass > 0 and high_mass > 0 and bool((~high).any()) and int(high.sum()) > 1:
        selected = score[high]
        if float(selected.max()) != float(selected.min()):
            kappa = min(c, .5 - c)
            mean = float((beta[high] * selected).sum()) / high_mass
            expected[high] += kappa * beta[high] * (selected - mean)
    tolerance = 2e-7 + 5e-6 * beta.abs()
    error = (candidate - expected).abs()
    mass_error = abs(float(candidate[high].sum()) - high_mass * (1 + c))
    checks = dict(finite=bool(torch.isfinite(candidate).all()), score_bounds=bool(((score >= 0) & (score <= 1)).all()),
                  native_coefficient_oracle=bool(((native - oracle_native).abs() <= tolerance).all()),
                  coefficient_oracle=bool((error <= tolerance).all()),
                  high_mass=mass_error <= float(tolerance[high].sum()),
                  low_exact=torch.equal(candidate[~high], native[~high]),
                  high_bounds=bool(((candidate[high] >= beta[high] - tolerance[high]) &
                                    (candidate[high] <= 1.5 * beta[high] + tolerance[high])).all()))
    result = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
                  repair_count=len(beta), high_count=int(high.sum()),
                  changed_high_positions=int(((candidate != native) & high).sum()),
                  structurally_active=bool(kappa > 0),
                  coefficient_max_error=float(error.max()) if error.numel() else 0.,
                  high_mass_error=mass_error, oracle_kappa=kappa, scalar_correction_not_used_as_activity=True)
    return result


def install_preservation_capture(trainer, case):
    """Fingerprint actual native C coefficients only during original forwards."""
    allocation = trainer.bundle.model.prefix_allocation
    previous_probe = allocation.coefficient_probe
    original_forward = trainer.author_strategy.forward_loss
    rows = []
    collecting = False

    def observe(payload):
        if previous_probe is not None:
            previous_probe(payload)
        if collecting:
            rows.append((payload['candidate'].detach().reshape(-1, 16).cpu(),
                         payload['prefix'].detach().reshape(-1, 16).cpu()))

    def forward(*args, **kwargs):
        nonlocal collecting
        rows.clear()
        collecting = True
        try:
            output = original_forward(*args, **kwargs)
            coefficients, mask = (torch.cat(parts) for parts in zip(*rows))
            case.setdefault('preservation_coefficients', []).append(
                vector_identity(['coefficients', 'prefix_mask'], [coefficients, mask]))
            return output
        finally:
            collecting = False
            rows.clear()

    allocation.coefficient_probe = observe
    trainer.author_strategy.forward_loss = forward


def scientific_case(case):
    rows = deepcopy(case['rows'])
    for row in rows:
        row.pop('training_s', None)
        row['response_observation'].pop('event_id', None)
    # JSON-normalize tuple optimizer options identically on both sides.
    return json.loads(json.dumps(dict(rows=rows, gradient=case['gradient'], endpoint=case['endpoint'],
                                     gradient_norm=case['gradient_norm'])))


def run_case(plan, arm, fixture, output, ledger):
    features = json.loads(Path(plan['feature_manifest']).read_text())
    event = json.loads(Path(plan['retained_event']).read_text())
    trainer = None
    before_runtime = None
    case = dict(status='running', arm=arm, fixture=fixture, rows=[], coefficient_checks=[],
                schedule_advance_work=dict(response_forwards=0, backwards=0, optimizer_steps=0))
    started = time.perf_counter()
    try:
        load_start = time.perf_counter()
        trainer = WarmContinuationTrainer(plan['configs'][arm], output/'trainer', arm=arm,
            binding=plan['binding'], parent_plan=plan['parent_plan'], retained_event=event,
            qualification_only=True, adam_ledger=ledger)
        case['loading_s'] = time.perf_counter() - load_start
        case['startup'] = discarded_startup(trainer, [r['path'] for r in features], output/'startup')
        before_runtime = runtime_state(trainer.bundle.model)
        case['initial_state'] = _endpoint_identity(trainer)
        trainer._active_event = 'qualification_window'
        (trainer.events_root/trainer._active_event).mkdir()
        trainer.bundle.model.begin_event()
        schedule = PresentationSchedule(event, [r['record_id'] for r in features], 2)
        trainer._presentation_schedule = schedule
        order = torch.randperm(len(features)).tolist()
        expected_order = [r['record_id'] for r in event['records'][:128]]
        if [features[i]['record_id'] for i in order] != expected_order:
            raise ValueError('Natural first-pass shuffle differs from retained event')
        # For the sole second fixture, advance the actual two native forwards and
        # backwards without Adam. Counters stay at PAIR1; this is not step 130.
        for index in range(2 * fixture):
            feature = features[order[index]]
            schedule.begin_presentation(0, feature['record_id'])
            raw = torch.load(feature['path'], map_location='cpu', weights_only=True)
            trainer._anchor_observations.clear()
            result = trainer.strategy.forward_loss(trainer._batch(raw), StepContext(
                global_step=128, total_steps=trainer.horizon, collect_detailed_metrics=True))
            case['schedule_advance_work']['response_forwards'] += 1
            trainer.backend.backward(result.loss / 2, is_boundary=index % 2 == 1)
            case['schedule_advance_work']['backwards'] += 1
            schedule.finish_presentation()
        trainer.backend.optimizer.zero_grad()
        case['schedule_advance_physical_work'] = dict(trainer.work)
        trainer.response_records.clear()
        trainer.window_records.clear()
        trainer.work.clear()
        install_preservation_capture(trainer, case)
        extension = getattr(trainer.bundle.model, 'continuation_repair', None)
        if extension is not None:
            def probe(payload):
                checked = coefficient_check(payload)
                checked['preservation_coefficients'] = vector_identity(
                    ['coefficients', 'prefix_mask'],
                    [payload['preservation_coefficients'], payload['preservation_mask']])
                checked['preservation_terms'] = vector_identity(
                    ['delta', 'counts', 'by_slot', 'by_length'], payload['preservation_terms'])
                case['coefficient_checks'].append(checked)
                write_json(output/'RESULT.json', case)
                if checked['status'] != 'passed':
                    raise ValueError('Native coefficient invariant failed')
            extension.coefficient_probe = probe
        original_step = trainer.backend.optimizer.step
        def capture_step():
            named = list(trainer.bundle.draft_model.named_parameters())
            case['gradient'] = vector_identity([n for n, _ in named], [p.grad for _, p in named])
            write_json(output/'RESULT.json', case)
            return original_step()
        trainer.backend.optimizer.step = capture_step
        for ordinal in range(2 * fixture, 2 * fixture + 2):
            feature = features[order[ordinal]]
            schedule.begin_presentation(0, feature['record_id'])
            raw = torch.load(feature['path'], map_location='cpu', weights_only=True)
            trainer._anchor_observations.clear()
            result = trainer.core.train_step(trainer._batch(raw), StepContext(global_step=trainer.global_step,
                total_steps=trainer.horizon, collect_detailed_metrics=True))
            schedule.finish_presentation()
            metrics = result.materialize_metrics()
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
            row = dict(record_id=feature['record_id'], pass_index=0, optimizer_step_after=trainer.global_step,
                optimizer_stepped=result.optimizer_stepped, metrics=metrics,
                sampled_anchors=deepcopy(trainer._anchor_observations),
                response_observation=deepcopy(trainer.response_records[-1]))
            if row['sampled_anchors'] != event['records'][ordinal]['sampled_anchors']:
                raise ValueError('Natural qualification anchors differ')
            case['rows'].append(row)
            write_json(output/'RESULT.json', case)
        trainer._presentation_schedule = None
        case.update(gradient_norm=trainer.window_records[-1]['gradient_norm'], endpoint=_endpoint_identity(trainer),
                    actual_work=dict(trainer.work), objective_costs=trainer.bundle.model.end_event(),
                    counters=[trainer.global_step, trainer.presentations], status='completed')
        if case['counters'] != [129, 258]:
            raise ValueError('Disposable qualification did not commit exactly one Adam step')
        if arm == 'BASE' and fixture == 0:
            observed = case['rows']
            for index in range(2):
                old = event['records'][index]['response_observation']
                current = observed[index]['response_observation']
                for key in ('loss', 'ratio_metrics', 'sum_metrics', 'sampled_anchors', 'prefix_allocation'):
                    if current[key] != old[key]:
                        raise ValueError('Retained BASE first-window observation differs: ' + key)
            case['retained_BASE_first_window_exact'] = True
        if arm == 'TAIL':
            if len(case['coefficient_checks']) != 2:
                raise ValueError('Coefficient probe omitted or double-counted an original response')
            extension.coefficient_probe = None
        write_json(output/'RESULT.json', case)
    except BaseException as error:
        case.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if trainer is not None:
                case['actual_work'] = dict(trainer.work)
                try:
                    # Even a failed case restores its complete original state before disposal.
                    if before_runtime is not None:
                        trainer._presentation_schedule = None
                        trainer.backend.optimizer.zero_grad()
                        trainer.core._ratio_totals.clear()
                        trainer.core._sum_totals.clear()
                        trainer._restore(trainer.parent_recovery)
                        restore_runtime(trainer.bundle.model, before_runtime)
                        case['restored_state'] = _endpoint_identity(trainer)
                        case['restoration_exact'] = case['restored_state'] == case['initial_state']
                        if not case['restoration_exact']:
                            raise ValueError('Disposable case did not restore PAIR1 exactly')
                except BaseException as error:
                    already_failed = case['status'] == 'failed'
                    case.update(status='failed', restoration_error=repr(error))
                    if not already_failed:
                        raise
                finally:
                    if case['status'] != 'completed':
                        trainer.failed = True
                    try:
                        trainer.close()
                    except BaseException as error:
                        already_failed = case['status'] == 'failed'
                        case.update(status='failed', close_error=repr(error))
                        if not already_failed:
                            raise
        finally:
            case['wall_s'] = time.perf_counter() - started
            write_json(output/'RESULT.json', case)
    return case


def qualify(plan):
    from .launch import AdamLedger, boundary
    root = Path(plan['root'])/'qualification'
    root.mkdir(parents=True, exist_ok=False)
    boundary(plan, remaining_write_bytes=10_000_000_000)
    pin_retained_graph_runtime(root)
    result = dict(status='running', binding=plan['binding'], execution_commit=plan['execution_commit'], cases=[])
    ledger = AdamLedger(Path(plan['root'])/'ADAM_LEDGER.json', phase='qualification')
    try:
        for fixture in (0, 1):
            cases = {}
            for arm in ('BASE', 'TAIL_DISABLED', 'TAIL'):
                output = root/f'fixture_{fixture}_{arm}'
                output.mkdir()
                cases[arm] = run_case(plan, arm, fixture, output, ledger)
                result['cases'].append(dict(arm=arm, fixture=fixture, path=str(output/'RESULT.json')))
                write_json(root/'RESULT.json', result)
            mismatch = differences(scientific_case(cases['BASE']), scientific_case(cases['TAIL_DISABLED']))
            result.setdefault('neutrality_checks', []).append(dict(fixture=fixture, mismatches=mismatch[:32]))
            write_json(root/'RESULT.json', result)
            if mismatch:
                raise ValueError('Disabled continuation wrapper differs from native BASE')
            preservation = {arm: cases[arm]['preservation_coefficients'] for arm in cases}
            checked_tail = [row['preservation_coefficients'] for row in cases['TAIL']['coefficient_checks']]
            if (len(preservation['BASE']) != 2 or preservation['BASE'] != preservation['TAIL_DISABLED']
                    or preservation['BASE'] != preservation['TAIL'] or preservation['TAIL'] != checked_tail):
                raise ValueError('Actual native C preservation coefficients changed')
            result.setdefault('preservation_checks', []).append(dict(fixture=fixture,
                same_state_native_C_coefficients_exact=True, original_response_forwards=2))
            expected_work = cases['BASE']['actual_work']
            if (any(cases[arm]['actual_work'] != expected_work for arm in ('TAIL_DISABLED', 'TAIL'))
                    or expected_work.get('response_forwards') != 2
                    or expected_work.get('backwards_attempted') != 2
                    or expected_work.get('backwards_completed') != 2):
                raise ValueError('TAIL changed native forward/backward work')
            result.setdefault('work_checks', []).append(dict(fixture=fixture, exact_native_work=expected_work))
            activity = sum(r['changed_high_positions'] for r in cases['TAIL']['coefficient_checks'])
            result.setdefault('activity', []).append(dict(fixture=fixture, changed_high_positions=activity,
                gradient_changed=cases['TAIL']['gradient'] != cases['BASE']['gradient'],
                BF16_parameters_changed=cases['TAIL']['endpoint']['parameters'] != cases['BASE']['endpoint']['parameters']))
            structurally_active = any(row['structurally_active'] for row in cases['TAIL']['coefficient_checks'])
            result['activity'][-1]['structurally_active'] = structurally_active
            if structurally_active:
                break
        if not cases['BASE'].get('retained_BASE_first_window_exact') and fixture == 0:
            raise ValueError('BASE reuse did not authenticate its first original window')
        result.update(status='passed', base_disabled_exact=True, retained_BASE_reusable=True,
                      inactive_rule_allowed=True, qualification_work=ledger.value)
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(root/'RESULT.json', result)
    return result
