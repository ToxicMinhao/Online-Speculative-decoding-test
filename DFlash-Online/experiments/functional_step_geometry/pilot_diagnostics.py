"""Predetermined fitting/off-fit counterfactuals; never alter a serving update.

The production native-gradient columns are reused. Uniform geometry and the
expanded sketch's raw-FSG direction require one additional VJP per response
each. Candidate scoring uses the original reference support for every method.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from specforge.core.chunking import checkpointed_chunk_reduce
from online_feedback import functional_geometry_math as geometry
from online_feedback.first_error import first_error_mask, valid_prefix_mask
from online_feedback.functional_geometry import FEASIBLE_REALIZATION, _cleanup
from online_feedback.prefix_correction import canonical_margin, repair_transitions
from online_feedback.prefix_guard import prefix_changes
from online_feedback.ranking import matching_prefix_mask
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json, save_torch
from .qualification_checks import restored_check
from .work import QualificationWork, LIMITS


BUDGET = dict(additional_response_forwards=40, additional_backwards=4)


@dataclass
class FixedResponse:
    response: dict
    hidden: torch.Tensor
    weights: torch.Tensor
    proposals: torch.Tensor
    prefix_mask: torch.Tensor
    teacher_hidden: torch.Tensor
    record_id: str
    role: str


def validate_bindings(hook, fitting_batches, off_fit_batches, binding):
    """Confirm the declared A/D identities and complete current A tensors."""
    if (hook.arm != 'FSG' or hook.realization_revision != FEASIBLE_REALIZATION
            or binding['window'] not in (1, 32, 65, 128)
            or binding['event'] not in (5, 6)
            or len(fitting_batches) != 2 or len(off_fit_batches) != 2
            or len(hook.responses) != 2):
        raise ValueError('Diagnostics require a declared R1 FSG window and two A/D pairs')
    if ([batch['record_id'] for batch in fitting_batches] != binding['fitting_ids']
            or [batch['record_id'] for batch in off_fit_batches] != binding['off_fit_ids']
            or len(set(binding['fitting_ids'] + binding['off_fit_ids'])) != 4):
        raise ValueError('Diagnostic pair identities differ from the sealed schedule')
    if (binding['dataset'] not in ('gsm8k', 'mathqa') or binding['diagnostic_rng_key'] !=
            [20261108, ('gsm8k', 'mathqa').index(binding['dataset']), binding['event'], binding['window']]):
        raise ValueError('Diagnostic sampling key differs from the fixed declared rule')
    for captured, batch in zip(hook.responses, fitting_batches):
        for key in ('input_ids', 'hidden_states', 'loss_mask'):
            if not torch.equal(captured[key], batch[key].detach().cpu()):
                raise ValueError('Fitting diagnostic tensors differ: ' + key)
    for batch in [*fitting_batches, *off_fit_batches]:
        if 'target_last_hidden_states' not in batch:
            raise ValueError('Fixed-support teacher KL requires the retained final teacher features')


@contextmanager
def separate_diagnostic_work(hook, root, receipt):
    """Charge diagnostic work to its own durable ledger, not main training."""
    saved = hook.work, hook.costs, hook.forward_calls, hook.work_callback, hook.phase
    hook.work = {name: 0 for name in hook.work}
    hook.costs, hook.forward_calls = {}, {}
    limits = dict(LIMITS, adam=0, ordinary_forward=0, ordinary_backward=0,
                  additional_forward=40, additional_backward=4, serving_generation=0)
    ledger = QualificationWork(Path(root) / 'MODEL_WORK.jsonl', limits=limits)
    hook.work_callback = ledger.callback
    try:
        yield ledger
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), model_work=ledger.summary())
        _cleanup(lambda: write_json(Path(root) / 'RESULT.json', receipt), error,
                 receipt, 'diagnostic failure receipt')
        raise
    finally:
        hook.work, hook.costs, hook.forward_calls, hook.work_callback, hook.phase = saved


def _aligned_teacher(hook, response, batch):
    device = hook.active_parameters[0].device
    _, _, indices = hook.model._block_labels_and_weights(
        response['input_ids'].to(device), response['loss_mask'].to(device),
        response['anchors'].to(device), response['keep'].to(device))
    teacher = batch['target_last_hidden_states'].detach().to(device)
    return hook.model._aligned_target_hidden(teacher, indices).detach()


def fixed_responses(hook, fitting_batches, off_fit_batches, binding):
    """Reuse fitting references and draw off-fit anchors only under local RNG."""
    references = []
    for original, batch in zip(hook.window_context['references'], fitting_batches):
        references.append(FixedResponse(original.response, original.hidden,
            original.weights, original.proposals, original.prefix_mask,
            _aligned_teacher(hook, original.response, batch), batch['record_id'], 'A'))
    device = hook.active_parameters[0].device
    seed = int(np.random.SeedSequence(binding['diagnostic_rng_key']).generate_state(1, dtype=np.uint64)[0])
    with preserve_rng():
        torch.manual_seed(seed)
        for batch in off_fit_batches:
            inputs, features, mask = [batch[key].to(device) for key in
                                     ('input_ids', 'hidden_states', 'loss_mask')]
            # Bypass the training observation wrapper so diagnostics cannot append
            # ordinary anchor records or advance the sealed presentation schedule.
            anchors, keep = hook.trainer._original_sampler(inputs.shape[1], mask, device)
            labels, weights, _ = hook.model._block_labels_and_weights(inputs, mask, anchors, keep)
            response = {key: value.detach().cpu() for key, value in dict(
                input_ids=inputs, hidden_states=features, loss_mask=mask, anchors=anchors,
                keep=keep, labels=labels, validity_weights=weights).items()}
            hidden = hook.forward_hidden(response, 'diagnostic_off_fit_prestate', grad=False).detach()
            protection, proposals, prefix, _ = hook._scan_reference(
                hidden, response, hook.model.objective_chunk_blocks)
            references.append(FixedResponse(response, hidden, protection, proposals, prefix,
                _aligned_teacher(hook, response, batch), batch['record_id'], 'D'))
    return references, seed


@torch.no_grad()
def measure_hidden(hook, reference, hidden):
    """Fixed-support draft drift, teacher KL and native prediction observations."""
    response = reference.response
    labels = response['labels'].to(hidden.device)
    validity = response['validity_weights'].to(hidden.device)
    first_error = first_error_mask(reference.proposals, labels, validity,
                                  hook.model.eos_token_ids) & ~reference.prefix_mask
    support = dict(prefix=reference.prefix_mask, first_error=first_error)
    teacher_numerators = {name: 0. for name in support}
    teacher_denominators = {name: float((validity * mask).double().sum())
                            for name, mask in support.items()}
    proposals, margins, risk_terms = [], [], []
    chunk = hook.model.objective_chunk_blocks
    for start in range(0, hidden.shape[1], chunk):
        stop = start + chunk
        before = hook.model.lm_head(reference.hidden[:, start:stop])
        logits = hook.model.lm_head(hidden[:, start:stop])
        geometry.finite(logits, 'diagnostic native logits')
        risk_terms.append(geometry.risk_terms(before, logits, reference.weights[:, start:stop]))
        proposals.append(logits.argmax(-1))
        margins.append(canonical_margin(logits, labels[:, start:stop]))
        teacher_logits = hook.model.lm_head(reference.teacher_hidden[:, start:stop]).to(hook.model.loss_dtype)
        hook.work['teacher_head_calls'] = hook.work.get('teacher_head_calls', 0) + 1
        token_kl = F.kl_div(F.log_softmax(logits.to(hook.model.loss_dtype), dim=-1),
                           F.softmax(teacher_logits, dim=-1), reduction='none').sum(-1)
        geometry.finite(token_kl, 'diagnostic teacher KL')
        for name, mask in support.items():
            weighted = validity[:, start:stop] * mask[:, start:stop]
            teacher_numerators[name] += float((token_kl.double() * weighted.double()).sum())
    raw_risk = math.fsum(float(term[0]) for term in risk_terms)
    absolute = math.fsum(float(term[1]) for term in risk_terms)
    predictions, margin = torch.cat(proposals, 1), torch.cat(margins, 1)
    valid = valid_prefix_mask(labels, validity, hook.model.eos_token_ids)
    teacher = {name: dict(numerator=teacher_numerators[name], denominator=denominator,
                         mean=teacher_numerators[name] / denominator if denominator else None,
                         occurrences=int(support[name].sum()))
               for name, denominator in teacher_denominators.items()}
    return dict(record_id=reference.record_id, role=reference.role,
        risk_unaveraged=raw_risk, absolute_sum_unaveraged=absolute,
        teacher_kl=teacher, valid_occurrences=int(valid.sum()),
        protected_occurrences=int(reference.prefix_mask.sum()),
        tie_occurrences=int((valid & (margin == 0)).sum()),
        wrong_ties=int((valid & (predictions != labels) & (margin == 0)).sum()),
        predictions=predictions.detach().cpu(), margins=margin.detach().cpu())


def _response_metrics(reference, before, native, current, eos):
    labels, validity = reference.response['labels'], reference.response['validity_weights']
    old, now = before['predictions'], current['predictions']
    row = {name: value for name, value in current.items() if name not in ('predictions', 'margins')}
    row.update(prefix_changes(old, now, labels, validity, eos))
    row.update(repair_transitions(old, native['predictions'], now, labels, validity, eos))
    valid = valid_prefix_mask(labels, validity, eos)
    difference = current['margins'] - before['margins']
    old_length, native_length, current_length = [matching_prefix_mask(predictions, labels, validity, eos).sum(-1)
        for predictions in (old, native['predictions'], now)]
    native_gain = (native_length - old_length).clamp_min(0)
    current_gain = (current_length - old_length).clamp_min(0)
    row.update(gained_prefix_occurrences=row['net_matching_prefix_change'] + row['old_prefix_tokens_lost'],
        native_gained_prefix_occurrences=int(native_gain.sum()),
        native_prefix_gains_retained=int(torch.minimum(native_gain, current_gain).sum()),
        native_prefix_gains_lost=int((native_gain - current_gain).clamp_min(0).sum()),
        prefix_gains_beyond_native=int((current_gain - native_gain).clamp_min(0).sum()),
        canonical_margin_increases=int((valid & (difference > 0)).sum()),
        canonical_margin_decreases=int((valid & (difference < 0)).sum()),
        canonical_margin_ties=int((valid & (difference == 0)).sum()))
    return row


def extra_direction(hook, reference, native_hidden, raw_hidden, *, uniform):
    """One extra VJP, retaining the production native gradient for expansion."""
    hidden = hook.forward_hidden(reference.response, 'diagnostic_direction_prestate', grad=True)
    if not torch.equal(hidden.detach(), reference.hidden):
        raise ValueError('Counterfactual prestate hidden differs from the production reference')
    weights = reference.prefix_mask.float()
    count = weights.sum()
    if float(count) > 0:
        weights = weights / count

    def terms(live, native, raw, original_weights, uniform_weights):
        hook.work['checkpoint_proxy_calls'] += 1
        logits = hook.model.lm_head(live)
        with torch.no_grad():
            native_logits = hook.model.lm_head(native)
            if uniform:
                cotangent, statistic = geometry.geometry_terms(logits, native_logits, uniform_weights)
            else:
                raw_logits = hook.model.lm_head(raw)
                directions, statistic = geometry.expanded_terms(logits, native_logits, raw_logits, original_weights)
                cotangent = directions[1]
        return (logits.float() * cotangent).sum(dtype=torch.float32), statistic

    hook.phase = 'diagnostic_direction_proxy'
    proxy, statistic = checkpointed_chunk_reduce(terms, hidden, native_hidden, raw_hidden,
        reference.weights, weights, chunk_size=hook.model.objective_chunk_blocks, dim=1, checkpoint=True)
    hook.phase = 'diagnostic_direction_vjp'
    with hook._work('additional_backwards'), hook._timed('diagnostic_extra_vjp_s'):
        gradients = torch.autograd.grad(proxy, hook.active_parameters, allow_unused=True)
    for gradient in gradients:
        if gradient is not None:
            geometry.finite(gradient, 'diagnostic VJP')
    return tuple(None if g is None else g.detach() for g in gradients), statistic.detach()


def run_window_diagnostics(hook, fitting_batches, off_fit_batches, *, output, binding):
    """Score the fixed candidates and restore the actual committed endpoint."""
    validate_bindings(hook, fitting_batches, off_fit_batches, binding)
    root = Path(output)
    receipt = dict(binding=deepcopy(binding), realization_revision=hook.realization_revision,
        teacher_kl_definition='qualified BF16 KL, fixed prefix/first-error support; original validity/decay weights',
        teacher_preparation=0, extra_adam_steps=0,
        historical_cross_layout_gate_passed=False)
    with separate_diagnostic_work(hook, root, receipt) as ledger:
        with restored_check(hook, root / 'state', receipt, BUDGET) as committed:
            if hook.window_context is None:
                # A zero native displacement legitimately bypasses production
                # geometry. Its scheduled diagnostic still observes the fixed
                # responses, without introducing another update or nonzero VJP.
                if not all(torch.equal(old, native) for old, native in zip(hook.theta0, hook.theta_native)):
                    raise ValueError('Missing production references for a nonzero native displacement')
                native_hidden = [hook.forward_hidden(response, 'diagnostic_zero_native', grad=False)
                                 for response in hook.responses]
                hook.restore_endpoint(hook.theta0)
                rows = [hook.build_reference(response, hidden, hook.active_parameters)
                        for response, hidden in zip(hook.responses, native_hidden)]
                sketch = geometry.make_sketch([row.gradients for row in rows], [row.c for row in rows],
                                               hook.theta0, hook.theta_native)
                hook.window_context = dict(references=rows, sketch=sketch,
                    native_risk=geometry.pooled_risk([row.native_risk_terms for row in rows], hook.vocab_size),
                    row=dict(decision=dict(accepted=False, reason='zero_native_displacement')))
                del native_hidden
            context = hook.window_context
            production = context['sketch']
            production_row = context['row']
            names = [hook.parameter_names[id(p)] for p in hook.active_parameters]
            def save_endpoint(path, active_values):
                active = dict(zip(names, active_values))
                # Inactive tensors can contain earlier updates. Keep their
                # current values too, so each endpoint is reconstructible even
                # when this window has a narrower native gradient support.
                weights = {name: active.get(name, value).detach().cpu()
                    for name, value in hook.trainer.bundle.draft_model.state_dict().items()}
                save_torch(path, weights)
                return len(weights)
            with hook._timed('diagnostic_endpoint_persistence_s'):
                prestate_tensors = save_endpoint(root / 'prestate.pt', hook.theta0)
                native_tensors = save_endpoint(root / 'native.pt', hook.theta_native)
            receipt['saved_endpoint_scope'] = dict(kind='complete_draft_state_dict',
                active_parameter_names=names, prestate_tensors=prestate_tensors,
                native_tensors=native_tensors, inactive_tensors='current values unchanged by this window')
            hook.restore_endpoint(hook.theta0)
            references, seed = fixed_responses(hook, fitting_batches, off_fit_batches, binding)
            receipt['diagnostic_seed'] = seed
            write_json(root / 'ANCHORS.json', [dict(record_id=r.record_id, role=r.role,
                anchors=r.response['anchors'].tolist(), keep=r.response['keep'].tolist()) for r in references])
            scores = {'prestate': [measure_hidden(hook, ref, ref.hidden) for ref in references]}
            candidate_rows = {}
            saved_hidden = {}

            def write_raw(sketch):
                with hook._timed('diagnostic_parameter_construction_s'):
                    result = geometry.write_candidate(hook.active_parameters, hook.theta0,
                                                       hook.theta_native, sketch)
                hook.work['diagnostic_candidate_write_passes'] = hook.work.get('diagnostic_candidate_write_passes', 0) + 1
                hook.work['diagnostic_endpoint_statistic_scans'] = hook.work.get('diagnostic_endpoint_statistic_scans', 0) + 1
                return result

            def write_scaled(sketch, alpha):
                with hook._timed('diagnostic_parameter_construction_s'):
                    result = geometry.write_scaled_candidate(hook.active_parameters, hook.theta0,
                                                              hook.theta_native, sketch, alpha)
                hook.work['diagnostic_candidate_write_passes'] = hook.work.get('diagnostic_candidate_write_passes', 0) + 1
                hook.work['diagnostic_endpoint_statistic_scans'] = hook.work.get('diagnostic_endpoint_statistic_scans', 0) + 1
                return result

            def score(name, *, keep_hidden=False):
                identities = geometry.endpoint_identities(hook.active_parameters, hook.theta0, hook.theta_native)
                alias = 'prestate' if identities['identical_prestate'] else (
                    'native' if identities['identical_native'] and 'native' in scores else None)
                if alias is not None and not keep_hidden:
                    scores[name] = scores[alias]
                    return alias
                measurements, hidden_a = [], []
                for ref in references:
                    hidden = hook.forward_hidden(ref.response, 'diagnostic_' + name, grad=False).detach()
                    measurements.append(measure_hidden(hook, ref, hidden))
                    if ref.role == 'A' and keep_hidden:
                        hidden_a.append(hidden)
                scores[name] = measurements
                if keep_hidden:
                    saved_hidden[name] = hidden_a
                return None

            def guarded(name, statistics):
                risk = geometry.pooled_risk([(row['risk_unaveraged'], row['absolute_sum_unaveraged'])
                                             for row in scores[name][:2]], hook.vocab_size)
                if statistics['identical_prestate'] and not statistics['identical_native']:
                    decision = dict(accepted=False, reason='rounded_to_prestate')
                elif statistics['identical_native']:
                    decision = dict(accepted=False, reason='identical_native_endpoint')
                else:
                    decision = geometry.gate(risk, common_native_risk, statistics['norm'], statistics['native_norm'])
                return dict(candidate=statistics, candidate_risk=risk.as_dict(), decision=decision,
                            guarded_score=name if decision['accepted'] else 'native')

            hook.restore_endpoint(hook.theta_native)
            score('native', keep_hidden=True)
            common_native_risk = geometry.pooled_risk([
                (row['risk_unaveraged'], row['absolute_sum_unaveraged'])
                for row in scores['native'][:2]], hook.vocab_size)
            receipt.update(common_native_risk=common_native_risk.as_dict(),
                production_native_risk=context['native_risk'].as_dict(),
                native_risk_remeasurement_difference=common_native_risk.raw - context['native_risk'].raw)
            if not production_row['decision']['accepted'] and not all(
                    torch.equal(parameter, saved) for parameter, saved in zip(hook.active_parameters, committed)):
                raise ValueError('Production fallback differs from its native endpoint')
            write_raw(production)
            score('raw_fsg', keep_hidden=True)
            candidate_rows['raw_fsg'] = dict(candidate=geometry.endpoint_statistics(
                hook.active_parameters, hook.theta0, hook.theta_native), diagnostic_only=True,
                role='unadjusted a=1; expanded second direction')
            alpha = production_row.get('realization', {}).get('alpha', 1.)
            feasible = write_scaled(production, alpha)
            feasible.update(geometry.endpoint_identities(hook.active_parameters, hook.theta0, hook.theta_native))
            if alpha == 1.:
                scores['feasible_fsg'] = scores['raw_fsg']
                saved_hidden['feasible_fsg'] = saved_hidden['raw_fsg']
            else:
                score('feasible_fsg', keep_hidden=True)
            candidate_rows['feasible_fsg'] = guarded('feasible_fsg', feasible)
            if production_row['decision']['accepted'] and not all(
                    torch.equal(parameter, saved) for parameter, saved in zip(hook.active_parameters, committed)):
                raise ValueError('Production accepted endpoint differs from its feasible construction')
            committed_alias = 'feasible_fsg' if production_row['decision']['accepted'] else 'native'
            scores['committed'] = scores[committed_alias]
            saved_hidden['committed'] = saved_hidden[committed_alias]
            candidate_rows['committed'] = dict(production_decision=production_row['decision'],
                                               exact_score_reuse=committed_alias)

            target = feasible['norm']
            if target == 0:
                hook.restore_endpoint(hook.theta_native)
                scalar = geometry.endpoint_statistics(hook.active_parameters, hook.theta0, hook.theta_native)
                scalar.update(target_norm=0., bypass='zero_hypothetical_displacement')
            else:
                with hook._timed('diagnostic_scalar_matching_s'):
                    scalar = geometry.scalar_match(hook.active_parameters, hook.theta0, hook.theta_native, target)
                hook.work['diagnostic_scalar_norm_scan_passes'] = scalar['norm_scan_passes']
                hook.work['diagnostic_scalar_write_passes'] = 1
            scalar.update(geometry.endpoint_identities(hook.active_parameters, hook.theta0, hook.theta_native))
            score('scalar')
            candidate_rows['scalar'] = guarded('scalar', scalar)

            # Uniform and expanded gradients are formed in separate phases so
            # at most four native parameter-sized VJP vectors coexist.
            for name, uniform in (('uniform', True), ('expanded', False)):
                if not uniform and production.sigma == 0:
                    scores[name] = scores['native']
                    candidate_rows[name] = dict(bypass='zero_production_sigma', guarded_score='native')
                    continue
                hook.restore_endpoint(hook.theta0)
                extra = [extra_direction(hook, ref, native, raw, uniform=uniform)
                    for ref, native, raw in zip(context['references'], saved_hidden['native'], saved_hidden['raw_fsg'])]
                if uniform:
                    sketch = geometry.make_sketch([row[0] for row in extra], [float(row[1]) for row in extra],
                                                   hook.theta0, hook.theta_native)
                else:
                    sketch = geometry.make_expanded_sketch(
                        [(ref.gradients, row[0]) for ref, row in zip(context['references'], extra)],
                        [row[1] for row in extra], hook.theta0, hook.theta_native,
                        production_sigma=production.sigma)
                with hook._timed('diagnostic_feasible_realization_s'):
                    realization = geometry.realize_feasible_candidate(hook.active_parameters, hook.theta0,
                                                                       hook.theta_native, sketch)
                for key in ('probe_count', 'norm_scan_passes', 'construction_scan_passes', 'identity_scan_passes', 'write_passes'):
                    name_key = 'diagnostic_feasibility_' + key
                    hook.work[name_key] = hook.work.get(name_key, 0) + realization[key]
                write_raw(sketch)
                score(name + '_raw')
                candidate_rows[name + '_raw'] = dict(candidate=realization['raw_candidate'],
                    diagnostic_only=True, role='unadjusted a=1 counterfactual')
                write_scaled(sketch, realization['alpha'])
                if realization['alpha'] == 1.:
                    scores[name] = scores[name + '_raw']
                else:
                    score(name)
                candidate_rows[name] = dict(guarded(name, realization['candidate']), realization=realization,
                                            sketch=sketch.statistics)
                del extra, sketch

            hook.restore_endpoint(committed)
            for ref, before in zip(references[:2], saved_hidden['committed']):
                after = hook.forward_hidden(ref.response, 'diagnostic_committed_restoration', grad=False)
                if not torch.equal(after, before):
                    raise ValueError('Diagnostic physical restore changed committed hidden states')
            metrics = {name: [_response_metrics(ref, before, native, current, hook.model.eos_token_ids)
                for ref, before, native, current in zip(references, scores['prestate'], scores['native'], rows)]
                for name, rows in scores.items()}
            role_risks = {name: {role: geometry.pooled_risk([
                (row['risk_unaveraged'], row['absolute_sum_unaveraged'])
                for row in rows if row['role'] == role], hook.vocab_size).as_dict()
                for role in ('A', 'D')} for name, rows in scores.items()}
            save_torch(root / 'PREDICTIONS.pt', {name: [dict(predictions=row['predictions'], margins=row['margins'])
                                                       for row in rows] for name, rows in scores.items()})
            receipt.update(candidates=candidate_rows, measurements=metrics, role_risks=role_risks,
                           physical_restore_hidden_exact=True, costs=dict(hook.costs),
                           model_work=ledger.summary())
        receipt.update(status='completed', dataset=binding['dataset'], event=binding['event'],
            window=binding['window'], restoration_exact=all(receipt[name] for name in (
                'committed_weights_exact', 'rng_exact', 'modes_exact', 'gradients_unchanged',
                'optimizer_tensor_identity_and_versions_exact')),
            response_forwards=receipt['actual_work']['additional_response_forwards_attempted'],
            backwards=receipt['actual_work']['additional_backwards_attempted'],
            physical_head_calls=receipt['actual_work']['head_projection_calls'])
        write_json(root / 'RESULT.json', receipt)
    return receipt
