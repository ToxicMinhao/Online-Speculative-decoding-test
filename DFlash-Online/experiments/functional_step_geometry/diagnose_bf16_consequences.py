"""Compare guarded BF16 FSG consequences on one retained complete window.

Both layouts use the production-eight reference membership and weights. Only
the native vocabulary-head projection chunk changes to four. No optimizer runs,
and no temporary proposal becomes a training or serving checkpoint.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch

from experiments.ok_continuation.resume import differences
from online_feedback import functional_geometry_math as geometry
from online_feedback.fsg_consequence_diagnostic import (
    compare_endpoints, compare_sketches, decision_summary, prefix_consequences,
)
from online_feedback.storage import write_json
from online_feedback.training import validate_feature
from specforge.core.chunking import checkpointed_chunk_reduce
from .diagnose_gradients import run
from .native_oracles import gradient_comparison


BUDGET = dict(response_forward=10, draft_backward=4, optimizer_step=0,
              feedback_generation=0, teacher_preparation=0, serving_generation=0)
LAYOUTS = (8, 4)
REPRODUCTION_FIELDS = ('native', 'responses', 'sketch', 'native_risk', 'raw_fsg', 'decision')


def load_window(trainer, feature_paths, presentations_path):
    """Load the original two responses and every retained anchor, without draws."""
    rows = json.loads(Path(presentations_path).read_text())
    if len(feature_paths) != 2 or len(rows) < 2:
        raise ValueError('Require the first complete retained two-response window')
    responses, metadata = [], []
    for path, row in zip(feature_paths, rows[:2]):
        raw = torch.load(path, map_location='cpu', weights_only=False)
        checked = validate_feature(raw, max_length=trainer.max_length)
        if row['record_id'] != raw['record_id'] or len(row['sampled_anchors']) != 1:
            raise ValueError('Retained full-window feature and anchor identities differ')
        draw = row['sampled_anchors'][0]
        anchors = torch.tensor(draw['positions'], dtype=torch.int64)
        keep = torch.tensor(draw['keep_mask'], dtype=torch.bool)
        if (anchors.ndim != 2 or anchors.shape[0] != 1 or anchors.shape != keep.shape
                or anchors.shape[1] == 0 or not bool(keep.any())):
            raise ValueError('Retained full-window anchors are unavailable')
        batch = trainer._batch(raw).tensors
        labels, weights, _ = trainer.bundle.model._block_labels_and_weights(
            batch['input_ids'], batch['loss_mask'], anchors, keep)
        responses.append(dict(batch, anchors=anchors, keep=keep, labels=labels,
                              validity_weights=weights, record_id=raw['record_id']))
        metadata.append(dict(checked, anchors=anchors.tolist(), keep=keep.tolist(),
            original_anchor_count=anchors.shape[1], feature_path=str(path),
            selection='all original anchors in their retained order; no sampler call'))
    if responses[0]['record_id'] == responses[1]['record_id']:
        raise ValueError('The retained window must contain two distinct responses')
    return responses, metadata


def reproduction_receipt(observed, retained):
    expected = {key: retained[key] for key in REPRODUCTION_FIELDS}
    actual = {key: observed[key] for key in REPRODUCTION_FIELDS}
    mismatch = differences(actual, expected)
    return dict(status='passed' if not mismatch else 'failed', differences=mismatch,
        observed=actual, expected=expected, exact_retained_receipt_comparison=True,
        historical_raw_parameter_tensors_available=False,
        interpretation='exact retained scientific metrics, not historical raw-candidate tensor identity')


def require_reproduction(observed, retained, root):
    receipt = reproduction_receipt(observed, retained)
    write_json(Path(root) / 'NATIVE_REPRODUCTION.json', receipt)
    if receipt['status'] != 'passed':
        raise ValueError('Production-eight retained metrics did not reproduce; alternate layout not executed')
    return receipt


@dataclass
class RetainedWindow:
    hook: object
    responses: list
    parameters: tuple
    names: list
    theta0: tuple
    theta_native: tuple
    work: object
    root: Path
    retained: dict


def prepare_window_case(*, trainer, hook, plan, work, root, parameters, names, before, native):
    if tuple(plan['layouts']) != LAYOUTS or hook.model.objective_chunk_blocks != 8:
        raise ValueError('Keep production-eight and prespecified alternate-four projection layouts')
    responses, metadata = load_window(trainer, plan['features'], plan['presentations'])
    retained = json.loads(Path(plan['retained_step']).read_text())
    if retained['active_parameter_names'] != names or retained['strength'] != 1.:
        raise ValueError('Retained ordinary-active parameters or FSG strength changed')
    theta0 = tuple(parameter.detach().clone() for parameter in parameters)
    theta_native = tuple(native['weights'][name].to(parameter.device).clone()
                         for name, parameter in zip(names, parameters))
    hook.theta0, hook.theta_native = theta0, theta_native
    kinds = {'additional_response_forwards': 'response_forward',
             'additional_backwards': 'draft_backward'}
    hook.work_callback = lambda kind, status: work.record(kinds[kind], status)
    case = RetainedWindow(hook, responses, parameters, names, theta0, theta_native,
                          work, root, retained)
    return case, dict(fixture=metadata, layouts=list(LAYOUTS),
        reference_policy='production-eight prefixes and weights fixed for both layouts',
        protection_responses=2, anchor_subsampling=False)


def alternate_reference(hook, response, native_hidden, reference):
    """One original BF16 VJP with fixed production reference weights, chunk four."""
    hidden = hook.forward_hidden(response, 'alternate_four_prestate', grad=True)
    if not bool(reference.prefix_mask.any()):
        return tuple(None for _ in hook.active_parameters), 0.
    hook.phase = 'alternate_four_proxy'
    with hook._timed('alternate_four_proxy_s'):
        proxy, energy, _, _ = checkpointed_chunk_reduce(hook._proxy_chunk,
            hidden, native_hidden.detach(), reference.weights, chunk_size=4,
            dim=1, checkpoint=True)
    if float(energy) == 0.:
        return tuple(None for _ in hook.active_parameters), 0.
    hook.phase = 'alternate_four_vjp'
    with hook._work('additional_backwards'), hook._timed('alternate_four_vjp_s'):
        values = torch.autograd.grad(proxy, hook.active_parameters, allow_unused=True)
    for value in values:
        if value is not None:
            geometry.finite(value, 'alternate-four native gradient')
    return tuple(None if value is None else value.detach() for value in values), float(energy)


@torch.no_grad()
def score_hidden(hook, references, hidden_states):
    """Common production-eight q0/weights and precision for every endpoint."""
    risk_parts, rows, proposals = [], [], []
    hook.phase = 'common_eight_score'
    for reference, hidden in zip(references, hidden_states):
        risk = hidden.new_zeros((), dtype=torch.float64)
        absolute = hidden.new_zeros((), dtype=torch.float64)
        prediction_chunks = []
        for start in range(0, hidden.shape[1], 8):
            stop = start + 8
            old_logits = hook.model.lm_head(reference.hidden[:, start:stop])
            logits = hook.model.lm_head(hidden[:, start:stop])
            values = geometry.risk_terms(old_logits, logits, reference.weights[:, start:stop])
            risk += values[0]
            absolute += values[1]
            prediction_chunks.append(logits.argmax(-1))
        risk_parts.append((float(risk), float(absolute)))
        rows.append(dict(record_id=reference.response['record_id'],
                         risk_unaveraged=float(risk), absolute_sum_unaveraged=float(absolute)))
        proposals.append(torch.cat(prediction_chunks, 1).detach())
    return geometry.pooled_risk(risk_parts, hook.vocab_size), rows, proposals


def raw_candidate(case, references, sketch, native_risk, label):
    hook = case.hook
    with hook._timed(label + '_formation_s'):
        candidate = geometry.write_candidate(case.parameters, case.theta0, case.theta_native, sketch)
    # These are diagnostic raw-candidate evaluations even if the norm veto is
    # already known. They do not bypass or alter the subsequent production gate.
    hidden = [hook.forward_hidden(response, label + '_candidate', grad=False).detach()
              for response in case.responses]
    with hook._timed(label + '_common_score_s'):
        risk, risks, proposals = score_hidden(hook, references, hidden)
    del hidden
    decision = geometry.gate(risk, native_risk, candidate['norm'], candidate['native_norm'],
                             identical=candidate['identical_native'])
    if sketch.sigma == 0.:
        decision = dict(accepted=False, reason='empty_sketch')
    snapshot = tuple(parameter.detach().clone() for parameter in case.parameters)
    row = dict(raw_fsg=candidate, sketch=sketch.statistics, candidate_risk=risk.as_dict(),
        candidate_response_risks=risks, decision=decision,
        raw_scored_despite_norm_rejection=candidate['norm'] > candidate['native_norm'])
    write_json(case.root / (label.upper() + '.json'), row)
    return row, snapshot, proposals


def diagnose_consequences(case):
    hook = case.hook
    started = time.perf_counter()
    hook.restore_endpoint(case.theta_native)
    native_stats = geometry.endpoint_statistics(case.parameters, case.theta0, case.theta_native)
    native_hidden = [hook.forward_hidden(response, 'retained_native', grad=False).detach()
                     for response in case.responses]
    hook.restore_endpoint(case.theta0)
    references = [hook.build_reference(response, hidden, case.parameters, chunk_blocks=8)
                  for response, hidden in zip(case.responses, native_hidden)]
    native_risk = geometry.pooled_risk([row.native_risk_terms for row in references], hook.vocab_size)
    common_native_risk, native_rows, native_proposals = score_hidden(hook, references, native_hidden)
    if differences(native_risk.as_dict(), common_native_risk.as_dict()):
        write_json(case.root / 'COMMON_RISK_MISMATCH.json',
            dict(original=native_risk.as_dict(), rescored=common_native_risk.as_dict()))
        raise ValueError('Common production-eight risk did not reproduce native risk')
    sketch8 = geometry.make_sketch([row.gradients for row in references],
        [row.c for row in references], case.theta0, case.theta_native)
    row8, weights8, proposals8 = raw_candidate(case, references, sketch8, native_risk, 'production_eight')
    observed = dict(native=native_stats, responses=[row.statistics for row in references],
        sketch=sketch8.statistics, native_risk=native_risk.as_dict(),
        raw_fsg=row8['raw_fsg'], decision=row8['decision'])
    reproduced = require_reproduction(observed, case.retained, case.root)

    # Every production graph has completed its VJP. Restore the pre-state and
    # evaluate the second numerical layout on exactly the same fixed objective.
    hook.restore_endpoint(case.theta0)
    alternate = [alternate_reference(hook, response, hidden, reference)
                 for response, hidden, reference in zip(case.responses, native_hidden, references)]
    gradient_rows = [dict(record_id=response['record_id'],
        comparison=gradient_comparison(reference.gradients, changed[0], case.names))
        for response, reference, changed in zip(case.responses, references, alternate)]
    write_json(case.root / 'GRADIENT_COMPARISONS.json', gradient_rows)
    sketch4 = geometry.make_sketch([row[0] for row in alternate], [row[1] for row in alternate],
                                  case.theta0, case.theta_native)
    row4, weights4, proposals4 = raw_candidate(case, references, sketch4, native_risk, 'alternate_four')
    raw_comparison = compare_endpoints(weights8, weights4, case.theta0, case.theta_native)
    committed8 = weights8 if row8['decision']['accepted'] else case.theta_native
    committed4 = weights4 if row4['decision']['accepted'] else case.theta_native
    committed_comparison = compare_endpoints(committed8, committed4, case.theta0, case.theta_native)
    prefix_rows = []
    for index, reference in enumerate(references):
        response = reference.response
        common = dict(labels=response['labels'].to(reference.proposals.device),
                      validity_weights=response['validity_weights'].to(reference.proposals.device),
                      eos_token_ids=hook.model.eos_token_ids)
        raw = prefix_consequences(reference.proposals, native_proposals[index],
                                  proposals8[index], proposals4[index], **common)
        committed = prefix_consequences(reference.proposals, native_proposals[index],
            proposals8[index] if row8['decision']['accepted'] else native_proposals[index],
            proposals4[index] if row4['decision']['accepted'] else native_proposals[index], **common)
        prefix_rows.append(dict(record_id=response['record_id'], raw=raw, committed=committed))
    result = dict(policy='native_bf16_same_fixed_production_reference',
        reproduction=reproduced, native_risk=native_risk.as_dict(), native_response_risks=native_rows,
        production_eight=row8, alternate_four=row4,
        gradient_comparisons=gradient_rows,
        sketches=compare_sketches(sketch8, sketch4, case.theta0, case.theta_native),
        raw_endpoint_comparison=raw_comparison, committed_endpoint_comparison=committed_comparison,
        decisions=decision_summary(row8['decision'], row4['decision'], raw_comparison, committed_comparison),
        prefix_consequences=prefix_rows, duration_s=time.perf_counter()-started,
        original_numerical_gate_waived=False, production_qualification_completed=False,
        no_proposal_published=True,
        rng_scope='outer isolated evaluation runtime; qualified dropout zero and fixed retained anchors',
        interpretation='one retained full-window functional sensitivity diagnostic, not serving or efficacy')
    write_json(case.root / 'CONSEQUENCES.json', result)
    # Shared cleanup restores the exact retained origin, including RNG and Adam.
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    run(args.plan, case_runner=diagnose_consequences, case_preparer=prepare_window_case,
        work_limits=BUDGET)
