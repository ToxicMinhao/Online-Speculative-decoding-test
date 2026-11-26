"""Bounded Point1 readouts with frozen origins and native BF16 head shapes.

These draft-only evaluations never sample anchors, train, or publish a model.
The caller authenticates packets and chooses the declared endpoint. Readouts
retain private block records; only compact reductions belong in public reports.
"""

from copy import deepcopy
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.functional_step_geometry.qualify import runtime_state, restore_runtime
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from .allocation_witness import block_readouts, METRICS, LOWER_BETTER
from .math import require
from .native import FrozenResponse, NativeHidden, rng_digest, trainable_parameters


ENDPOINTS = ('parent', 'Q_BETA', 'Q_C', 'Q_F', 'N_BETA', 'N_C', 'N_F')


def validate_disjoint(qualification_rows, witness_rows):
    """Check both request identities and resolved packet paths, not one alone."""
    def identities(rows):
        identifiers = [str(row['record_id']) for row in rows]
        paths = [str(Path(row['feature_path']).resolve()) for row in rows]
        require(len(set(identifiers)) == len(rows), 'Duplicate readout response identity')
        require(len(set(paths)) == len(rows), 'Duplicate readout feature path')
        return set(identifiers), set(paths)

    old_ids, old_paths = identities(qualification_rows)
    new_ids, new_paths = identities(witness_rows)
    require(not old_ids & new_ids, 'Witness response identities overlap qualification')
    require(not old_paths & new_paths, 'Witness feature paths overlap qualification')
    return dict(record_ids_disjoint=True, feature_paths_disjoint=True)


def witness_responses(trainer, binding_rows):
    """Load the already authenticated new-buffer rows without drawing anchors."""
    model = trainer.bundle.model
    device = next(model.draft_model.parameters()).device
    frozen = []
    state = capture_rng()
    for row in binding_rows:
        raw = torch.load(row['feature_path'], map_location='cpu', weights_only=True)
        require(str(raw['record_id']) == str(row['record_id']), 'Witness packet ID changed')
        batch = trainer._batch(raw)
        tensors = {
            key: batch.tensors[key].to(device)
            for key in ('input_ids', 'hidden_states', 'loss_mask')
        }
        anchors = torch.tensor([row['anchors']], device=device, dtype=torch.long)
        require(anchors.shape[1] <= 8, 'Witness exceeds retained eight-anchor limit')
        require(len(set(row['anchors'])) == len(row['anchors']), 'Duplicate witness anchor')
        frozen.append(FrozenResponse(
            str(row['record_id']), tensors, anchors,
            torch.ones_like(anchors, dtype=torch.bool), state, state,
        ))
    require(rng_digest(capture_rng()) == rng_digest(state), 'Witness loading consumed RNG')
    return frozen


def _boundary(trainer):
    """Point1 alone admits disposable step129; old production guards stay intact."""
    counters = (trainer.global_step, trainer.presentations, len(trainer.events))
    require(counters in ((128, 256, 1), (129, 258, 1)), 'Unexpected Point1 boundary')
    require(not trainer.core.accumulation_remainder, 'Readout at an accumulation boundary only')
    return counters


def _chunk_assessment(blocks, device):
    """Reproduce the scorer's FP32 selected-position reduction for one head chunk.

    The individual terms use its FP32 margin, softplus, product and division.
    Their represented values are also retained for matched FP64 subtraction and
    reduction. This does not replace the original FP32 assessment with FP64 math.
    """
    positions = [position for block in blocks for position in block['positions']]
    margins = torch.tensor([p['margin'] for p in positions], device=device, dtype=torch.float32)
    weights = torch.tensor(
        [block['origin']['K'] - p['position'] + 1
         for block in blocks for p in block['positions']],
        device=device, dtype=torch.float32,
    )
    contributions = weights * F.softplus(-margins) / math.log(2.0)
    require(torch.isfinite(contributions).all(), 'Nonfinite Point1 assessment contribution')
    values = contributions.cpu().tolist()
    offset = 0
    for block in blocks:
        count = len(block['positions'])
        for position, value in zip(block['positions'], values[offset:offset + count]):
            position['assessment_contribution_fp32'] = value
        block['assessment_sum_fp64'] = math.fsum(values[offset:offset + count])
        offset += count
    return contributions.sum()


def _origins_rows(origins):
    return origins['responses'] if isinstance(origins, dict) else origins


def evaluate_responses(trainer, responses, *, endpoint, origins=None, parameters=None, work=None):
    """Point1's public endpoint contract remains restricted to Point1 labels."""
    require(endpoint in ENDPOINTS, 'Unknown Point1 endpoint')
    return _evaluate_responses(
        trainer, responses, endpoint=endpoint, origins=origins,
        parameters=parameters, work=work, readout_name='point1',
    )


@torch.no_grad()
def _evaluate_responses(trainer, responses, *, endpoint, origins=None, parameters=None,
                        work=None, readout_name, boundary_check=_boundary,
                        chunk_assessment=_chunk_assessment):
    """Read a parent or alternative endpoint, retaining native eight-block heads.

    Functional dictionaries use NativeHidden parameter names. FP32 shadow leaves
    are cast to each native parameter's BF16 dtype before functional execution.
    No counter is forged when a functional endpoint is evaluated at the parent.
    Existing modes, gradients, observer buffers, costs and RNG are preserved.
    """
    before = boundary_check(trainer)
    model = trainer.bundle.model
    require(model.block_size == 16 and model.objective_chunk_blocks == 8,
            'Point1 requires native sixteen-slot/eight-block shapes')
    require(model.loss_dtype == torch.bfloat16, 'Point1 head must stay native BF16')
    hidden_model = NativeHidden(model)
    named = trainable_parameters(hidden_model)
    device = next(model.draft_model.parameters()).device
    represented = None
    if parameters is not None:
        require(set(parameters) == set(named), 'Functional endpoint parameter names changed')
        represented = {
            name: parameters[name].to(device=device, dtype=parameter.dtype)
            for name, parameter in named.items()
        }
        require(all(value.dtype == torch.bfloat16 and torch.isfinite(value).all()
                    for value in represented.values()), 'Invalid represented BF16 endpoint')
    frozen_origins = None if origins is None else _origins_rows(origins)
    if frozen_origins is not None:
        require(len(frozen_origins) == len(responses), 'Readout response count changed')
    runtime = runtime_state(model)
    versions = [(parameter, parameter._version) for parameter in model.parameters()]
    gradients = [(parameter, parameter.grad,
                  None if parameter.grad is None else parameter.grad._version)
                 for parameter in model.parameters()]
    saved_work, saved_costs = trainer.work, model.costs
    observations = deepcopy(trainer._anchor_observations)
    history = deepcopy(trainer.events)
    rng_before = rng_digest(capture_rng())
    trainer.work, model.costs = deepcopy(saved_work), {}
    result = dict(
        schema=readout_name + '_native_endpoint_readout_v1', endpoint=endpoint,
        functional_endpoint=parameters is not None, live_counters=list(before),
        responses=[], backwards=0, optimizer_steps=0, teacher_forwards=0,
        head_calls=0, draft_forwards=0,
        units='overlapping frozen blocks from TRAIN responses; not held-out serving',
    )
    try:
        with preserve_rng():
            model.eval()
            response_values = []
            for index, response in enumerate(responses):
                first = None if frozen_origins is None else frozen_origins[index]
                anchors = response.anchors[0, response.keep[0]].cpu().tolist()
                if first is not None:
                    require(str(first['record_id']) == response.record_id,
                            'Frozen readout response identity changed')
                    require(first['anchors'] == anchors, 'Frozen readout anchors changed')
                blocks, numerator = [], None
                if response.anchors.shape[1]:
                    restore_rng(response.rng_after_sampling)
                    if work is not None:
                        work.start(readout_name + '_readout_backbone_forward')
                    hidden = (hidden_model(response) if represented is None
                              else torch.func.functional_call(hidden_model, represented, (response,)))
                    if work is not None:
                        work.finish(readout_name + '_readout_backbone_forward')
                    result['draft_forwards'] += 1
                    labels, weights, _ = model._block_labels_and_weights(
                        response.tensors['input_ids'], response.tensors['loss_mask'],
                        response.anchors, response.keep,
                    )
                    hidden = hidden.reshape(1, response.anchors.shape[1], 16, -1)
                    for start in range(0, hidden.shape[1], 8):
                        part = slice(start, start + 8)
                        if work is not None:
                            work.start(readout_name + '_readout_head_chunk')
                        logits = model.lm_head(hidden[:, part]).to(model.loss_dtype)
                        if work is not None:
                            work.finish(readout_name + '_readout_head_chunk')
                        result['head_calls'] += 1
                        chunk = block_readouts(
                            logits, labels[:, part], weights[:, part], model.eos_token_ids,
                            origin=None if first is None else first['blocks'][part],
                        )
                        chunk_numerator = chunk_assessment(chunk, device)
                        numerator = chunk_numerator if numerator is None else numerator + chunk_numerator
                        blocks.extend(chunk)
                        del logits
                    del hidden
                if first is not None:
                    require(len(blocks) == len(first['blocks']), 'Frozen readout block count changed')
                nonempty = sum(block['origin']['m'] > 0 for block in blocks)
                if numerator is None:
                    numerator = torch.zeros((), device=device, dtype=torch.float32)
                value = numerator / nonempty if nonempty else numerator
                response_values.append(value)
                result['responses'].append(dict(
                    record_id=response.record_id, anchors=anchors, blocks=blocks,
                    nonempty_blocks=nonempty, response_assessment_fp32=float(value),
                    response_assessment_fp64=(
                        math.fsum(block['assessment_sum_fp64'] for block in blocks) / nonempty
                        if nonempty else 0.0
                    ),
                    existing_block_reduced_assessment=(
                        sum(block['metrics']['assessment'] for block in blocks) / nonempty
                        if nonempty else 0.0
                    ),
                ))
            count = len(response_values)
            require(count > 0, 'Point1 readout needs a declared response set')
            result['mean_response_assessment_fp32'] = float(sum(response_values) / count)
            result['mean_response_assessment'] = sum(float(value) for value in response_values) / count
            result['mean_response_assessment_fp64'] = math.fsum(
                row['response_assessment_fp64'] for row in result['responses']) / count
            result['physical_work'] = {
                key: value - saved_work.get(key, 0) for key, value in trainer.work.items()
            }
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        require(boundary_check(trainer) == before and trainer.events == history,
                'Readout changed counters or event history')
        require(all(parameter._version == version for parameter, version in versions),
                'Readout changed native parameters')
        require(all(parameter.grad is gradient and
                    (gradient is None or gradient._version == version)
                    for parameter, gradient, version in gradients), 'Readout changed training gradients')
        require(rng_digest(capture_rng()) == rng_before, 'Readout changed RNG')
        result.update(state_preserved=True, status='passed')
    finally:
        restore_runtime(model, runtime)
        trainer.work, model.costs = saved_work, saved_costs
        trainer._anchor_observations[:] = observations
    return result


def compare_readouts(first, second):
    """Matched FP64 benefit and discrete changes; no independent-anchor inference."""
    require(len(first['responses']) == len(second['responses']), 'Response counts differ')
    response_rows, block_rows, position_terms = [], [], []
    for ordinal, (old, new) in enumerate(zip(first['responses'], second['responses'])):
        require(old['record_id'] == new['record_id'] and old['anchors'] == new['anchors'],
                'Paired readout identities or anchors differ')
        require(len(old['blocks']) == len(new['blocks']), 'Paired block counts differ')
        nonempty = old['nonempty_blocks']
        require(nonempty == new['nonempty_blocks'], 'Frozen response denominator changed')
        paired = []
        response_metric_changes = {key: [] for key in METRICS}
        for block_index, (a, b) in enumerate(zip(old['blocks'], new['blocks'])):
            require(a['origin'] == b['origin'], 'Paired origins differ')
            require(len(a['positions']) == len(b['positions']), 'Assessment support changed')
            terms = []
            for left, right in zip(a['positions'], b['positions']):
                require(left['position'] == right['position'] and left['canonical'] == right['canonical'],
                        'Matched assessment position changed')
                terms.append(left['assessment_contribution_fp32'] - right['assessment_contribution_fp32'])
            benefit = math.fsum(terms)
            for metric in METRICS:
                response_metric_changes[metric].append(b['metrics'][metric] - a['metrics'][metric])
            block_rows.append(dict(response_ordinal=ordinal, block_index=block_index,
                                   assessment_benefit_fp64=benefit,
                                   metric_changes={key: b['metrics'][key] - a['metrics'][key]
                                                   for key in METRICS}))
            paired.extend(terms)
        response_benefit = math.fsum(paired) / nonempty if nonempty else 0.0
        response_rows.append(dict(
            response_ordinal=ordinal, nonempty_blocks=nonempty,
            assessment_benefit_fp64=response_benefit,
            metric_changes={key: math.fsum(values) / (nonempty or 1) if key == 'assessment'
                            else math.fsum(values)
                            for key, values in response_metric_changes.items()},
        ))
        if nonempty:
            position_terms.extend(value / nonempty / len(first['responses']) for value in paired)
    deltas = [row['metric_changes'] for row in block_rows]
    comparisons, response_comparisons = {}, {}
    for metric in METRICS:
        sign = -1 if metric in LOWER_BETTER else 1
        values = [row[metric] for row in deltas]
        comparisons[metric] = dict(delta=math.fsum(values),
                                   gains=sum(sign * value > 0 for value in values),
                                   losses=sum(sign * value < 0 for value in values),
                                   ties=sum(value == 0 for value in values))
        response_changes = [row['metric_changes'][metric] for row in response_rows]
        response_comparisons[metric] = dict(
            delta=math.fsum(response_changes),
            gains=sum(sign * value > 0 for value in response_changes),
            losses=sum(sign * value < 0 for value in response_changes),
            ties=sum(value == 0 for value in response_changes),
        )
    return dict(
        first=first['endpoint'], second=second['endpoint'],
        assessment_benefit_fp64=math.fsum(row['assessment_benefit_fp64'] for row in response_rows)
                                / len(response_rows),
        assessment_benefit_original_fp32=(first['mean_response_assessment_fp32']
                                         - second['mean_response_assessment_fp32']),
        positive_contribution_sum_fp64=math.fsum(value for value in position_terms if value > 0),
        negative_contribution_sum_fp64=math.fsum(value for value in position_terms if value < 0),
        positive_contributions=sum(value > 0 for value in position_terms),
        negative_contributions=sum(value < 0 for value in position_terms),
        zero_contributions=sum(value == 0 for value in position_terms),
        assessed_occurrences=len(position_terms), responses=response_rows, blocks=block_rows,
        block_metrics=comparisons, response_metrics=response_comparisons,
        difference_convention='benefit = first minus second assessment; discrete changes = second minus first',
    )
