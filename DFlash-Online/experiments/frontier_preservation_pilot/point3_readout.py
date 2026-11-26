"""Both fixed-K assessments and native agreement from one endpoint forward."""

import math

import torch

from .math import require
from .point1_readout import (
    _chunk_assessment, _evaluate_responses, compare_readouts, witness_responses,
)
from .point3_analysis import CONTRASTS
from .point3_math import max_terms_from_margins

ENDPOINTS = ('parent', 'C', 'F_C_SUM', 'F_C_MAX', 'N_C', 'N_F_C_SUM', 'N_F_C_MAX',
             'Q_C_SUM', 'Q_C_MAX')


def _boundary(trainer):
    """Admit this warm continuation without broadening historical readout gates."""
    counters = (trainer.global_step, trainer.presentations, len(trainer.events))
    step, presentations, events = counters
    require(128 <= step <= 256 and presentations == 2 * step,
            'Point3 readout is outside the authorized continuation')
    require(events == 1 or (step == 256 and events == 2), 'Point3 event boundary changed')
    require(not trainer.core.accumulation_remainder, 'Point3 readout requires an accumulation boundary')
    return counters


def _dual_chunk(blocks, device):
    """Keep historical SUM reduction and select MAX from the identical margins."""
    sum_numerator = _chunk_assessment(blocks, device)
    margins = torch.zeros((len(blocks), 16), dtype=torch.float32, device=device)
    mask = torch.zeros_like(margins, dtype=torch.bool)
    for index, block in enumerate(blocks):
        positions = block['positions']
        if positions:
            columns = [row['position'] for row in positions]
            margins[index, columns] = torch.tensor(
                [row['margin'] for row in positions], device=device, dtype=torch.float32,
            )
            mask[index, columns] = True
    terms, branches = max_terms_from_margins(margins, mask)
    max_numerator = terms[mask].sum()
    values = terms.cpu().tolist()
    bottlenecks = branches['bottleneck_indices'].cpu().tolist()
    multiplicities = branches['bottleneck_multiplicity'].cpu().tolist()
    ties = branches['tied_bottleneck_prefixes'].cpu().tolist()
    for index, block in enumerate(blocks):
        for position in block['positions']:
            j = position['position']
            position.update(
                max_assessment_contribution_fp32=values[index][j],
                bottleneck_position=bottlenecks[index][j],
                bottleneck_multiplicity=multiplicities[index][j],
                tied_bottleneck_prefix=ties[index][j],
            )
        block['max_assessment_sum_fp64'] = math.fsum(
            row['max_assessment_contribution_fp32'] for row in block['positions']
        )
        block['active_bottlenecks_by_role'] = {
            role: dict(
                distinct_positions=sum(multiplicities[index][j] > 0 for j in columns),
                multiplicity=sum(multiplicities[index][j] for j in columns),
            )
            for role, columns in (
                ('P', [j for j, flag in enumerate(block['origin']['prefix']) if flag]),
                ('E', [j for j, flag in enumerate(block['origin']['repair']) if flag]),
                ('C', [row['position'] for row in block['positions']
                       if not block['origin']['prefix'][row['position']]
                       and not block['origin']['repair'][row['position']]]),
            )
        }
    return sum_numerator, max_numerator


def evaluate_responses(trainer, responses, *, endpoint, origins=None, parameters=None, work=None):
    """Reuse isolated native forwards, carrying both assessment reductions.

    ``origins`` is a parent readout (or its response list), not scorer tensors.
    ``parameters`` optionally maps NativeHidden names to retained endpoint values.
    Witness sets remain TRAIN data and never supply a serving claim.
    """
    require(endpoint in ENDPOINTS, 'Unknown Point3 readout endpoint')
    max_chunks = []

    def chunk_assessment(blocks, device):
        sum_numerator, max_numerator = _dual_chunk(blocks, device)
        max_chunks.append(max_numerator)
        return sum_numerator

    result = _evaluate_responses(
        trainer, responses, endpoint=endpoint, origins=origins, parameters=parameters,
        work=work, readout_name='point3', boundary_check=_boundary,
        chunk_assessment=chunk_assessment,
    )
    cursor, max_responses = 0, []
    device = next(trainer.bundle.model.draft_model.parameters()).device
    for response in result['responses']:
        chunks = (len(response['blocks']) + 7) // 8
        numerator = torch.zeros((), device=device, dtype=torch.float32)
        for value in max_chunks[cursor:cursor + chunks]:
            numerator = numerator + value
        cursor += chunks
        count = response['nonempty_blocks']
        value = numerator / count if count else numerator
        max_responses.append(value)
        response['assessments'] = dict(
            SUM=dict(fp32=response['response_assessment_fp32'], fp64=response['response_assessment_fp64']),
            MAX=dict(fp32=float(value), fp64=(math.fsum(
                block['max_assessment_sum_fp64'] for block in response['blocks']) / count if count else 0.0)),
        )
    require(cursor == len(max_chunks), 'Point3 assessment chunks lost a response binding')
    result['assessments'] = dict(
        SUM=dict(fp32=result['mean_response_assessment_fp32'], fp64=result['mean_response_assessment_fp64']),
        MAX=dict(fp32=float(sum(max_responses) / len(max_responses)), fp64=math.fsum(
            row['assessments']['MAX']['fp64'] for row in result['responses']) / len(result['responses'])),
    )
    result['assessment_comparison_scope'] = 'Within-assessment endpoint changes only; MAX<=SUM is structural'
    return result


def compare_endpoints(reference, candidate):
    """Preserve native discrete metrics; subtract matched prefix terms in FP64."""
    result = compare_readouts(reference, candidate)
    response_benefits, blocks = [], []
    for ordinal, (old, new) in enumerate(zip(reference['responses'], candidate['responses'], strict=True)):
        terms = []
        for index, (first, second) in enumerate(zip(old['blocks'], new['blocks'], strict=True)):
            differences = [a['max_assessment_contribution_fp32'] - b['max_assessment_contribution_fp32']
                           for a, b in zip(first['positions'], second['positions'], strict=True)]
            terms.extend(differences)
            blocks.append(dict(response_ordinal=ordinal, block_index=index,
                               max_assessment_benefit_fp64=math.fsum(differences)))
        count = old['nonempty_blocks']
        response_benefits.append(math.fsum(terms) / count if count else 0.0)
    result['assessments'] = dict(
        SUM=dict(benefit_fp32=result['assessment_benefit_original_fp32'],
                 benefit_fp64=result['assessment_benefit_fp64']),
        MAX=dict(benefit_fp32=reference['assessments']['MAX']['fp32'] - candidate['assessments']['MAX']['fp32'],
                 benefit_fp64=math.fsum(response_benefits) / len(response_benefits),
                 response_benefits_fp64=response_benefits, blocks=blocks),
    )
    return result


def paired_contrasts(readouts):
    return {candidate + '_vs_' + reference: compare_endpoints(readouts[reference], readouts[candidate])
            for candidate, reference in CONTRASTS if reference in readouts and candidate in readouts}


def evaluate_witness_sets(trainer, bindings, parent_origins, *, endpoint, work=None,
                          parameters=None):
    """Read both retained eight-by-eight TRAIN sets with common PAIR1 origins.

    Preparation authenticates these two JSON inputs and feature hashes. This
    function only checks their semantic bindings and reuses each retained anchor.
    """
    names = ('new_buffer', 'parent_training')
    require(set(bindings['sets']) == set(parent_origins['sets']) == set(names),
            'Point3 requires both retained TRAIN witness sets')
    result = dict(endpoint=endpoint, sets={},
                  scope='TRAIN occurrences; new_buffer is trained during this event',
                  teacher_forwards=0, backwards=0, optimizer_steps=0)
    identifiers, paths = set(), set()
    for name in names:
        rows, origins = bindings['sets'][name], parent_origins['sets'][name]
        require(len(rows) == len(origins) == 8, 'Point3 requires eight responses per witness set')
        for row, origin in zip(rows, origins, strict=True):
            require(row['record_id'] == origin['record_id'] and row['anchors'] == origin['anchors'],
                    'Point3 witness response or anchor binding changed')
            require(len(row['anchors']) <= 8, 'Point3 witness anchor limit changed')
            require(row['record_id'] not in identifiers and row['feature_path'] not in paths,
                    'Point3 witness sets overlap')
            identifiers.add(row['record_id'])
            paths.add(row['feature_path'])
        responses = witness_responses(trainer, rows)
        result['sets'][name] = evaluate_responses(
            trainer, responses, endpoint=endpoint, origins=origins,
            parameters=parameters, work=work,
        )
    result['draft_forwards'] = sum(row['draft_forwards'] for row in result['sets'].values())
    result['head_calls'] = sum(row['head_calls'] for row in result['sets'].values())
    result['status'] = 'passed'
    return result
