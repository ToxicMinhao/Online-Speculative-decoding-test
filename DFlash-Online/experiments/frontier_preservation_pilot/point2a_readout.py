"""Point2A native endpoints using the unchanged PR61 assessment and readouts."""

import json
from pathlib import Path

import torch

from online_feedback.storage import write_json
from .math import require
from .point1_readout import (
    _evaluate_responses, compare_readouts, validate_disjoint, witness_responses,
)


CASES = ('C', 'F_BETA', 'F_C')
ENDPOINTS = ('parent', 'N_C', 'N_F_BETA', 'N_F_C')
CONTRASTS = (('F_C', 'F_BETA'), ('F_C', 'C'), ('F_BETA', 'C'))


def evaluate_responses(trainer, responses, *, endpoint, origins=None, parameters=None, work=None):
    """Validate Point2A names directly, then use PR61's isolated evaluator."""
    require(endpoint in ENDPOINTS, 'Unknown Point2A endpoint')
    return _evaluate_responses(
        trainer, responses, endpoint=endpoint, origins=origins,
        parameters=parameters, work=work, readout_name='point2a',
    )


def paired_contrasts(readouts):
    """Return every available planned contrast, with candidate-minus-reference labels."""
    result = {}
    for candidate, reference in CONTRASTS:
        first, second = 'N_' + reference, 'N_' + candidate
        if first in readouts and second in readouts:
            result[candidate + '-' + reference] = compare_readouts(readouts[first], readouts[second])
    return result


def endpoint_readouts(trainer, scorer, retained, contract, work, output):
    """Read parent plus three retained endpoints without changing restored PAIR1."""
    output = Path(output)
    require(set(retained['native']) == set(CASES), 'Point2A native cases differ')
    window_parent = evaluate_responses(trainer, scorer.responses, endpoint='parent', work=work)
    require(len(window_parent['responses']) == len(scorer.snapshots), 'Frozen response count changed')
    for row, snapshot in zip(window_parent['responses'], scorer.snapshots, strict=True):
        require([block['proposals'] for block in row['blocks']] == snapshot['proposals'][0].cpu().tolist(),
                'Parent readout differs from frozen scoring proposals')
        for key in ('labels', 'valid', 'prefix', 'repair', 'L', 'e', 'H', 'K', 'm'):
            require([block['origin'][key] for block in row['blocks']] == snapshot['roles'][key][0].cpu().tolist(),
                    'Parent readout differs from frozen scoring roles: ' + key)
    write_json(output / 'window_parent.json', window_parent)
    witness = contract['witness']
    witness_rows = witness['items'] if witness['status'] == 'authenticated' else []
    witnesses = witness_responses(trainer, witness_rows) if witness_rows else []
    results = {'window': {'parent': window_parent}, 'new_buffer': {}}
    if witnesses:
        require(len(witnesses) == 8, 'Point2A requires the eight authenticated TRAIN witnesses')
        historical = json.loads(Path(witness['parent_path']).read_text())
        witness_parent = evaluate_responses(
            trainer, witnesses, endpoint='parent', origins=historical['sets']['new_buffer'], work=work,
        )
        results['new_buffer']['parent'] = witness_parent
        write_json(output / 'witness_parent.json', witness_parent)
    device = scorer.parameters[0].device
    for name in CASES:
        require(len(scorer.parameter_names) == len(retained['native'][name]),
                'Point2A retained endpoint parameter count changed')
        parameters = {
            key: value.to(device=device, dtype=torch.bfloat16)
            for key, value in zip(scorer.parameter_names, retained['native'][name], strict=True)
        }
        endpoint = 'N_' + name
        result = evaluate_responses(
            trainer, scorer.responses, endpoint=endpoint, origins=window_parent,
            parameters=parameters, work=work,
        )
        results['window'][endpoint] = result
        write_json(output / ('window_' + endpoint + '.json'), result)
        if witnesses:
            result = evaluate_responses(
                trainer, witnesses, endpoint=endpoint, origins=results['new_buffer']['parent'],
                parameters=parameters, work=work,
            )
            results['new_buffer'][endpoint] = result
            write_json(output / ('witness_' + endpoint + '.json'), result)
        del parameters
    results['contrasts'] = {scope: paired_contrasts(results[scope]) for scope in ('window', 'new_buffer')}
    return results
