"""Bind PR9 training and completed PR10 comparisons without replacing evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from experiments.dflash_baseline_learning_curve.experiment import require, digest
from experiments.dflash_kl_native_diagnostic.evidence import (
    load_evidence as load_baseline, checked_file, file_binding, source_loss, heldout_records)
from experiments.dflash_kl_native_diagnostic.experiment import validate_pair

HERE = Path(__file__).resolve().parent


def load_evidence(baseline_root, diagnostic_root):
    bundle = load_baseline(baseline_root)
    root = Path(diagnostic_root).resolve()
    pins = json.loads((HERE / 'reference_bindings.json').read_text())
    files = {name: checked_file(root / name, pin) for name, pin in pins['files'].items()}
    result = json.loads((root / 'RESULT.json').read_text())
    source = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    config = json.loads((root / 'RUNTIME_CONFIG.json').read_text())
    require(result['status'] == 'complete' and result['unexecuted_groups'] == [] and
        result['optimizer_steps'] == result['explicit_warmups'] == 0 and
        result['actual_source_generations'] == 24 and result['actual_response_kl_evaluations'] == 48,
        'Completed bounded PR10 evidence required')
    require(source['git']['commit'] == pins['diagnostic_commit'] and not source['git']['dirty'],
        'PR10 revision differs')
    require([g['group_id'] for g in result['groups']] == [g['group_id'] for g in bundle['groups']],
        'Diagnostic group identities/order differ')
    for key, value in bundle['runtime_config'].items():
        if key != 'output_root':
            require(config[key] == value, 'PR9/PR10 runtime configuration differs: ' + key)
    from experiments.dflash_baseline_learning_curve.native import normalize_rounds
    traces = []
    for group, prior in zip(bundle['groups'], result['groups']):
        require(prior['status'] == 'complete' and prior['domain'] == group['domain'],
            'Diagnostic group/domain incomplete or changed')
        group['reference_pairs'] = {}
        for population in ('source', 'heldout'):
            originals = group['feedback'] if population == 'source' else heldout_records(group)
            pairs = prior[population]
            require(len(pairs) == 4 and [p['request_id'] for p in pairs] == [r['request_id'] for r in originals],
                'Original response population/order differs')
            for i, (pair, original) in enumerate(zip(pairs, originals)):
                require(pair['validation']['status'] == 'passed' and validate_pair(pair)['status'] == 'passed',
                    'Invalid retained F/S8 pair')
                require(pair['group_id'] == group['group_id'] and pair['domain'] == group['domain'] and
                    pair['population'] == population and pair['F']['native'] == original,
                    'Retained F input/response/identity differs')
                for label in ('F', 'S8'):
                    require(pair['checkpoint_bindings'][label] == group['checkpoints'][label]['checkpoint_identity_sha256'],
                        'Reference uses another group checkpoint')
                if population == 'source':
                    require(all(pair[label]['kl'] == source_loss(group, label, original) for label in ('F', 'S8')),
                        'PR9 fixed source KL changed')
                    row = pair['S8']['native']
                    path = Path(row['trajectory']['path'])
                    require(path.is_relative_to(Path(pins['original_root']) / 'native' / group['group_id']),
                        'PR10 source trace belongs to another group')
                    local = root / path.relative_to(pins['original_root'])
                    traces.append(checked_file(local, row['trajectory']))
                    document = json.loads(local.read_text())
                    rounds, initial = normalize_rounds(document, row, row['input_ids'])
                    require(rounds == row['rounds'] and initial == row['initial_output_tokens'] and
                        document['max_new_tokens'] == 1024, 'PR10 native trace/limits differ')
                else:
                    require(pair['S8']['native'] == heldout_records(group, 'S8')[i], 'PR9 S8 held-out record differs')
            group['reference_pairs'][population] = copy.deepcopy(pairs)
        training = group['training']
        batches = [batch for update in training['updates'] for batch in update['accounting']['batches']]
        require(len(batches) == 8 and [p['record_id'] for b in batches for p in b['presentations']] ==
            [r['record_id'] for r in group['support']] * 4, 'Original training presentation order/count changed')
    bundle['runtime_config'] = config
    bundle['bindings']['diagnostic'] = dict(files=files, trajectories=traces,
        diagnostic_commit=pins['diagnostic_commit'], complete_groups=6, pairs=48)
    return bundle
