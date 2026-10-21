"""Read and bind the original six-group evidence; never reconstruct missing data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from experiments.dflash_baseline_learning_curve.experiment import (
    digest, require, validate_groups, validate_measurement)

HERE = Path(__file__).resolve().parent


def file_binding(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def checked_file(path, expected):
    actual = file_binding(path)
    require(all(actual[k] == expected[k] for k in ('bytes', 'sha256')), 'Original evidence changed: ' + str(path))
    return actual


def _one(rows, predicate, description):
    matches = [r for r in rows if predicate(r)]
    require(len(matches) == 1, 'Missing/duplicate ' + description)
    return matches[0]


def source_loss(group, label, record):
    block = _one(group['training']['source_losses'], lambda r: r['checkpoint_id'] == label, 'source checkpoint loss')
    require(block['checkpoint_sha256'] == group['checkpoints'][label]['checkpoint_identity_sha256'],
            'Original source-loss checkpoint differs')
    require(block['training_rng_preserved'] is True and block['diagnostic_model_mode'] == 'eval_no_grad',
            'Original source diagnostic contract changed')
    row = copy.deepcopy(_one(block['response_losses'], lambda r: r['record_id'] == record['request_id'], 'source response loss'))
    row.update(request_id=record['request_id'], checkpoint_id=label,
        checkpoint_identity_sha256=block['checkpoint_sha256'], input_ids_sha256=digest(record['input_ids']),
        output_ids_sha256=digest(record['output_ids']), provenance='retained_source_loss',
        excluded_from_loss_improvement=bool(row['zero_supervision']))
    return row


def heldout_records(group, label='F'):
    require(label in ('F', 'S8'), 'Only F/S8 may be evaluated')
    return [copy.deepcopy(_one(group['scores'], lambda r: r['request_id'] == request['request_id'] and
            r['checkpoint_id'] == label, 'held-out native record')) for request in group['evaluation_requests']]


def load_evidence(root):
    """Allow a verified compact local mirror, preserving every original path binding."""
    root = Path(root).resolve()
    pins = json.loads((HERE / 'baseline_bindings.json').read_text())
    files = {name: checked_file(root / name, expected) for name, expected in pins['files'].items()}
    result, manifest, source, config = (json.loads((root / name).read_text()) for name in
        ('RESULT.json', 'SAMPLE_MANIFEST.json', 'SOURCE_MANIFEST.json', 'RUNTIME_CONFIG.json'))
    require(result['status'] == 'complete' and result['unexecuted_groups'] == [], 'Original campaign is incomplete')
    require(source['git']['commit'] == pins['baseline_commit'] and source['git']['dirty'] is False,
            'Original campaign revision differs')
    require(source['sample_manifest']['sha256'] == pins['files']['SAMPLE_MANIFEST.json']['sha256'],
            'Sealed sample binding differs')
    validate_groups(manifest['groups'])
    require([g['group_id'] for g in result['groups']] == [g['group_id'] for g in manifest['groups']],
            'Original group order differs from sealed manifest')
    traces = []
    from experiments.dflash_baseline_learning_curve.native import normalize_rounds
    for group, selected in zip(result['groups'], manifest['groups']):
        require(group['status'] == 'complete' and group['implementation_checks']['status'] == 'passed',
                'Original group did not qualify')
        require(len(group['feedback']) == 4 and len(group['scores']) == 12, 'Original group count differs')
        require(group['domain'] == selected['domain'] and
                [r['request_id'] for r in group['feedback']] == [r['request_id'] for r in selected['feedback']] and
                [r['request_id'] for r in group['evaluation_requests']] == [r['request_id'] for r in selected['evaluation']],
                'Original population, request identities, or order differ from manifest')
        for label in ('F', 'S8'):
            checkpoint = group['checkpoints'][label]
            require(checkpoint['checkpoint_identity_sha256'] == digest(checkpoint['files']), 'Checkpoint file identity differs')
            if label == 'S8':
                expected = Path(pins['original_root']) / 'native' / group['group_id'] / 'checkpoints/U2'
                require(Path(checkpoint['path']) == expected and checkpoint['checkpoint_id'] == 'U2',
                        'Wrong group S8 checkpoint')
        for role, rows in (('feedback', group['feedback']), ('evaluation', heldout_records(group) + heldout_records(group, 'S8'))):
            for row in rows:
                chosen = _one(selected[role], lambda r: r['request_id'] == row['request_id'], 'selected request')
                validate_measurement(row)
                require(row['input_ids'] == chosen['input_ids'], 'Saved prompt differs from sealed request')
                require(row['checkpoint_identity_sha256'] == group['checkpoints'][row['checkpoint_id']]['checkpoint_identity_sha256'],
                        'Native record uses another checkpoint')
                binding = row['trajectory']
                original = Path(binding['path'])
                own = Path(pins['original_root']) / 'native' / group['group_id']
                require(original.is_relative_to(own), 'Trace belongs to another group')
                local = root / original.relative_to(pins['original_root'])
                traces.append(checked_file(local, binding))
                document = json.loads(local.read_text())
                rounds, initial = normalize_rounds(document, row, row['input_ids'])
                require(rounds == row['rounds'] and initial == row['initial_output_tokens'], 'Saved trace/counts differ')
                require(document['max_new_tokens'] == 1024, 'Original native output limit differs')
        for row in group['feedback']:
            source_loss(group, 'F', row)
            source_loss(group, 'S8', row)
        for frozen, adapted in zip(heldout_records(group), heldout_records(group, 'S8')):
            require(frozen['input_ids'] == adapted['input_ids'] and frozen['output_ids'] == adapted['output_ids'] and
                    frozen['finish_reason'] == adapted['finish_reason'], 'Original held-out output/termination differs')
    return dict(groups=copy.deepcopy(result['groups']), manifest=manifest, runtime_config=config,
        bindings=dict(files=files, trajectories=traces, baseline_commit=pins['baseline_commit'],
                      compact_only=True, checkpoint_tensor_verification='required at native execution'))
