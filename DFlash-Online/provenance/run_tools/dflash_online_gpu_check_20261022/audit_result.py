"""Read-only audit of the completed qualification; no model generation/training."""
from pathlib import Path
import json
import torch
from safetensors import safe_open

ROOT = Path('/data/sources/minhao/runs/dflash_online_foundation_20261022/refactor_qualification_001')
OPS = Path('/data/sources/minhao/runs/dflash_online_gpu_check_20261022')
RELEASED = Path('/data/sources/minhao/models/qwen3_dflash_foundation_20261022/draft/model.safetensors')
result = json.loads((ROOT / 'RESULT.json').read_text())
assert result['status'] == 'qualified'
assert len(result['serving']) == 4 and len(result['events']) == 2 and len(result['replays']) == 3
assert all(row['all_equal'] for row in result['publication_checks'])
assert all(result['restart_check'].values())

audit = {'status': 'passed', 'scope': 'read-only independent result/weight audit',
         'generation_count': 7, 'optimizer_steps': 4, 'presentations': 8,
         'event_exposure': [], 'weight_changes': [], 'heldout': []}
previous = RELEASED
for event in result['events']:
    training = event['training']
    assert training['optimizer_steps'] == 2 and training['response_presentations'] == 4
    assert [row['optimizer_stepped'] for row in training['records']] == [False, True, False, True]
    assert all(not record['label_mismatches'] for record in event['preparation'])
    for feature in training['features']:
        data = torch.load(feature['path'], map_location='cpu', weights_only=True)
        prompt = data['prompt_length']
        assert torch.all(data['loss_mask'][:, :prompt] == 0)
        assert torch.all(data['loss_mask'][:, prompt:] == 1)
    audit['event_exposure'].append({'event': event['event_id'],
                                   'records': [row['record_id'] for row in training['records']],
                                   'passes': [row['pass'] for row in training['records']],
                                   'optimizer_steps': training['optimizer_steps']})
    updated = Path(training['checkpoint_path']) / 'model.safetensors'
    changed = 0
    coordinates = 0
    with safe_open(previous, framework='pt', device='cpu') as before, safe_open(updated, framework='pt', device='cpu') as after:
        assert set(before.keys()) == set(after.keys())
        for name in before.keys():
            a, b = before.get_tensor(name), after.get_tensor(name)
            different = int(torch.count_nonzero(a != b))
            changed += different > 0
            coordinates += different
    assert changed > 0
    audit['weight_changes'].append({'event': event['event_id'], 'changed_tensors': changed,
                                    'changed_coordinates': coordinates, 'reference': str(previous)})
    previous = updated

reference = result['replays'][0]['record']
for replay in result['replays']:
    row = replay['record']
    assert row['prompt_ids'] == reference['prompt_ids']
    assert row['response_ids'] == reference['response_ids']
    assert replay['output_equal']
    if 'learning_unchanged' in replay:
        assert replay['learning_unchanged']
    audit['heldout'].append({'label': replay['label'], 'output_tokens': len(row['response_ids']),
                             'calls': row['metadata']['verification_calls'],
                             'accepted_draft_tokens': row['metadata']['accepted_draft_tokens']})
assert audit['heldout'][0]['calls'] == audit['heldout'][2]['calls']
assert result['trainer_status']['global_step'] == 4
assert result['trainer_status']['presentations'] == 8
(OPS / 'AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
