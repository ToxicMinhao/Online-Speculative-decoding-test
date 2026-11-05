"""Retire only the completed foundation qualification's bulky restart payload."""
from pathlib import Path
import gc
import json
import shutil
import torch
from safetensors import safe_open

ROOT = Path('/data/sources/minhao/runs/projected_target_feasibility_20261022')
OLD = Path('/data/sources/minhao/runs/dflash_online_foundation_20261022/refactor_qualification_001')
ROOT.mkdir(parents=True, exist_ok=True)
state_path = OLD / 'trainer/training_state.pt'
result = json.loads((OLD / 'RESULT.json').read_text())
assert result['status'] == 'qualified', result['status']
state = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
assert state['global_step'] == 4 and state['presentations'] == 8
export = OLD / 'trainer/events/event_2/export/model.safetensors'
with safe_open(export, framework='pt', device='cpu') as file:
    assert set(file.keys()) == set(state['draft_state_dict'])
    for name, tensor in state['draft_state_dict'].items():
        assert torch.equal(tensor, file.get_tensor(name)), name
compact = {key: value for key, value in state.items() if key not in {'draft_state_dict', 'backend'}}
compact.update(retained_export=str(export), lost_capability='Exact Adam/master/scheduler continuation from this completed run; model weights and all compact results retained.')
compact_path = ROOT / 'retired_foundation_state_metadata.pt'
torch.save(compact, compact_path)
verified = torch.load(compact_path, map_location='cpu', weights_only=False)
assert verified['global_step'] == 4 and verified['events'] == compact['events']
receipt = {'path': str(state_path), 'bytes': state_path.stat().st_size,
           'free_before': shutil.disk_usage(ROOT).free, 'retained_export': str(export),
           'all_export_tensors_equal': len(state['draft_state_dict']),
           'retained_metadata': str(compact_path), 'reason': 'Completed and audited infrastructure qualification; storage admitted for new independent event.',
           'lost_capability': compact['lost_capability']}
(ROOT / 'STORAGE_RETIREMENT_PREPARED.json').write_text(json.dumps(receipt, indent=2))
del state, compact, verified, tensor
gc.collect()
state_path.unlink()
receipt.update(free_after=shutil.disk_usage(ROOT).free, deleted=True)
(ROOT / 'STORAGE_RETIREMENT.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
