"""Prepare candidate configs and storage receipts without model execution."""
import json
from pathlib import Path
import sys

import yaml

code = Path('/data/sources/minhao/DFlash-Online')
sys.path.insert(0, str(code))
from experiments.osd_persistent.retention import admit_storage

root = Path('/data/sources/minhao/runs/dflash_online_osd_rerun_20261023')
prep = root / 'preparation'
receipts = {}
for policy in ('remaining', 'fixed2048'):
    manifest = json.loads((prep / f'CANDIDATE_{policy}.json').read_text())
    config = yaml.safe_load((code / 'configs/osd.yaml').read_text())
    length = manifest['policy']['train_context_limit']
    config['data']['max_length'] = length
    config['run_id'] = 'osd-persistent-20261023'
    config['output_dir'] = str(root / 'campaign_001/trainer')
    (prep / f'CANDIDATE_{policy}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    admission_root = prep / ('admission_' + policy)
    admission_root.mkdir(exist_ok=True)
    receipts[policy] = admit_storage(
        admission_root, config['model']['draft_checkpoint_path'],
        max_sequence_length=length, receipt_path=admission_root / 'STORAGE_ADMISSION.json')
print(json.dumps({key: {field: value[field] for field in (
    'admitted', 'max_sequence_length', 'free_bytes', 'required_additional_and_reserve_bytes')}
    for key, value in receipts.items()}, indent=2))
