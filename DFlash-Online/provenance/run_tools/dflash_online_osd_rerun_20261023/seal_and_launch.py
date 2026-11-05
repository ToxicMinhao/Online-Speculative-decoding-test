"""Seal the user's selected length policy and start the authorized rerun once."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

root = Path('/data/sources/minhao/runs/dflash_online_osd_rerun_20261023')
code = Path('/data/sources/minhao/DFlash-Online')
prep = root / 'preparation'
sealed = root / 'sealed'
campaign = root / 'campaign_001'
assert not sealed.exists() and not campaign.exists() and not (root / 'LAUNCH.json').exists()
source = {}
for receipt in ('BASE_SOURCE.json', 'SOURCE_ADDITIONS.json'):
    for relative, expected in json.loads((prep / receipt).read_text()).items():
        actual = hashlib.sha256((code / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Installed source differs from tested preparation: ' + relative)
        source[relative] = actual

sys.path.insert(0, str(code))
from experiments.osd_persistent.data import validate_manifest
from experiments.osd_persistent.retention import admit_storage

manifest = json.loads((prep / 'CANDIDATE_fixed2048.json').read_text())
validate_manifest(manifest)
config = yaml.safe_load((prep / 'CANDIDATE_fixed2048.yaml').read_text())
assert config['data']['max_length'] == manifest['policy']['train_context_limit'] == 2272
assert all(row['max_new_tokens'] == 2048 for row in manifest['train_requests'] + manifest['eval_requests'])
assert manifest['policy']['serving_context_limit'] == 4096
assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'Preserve any scheduler GPU assignment'
gpu_processes = subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert not gpu_processes, 'Both planned GPUs must be idle'

sealed.mkdir()
shutil.copy2(prep / 'CANDIDATE_fixed2048.json', sealed / 'MANIFEST.json')
shutil.copy2(prep / 'CANDIDATE_fixed2048.yaml', sealed / 'config.yaml')
admission = admit_storage(sealed, config['model']['draft_checkpoint_path'],
                          max_sequence_length=2272,
                          receipt_path=sealed / 'STORAGE_ADMISSION.json')
versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'safetensors')}
command = ['bash', str(root / 'run_campaign.sh'), str(sealed / 'MANIFEST.json'),
           str(sealed / 'config.yaml'), str(campaign)]
seal = {
    'status': 'sealed', 'selected_by_user': '2048 outputs or EOS; training sequence up to 2272',
    'sealed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'manifest_sha256': hashlib.sha256((sealed / 'MANIFEST.json').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((sealed / 'config.yaml').read_bytes()).hexdigest(),
    'source_sha256': source, 'trainer_versions': versions,
    'trainer_python': sys.executable, 'trainer_gpu': 1, 'serving_gpu': 0,
    'command': command, 'timeout_hours': 48,
    'storage_required_bytes': admission['required_additional_and_reserve_bytes'],
    'free_bytes_at_admission': admission['free_bytes'],
    'qualification_gate': 'First full 128-response event; no efficacy gate',
}
(sealed / 'SEAL.json').write_text(json.dumps(seal, indent=2) + '\n')

wrapper = root / 'execute_sealed.sh'
wrapper.write_text('''#!/usr/bin/env bash
set -u
task_root=/data/sources/minhao/runs/dflash_online_osd_rerun_20261023
timeout --signal=TERM --kill-after=120s 172800 bash "$task_root/run_campaign.sh" \\
  "$task_root/sealed/MANIFEST.json" "$task_root/sealed/config.yaml" "$task_root/campaign_001"
run_exit=$?
printf '%s\\n' "$run_exit" > "$task_root/EXIT_CODE"
date -u +%Y-%m-%dT%H:%M:%SZ > "$task_root/FINISHED_UTC"
exit "$run_exit"
''')
with (root / 'campaign.log').open('x') as log:
    process = subprocess.Popen(['bash', str(wrapper)], cwd=code, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
launch = {'pid': process.pid, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'campaign': str(campaign), 'log': str(root / 'campaign.log'),
          'command': command, 'source_seal': str(sealed / 'SEAL.json')}
(root / 'LAUNCH.json').write_text(json.dumps(launch, indent=2) + '\n')
print(json.dumps(launch, indent=2))
