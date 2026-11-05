"""CPU provenance/storage gate; never changes the retained ordinary run."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.osd_one_event.protocol import validate_manifest
from experiments.prefix_ranking.run import admit_storage
from online_feedback.qualify_sglang import check_recipe
from online_feedback.storage import write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


root = Path(sys.argv[1]).resolve()
code = Path('/data/sources/minhao/DFlash-Online')
prior = Path('/data/sources/minhao/runs/dflash_osd_one_event_20261025')
implementation = Path('/data/sources/minhao/runs/dflash_prefix_ranking_implementation_20261025')
started = time.perf_counter()
seal = json.loads((root / 'SEAL.json').read_text())
for name, expected in seal['source_sha256'].items():
    if digest(code / name) != expected:
        raise ValueError('Reviewed source differs: ' + name)
old = json.loads((prior / 'sealed/SEAL.json').read_text())
for name, expected in old['input_sha256'].items():
    if digest(prior / 'sealed' / name) != expected:
        raise ValueError('Original sealed input differs: ' + name)
manifest = json.loads((prior / 'sealed/MANIFEST.json').read_text())
validate_manifest(manifest)
recipe = check_recipe(prior / 'sealed/config.yaml')
model_pin = json.loads((prior / 'sealed/MODEL_PIN.json').read_text())
models = Path('/data/sources/minhao/models/qwen3_dflash_foundation_20261022')
for name in ('target', 'draft'):
    verify_files(models / name, model_pin[name]['files'], exact_files=True)
runtime = verify_native_runtime('/data/conda_envs/minhao/adaflash_author/bin/python',
    json.loads((prior / 'sealed/RUNTIME_PIN.json').read_text()))

receipt_path = prior / 'campaign_001/trainer/events/event_001/RESULT.json'
receipt = json.loads(receipt_path.read_text())
if (receipt['status'], receipt['optimizer_steps'], receipt['response_presentations'],
        len(receipt['features'])) != ('completed', 128, 256, 128):
    raise ValueError('Ordinary event is incomplete')
previous_stats = json.loads((implementation / 'INSTALL.json').read_text())['retained_ordinary_artifact_stats']
export = Path(receipt['checkpoint_path'])
expected_export_files = {Path(name).name for name in previous_stats if Path(name).parent == export}
if {path.name for path in export.iterdir() if path.is_file()} != expected_export_files:
    raise ValueError('Retained O export filename membership changed')
retained = {}
for name, expected in previous_stats.items():
    path = Path(name)
    stat = path.stat()
    if [stat.st_size, stat.st_mtime_ns] != expected:
        raise ValueError('Retained O artifact changed since implementation: ' + name)
    retained[name] = {'bytes': stat.st_size, 'sha256': digest(path)}
write_json(root / 'RETAINED_ARTIFACTS.json', retained)
admission = admit_storage(root, models / 'draft')
write_json(root / 'PREFLIGHT.json', {
    'status': 'passed', 'recipe': recipe, 'runtime': runtime,
    'reviewed_source_files': len(seal['source_sha256']), 'original_inputs_unchanged': True,
    'retained_artifacts': len(retained), 'storage': admission,
    'new_feedback_or_teacher_preparation': False, 'wall_s': time.perf_counter() - started})
print('Source, retained artifacts, model/runtime and peak storage preflight passed.', flush=True)
