"""One CPU-only source/model/runtime check before the bounded GPU run."""
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.osd_components.prepare import verify_files, verify_native_runtime, verify_source
from experiments.osd_one_event.protocol import validate_manifest
from online_feedback.qualify_sglang import check_recipe
from online_feedback.storage import write_json

root = Path(sys.argv[1]).resolve()
code = Path('/data/sources/minhao/DFlash-Online')
started = time.perf_counter()
seal = json.loads((root / 'sealed/SEAL.json').read_text())
for name, digest in seal['new_source_sha256'].items():
    if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest:
        raise ValueError('One-event source differs from seal: ' + name)
for name, digest in seal['input_sha256'].items():
    if hashlib.sha256((root / 'sealed' / name).read_bytes()).hexdigest() != digest:
        raise ValueError('Sealed input differs: ' + name)
source = verify_source('/data/sources/minhao/runs/dflash_online_osd_rerun_20261023', code)
manifest = json.loads((root / 'sealed/MANIFEST.json').read_text())
validate_manifest(manifest)
recipe = check_recipe(root / 'sealed/config.yaml')
pin = json.loads((root / 'sealed/MODEL_PIN.json').read_text())
models = Path('/data/sources/minhao/models/qwen3_dflash_foundation_20261022')
for name in ('target', 'draft'):
    verify_files(models / name, pin[name]['files'])
runtime = verify_native_runtime('/data/conda_envs/minhao/adaflash_author/bin/python',
    json.loads((root / 'sealed/RUNTIME_PIN.json').read_text()))
write_json(root / 'PREFLIGHT.json', {
    'status': 'passed', 'unchanged_qualified_source': source, 'recipe': recipe,
    'native_runtime': runtime, 'models_match_original_pins': True,
    'new_source_matches_seal': True, 'manifest_matches_seal': True,
    'cpu_only': True, 'wall_s': time.perf_counter() - started,
})
print('CPU source/model/runtime preflight passed.', flush=True)
