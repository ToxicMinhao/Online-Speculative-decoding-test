"""Read-only remote check before deploying the reviewed startup integration."""
import hashlib
import json
from pathlib import Path

code = Path('/data/sources/minhao/DFlash-Online')
qualification = Path('/data/sources/minhao/runs/dflash_prefix_ranking_controlled_qualification_20261026')
seal = json.loads((qualification / 'SEAL.json').read_text())
for name, expected in seal['source_sha256'].items():
    with (code / name).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Live source changed since qualification: ' + name)
for name in ('experiments/prefix_ranking/controlled_startup.py', 'tests/test_prefix_ranking_controlled_startup.py'):
    if (code / name).exists():
        raise ValueError('New startup file already exists; inspect before replacing: ' + name)
print('Live qualified source unchanged; startup deployment is safe.')
