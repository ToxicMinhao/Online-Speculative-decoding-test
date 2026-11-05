set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python - <<'PY'
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

root = Path('/data/sources/minhao/DFlash-Online')
work = Path('/data/sources/minhao/runs/dflash_online_osd_schedule_20261023/implementation_001')
free = shutil.disk_usage(root).free
assert free >= 8 * 1024**3 + 256 * 1024**2, 'Source-only storage admission failed'
files = json.loads((work / 'INSTALL_FILES.json').read_text())
for name, expected in files.items():
    path = root / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    assert actual == expected['before'], f'Source changed since reviewed snapshot: {name}'
with tarfile.open(work / 'before.tar', 'x') as backup:
    for name in files:
        if (root / name).exists():
            backup.add(root / name, arcname=name)
with tarfile.open(work / 'install.tar') as archive:
    assert sorted(m.name for m in archive.getmembers()) == sorted(files)
    archive.extractall(root, filter='data')
for name, expected in files.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected['after'], name
receipt = {
    'files': files, 'backup': str(work / 'before.tar'),
    'free_before_bytes': free, 'free_after_bytes': shutil.disk_usage(root).free,
    'gpu_execution': False, 'schedule': {'responses_per_event': 128,
        'passes_per_event': 2, 'max_events': 58, 'optimizer_steps': 7424},
}
(work / 'INSTALL_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
PY
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/tests:$PWD/training:$PWD/inference"
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -m unittest discover -s tests -v > /data/sources/minhao/runs/dflash_online_osd_schedule_20261023/implementation_001/cpu_tests.log 2>&1
tail -5 /data/sources/minhao/runs/dflash_online_osd_schedule_20261023/implementation_001/cpu_tests.log
df -B1 /data/sources/minhao
