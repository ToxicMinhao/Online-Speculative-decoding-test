set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python - <<'PY'
from pathlib import Path
import json, shutil, tarfile
root = Path('/data/sources/minhao/DFlash-Online')
work = Path('/data/sources/minhao/runs/dflash_online_osd_recipe_20261023/implementation_001')
free = shutil.disk_usage(root).free
assert free >= (8 * 1024**3 + 256 * 1024**2), 'Source-only storage admission failed'
with tarfile.open(work / 'a800_source.tar') as old:
    original = {m.name: old.extractfile(m).read() for m in old.getmembers() if m.isfile()}
for name, content in original.items():
    assert (root / name).read_bytes() == content, f'Source changed since baseline: {name}'
files = json.loads((work / 'INSTALL_FILES.json').read_text())
with tarfile.open(work / 'install.tar') as new:
    assert sorted(m.name for m in new.getmembers()) == sorted(files)
    new.extractall(root, filter='data')
    for name in files:
        assert (root / name).read_bytes() == new.extractfile(name).read(), name
for name, content in original.items():
    if name not in files:
        assert (root / name).read_bytes() == content, name
receipt = dict(files=files, before_archive=str(work / 'a800_source.tar'),
               untouched_prior_files=sum(n not in files for n in original),
               free_before_bytes=free, free_after_bytes=shutil.disk_usage(root).free,
               gpu_execution=False, source_verified=True)
(work / 'INSTALL_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
PY
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/tests:$PWD/training:$PWD/inference"
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -m unittest discover -s tests -v > /data/sources/minhao/runs/dflash_online_osd_recipe_20261023/implementation_001/cpu_tests.log 2>&1
printf '%s\n' 'CPU tests exit status: 0'
tail -5 /data/sources/minhao/runs/dflash_online_osd_recipe_20261023/implementation_001/cpu_tests.log
df -B1 /data/sources/minhao
