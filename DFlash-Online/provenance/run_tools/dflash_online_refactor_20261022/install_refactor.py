"""Install only the reviewed loop files; leave upstream source and runs alone."""
import json
from pathlib import Path
import shutil
import tarfile

project = Path('/data/sources/minhao/DFlash-Online')
work = Path('/data/sources/minhao/runs/dflash_online_refactor_20261022')

def contents(path):
    with tarfile.open(path) as archive:
        return {item.name: archive.extractfile(item).read() for item in archive.getmembers() if item.isfile()}

before = contents(work / 'before.tar.gz')
after = contents(work / 'refactor.tar.gz')
free_before = shutil.disk_usage(project).free
if free_before < 8 * 2**30 + 32 * 2**20:
    raise RuntimeError('Code/unit-test allowance plus 8 GiB reserve does not fit')
# The user may have edited the live copy while it was being reviewed.
for name, data in before.items():
    if (project / name).read_bytes() != data:
        raise RuntimeError('Live file changed during refactor: ' + name)
for name in after:
    if name not in before and (project / name).exists():
        raise RuntimeError('New filename already belongs to other work: ' + name)
for name, data in after.items():
    destination = project / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
retired = [name for name in before if name.startswith(('online_feedback/', 'tests/')) and name not in after]
for name in retired:
    (project / name).unlink()
removed_caches = []
for directory in (project / 'online_feedback', project / 'tests'):
    for cache in directory.rglob('__pycache__'):
        removed_caches.append(str(cache))
        shutil.rmtree(cache)
for name, data in after.items():
    if (project / name).read_bytes() != data:
        raise RuntimeError('Installed bytes differ: ' + name)
receipt = {
    'installed_files': sorted(after), 'retired_files': retired,
    'retained_old_code': str(work / 'before.tar.gz'),
    'removed_disposable_caches': removed_caches,
    'free_bytes_before': free_before,
    'free_bytes_after': shutil.disk_usage(project).free,
    'upstream_training_inference_touched': False,
    'existing_run_artifacts_touched': False,
    'installed_bytes_match': True,
}
(work / 'INSTALL.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
