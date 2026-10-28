"""Reclaim exact duplicate immutable qualification tensors without losing paths."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/data/sources/minhao/runs/dflash_prefix_ranking_controlled_qualification_20261026/qualification')
GROUPS = [
    ('warmup/initial.pt', [f'{arm}/initial.pt' for arm in ('Oa', 'Ob', 'U0', 'P0', 'U1', 'P1')],
     'cc136915633b6a260399cdc99e3c7748986d0bda038558a7bc490055068e64f5'),
    ('Oa/final.pt', [f'{arm}/final.pt' for arm in ('Ob', 'U0', 'P0')],
     'bff989b7d577b2cf5ae4316778576ef75d1964d4befc897c5560931a1b98e4fa'),
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(destination):
    destination = Path(destination)
    receipt = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        free_before=shutil.disk_usage(ROOT).free, operations=[],
        reason='Byte-identical immutable completed qualification snapshots; preserve every original path and full bytes',
        lost_capability=None, scientific_inputs_changed=False)
    for source_name, duplicates, expected in GROUPS:
        source = ROOT / source_name
        if digest(source) != expected:
            raise ValueError('Retained source identity differs: ' + str(source))
        for relative in duplicates:
            path = ROOT / relative
            old, retained = path.stat(), source.stat()
            if path.is_symlink() or old.st_dev != retained.st_dev or old.st_mode != retained.st_mode:
                raise ValueError('Require ordinary same-filesystem identical-mode files')
            if digest(path) != expected:
                raise ValueError('Duplicate identity differs: ' + str(path))
            row = dict(path=str(path), retained_copy=str(source), sha256=expected,
                       bytes=old.st_size, old_inode=old.st_ino, retained_inode=retained.st_ino,
                       previous_links=old.st_nlink)
            if old.st_ino != retained.st_ino:
                temporary = path.with_name(path.name + '.dedup-link')
                os.link(source, temporary)
                os.replace(temporary, path)
            row['same_inode_after'] = path.stat().st_ino == source.stat().st_ino
            receipt['operations'].append(row)
            receipt['free_after'] = shutil.disk_usage(ROOT).free
            destination.write_text(json.dumps(receipt, indent=2) + '\n')
    receipt['status'] = 'completed'
    receipt['free_after'] = shutil.disk_usage(ROOT).free
    receipt['freed_bytes'] = receipt['free_after'] - receipt['free_before']
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'operations'}))


if __name__ == '__main__':
    main(sys.argv[1])
