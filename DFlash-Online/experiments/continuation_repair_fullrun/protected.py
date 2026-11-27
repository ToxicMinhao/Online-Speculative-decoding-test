"""Retirement is confined to this new run; historical dependencies stay read-only."""
from pathlib import Path
from experiments.ok_coding.storage import identity
from .runtime import read


def require_retirement_paths(root, paths):
    root = Path(root).resolve()
    for value in paths:
        path = Path(value).absolute()
        if (path != path.resolve() or not path.is_relative_to(root/'campaign_001')
                or not path.is_file()):
            raise ValueError('Only an explicit new-run artifact may be retired: '+str(path))


def check(root, *, purpose):
    """Boundary dependency checks use sealed small-file identities, not model rehashes."""
    seal = read(Path(root)/'PROTECTED_INPUTS.json')
    for entry in seal['files']:
        path = Path(entry['path'])
        stat = path.stat()
        if stat.st_size != entry['bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise ValueError('Historical dependency changed: '+str(path))
    return dict(status='passed', purpose=purpose, checked_files=len(seal['files']),
                full_hash_policy='Hashed before model work and at final postflight; size/mtime at boundaries')
