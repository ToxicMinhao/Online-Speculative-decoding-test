"""Small atomic writes for a single-process experiment.

A replacement is published only after its temporary file is written. These
helpers do not implement locking, content hashing, or automatic recovery.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def _temporary_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    """Write readable metadata, preserving the previous file if writing fails."""
    with _temporary_file(path) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def save_torch(path, payload):
    """Save a checkpoint without exposing a partially written replacement."""
    import torch

    with _temporary_file(path) as temporary:
        torch.save(payload, temporary)
