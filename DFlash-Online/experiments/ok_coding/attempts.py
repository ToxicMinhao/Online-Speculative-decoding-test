"""Exclusive, monotonic attempt names without touching scientific RNG."""

import os
from pathlib import Path
import re


def _next_path(folder, prefix, width, suffix):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix))
    numbers = [int(match.group(1)) for path in folder.iterdir()
               if (match := pattern.fullmatch(path.name))]
    return folder / f"{prefix}{max(numbers, default=0) + 1:0{width}d}{suffix}"


def reserve_directory(folder, prefix="attempt_", width=6):
    path = _next_path(folder, prefix, width, "")
    path.mkdir(exist_ok=False)
    return path


def reserve_receipt(folder, prefix="attempt_", width=3):
    path = _next_path(folder, prefix, width, ".json")
    # A crash after reservation leaves this number occupied. Never reuse a gap.
    with path.open("x") as stream:
        stream.write('{"status": "reserved"}\n')
        stream.flush()
        os.fsync(stream.fileno())
    return path
