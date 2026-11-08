"""Durable, cumulative qualification limits, including unsuccessful attempts."""
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


LIMITS = {
    'adam': 8,
    'ordinary_forward': 16,
    'ordinary_backward': 16,
    'additional_forward': 96,
    'additional_backward': 32,
    'serving_generation': 8,
}


class QualificationWork:
    """Append before model work so an interrupted attempt still consumes budget."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counts = Counter()
        self.context = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.counts[(row['kind'], row['status'])] += 1

    def record(self, kind, status, **details):
        if status == 'attempted' and kind in LIMITS:
            if self.counts[(kind, status)] >= LIMITS[kind]:
                raise RuntimeError('Qualification ceiling reached before ' + kind)
        row = dict(time=time.time(), kind=kind, status=status,
                   **self.context, **details)
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.counts[(kind, status)] += 1

    @contextmanager
    def operation(self, kind, **details):
        self.record(kind, 'attempted', **details)
        try:
            yield
        except BaseException as error:
            self.record(kind, 'failed', error=repr(error), **details)
            raise
        else:
            self.record(kind, 'completed', **details)

    def callback(self, kind, status):
        """The hook distinguishes model calls from unbounded head/chunk work."""
        translated = {'additional_response_forwards': 'additional_forward',
                      'additional_backwards': 'additional_backward',
                      'optimizer_steps': 'adam'}.get(kind, kind)
        self.record(translated, status)

    def summary(self):
        return {f'{kind}_{status}': count
                for (kind, status), count in sorted(self.counts.items())}
