"""Physical work accounting that is never part of a restorable model state."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


class WorkLimit(RuntimeError):
    pass


class Ledger:
    """Durable operation starts, explicit failures, and independent counters.

    Frequent layer/head hooks only increment counters. Every enclosing response
    boundary durably publishes those counters; an interrupted operation remains
    visibly incomplete instead of being counted as an unattempted response.
    Wall intervals can be nested and must not be added as disjoint GPU costs.
    """
    def __init__(self, path, *, limits=None, deadline=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x")
        self.limits = dict(limits or {})
        self.deadline = deadline
        self.context = {}
        self.attempts, self.completed, self.failed, self.physical = (Counter() for _ in range(4))
        self.scoped = Counter()
        self.serial = 0
        self.started = time.monotonic()

    def _write(self, row):
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def check(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise WorkLimit("Three occupied GPU-card-hour budget exhausted")

    @contextmanager
    def scope(self, **metadata):
        previous = self.context
        self.context = dict(previous, **metadata)
        try:
            yield
        finally:
            self.context = previous

    def tick(self, kind, count=1):
        """Count a physical hook without altering model/RNG state."""
        self.physical[kind] += count
        self.scoped[(self.context.get("category", "unspecified"), kind)] += count

    @contextmanager
    def operation(self, kind, **metadata):
        self.check()
        if kind in self.limits and self.attempts[kind] >= self.limits[kind]:
            raise WorkLimit("Physical operation limit reached: " + kind)
        self.serial += 1
        serial, began = self.serial, time.monotonic()
        self.attempts[kind] += 1
        context = dict(self.context, **metadata)
        receipt = dict(serial=serial, kind=kind, **context)
        self._write(dict(receipt, phase="started", elapsed_s=began-self.started))
        try:
            yield receipt
        except BaseException as error:
            self.failed[kind] += 1
            self._write(dict(receipt, phase="failed", error=repr(error),
                wall_s=time.monotonic()-began, physical=dict(self.physical)))
            raise
        else:
            self.completed[kind] += 1
            self._write(dict(receipt, phase="completed", wall_s=time.monotonic()-began,
                             physical=dict(self.physical)))

    def call(self, kind, function, *args, **kwargs):
        with self.operation(kind):
            return function(*args, **kwargs)

    def summary(self):
        return dict(attempted=dict(self.attempts), completed=dict(self.completed),
            failed=dict(self.failed), physical=dict(self.physical),
            unknown={k:self.attempts[k]-self.completed[k]-self.failed[k] for k in self.attempts},
            by_category={c:{k:v for (category,k),v in self.scoped.items() if category==c}
                         for c in sorted({c for c,k in self.scoped})},
            wall_s=time.monotonic()-self.started, nested_clocks_not_additive=True)

    def close(self):
        if not self.stream.closed:
            self._write(dict(phase="ledger_closed", **self.summary()))
            self.stream.close()


class TrainingInstrumentation:
    """Observe existing native boundaries; never replace their arithmetic."""
    def __init__(self, trainer, ledger):
        self.ledger, self.originals, self.handles = ledger, [], []
        self._wrap(trainer.author_strategy, "forward_loss", "response_forward")
        self._wrap(trainer.backend, "backward", "response_backward")
        self._wrap(trainer.backend.optimizer.optimizer, "step", "optimizer_step")
        model = trainer.bundle.model
        for module, kind in ((model.draft_model, "draft_forwards"),
                             (model.lm_head, "vocabulary_head_calls")):
            self.handles.append(module.register_forward_pre_hook(self._count(kind)))
        for layer in model.draft_model.layers:
            self.handles.append(layer.register_forward_pre_hook(self._count("layer_forwards")))
        self._wrap(model, "_chunk_loss", "objective_chunk", durable=False)

    def _count(self, kind):
        def count(_module, _inputs):
            self.ledger.tick(kind)
        return count

    def _wrap(self, owner, name, kind, *, durable=True):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        def counted(*args, **kwargs):
            if durable:
                with self.ledger.operation(kind):
                    return original(*args, **kwargs)
            self.ledger.tick(kind)
            return original(*args, **kwargs)
        setattr(owner, name, counted)

    def close(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        for handle in self.handles:
            handle.remove()


def read_ledger(path):
    """Independent accounting from durable records, including crashed starts."""
    starts, ends, failures, last_physical = {}, {}, {}, {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            phase = row["phase"]
            if phase == "started":
                if row["serial"] in starts:
                    raise ValueError("Duplicate physical-work serial")
                starts[row["serial"]] = row
            elif phase in ("completed", "failed"):
                serial = row["serial"]
                if serial not in starts or serial in ends or serial in failures:
                    raise ValueError("Unpaired physical-work completion")
                (ends if phase == "completed" else failures)[serial] = row
                last_physical = row.get("physical", last_physical)
            elif phase == "ledger_closed":
                last_physical = row["physical"]
    counts = lambda rows: dict(Counter(row["kind"] for row in rows.values()))
    unknown = {key:row for key,row in starts.items() if key not in ends and key not in failures}
    return dict(attempted=counts(starts), completed=counts(ends), failed=counts(failures),
                unknown=counts(unknown), physical=last_physical)
