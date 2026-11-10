"""Durable admission before model work, with explicit unfinished attempts.

A proposal plus its normal verification is one draft_block. Proposal-only or
standalone verifier replay is also one draft_block. Target calls and context
forwards are recorded separately; they do not increase that count twice.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time

from .contract import GPU_SECONDS, LIMITS


class BudgetExceeded(RuntimeError):
    """The selected request remains incomplete; it must not be replaced."""


class WorkBudget:
    def __init__(self, path, *, limits=None, gpu_seconds=GPU_SECONDS, clock=time.monotonic):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError('A fresh diagnostic ledger is required; no automatic retry')
        self.limits = dict(LIMITS if limits is None else limits)
        self.gpu_seconds_limit = gpu_seconds
        self.clock = clock
        self.counts = Counter()
        self.pending = {}
        self.completed = Counter()
        self.failed = Counter()
        self.gpu_started = None
        self.gpu_count = 0
        self.accumulated_gpu_seconds = 0.0
        self.serial = 0
        self.owned_processes = {}
        self._write(dict(kind='contract', limits=self.limits, gpu_seconds=gpu_seconds))

    def _write(self, row):
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    @property
    def gpu_seconds(self):
        elapsed = 0 if self.gpu_started is None else (self.clock() - self.gpu_started) * self.gpu_count
        return self.accumulated_gpu_seconds + elapsed

    def start_gpu(self, occupied_gpus=1):
        if self.gpu_started is not None or type(occupied_gpus) is not int or occupied_gpus < 1:
            raise ValueError('GPU occupancy must be explicit and nonoverlapping')
        self.gpu_started = self.clock()
        self.gpu_count = occupied_gpus
        self._write(dict(kind='gpu_start', occupied_gpus=occupied_gpus))
        self.check_time()

    def end_gpu(self):
        if self.gpu_started is None:
            return
        self.accumulated_gpu_seconds = self.gpu_seconds
        self.gpu_started = None
        self._write(dict(kind='gpu_end', gpu_seconds=self.accumulated_gpu_seconds))

    def check_time(self):
        if self.gpu_seconds >= self.gpu_seconds_limit:
            self._write(dict(kind='budget_stop', reason='gpu_time', gpu_seconds=self.gpu_seconds))
            raise BudgetExceeded('Cumulative occupied GPU time reached four-hour limit')

    def attempt(self, kind, **metadata):
        self.check_time()
        if kind in {'optimizer_step', 'backward', 'feedback_generation', 'timing_sweep'}:
            raise ValueError('Training, feedback collection and timing sweeps are forbidden')
        if kind not in self.limits and kind not in {'target_call', 'context_forward', 'model_load'}:
            raise ValueError('Unknown model-work category: ' + kind)
        if kind in self.limits and self.counts[kind] >= self.limits[kind]:
            self._write(dict(kind='budget_stop', reason=kind, counts=dict(self.counts)))
            raise BudgetExceeded(kind + ' allowance exhausted')
        self.serial += 1
        ticket = self.serial
        self.counts[kind] += 1
        self.pending[ticket] = kind
        self._write(dict(kind=kind, status='attempted', ticket=ticket,
                         gpu_seconds=self.gpu_seconds, metadata=metadata))
        return ticket

    def complete(self, ticket, **measurements):
        self._finish(ticket, 'completed', measurements)

    def fail(self, ticket, error):
        self._finish(ticket, 'failed', {'error': str(error)})

    def _finish(self, ticket, status, measurements):
        kind = self.pending.pop(ticket)
        (self.completed if status == 'completed' else self.failed)[kind] += 1
        self._write(dict(kind=kind, status=status, ticket=ticket,
                         gpu_seconds=self.gpu_seconds, measurements=measurements))

    def register_process(self, pid, pgid):
        if type(pid) is not int or type(pgid) is not int or pid <= 1 or pgid <= 1:
            raise ValueError('Expected an owned native process and process group')
        if os.getpgid(pid) != pgid:
            raise ValueError('Registered process group differs from the live process')
        self.owned_processes[pid] = dict(pid=pid, pgid=pgid)
        self._write(dict(kind='owned_process', pid=pid, pgid=pgid))

    def processes(self):
        return list(self.owned_processes.values())

    def summary(self):
        unknown = Counter(self.pending.values())
        return dict(attempted=dict(self.counts), completed=dict(self.completed),
                    failed=dict(self.failed), unknown=dict(unknown),
                    remaining={key: value - self.counts[key] for key, value in self.limits.items()},
                    gpu_seconds=self.gpu_seconds, gpu_seconds_limit=self.gpu_seconds_limit)


class BudgetService:
    """One local admission authority shared with native scheduler subprocesses.

    Workers call a proxy instead of writing competing ledger files. The random
    connection token belongs only to private execution configuration.
    """

    def __init__(self, budget):
        from multiprocessing.managers import BaseManager
        import threading

        class Manager(BaseManager):
            pass

        lock = threading.RLock()

        class SerializedBudget:
            def dispatch(self, method, args, kwargs):
                allowed = {'attempt', 'complete', 'fail', 'summary', 'check_time', 'start_gpu', 'end_gpu', 'register_process', 'processes'}
                if method not in allowed:
                    raise ValueError('Unknown budget operation')
                with lock:
                    return getattr(budget, method)(*args, **kwargs)

        shared = SerializedBudget()
        Manager.register('budget', callable=lambda: shared)
        self.authkey = os.urandom(32)
        manager = Manager(address=('127.0.0.1', 0), authkey=self.authkey)
        self.server = manager.get_server()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = dict(address=list(self.server.address), authkey_hex=self.authkey.hex())

    def close(self):
        self.server.stop_event.set()
        self.thread.join(timeout=2)


class BudgetClient:
    def __init__(self, proxy, manager):
        self.proxy, self.manager = proxy, manager

    def __getattr__(self, method):
        def invoke(*args, **kwargs):
            return self.proxy.dispatch(method, args, kwargs)
        return invoke


def connect_budget(connection):
    from multiprocessing.managers import BaseManager

    class Manager(BaseManager):
        pass

    Manager.register('budget')
    manager = Manager(address=tuple(connection['address']),
                      authkey=bytes.fromhex(connection['authkey_hex']))
    manager.connect()
    return BudgetClient(manager.budget(), manager)
