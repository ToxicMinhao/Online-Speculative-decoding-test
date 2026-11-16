"""Run one bounded card qualification worker; no production trajectory here."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import threading
import time
import traceback

from online_feedback.storage import write_json
from experiments.threatened_repair_replay.work import Ledger, TrainingInstrumentation
from .qualification import CASES, qualify
from .training import PersistentPrefixTrainer


def install_interrupt_handlers():
    """Let an owned deadline interruption reach evidence and runtime cleanup."""
    def interrupt(signum, _frame):
        raise InterruptedError(f"Qualification interrupted by {signal.Signals(signum).name}; no automatic retry")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupt)


class HostMemorySample:
    """Sample this process and descendants; report it as a sampled RSS bound."""
    def __init__(self):
        self.maximum_bytes = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import psutil
        process = psutil.Process()
        while not self.stop.is_set():
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.maximum_bytes = max(total, self.maximum_bytes)
            self.stop.wait(.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def run(plan, slot):
    import torch
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.functional_step_geometry.native_pilot import discarded_startup

    if slot not in CASES:
        raise ValueError("Qualification slot must be A or B")
    device = plan['devices'][slot]
    uuid = device['uuid'] if isinstance(device, dict) else device
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid or torch.cuda.device_count() != 1:
        raise ValueError("Worker must expose exactly its sealed physical GPU as logical cuda:0")
    root = Path(plan['root'])/'qualification'/slot
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    deadline = float(plan.get('gpu_deadline_monotonic', began + 14400))
    case_count = len(CASES[slot])
    limits = dict(model_load=1, optimizer_step=case_count,
                  response_forward=2+2*case_count, response_backward=2+2*case_count)
    ledger = Ledger(root/'PHYSICAL_WORK.jsonl', limits=limits, deadline=deadline)
    trainer = instrumentation = None
    report = dict(status='running', slot=slot, physical_gpu_uuid=uuid,
        execution_commit=plan['execution_commit'], limits=limits, optimizer_steps_startup=0,
        no_production_training=True, no_export=True, complete_generations=0,
        occupied_started_unix=time.time(), occupied_started_monotonic=began)
    write_json(root/'STATUS.json', report)
    monitor = HostMemorySample()
    try:
        with monitor:
            pin_retained_graph_runtime(root)
            start = time.perf_counter()
            with ledger.scope(category='loading'), ledger.operation('model_load'):
                trainer = PersistentPrefixTrainer(plan['configs']['N'], root/'trainer',
                    arm='N', binding=plan['binding'], origin_state=plan['origin_state'],
                    origin_export=plan['origin_export'],
                    original_config_text=Path(plan['origin_config']).read_text(),
                    completed_event=6, schedule=json.loads(Path(plan['event7_schedule']).read_text()),
                    qualification_only=True, expected_cuda_rng_states=1)
            report['loading_s'] = time.perf_counter()-start
            instrumentation = TrainingInstrumentation(trainer, ledger)
            with ledger.scope(category='startup'):
                report['startup'] = discarded_startup(trainer, plan['event7_feature_paths'], root/'startup')
            report['startup_s'] = report['startup']['wall_s']
            write_json(root/'STATUS.json', report)
            with ledger.scope(category='qualification'):
                report['qualification'] = qualify(trainer, plan['event7_feature_paths'], root/'cases', device_slot=slot)
            report['qualification_s'] = report['qualification']['wall_s']
            expected = dict(model_load=1, optimizer_step=case_count,
                            response_forward=2+2*case_count, response_backward=2+2*case_count)
            if dict(ledger.completed) != expected or ledger.failed:
                raise ValueError('Qualification physical work differs from its fixed accounting')
            report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        if trainer is not None:
            trainer.failed = True
        write_json(root/'STATUS.json', report)
        raise
    finally:
        cleanup_errors = []
        if trainer is not None:
            report['peak_gpu_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
            report['peak_gpu_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
        for owner in (instrumentation, trainer):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    cleanup_errors.append(dict(object=type(owner).__name__, error=repr(error)))
        if cleanup_errors:
            report.update(status='failed', cleanup_errors=cleanup_errors)
        report.update(physical_work=ledger.summary(), wall_s=time.monotonic()-began,
            occupied_card_s=time.monotonic()-began, max_main_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes,
            rss_sample_interval_s=.25, rss_not_continuous_peak=True)
        ledger.close()
        write_json(root/'STATUS.json', report)
    if report.get("cleanup_errors"):
        raise RuntimeError("Qualification cleanup failed; retained STATUS.json describes the failure")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--slot', required=True, choices=tuple(CASES))
    args = parser.parse_args()
    install_interrupt_handlers()
    run(json.loads(args.plan.read_text()), args.slot)


if __name__ == '__main__':
    main()
