"""One separately authorized constructor trace; no startup, forward or Adam step.

This manual diagnostic is never called by the default campaign. Review and
publish its identity before a separately authorized execution. It reuses the existing work/clock
ledgers and never starts a new campaign or resets a seed itself.
"""
import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import os
import platform
import random
import sys
import time
import traceback

# Match the qualification entry's imports, including lazy imports performed before
# pin_retained_graph_runtime and FullCTrainer construction. Do not warm up models.
import experiments.full_c_gsm8k.entry
import experiments.full_c_gsm8k.qualification
import torch
from experiments.functional_step_geometry.qualify import runtime_state
from experiments.functional_step_geometry.native_pilot import discarded_startup
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.persistent_prefix_withholding.qualification import scientific_identity, tensor_identity, tree_identity
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.training import SpecForgeEventTrainer
from specforge.export.to_hf import export_to_hf
from experiments.full_c_gsm8k.training import FullCTrainer
from experiments.full_c_gsm8k.runtime import WorkLedger, boundary, occupied_card, read
from online_feedback.storage import write_json


def runtime_versions():
    """Read installed metadata without loading a package earlier than native startup."""
    return {'python':platform.python_version(), 'torch':str(torch.__version__),
            'transformers':version('transformers')}


def module_sources():
    names = ('torch', 'transformers', 'specforge', 'specforge.distributed',
             'specforge.cli', 'rich', 'rich.style', 'online_feedback.training',
             'experiments.full_c_gsm8k.training')
    result = {}
    for name in names:
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        result[name] = {'imported':module is not None, 'path':path}
        if path is not None and Path(path).is_file():
            result[name]['file_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


class PythonRandomTrace:
    """Observe only the global Python generator; never draw or restore it."""
    def __init__(self, destination, *, max_records=1024):
        self.destination = Path(destination)
        self.events = []
        self.pending = {}
        self.max_records = max_records
        self.dropped_calls = 0
        self.stage = 'ready'
        self.previous = None

    @staticmethod
    def state():
        value = random.getstate()
        return {'index':value[1][-1], 'sha256':hashlib.sha256(repr(value).encode()).hexdigest()}

    def save(self):
        write_json(self.destination, {'stage':self.stage, 'global_random_calls':self.events,
            'dropped_call_records':self.dropped_calls, 'python_rng':self.state(),
            'observer_draws':0, 'observer_seed_or_restore_calls':0})

    def profile(self, frame, event, function):
        method = getattr(function, '__name__', '')
        c_random = (getattr(function, '__self__', None) is random._inst and method != 'getstate')
        python_random = (frame.f_code.co_filename.endswith('/random.py')
            and frame.f_locals.get('self') is random._inst
            and frame.f_code.co_name in ('seed', 'setstate'))
        starts = (event == 'c_call' and c_random) or (event == 'call' and python_random)
        ends = (event in ('c_return', 'c_exception') and c_random) or (event == 'return' and python_random)
        key = ('c' if c_random else 'python', id(frame),
               method if c_random else frame.f_code.co_name)
        if starts:
            if len(self.events) >= self.max_records:
                self.dropped_calls += 1
                return
            row = {'stage':self.stage, 'function':key[2], 'call_type':key[0], 'event':event,
                'before':self.state(), 'stack':traceback.format_stack(frame, limit=32)}
            self.pending[key] = row
            self.events.append(row)
            self.save()
        elif ends and key in self.pending:
            row = self.pending.pop(key)
            row['after'] = self.state()
            row['changed'] = row['before'] != row['after']
            self.save()

    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError('An existing Python profiler must not be replaced')
        sys.setprofile(self.profile)
        self.save()
        return self

    def __exit__(self, *_):
        sys.setprofile(self.previous)
        self.save()


def run(plan, output, execution_commit):
    root, base = Path(plan['root']), Path(output)
    if root.resolve() not in base.resolve().parents:
        raise ValueError('Diagnostic evidence must stay inside the existing run')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['devices']['B']:
        raise ValueError('Constructor trace must own the qualified training card')
    for name in ('WORK.json', 'CLOCK.json'):
        if not (root/name).is_file():
            raise FileNotFoundError('Refuse to create or reset campaign ledgers: '+name)
    base.mkdir(parents=True, exist_ok=False)
    ledger = WorkLedger(root, 'constructor_trace_'+base.name)
    report = {'status':'running', 'execution_commit':execution_commit,
        'original_plan_execution_commit':plan['execution_commit'], 'binding':plan['binding'],
        'entry_path':str(Path(__file__).resolve()),
        'entry_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_epoch':time.time(), 'work_before':read(root/'WORK.json'),
        'runtime':runtime_versions(),
        'sources_before':module_sources(), 'planned_initializations':1,
        'planned_forward_backward_adam_generations':0, 'rng_override':False}
    destination = base/'DETAILS.json'
    write_json(destination, report)
    trainer = None
    failure = None
    occupancy = ExitStack()
    try:
        occupancy.enter_context(occupied_card(root, 'constructor_trace_'+base.name, plan['devices']['B']))
        report['admission'] = boundary(plan)
        with PythonRandomTrace(base/'PYTHON_RANDOM_CALLS.json') as trace:
            trace.stage = 'pin_runtime'
            report['runtime_compatibility'] = pin_retained_graph_runtime(base)
            trace.stage = 'constructor'
            began = time.perf_counter()
            trainer = ledger.call({'training_initialization':1}, 'constructor', FullCTrainer,
                plan['config_path'], base/'trainer', binding=plan['binding'],
                original_config_text=Path(plan['original_config']).read_text(), qualification_only=True)
            report['loading_s'] = time.perf_counter()-began
            report['sources_after_constructor'] = module_sources()
            trace.stage = 'initial_state_check'
            # The new evidence helper records all RNG payloads before the
            # unchanged exact comparator raises. A pass saves only step zero.
            began = time.perf_counter()
            report['initialization'] = trainer.save_initial(plan['initial_witness'])
            report['initialization_check_s'] = time.perf_counter()-began
            report['status'] = 'passed'
    except BaseException as error:
        failure = error
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            report['physical_work'] = dict(trainer.work)
            try:
                trainer.close()
            except BaseException as error:
                report['cleanup_error'] = repr(error)
                if failure is not None:
                    failure.add_note('Constructor-trace cleanup also failed: '+repr(error))
                else:
                    report.update(status='failed', error=repr(error))
                    failure = error
        occupancy.close()
        report.update(work_after=read(root/'WORK.json'), clock_after=read(root/'CLOCK.json'),
            wall_s=time.time()-report['started_epoch'], sources_final=module_sources(),
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()) if torch.cuda.is_initialized() else 0)
        write_json(destination, report)
    if failure is not None:
        raise failure
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execution-commit', required=True)
    args = parser.parse_args()
    run(read(args.plan), args.output, args.execution_commit)
