"""Isolate startup and auxiliary-work effects on ordinary K+ backwards.

Every window stops at the pre-Adam boundary. Cold/warm comparisons diagnose the
test procedure; they never replace the unchanged ordinary-gradient exact gate.
"""
import argparse
from copy import deepcopy
import gc
import inspect
import json
from pathlib import Path
import shutil
import time
import traceback

import torch

from experiments.ok_coding.storage import identity
from experiments.ok_continuation.resume import differences
from experiments.prefix_guard.qualify import persist_comparison
from experiments.prefix_ranking.qualify import override
from online_feedback.functional_geometry import install_functional_geometry
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json
from specforge.training.strategies.base import StepContext
from .diagnose_gradients import DiagnosticWork, forbid_step, pin_diagnostic_runtime
from .ordinary_gradient_checks import compare_gradients
from .qualify import (cleanup, physical_work, restore_origin, runtime_state,
                      scientific_state, set_observation_directory)
from .resume_qualification import retained_reference, restored_gsm_oracles
from .training import FunctionalGeometryTrainer


BUDGET = dict(response_forward=26, response_backward=18, optimizer_step=0,
              serving_generation=0, feedback_generation=0, teacher_preparation=0)


class CapturedBoundary(Exception):
    """The intended stopping point, before clipping or Adam."""


def compiler_state():
    """Observe the actual imported compiler state without resetting or tuning it."""
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    from specforge.modeling.draft import flex_attention as wrapper
    from torch.nn.attention.flex_attention import flex_attention
    entries = dynamo.eval_frame._debug_get_cache_entry_list(flex_attention)
    return dict(wrapper_file=inspect.getfile(wrapper),
        wrapper_identity=identity(inspect.getfile(wrapper)),
        automatic_dynamic_shapes=dynamo.config.automatic_dynamic_shapes,
        assume_static_by_default=dynamo.config.assume_static_by_default,
        automatic_dynamic_exclusion_guard=dynamo.config.automatic_dynamic_exclusion_guard,
        counters={str(k): {str(a): int(b) for a, b in v.items()} for k, v in counters.items()},
        flex_entries=[dict(compile_id=str(e.compile_id), code=e.code.co_name,
                           guards=str(e.guard_manager)) for e in entries])


def capture_window(trainer, features, label, work, baseline, root, expected_state,
                   expected_observations, *, scalar=False):
    """Use real TrainerCore forwards/backwards and intercept only the boundary."""
    root = Path(root); root.mkdir()
    restore_origin(trainer, baseline, root)
    set_observation_directory(trainer, label)
    trainer.bundle.model.begin_event()
    persist_comparison(root/'START_STATE.json', scientific_state(trainer), expected_state)
    gradients, rows = {}, []
    receipt = dict(status='running', scalar_capture=scalar, optimizer_steps=0,
                   before_compiler=compiler_state(), responses=[])
    hook = None; failure = None; intercepted = False
    forward, backward = trainer.author_strategy.forward_loss, trainer.backend.backward
    started = time.perf_counter()

    def boundary():
        gradients.update({name: None if p.grad is None else p.grad.detach().cpu().clone()
                          for name, p in trainer.bundle.draft_model.named_parameters()})
        nonfinite = {name:int((~torch.isfinite(value)).sum())
                     for name,value in gradients.items()
                     if value is not None and not bool(torch.isfinite(value).all())}
        if nonfinite:
            write_json(root/'NONFINITE_GRADIENTS.json',nonfinite)
            raise ValueError('Nonfinite ordinary gradients before the diagnostic boundary')
        raise CapturedBoundary()

    def observer(phase, active_hook):
        if phase != 'pre':
            raise AssertionError('A zero-Adam diagnostic reached a later optimizer phase')
        boundary()

    def counted_forward(*args, **kwargs):
        with work.operation('response_forward'):
            return forward(*args, **kwargs)

    def counted_backward(*args, **kwargs):
        with work.operation('response_backward'):
            result = backward(*args, **kwargs)
        receipt.setdefault('response_compiler', []).append(compiler_state())
        return result

    try:
        with override(trainer.backend.optimizer.optimizer, 'step', forbid_step), \
             override(trainer.author_strategy, 'forward_loss', counted_forward), \
             override(trainer.backend, 'backward', counted_backward):
            if scalar:
                hook = install_functional_geometry(trainer, arm='SCALAR', observer=observer)
            with override(trainer.backend.optimizer, 'step',
                          trainer.backend.optimizer.step if scalar else boundary):
                try:
                    for path in features:
                        raw = torch.load(path, map_location='cpu', weights_only=True)
                        trainer._anchor_observations.clear()
                        result = trainer.core.train_step(trainer._batch(raw), StepContext(
                            global_step=512, total_steps=trainer.horizon, collect_detailed_metrics=True))
                        rows.append(dict(record_id=raw['record_id'],
                            sampled_anchors=deepcopy(trainer._anchor_observations),
                            metrics=result.materialize_metrics()))
                        trainer.presentations += 1
                        del raw, result
                except CapturedBoundary:
                    intercepted = True
        if not intercepted or len(trainer.response_records) != 2 or not gradients:
            raise ValueError('The diagnostic did not intercept exactly one complete boundary')
        receipt['responses'] = deepcopy(trainer.response_records)
        actual = [{k:v for k,v in r.items() if k != 'event_id'} for r in trainer.response_records]
        expected = [{k:v for k,v in r.items() if k != 'event_id'} for r in expected_observations]
        persist_comparison(root/'ANCHOR_LOSS_OBSERVATIONS.json', actual, expected)
        after = scientific_state(trainer)
        for key in ('weights','optimizer','global_step'):
            persist_comparison(root/('UNCHANGED_'+key.upper()+'.json'), after[key], expected_state[key])
        del after
        receipt.update(status='captured_before_adam', intended_boundary_stop=True,
                       after_compiler=compiler_state())
        return gradients, receipt
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['physical_work'] = physical_work(trainer)
        if hook is not None:
            receipt['capture_hook'] = hook.report()
        finalization_error = None
        actions = ([] if hook is None else [('close capture hook', hook.close)]) + [
            ('restore origin', lambda: restore_origin(trainer, baseline, root)),
            ('exact final state', lambda: persist_comparison(root/'RESTORATION.json',
                 scientific_state(trainer), expected_state))]
        for label, action in actions:
            try:
                action()
            except BaseException as error:
                receipt.setdefault('cleanup_errors', []).append(dict(
                    operation=label, error=repr(error), traceback=traceback.format_exc()))
                receipt['status'] = 'failed'
                if finalization_error is None:
                    finalization_error = error
        receipt.update(duration_s=time.perf_counter()-started, work=dict(work.counts))
        cleanup(lambda: write_json(root/'RESULT.json', receipt),
                failure or finalization_error, receipt, 'capture receipt')
        if failure is None and finalization_error is not None:
            raise finalization_error



class AuxiliaryWork:
    def __init__(self, work): self.work = work
    def callback(self, kind, status):
        translated = {'additional_response_forwards':'response_forward',
                      'additional_backwards':'response_backward',
                      'optimizer_steps':'optimizer_step'}[kind]
        self.work.record(translated, status)


def run(plan_path):
    plan = json.loads(Path(plan_path).read_text()); root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=False); write_json(root/'PLAN.json', plan)
    work = DiagnosticWork(root, BUDGET)
    receipt = dict(status='running', execution_commit=plan['execution_commit'],
        interpretation='zero-Adam diagnosis, not qualification acceptance', budget=BUDGET,
        comparisons={}, main_pilot_started=False)
    trainer = None; failure = None; started = time.perf_counter()
    try:
        free = shutil.disk_usage(root).free
        if free < plan['storage']['maximum_additional_bytes']+plan['storage']['reserve_bytes']:
            raise ValueError('Insufficient storage for separately budgeted diagnostic')
        admitted = [{**item, 'matched':identity(item['path']) == item['identity']}
                    for item in plan['dependencies']]
        write_json(root/'ADMISSION.json', dict(free_bytes=free, files=admitted))
        if not all(item['matched'] for item in admitted):
            raise ValueError('Diagnostic source/input identity changed')
        original = json.loads(Path(plan['qualification_plan']).read_text())
        item = original['datasets']['gsm8k']
        pin_diagnostic_runtime(root)
        gpu_free,gpu_total=torch.cuda.mem_get_info()
        receipt['gpu_admission']=dict(free_bytes=gpu_free,total_bytes=gpu_total,
            required_free_bytes=plan['minimum_gpu_free_bytes'])
        if gpu_free < plan['minimum_gpu_free_bytes']:
            raise ValueError('Insufficient free GPU memory for the fixed resource fixture')
        torch.cuda.reset_peak_memory_stats()
        trainer = FunctionalGeometryTrainer(item['configs']['KPLUS'],root/'trainer',
            arm='KPLUS',dataset='gsm8k',binding=original['binding'],
            original_config_text=Path(original['original_config']).read_text(),
            origin_state=item['origin_state'],origin_export=item['origin_export'])
        baseline = runtime_state(trainer.bundle.model)
        before = scientific_state(trainer)
        expected = [json.loads(line) for line in Path(plan['observations']).read_text().splitlines()]

        def capture(label, *, scalar=False):
            gradients, result = capture_window(trainer,plan['features'],label,work,baseline,
                root/label,before,expected,scalar=scalar)
            receipt.setdefault('cases',{})[label] = dict(status=result['status'],duration_s=result['duration_s'])
            return gradients

        def compare(label, left, right):
            result = compare_gradients(left,right)
            write_json(root/(label+'.json'),result)
            receipt['comparisons'][label] = {k:v for k,v in result.items() if k!='parameters'}
            write_json(root/'RESULT.json',receipt)
            if result['global']['nonfinite_coordinates']:
                raise ValueError('Nonfinite diagnostic gradient')
            return result['exact_equal']

        cold = capture('cold_plain')
        warm = capture('warm_plain')
        compare('cold_vs_warm',cold,warm); del cold
        repeated = capture('repeat_plain')
        repeat_equal = compare('warm_plain_repeat',repeated,warm); del repeated
        scalar = capture('warm_scalar',scalar=True)
        scalar_equal = compare('warm_scalar_vs_plain',scalar,warm); del scalar
        if repeat_equal and scalar_equal:
            # Load the retained actual native endpoint. No optimizer is applied.
            retained = retained_reference(original)
            native = torch.load(retained['payload'],map_location='cpu',weights_only=False,mmap=True)
            trainer.bundle.draft_model.load_state_dict(native['weights'],strict=True)
            trainer.backend.optimizer.load_state_dict(deepcopy(native['optimizer']))
            from .qualify import restore_runtime
            from online_feedback.rng import restore_rng
            trainer.global_step, trainer.presentations = native['global_step'],native['presentations']
            trainer.core._micro=native['micro']
            restore_runtime(trainer.bundle.model,native['runtime']);restore_rng(native['rng'])
            with override(trainer.backend.optimizer.optimizer,'step',forbid_step):
                receipt['auxiliary_intervention']=restored_gsm_oracles(trainer,plan['features'],
                    retained,original,AuxiliaryWork(work),root/'intervening_oracles')
            del native
            post = capture('post_aux_plain')
            compare('post_aux_vs_pre_aux_plain',post,warm)
            scalar = capture('post_aux_scalar',scalar=True)
            compare('post_aux_scalar_vs_plain',scalar,post)
            del post,scalar
        else:
            receipt['auxiliary_intervention']='not executed; warm ordinary/control parity unresolved'
        del warm
        receipt['status']='completed'
    except BaseException as error:
        failure=error;receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if trainer is not None:
            cleanup(trainer.close,failure,receipt,'close diagnostic trainer')
        receipt.update(duration_s=time.perf_counter()-started,work=work.counts,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        def postcheck():
            post=[dict(path=x['path'],matched=identity(x['path'])==x['identity'])
                  for x in plan['dependencies']]
            receipt['dependency_postcheck']=post
            if not all(x['matched'] for x in post):
                raise ValueError('Diagnostic source/input changed during execution')
        if failure is None:
            try:
                postcheck()
            except BaseException as error:
                failure=error;receipt.update(status='failed',dependency_failure=repr(error))
        else:
            cleanup(postcheck,failure,receipt,'dependency postcheck')
        write_json(root/'RESULT.json',receipt)
    if failure is not None: raise failure
    if receipt['status']!='completed': raise RuntimeError('Diagnostic integrity check failed')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan')
    result=run(parser.parse_args().plan)
    print(json.dumps({k:result[k] for k in ('status','duration_s','work')}))
