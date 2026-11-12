"""Opt-in native worker; ordinary online serving remains unchanged."""
from __future__ import annotations

import argparse
import functools
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import time
import traceback

from online_feedback import sglang_hooks, sglang_worker
from online_feedback.storage import write_json
from .budget import connect_budget
from .native import NativeSuccessors, acceptance_totals, run_counted

_BUDGET = None
_DRAFT_ADMITTED = 0
_TARGET_ADMITTED = 0
_SESSION = None
_SESSIONS = []
_INSTALLED = False


def _batch(args, kwargs):
    value = kwargs.get('forward_batch')
    if value is None:
        value = next((item for item in args if hasattr(item, 'forward_mode')), None)
    if value is None:
        raise ValueError('Native model call does not expose its ForwardBatch')
    return value


def _batch_size(batch):
    size = batch.batch_size
    return int(size() if callable(size) else size)


def _startup_model_call(function, model, args, kwargs, *, draft):
    """Account graph warmup/capture forwards before a live request exists."""
    batch = _batch(args, kwargs)
    tickets = []
    try:
        if draft or batch.forward_mode.is_target_verify():
            # Startup graphs may contain several request rows. Each proposal or
            # standalone verification block consumes one block allowance.
            for _ in range(_batch_size(batch)):
                tickets.append(_BUDGET.attempt('draft_block', operation='startup_draft' if draft
                    else 'startup_standalone_verifier', startup=True))
        if not draft:
            tickets.append(_BUDGET.attempt('target_call', operation='startup_target', startup=True))
            if not batch.forward_mode.is_target_verify():
                tickets.append(_BUDGET.attempt('context_forward', operation='startup_target_context',
                    tokens=int(batch.input_ids.numel()), startup=True))
        result = function(model, *args, **kwargs)
    except BaseException as error:
        for ticket in tickets:
            _BUDGET.fail(ticket, repr(error))
        raise
    for ticket in tickets:
        _BUDGET.complete(ticket)
    return result


def install_accounting():
    """Install before native engine loading, including CUDA-graph warmups."""
    global _BUDGET, _INSTALLED
    if _INSTALLED:
        return
    _BUDGET = connect_budget(json.loads(os.environ['DFLASH_TRANSITION_BUDGET']))
    from sglang.srt.models.dflash import DFlashDraftModel
    from sglang.srt.models.qwen3 import Qwen3ForCausalLM
    from sglang.srt.speculative.dflash_worker import DFlashWorker
    from sglang.srt.managers.tp_worker import TpModelWorker

    draft_forward = DFlashDraftModel.forward
    target_forward = Qwen3ForCausalLM.forward
    prepare = DFlashWorker._prepare_for_speculative_decoding
    target_generate = TpModelWorker.forward_batch_generation

    @functools.wraps(draft_forward)
    def draft_model(model, *args, **kwargs):
        if _DRAFT_ADMITTED:
            return draft_forward(model, *args, **kwargs)
        return _startup_model_call(draft_forward, model, args, kwargs, draft=True)

    @functools.wraps(target_forward)
    def target_model(model, *args, **kwargs):
        if _TARGET_ADMITTED:
            return target_forward(model, *args, **kwargs)
        return _startup_model_call(target_forward, model, args, kwargs, draft=False)

    @functools.wraps(prepare)
    def prepare_block(worker, batch, info):
        global _DRAFT_ADMITTED
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return prepare(worker, batch, info)
        if _batch_size(batch) != 1:
            raise ValueError('Live transition replay requires concurrency one')
        def call():
            global _DRAFT_ADMITTED
            _DRAFT_ADMITTED += 1
            try:
                return prepare(worker, batch, info)
            finally:
                _DRAFT_ADMITTED -= 1
        return run_counted(_BUDGET, 'draft_block', call,
                           metadata=dict(operation='native_proposal_plus_verification'))

    @functools.wraps(target_generate)
    def target_batch(worker, batch, **kwargs):
        global _TARGET_ADMITTED
        if isinstance(worker.model_runner.model, DFlashDraftModel):
            return target_generate(worker, batch, **kwargs)
        def call():
            global _TARGET_ADMITTED
            _TARGET_ADMITTED += 1
            try:
                return target_generate(worker, batch, **kwargs)
            finally:
                _TARGET_ADMITTED -= 1
        operation = 'native_target_verify' if kwargs.get('is_verify') else 'native_target_prefill'
        def target_call():
            return run_counted(_BUDGET, 'target_call', call, metadata=dict(operation=operation))
        if kwargs.get('is_verify'):
            return target_call()
        return run_counted(_BUDGET, 'context_forward', target_call,
            metadata=dict(operation=operation, tokens=int(batch.input_ids.numel())))

    DFlashDraftModel.forward = draft_model
    Qwen3ForCausalLM.forward = target_model
    DFlashWorker._prepare_for_speculative_decoding = prepare_block
    TpModelWorker.forward_batch_generation = target_batch
    _INSTALLED = True


def transition_begin(self, request, checkpoint, capture_rounds, outputfile):
    global _SESSION
    sglang_hooks.strict_idle(self)
    if _SESSION is not None:
        raise ValueError('Release the preceding request before collecting another')
    _SESSION = NativeSuccessors(self, request, checkpoint, capture_rounds, _BUDGET)
    _SESSION.install()
    receipt = dict(status='armed', request_id=request['request_id'])
    write_json(outputfile, receipt)
    return receipt


def transition_walk_finished(self, outputfile):
    global _SESSION
    sglang_hooks.strict_idle(self)
    _SESSION.close()
    receipt = dict(status='captured', native_rounds=_SESSION.round_index,
        successors={str(index): state.metadata() for index, state in _SESSION.successors.items()})
    _SESSIONS.append(_SESSION)
    _SESSION = None
    write_json(outputfile, receipt)
    return receipt


def transition_continue(self, handle, checkpoint, outputfile):
    """Run native decode from restored target state and rebuilt draft context."""
    import torch

    sglang_hooks.strict_idle(self)
    global _SESSION
    if _SESSION is not None:
        raise ValueError('A native request capture is still active')
    matches = [(session, value) for session in _SESSIONS
               for value in session.successors.values() if value.handle == handle]
    if len(matches) != 1:
        raise ValueError('The selected live request state is unavailable')
    _SESSION, state = matches[0]
    if str(checkpoint) != str(self._online_publication['checkpoint_path']):
        raise ValueError('Destination native weights were not published')
    request = _SESSION.request
    trace_path = str(Path(outputfile).with_suffix('.trace.json'))
    recorder = sglang_hooks.RequestCapture(request['prompt_ids'], trace_path, False)
    recorder.prefill_rows = len(request['prompt_ids'])
    recorder.covered_ids = request['prompt_ids'] + state.output_ids[:-1]
    recorder.teacher_ids = state.output_ids
    recorder.block_size = 16
    sglang_hooks._ACTIVE = recorder
    _SESSION.capture_enabled = False
    _SESSION.install()
    started = time.perf_counter()
    failure = None
    try:
        with torch.inference_mode():
            batch = _SESSION.rebuild(state, checkpoint)
            req = batch.reqs[0]
            while not req.finished():
                _BUDGET.check_time()
                _SESSION.worker.forward_batch_generation(batch)
            torch.cuda.synchronize()
        observation = recorder.finish(list(req.output_ids))
        response = list(req.output_ids)
        finish = 'eos' if response[-1] in request['stop_token_ids'] else 'length'
        if finish == 'length' and len(response) != state.original_max_new_tokens:
            raise ValueError('Native continuation ended before the original cap')
        canonical = request['canonical_response_ids']
        expected_finish = request['canonical_finish_reason']
        record = dict(record_id=request['request_id'], checkpoint_id=str(checkpoint),
            prompt_ids=request['prompt_ids'], response_ids=response, finish_reason=finish,
            max_new_tokens=state.original_max_new_tokens, stop_token_ids=request['stop_token_ids'],
            metadata=dict(trace=observation['blocks'], verification_calls=len(observation['blocks']),
                **acceptance_totals(observation['blocks']),
                committed_progress=len(response)-len(state.output_ids), untimed_diagnostic=True,
                runtime='pinned_SGLang_FA3', block_size=16))
        receipt = dict(status='completed', record=record, verification_calls=len(observation['blocks']),
            output_matches_canonical=response == canonical, termination_matches=finish == expected_finish,
            reconstruction=_SESSION.reconstruction, replay_wall_s=time.perf_counter()-started,
            original_remaining_budget=state.original_max_new_tokens-len(state.output_ids))
        write_json(outputfile, receipt)
        if not receipt['output_matches_canonical'] or not receipt['termination_matches']:
            raise ValueError('Native crossed continuation changed canonical output or termination')
        return receipt
    except BaseException as error:
        failure = error
        write_json(outputfile, dict(status='failed', error=repr(error),
            original_remaining_budget=state.original_max_new_tokens-len(state.output_ids),
            reconstruction=_SESSION.reconstruction, completed_trace=recorder.trace,
            emitted_prefix=recorder.teacher_ids, replay_wall_s=time.perf_counter()-started))
        raise
    finally:
        _SESSION.close()
        _SESSION = None
        sglang_hooks._ACTIVE = None
        # The manual batch is private, and scheduler state remains idle. The
        # next ordinary publication/request starts through its qualified flush.
        try:
            sglang_hooks.flush_idle(self)
        except BaseException as cleanup_error:
            if failure is None:
                raise
            saved = json.loads(Path(outputfile).read_text())
            saved['cleanup_error'] = repr(cleanup_error)
            write_json(outputfile, saved)


def transition_release(self, outputfile):
    global _SESSION
    sglang_hooks.strict_idle(self)
    if _SESSION is not None:
        _SESSION.close()
    _SESSION = None
    _SESSIONS.clear()
    write_json(outputfile, dict(status='released'))
    return dict(status='released')


def scheduler_process(*args, **kwargs):
    install_accounting()
    sglang_hooks.install()
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
    for function in (transition_begin, transition_walk_finished, transition_continue, transition_release):
        setattr(Scheduler, function.__name__, sglang_hooks._rpc(function))
    return run_scheduler_process(*args, **kwargs)


class NativeWorker(sglang_worker.NativeWorker):
    def __init__(self, settings, root):
        from sglang.srt.entrypoints.engine import Engine

        self.root = root
        self.released = settings['speculative_draft_model_path']
        self.layout = sglang_worker.checkpoint_layout(self.released)
        self.config = sglang_worker.semantic_config(self.released)
        self.blocked = False
        self.budget = connect_budget(json.loads(os.environ['DFLASH_TRANSITION_BUDGET']))
        Engine.run_scheduler_process_func = staticmethod(scheduler_process)
        # The host admits and times this one engine initialization before it
        # spawns us. Counting a second load here would duplicate that operation.
        self.engine = Engine(**settings)

    def publish(self, request):
        return run_counted(self.budget, 'model_load', lambda: super(NativeWorker, self).publish(request),
                           metadata=dict(operation='native_draft_publication'))

    def rpc(self, request):
        operation = request['operation']
        filename = request['outputfile']
        arguments = {key:value for key,value in request.items() if key != 'operation'}
        self.engine.collective_rpc(operation, **arguments)
        return json.loads(Path(filename).read_text())


class NativeTransitionBackend:
    """Host adapter over the same qualified subprocess/runtime configuration."""

    def __init__(self, target_path, draft_path, *, worker_python, output_root,
                 budget_connection, serving_gpu='0', context_limit=4096, block_size=16):
        from online_feedback.sglang_backend import SGLangDFlashBackend

        if context_limit != 4096 or block_size != 16:
            raise ValueError('Transition diagnostic requires the unchanged 4096/b16 serving configuration')
        variable = 'DFLASH_TRANSITION_BUDGET'
        previous = os.environ.get(variable)
        os.environ[variable] = json.dumps(budget_connection)
        try:
            self.backend = SGLangDFlashBackend(target_path, draft_path,
                worker_python=worker_python, output_root=output_root, serving_gpu=serving_gpu,
                context_limit=context_limit, block_size=block_size,
                worker_module='experiments.verification_transition.native_worker')
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
        self.loading_s = self.backend.loading_s
        self.serial = 0

    def _rpc(self, name, **kwargs):
        self.serial += 1
        output = self.backend.root/f'transition_{self.serial:06d}.json'
        return self.backend._call(name, outputfile=str(output), **kwargs)

    def native_walk(self, request, checkpoint, capture_rounds):
        if request['stop_token_ids'] != self.backend.stop_tokens:
            raise ValueError('Sealed stop-token IDs differ from the qualified native target')
        self.backend.publish(checkpoint)
        native_request = dict(request,
            canonical_response_ids=request['canonical']['response_ids'],
            canonical_finish_reason=request['canonical']['finish_reason'])
        self._rpc('transition_begin', request=native_request, checkpoint=str(checkpoint),
                  capture_rounds=capture_rounds)
        record = self.backend.generate(request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
            request_id=request['request_id'], session_id='verification_transition',
            request_index=request['panel_ordinal'], purpose='evaluation')
        capture = self._rpc('transition_walk_finished')
        if capture['native_rounds'] != record['metadata']['verification_calls']:
            raise ValueError('Captured native boundaries disagree with recorded verification calls')
        record['metadata']['untimed_diagnostic'] = True
        return dict(record=record, successors=capture['successors'])

    def continue_from(self, successor, checkpoint, kind):
        if kind not in ('neutral_suffix', 'crossed_suffix'):
            raise ValueError('Only declared neutral/crossed continuations are supported')
        self.backend.publish(checkpoint)
        return self._rpc('transition_continue', handle=successor['handle'], checkpoint=str(checkpoint))

    def release_request(self):
        return self._rpc('transition_release')

    def close(self):
        self.backend.close()


def main():
    budget = connect_budget(json.loads(os.environ['DFLASH_TRANSITION_BUDGET']))
    budget.register_process(os.getpid(), os.getpgrp())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    worker = None
    try:
        started = time.perf_counter()
        worker = NativeWorker(json.loads(args.settings.read_text()), args.output)
        with Listener(('127.0.0.1', 0), authkey=b'dflash-online-local') as listener:
            write_json(args.output/'ready.json', dict(address=list(listener.address),
                loading_s=time.perf_counter()-started, status=worker.status()))
            with listener.accept() as connection:
                while True:
                    request = connection.recv()
                    if request['operation'] == 'close':
                        connection.send(dict(ok=True, result=None))
                        break
                    try:
                        if worker.blocked:
                            raise RuntimeError('Native transition worker is blocked after a failure')
                        if request['operation'].startswith('transition_'):
                            result = worker.rpc(request)
                        else:
                            methods = dict(generate=worker.generate, publish=worker.publish,
                                           teacher=worker.teacher, check_weights=worker.check_weights)
                            result = methods[request['operation']](request)
                        connection.send(dict(ok=True, result=result))
                    except BaseException as error:
                        worker.blocked = True
                        failure = dict(type=type(error).__name__, message=str(error),
                            traceback=traceback.format_exc(), operation=request['operation'])
                        write_json(args.output/'last_failure.json', failure)
                        connection.send(dict(ok=False, error=failure))
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
