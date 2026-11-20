"""Target-only native AR with the existing worker transport and timing boundary."""
from __future__ import annotations

import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import subprocess
import time

from online_feedback.rng import preserve_rng
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.storage import write_json

from .validity import validate_ar_output, validate_measurement


def engine_settings(target_path, context_limit=4096):
    """4112 physical positions, 4096 logical budget; no speculative model at all."""
    if context_limit != 4096:
        raise ValueError('The CODE study fixes its logical context at 4096')
    return dict(model_path=str(target_path), speculative_algorithm=None,
                dtype='bfloat16', attention_backend='fa3', tp_size=1,
                max_running_requests=1, mem_fraction_static=0.65,
                context_length=context_limit + 16, chunked_prefill_size=8192,
                cuda_graph_max_bs=8, disable_cuda_graph=False,
                random_seed=17, trust_remote_code=True, skip_server_warmup=True,
                log_level='info')


class SGLangARBackend:
    """Separate engine lifecycle; shares only the qualified IPC/cleanup helpers.

    No DFlash constructor, publication, teacher capture, or observer is used.
    In particular this cannot be activated by publishing a dummy draft export.
    """
    _call = SGLangDFlashBackend._call
    _stop_process = SGLangDFlashBackend._stop_process
    close = SGLangDFlashBackend.close
    render = SGLangDFlashBackend.render

    def __init__(self, target_path, *, worker_python, output_root,
                 serving_gpu='0', context_limit=4096, reasoning='off'):
        from transformers import AutoTokenizer

        if reasoning != 'off':
            raise ValueError('The CODE study uses no-thinking prompts')
        settings = engine_settings(target_path, context_limit)
        self.root = Path(output_root).absolute()
        self.root.mkdir(parents=True, exist_ok=False)
        self.context_limit, self.reasoning = context_limit, reasoning
        self.checkpoint_id = str(target_path)
        self.connection = self.worker = None
        self.failed = False
        self.serial = 0
        self.tokenizer = AutoTokenizer.from_pretrained(target_path)
        target_config = json.loads((Path(target_path) / 'config.json').read_text())
        generation_path = Path(target_path) / 'generation_config.json'
        generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        eos = generation.get('eos_token_id') or target_config['eos_token_id']
        self.stop_tokens = list(eos) if isinstance(eos, list) else [eos]
        if self.tokenizer.eos_token_id not in self.stop_tokens:
            self.stop_tokens.append(self.tokenizer.eos_token_id)
        write_json(self.root / 'engine_settings.json', settings)
        write_json(self.root / 'measurement_contract.json', dict(
            logical_context_limit=context_limit, physical_context_length=settings['context_length'],
            feedback_capture=False, speculative_trace_instrumentation=False,
            cuda_graphs_enabled=True, speculation_disabled=True,
            native_overlap_schedule='retained runtime default; actual value in ready.json',
            request_timer='complete worker IPC including cache preparation and completion records',
            engine_timer='Engine.generate only; excludes cache preparation and completion records',
            status='requires_native_qualification'))

        # Match the DFlash worker environment rather than importing its runtime
        # into the controller/training interpreter.
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=str(serving_gpu), SGLANG_ENABLE_SPEC_V2='0',
                           PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        worker_lib = Path(worker_python).resolve().parent.parent / 'lib'
        environment['PATH'] = str(worker_lib.parent / 'bin') + os.pathsep + environment['PATH']
        environment['CONDA_PREFIX'] = str(worker_lib.parent)
        environment['LD_LIBRARY_PATH'] = ':'.join([
            str(worker_lib / 'adaflash_cuda12'), str(worker_lib),
            environment.get('LD_LIBRARY_PATH', '')])
        if 'CXX' in environment:
            environment['CUDAHOSTCXX'] = environment['CXX']
            environment['NVCC_CCBIN'] = environment['CXX']
        self.log = (self.root / 'worker.log').open('w')
        started = time.perf_counter()
        self.worker = subprocess.Popen([
            str(worker_python), '-u', '-m', 'experiments.opencode_native_comparison.ar_worker',
            '--settings', str(self.root / 'engine_settings.json'), '--output', str(self.root)],
            env=environment, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        ready_path = self.root / 'ready.json'
        try:
            while not ready_path.exists():
                if self.worker.poll() is not None:
                    raise RuntimeError('AR startup failed; inspect ' + str(self.root / 'worker.log'))
                if time.perf_counter() - started > 600:
                    raise TimeoutError('AR startup exceeded ten minutes; inspect worker.log')
                time.sleep(0.25)
            self.ready = json.loads(ready_path.read_text())
            self.connection = Client(tuple(self.ready['address']), authkey=b'dflash-online-local')
        except BaseException:
            self._stop_process()
            raise
        self.loading_s = time.perf_counter() - started

    def generate(self, prompt_ids, *, max_new_tokens, request_id, session_id,
                 request_index, purpose='evaluation', save_to=None):
        if purpose != 'evaluation':
            raise ValueError('AR supplies canonical/evaluation outputs, never learning feedback')
        if (not prompt_ids or max_new_tokens < 1 or max_new_tokens > 2048
                or len(prompt_ids) + max_new_tokens > self.context_limit):
            raise ValueError('Request exceeds the declared CODE context/output budget')
        self.serial += 1
        path = self.root / f'request_{self.serial:06d}.json'
        started = time.perf_counter()
        with preserve_rng():
            native = self._call('generate', prompt_ids=list(prompt_ids),
                max_new_tokens=max_new_tokens, trace_path=str(path), stop_token_ids=self.stop_tokens)
        latency = time.perf_counter() - started
        record = dict(record_id=request_id, session_id=session_id, request_index=request_index,
            checkpoint_id=self.checkpoint_id, purpose=purpose, prompt_ids=list(prompt_ids),
            response_ids=native['output_ids'], finish_reason=None,
            stop_token_ids=self.stop_tokens, max_new_tokens=max_new_tokens,
            metadata=dict(request_latency_s=latency, native_generate_s=native['native_generate_s'],
                cache_maintenance_s=native['cache_maintenance_s'], verification_calls=None,
                accepted_draft_tokens=None, committed_progress=None, committed_progress_per_call=None,
                calls_per_1000_output_tokens=None, block_size=None, context_limit=self.context_limit,
                physical_context_length=self.context_limit + 16, trace=None,
                native_trace_path=str(path), native_meta_info=native['meta_info'],
                cache_policy='idle_full_flush_before_every_request_and_publication',
                feedback_capture_in_timed_request=False, trace_instrumentation_in_timed_request=False,
                capture=None, runtime='pinned_SGLang_FA3', speculation_disabled=True),
            validation=dict(status='pending', errors=[]))
        record_path = Path(save_to) if save_to is not None else path.with_suffix('.record.json')
        write_json(record_path, record)
        try:
            _, finish = validate_ar_output(prompt_ids, native['output_ids'], native['meta_info'],
                max_new_tokens=max_new_tokens, stop_token_ids=self.stop_tokens,
                context_limit=self.context_limit)
            record['finish_reason'] = finish
            validate_measurement(record, arm='AR')
        except (ValueError, KeyError, TypeError) as error:
            record['validation'] = dict(status='failed', errors=[str(error)])
            write_json(record_path, record)
            self.failed = True
            raise
        record['validation']['status'] = 'passed'
        write_json(record_path, record)
        return record
