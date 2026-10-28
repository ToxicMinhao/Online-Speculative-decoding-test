"""SGLang/FA3 implementation of the online controller's backend interface.

Inference and native target capture run in the pinned SGLang environment.
The existing SpecForge trainer consumes ordinary CPU feature files unchanged.
"""
from __future__ import annotations

import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import signal
import subprocess
import time

from .native import validate_generation
from .rng import preserve_rng
from .storage import save_torch, write_json


class SGLangDFlashBackend:
    def __init__(self, target_path, draft_path, *, worker_python, output_root,
                 serving_gpu='0', context_limit=2048, block_size=16, reasoning='off'):
        from transformers import AutoTokenizer

        self.root = Path(output_root).absolute()
        self.root.mkdir(parents=True, exist_ok=False)
        self.context_limit, self.block_size = context_limit, block_size
        self.reasoning = reasoning
        self.checkpoint_id = str(draft_path)
        self.connection = None
        self.worker = None
        self.failed = False
        self.serial = 0
        self.tokenizer = AutoTokenizer.from_pretrained(target_path)
        target_config = json.loads((Path(target_path) / 'config.json').read_text())
        generation_path = Path(target_path) / 'generation_config.json'
        generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        eos = generation_config.get('eos_token_id') or target_config['eos_token_id']
        self.stop_tokens = eos if isinstance(eos, list) else [eos]
        if self.tokenizer.eos_token_id not in self.stop_tokens:
            self.stop_tokens.append(self.tokenizer.eos_token_id)
        draft_config = json.loads((Path(draft_path) / 'config.json').read_text())
        self.target_layer_ids = draft_config['dflash_config']['target_layer_ids']
        if draft_config['block_size'] != block_size:
            raise ValueError('Block size differs from released draft')
        settings = {
            'model_path': str(target_path), 'speculative_draft_model_path': str(draft_path),
            'speculative_algorithm': 'DFLASH', 'speculative_num_draft_tokens': block_size,
            'dtype': 'bfloat16', 'attention_backend': 'fa3', 'tp_size': 1,
            'max_running_requests': 1, 'mem_fraction_static': 0.65,
            'context_length': context_limit + block_size,
            'chunked_prefill_size': 8192, 'cuda_graph_max_bs': 8,
            'random_seed': 17, 'trust_remote_code': True, 'skip_server_warmup': True,
            'log_level': 'info',
        }
        write_json(self.root / 'engine_settings.json', settings)
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=str(serving_gpu), SGLANG_ENABLE_SPEC_V2='0',
                           PYTHONPATH=str(Path(__file__).resolve().parent.parent))
        worker_lib = Path(worker_python).resolve().parent.parent / 'lib'
        environment['PATH'] = str(worker_lib.parent / 'bin') + os.pathsep + environment['PATH']
        environment['CONDA_PREFIX'] = str(worker_lib.parent)
        environment['LD_LIBRARY_PATH'] = ':'.join([
            str(worker_lib / 'adaflash_cuda12'), str(worker_lib),
            environment.get('LD_LIBRARY_PATH', ''),
        ])
        if 'CXX' in environment:
            environment['CUDAHOSTCXX'] = environment['CXX']
            environment['NVCC_CCBIN'] = environment['CXX']
        self.log = (self.root / 'worker.log').open('w')
        started = time.perf_counter()
        self.worker = subprocess.Popen([
            str(worker_python), '-u', '-m', 'online_feedback.sglang_worker',
            '--settings', str(self.root / 'engine_settings.json'), '--output', str(self.root),
        ], env=environment, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        ready_path = self.root / 'ready.json'
        try:
            while not ready_path.exists():
                if self.worker.poll() is not None:
                    raise RuntimeError('SGLang startup failed; inspect ' + str(self.root / 'worker.log'))
                if time.perf_counter() - started > 600:
                    raise TimeoutError('SGLang startup exceeded ten minutes; inspect worker.log')
                time.sleep(0.25)
            ready = json.loads(ready_path.read_text())
            self.connection = Client(tuple(ready['address']), authkey=b'dflash-online-local')
        except BaseException:
            self._stop_process()
            raise
        self.loading_s = time.perf_counter() - started

    def _call(self, operation, **kwargs):
        if getattr(self, 'failed', False):
            raise RuntimeError('Backend blocked after a failed native operation')
        # A transport failure can leave an unanswered operation. Never reuse
        # that connection and misinterpret a late reply as the next result.
        self.failed = True
        self.connection.send({'operation': operation, **kwargs})
        if not self.connection.poll(600):
            raise TimeoutError('SGLang operation exceeded ten minutes: ' + operation)
        response = self.connection.recv()
        if not response['ok']:
            raise RuntimeError('SGLang operation failed: ' + json.dumps(response['error']))
        self.failed = False
        return response['result']

    def render(self, messages):
        from dflash.benchmark import apply_chat_template

        text = apply_chat_template(self.tokenizer, messages, self.reasoning)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(self, prompt_ids, *, max_new_tokens, request_id, session_id,
                 request_index, purpose='serving', save_to=None):
        if purpose not in {'serving', 'evaluation'}:
            raise ValueError('Specify serving or evaluation purpose')
        if not prompt_ids or max_new_tokens < 1 or len(prompt_ids) + max_new_tokens > self.context_limit:
            raise ValueError('Request exceeds full-context budget; no truncation is allowed')
        self.serial += 1
        trace_path = self.root / f'request_{self.serial:06d}.json'
        started = time.perf_counter()
        with preserve_rng():
            native = self._call('generate', prompt_ids=list(prompt_ids),
                                max_new_tokens=max_new_tokens, capture=purpose == 'serving',
                                trace_path=str(trace_path), stop_token_ids=self.stop_tokens)
        latency = time.perf_counter() - started
        observation = native['trace']
        blocks = observation['blocks']
        response, finish = validate_generation(
            list(prompt_ids), list(prompt_ids) + native['output_ids'], blocks,
            max_new_tokens=max_new_tokens, stop_tokens=self.stop_tokens)
        calls = len(blocks)
        record = {
            'record_id': request_id, 'session_id': session_id, 'request_index': request_index,
            'checkpoint_id': self.checkpoint_id, 'purpose': purpose,
            'prompt_ids': list(prompt_ids), 'response_ids': response,
            'finish_reason': finish, 'stop_token_ids': self.stop_tokens,
            'max_new_tokens': max_new_tokens,
            'metadata': {
                'request_latency_s': latency, 'native_generate_s': native['native_generate_s'],
                'verification_calls': calls,
                'accepted_draft_tokens': sum(b['accepted_draft_tokens'] for b in blocks),
                'committed_progress': len(response) - 1,
                'committed_progress_per_call': (len(response) - 1) / calls if calls else None,
                'calls_per_1000_output_tokens': 1000 * calls / len(response),
                'block_size': self.block_size, 'context_limit': self.context_limit,
                'trace': blocks, 'native_trace_path': str(trace_path),
                'cache_policy': 'idle_full_flush_before_every_request_and_publication',
                'feedback_capture_in_timed_request': purpose == 'serving',
                'trace_instrumentation_in_timed_request': True,
                'capture': observation,
                'runtime': 'pinned_SGLang_FA3',
            },
        }
        if save_to is not None:
            write_json(save_to, record)
        return record

    def prepare_feedback(self, record, path):
        """Use native captured rows, then supply the unused terminal feature row.

        The extra target forward processes exact committed IDs. Only its final
        feature is retained; all teacher rows that supply training labels remain
        those captured during native generation.
        """
        import torch

        if record['purpose'] != 'serving':
            raise ValueError('Evaluation cannot supply feedback')
        prompt, response = record['prompt_ids'], record['response_ids']
        if len(response) < 2:
            return {'status': 'ineligible_short_response', 'record_id': record['record_id'],
                    'supervised_positions': len(response), 'teacher_preparation_s': 0.0}
        path = Path(path)
        if path.exists():
            raise FileExistsError('Feedback feature file already exists')
        started = time.perf_counter()
        capture = record['metadata']['capture']
        packet = torch.load(capture['feature_path'], map_location='cpu', weights_only=False)
        combined = prompt + response
        if (capture['native_capture_validated'] is not True
                or capture['collect_features'] is not True
                or capture['covered_token_ids'] != combined[:-1]
                or capture['label_binding_scope'] != 'native_prefill_and_committed_verification'
                or capture['target_argmax_ids'] != response
                or capture['prompt_ids'] != prompt
                or packet['input_ids'] != combined[:-1]
                or packet['target_argmax_ids'] != response):
            raise ValueError('Native feature coverage differs from canonical tokens')
        terminal = self._call('teacher', input_ids=combined,
                              outputfile=str(path.with_suffix('.terminal.json')))
        last = torch.load(terminal['tensor_path'], map_location='cpu', weights_only=False)
        for evidence in (terminal, last):
            if (evidence['input_ids'] != combined
                    or evidence['positions'] != [len(combined) - 1]
                    or evidence['final_feature_logits_used_for_labels'] is not False):
                raise ValueError('Terminal feature does not belong to the final canonical token')
        for key in ('hidden_states', 'target_last_hidden_states'):
            if (packet[key].ndim != 2 or packet[key].shape[0] != len(combined) - 1
                    or last[key].shape != (1, packet[key].shape[1])
                    or packet[key].dtype != torch.bfloat16 or last[key].dtype != torch.bfloat16):
                raise ValueError('Native feature shape/dtype differs: ' + key)
        aux = torch.cat([packet['hidden_states'], last['hidden_states']], dim=0).unsqueeze(0)
        hidden = torch.cat([packet['target_last_hidden_states'], last['target_last_hidden_states']], dim=0).unsqueeze(0)
        ids = torch.tensor([combined], dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.float32)
        mask[:, len(prompt):] = 1
        receipt = {
            'record_id': record['record_id'], 'prompt_length': len(prompt),
            'response_length': len(response), 'input_ids': combined,
            'target_argmax_ids': capture['target_argmax_ids'], 'label_mismatches': [],
            'target_layer_ids': self.target_layer_ids, 'feature_shape': list(aux.shape),
            'capture_backend': 'same_frozen_SGLang_native_capture',
            'native_capture_validated': True,
            'label_binding_scope': 'native_prefill_and_committed_verification',
            'final_feature_logits_used_for_labels': False, 'response_only_loss': True,
            'terminal_feature': terminal,
        }
        payload = {'input_ids': ids, 'loss_mask': mask, 'hidden_states': aux,
                   'target_last_hidden_states': hidden, 'record_id': record['record_id'],
                   'prompt_length': len(prompt), 'label_binding': receipt}
        preparation_s = time.perf_counter() - started
        started = time.perf_counter()
        save_torch(path, payload)
        receipt.update(status='prepared', feature_path=str(path),
                       teacher_preparation_s=preparation_s,
                       feature_persistence_s=time.perf_counter() - started,
                       supervised_positions=len(response))
        write_json(path.with_suffix('.json'), receipt)
        return receipt

    def publish(self, checkpoint_path):
        started = time.perf_counter()
        with preserve_rng():
            receipt = self._call('publish', checkpoint=str(checkpoint_path))
        self.checkpoint_id = str(checkpoint_path)
        return {**receipt, 'checkpoint_id': self.checkpoint_id,
                'activation_s': time.perf_counter() - started,
                'cache_policy': 'full_flush_and_fused_KV_helper_refresh'}

    def check_weights(self, checkpoint_path):
        """Qualification-only tensor equality; never on the serving fast path."""
        return self._call('check_weights', checkpoint=str(checkpoint_path))

    def close(self):
        try:
            if self.connection is not None and self.worker.poll() is None:
                self.connection.send({'operation': 'close'})
                if self.connection.poll(10):
                    self.connection.recv()
                    self.worker.wait(timeout=20)
        finally:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
            self._stop_process()

    def _stop_process(self):
        """Clean up only this backend's isolated process group on failure."""
        if self.worker is not None:
            # Engine owns child processes, including when its parent failed.
            try:
                os.killpg(self.worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.worker.pid, signal.SIGKILL)
                self.worker.wait(timeout=10)
        self.log.close()
