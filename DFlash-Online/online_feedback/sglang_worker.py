"""SGLang process boundary: one native engine, serialized online operations.

Run with the qualified SGLang interpreter. SpecForge stays in its own process
and environment; neither package set has to be modified to accommodate the other.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

from .storage import write_json


def checkpoint_layout(path, *, check_finite=False):
    """Read checkpoint structure without loading a second draft on the GPU."""
    from safetensors import safe_open

    layout = {}
    for filename in sorted(Path(path).glob('*.safetensors')):
        with safe_open(filename, framework='pt', device='cpu') as source:
            for name in source.keys():
                value = source.get_slice(name)
                if name in layout:
                    raise ValueError('Duplicate checkpoint tensor: ' + name)
                layout[name] = (value.get_shape(), value.get_dtype())
                if check_finite:
                    import torch
                    if not bool(torch.isfinite(source.get_tensor(name)).all()):
                        raise ValueError('Nonfinite checkpoint tensor: ' + name)
    if not layout:
        raise ValueError('No safetensors draft weights found')
    return layout


def semantic_config(path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path).to_dict()
    for key in ('_name_or_path', '_commit_hash', 'transformers_version',
                'architectures', 'dtype', 'torch_dtype'):
        config.pop(key, None)
    config.setdefault('draft_vocab_size', config['vocab_size'])
    # Transformers 5 exports an explicit alias for the legacy default RoPE.
    # Accept only that redundant spelling; changed theta/scaling stays visible.
    default_rope = {'rope_theta': config.get('rope_theta'), 'rope_type': 'default'}
    if config.get('rope_scaling') is None and config.get('rope_parameters') == default_rope:
        config.pop('rope_parameters')
    return config


class NativeWorker:
    def __init__(self, settings, root):
        from sglang.srt.entrypoints.engine import Engine
        from .sglang_hooks import scheduler_process

        self.root = root
        self.released = settings['speculative_draft_model_path']
        self.layout = checkpoint_layout(self.released)
        self.config = semantic_config(self.released)
        self.blocked = False
        Engine.run_scheduler_process_func = staticmethod(scheduler_process)
        self.engine = Engine(**settings)

    def status(self):
        path = self.root / 'native_status.json'
        self.engine.collective_rpc('online_status', outputfile=str(path))
        return json.loads(path.read_text())

    def generate(self, request):
        path = Path(request['trace_path'])
        self.engine.collective_rpc(
            'online_begin', outputfile=str(path),
            collect_features=request['capture'], prompt_ids=request['prompt_ids'])
        started = time.perf_counter()
        result = self.engine.generate(
            input_ids=request['prompt_ids'],
            sampling_params={'temperature': 0, 'top_p': 1, 'top_k': 1,
                             'max_new_tokens': request['max_new_tokens'],
                             'ignore_eos': False, 'stop_token_ids': request['stop_token_ids']},
        )
        native_s = time.perf_counter() - started
        self.engine.collective_rpc('online_finish', output_ids=result['output_ids'])
        trace = json.loads(path.read_text())
        return {'output_ids': result['output_ids'], 'meta_info': result['meta_info'],
                'native_generate_s': native_s, 'trace': trace}

    def teacher(self, request):
        path = Path(request['outputfile'])
        self.engine.collective_rpc('online_teacher', input_ids=request['input_ids'],
                                   outputfile=str(path))
        return json.loads(path.read_text())

    def publish(self, request):
        path = request['checkpoint']
        if checkpoint_layout(path, check_finite=True) != self.layout:
            raise ValueError('Export tensor keys, shapes or dtypes differ from released draft')
        if semantic_config(path) != self.config:
            raise ValueError('Export changes semantic draft configuration')
        # Loading mutates the native model. A failure must never permit serving
        # from a potentially partially updated draft.
        self.blocked = True
        started = time.perf_counter()
        result = self.engine.update_weights_from_disk(
            model_path=path, update_speculative_draft=True)
        if not result[0]:
            raise RuntimeError('Native draft publication failed: ' + str(result))
        native_s = time.perf_counter() - started
        status = self.status()
        if status['status'] != 'passed' or status['publication']['status'] != 'passed':
            raise RuntimeError('Native publication/helper validation failed: ' + str(status))
        self.blocked = False
        return {'native_activation_s': native_s, 'native_status': status,
                'checkpoint_structure_valid': True}

    def check_weights(self, request):
        path = self.root / 'weight_check.json'
        self.engine.collective_rpc('online_check_weights',
                                   checkpoint=request['checkpoint'], outputfile=str(path))
        return json.loads(path.read_text())

    def close(self):
        self.engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    worker = None
    try:
        started = time.perf_counter()
        worker = NativeWorker(json.loads(args.settings.read_text()), root)
        with Listener(('127.0.0.1', 0), authkey=b'dflash-online-local') as listener:
            write_json(root / 'ready.json', {
                'address': list(listener.address), 'loading_s': time.perf_counter() - started,
                'status': worker.status(),
            })
            with listener.accept() as connection:
                while True:
                    request = connection.recv()
                    if request['operation'] == 'close':
                        connection.send({'ok': True, 'result': None})
                        break
                    try:
                        if worker.blocked:
                            raise RuntimeError('Native worker blocked after publication failure')
                        operation = request['operation']
                        methods = {'generate': worker.generate, 'teacher': worker.teacher,
                                   'publish': worker.publish, 'check_weights': worker.check_weights}
                        result = methods[operation](request)
                        connection.send({'ok': True, 'result': result})
                    except Exception as error:
                        worker.blocked = True
                        failure = {'type': type(error).__name__, 'message': str(error),
                                   'traceback': traceback.format_exc(), 'operation': request['operation']}
                        write_json(root / 'last_failure.json', failure)
                        connection.send({'ok': False, 'error': failure})
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
