"""Native generation, response-only teacher replay, and draft publication.

The target stays frozen. Generation uses the author's fresh per-request caches;
feedback replays the exact returned tokens and is timed separately from serving.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from .rng import preserve_rng
from .storage import save_torch, write_json


_PUBLICATION_METADATA_KEYS = {
    '_name_or_path', '_commit_hash', 'transformers_version', 'architectures',
    'dtype', 'torch_dtype',
}


def _publication_config(model) -> dict:
    """Compare forward settings while allowing export provenance to change."""
    config = model.config.to_dict()
    # SpecForge materializes this omitted Qwen3 DFlash default during loading.
    # Explicit values (including null) and other model families remain exact.
    if (config.get('model_type') == 'qwen3'
            and isinstance(config.get('dflash_config'), dict)
            and type(config.get('vocab_size')) is int
            and config['vocab_size'] > 0
            and 'draft_vocab_size' not in config):
        config['draft_vocab_size'] = config['vocab_size']
    return {
        key: value for key, value in config.items()
        if key not in _PUBLICATION_METADATA_KEYS
    }


def validate_publication(candidate, current, loading_info: dict, *, tensor_isfinite) -> dict:
    """Reject incompatible or partial exports before replacing the active draft."""
    for field in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if loading_info[field]:
            raise ValueError('strict checkpoint loading failed: ' + field)
    for field in ('block_size', 'mask_token_id'):
        if getattr(candidate, field) != getattr(current, field):
            raise ValueError('published draft changes ' + field)
    if list(candidate.target_layer_ids) != list(current.target_layer_ids):
        raise ValueError('published draft changes target-layer feature contract')

    # JSON comparison preserves distinctions such as true versus 1.
    new_config = json.dumps(_publication_config(candidate), sort_keys=True, allow_nan=False)
    old_config = json.dumps(_publication_config(current), sort_keys=True, allow_nan=False)
    if new_config != old_config:
        raise ValueError('published draft changes semantic model configuration')

    new_state = candidate.state_dict()
    old_state = current.state_dict()
    if set(new_state) != set(old_state):
        raise ValueError('published draft changes state keys')
    for name, tensor in new_state.items():
        expected = old_state[name]
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise ValueError('published draft changes tensor shape/dtype: ' + name)
        if not tensor_isfinite(tensor):
            raise FloatingPointError('nonfinite published tensor: ' + name)
    return {
        'strict_loading': True,
        'tensor_count': len(new_state),
        'shape_dtype_match': True,
        'all_tensors_finite': True,
        'semantic_config_match': True,
    }


def validate_generation(prompt, combined, trace, *, max_new_tokens, stop_tokens):
    """Check that native output and verification blocks describe the same answer."""
    if combined[:len(prompt)] != prompt:
        raise RuntimeError('generation changed canonical prompt')
    response = combined[len(prompt):]
    if not response or len(response) > max_new_tokens:
        raise RuntimeError('invalid generated response length')
    if any(token in stop_tokens for token in response[:-1]):
        raise RuntimeError('generation continued after first EOS')
    finish_reason = 'eos' if response[-1] in stop_tokens else 'length'
    if finish_reason == 'length' and len(response) != max_new_tokens:
        raise RuntimeError('generation ended before EOS or cap')

    offset = 0
    for block in trace:
        if block['output_start'] != offset or block['anchor_token'] != response[offset]:
            raise RuntimeError('block start/anchor mismatch')
        progress = block['committed_progress']
        if block['committed_tokens'] != response[offset + 1:offset + 1 + progress]:
            raise RuntimeError('committed tokens differ from canonical response')
        offset += progress
    if offset != len(response) - 1:
        raise RuntimeError('verification progress does not reconstruct answer')
    return response, finish_reason


class NativeDFlashBackend:
    def __init__(self, target_path: str, draft_path: str, *, device='cuda:0',
                 context_limit=2048, block_size=16, reasoning='off'):
        import torch
        from dflash.benchmark import load_transformers_models, stop_token_ids

        self.torch = torch
        self.device = torch.device(device)
        self.context_limit = int(context_limit)
        self.block_size = int(block_size)
        self.reasoning = reasoning
        self.released_path = Path(draft_path)
        started = time.perf_counter()
        with preserve_rng():
            self.target, self.draft, self.tokenizer = load_transformers_models(
                target_path, draft_path, self.device)
        self.target.eval().requires_grad_(False)
        self.draft.eval().requires_grad_(False)
        self.stop_tokens = stop_token_ids(self.target, self.tokenizer)
        if self.draft.block_size != self.block_size:
            raise ValueError('configured block size differs from released DFlash')
        self.checkpoint_id = str(self.released_path)
        torch.cuda.synchronize(self.device)
        self.loading_s = time.perf_counter() - started

    def render(self, messages: list[dict]) -> list[int]:
        """Tokenize the chat prompt once; these exact ids remain canonical."""
        from dflash.benchmark import apply_chat_template

        text = apply_chat_template(self.tokenizer, messages, self.reasoning)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(self, prompt_ids: list[int], *, max_new_tokens: int,
                 request_id: str, session_id: str, request_index: int,
                 purpose='serving', save_to: Path | None = None) -> dict:
        from dflash.model import dflash_generate

        torch = self.torch
        if purpose not in {'serving', 'evaluation'}:
            raise ValueError('purpose must explicitly isolate serving and evaluation')
        if (not prompt_ids or max_new_tokens < 1
                or len(prompt_ids) + max_new_tokens > self.context_limit):
            raise ValueError('request violates explicit full-context budget')

        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        trace = []
        self.draft.eval()
        self.target.eval()
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with preserve_rng():
            # dflash_generate constructs both caches inside this invocation.
            result = dflash_generate(
                self.draft, self.target, ids, max_new_tokens, self.stop_tokens,
                temperature=0, block_size=self.block_size, return_stats=True,
                trace_callback=trace.append)
        torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started

        response, finish_reason = validate_generation(
            prompt_ids, result.output_ids[0].tolist(), trace,
            max_new_tokens=max_new_tokens, stop_tokens=self.stop_tokens)
        calls = len(trace)
        progress = len(response) - 1
        record = {
            'record_id': request_id,
            'session_id': session_id,
            'request_index': request_index,
            'checkpoint_id': self.checkpoint_id,
            'purpose': purpose,
            'prompt_ids': prompt_ids,
            'response_ids': response,
            'finish_reason': finish_reason,
            'stop_token_ids': self.stop_tokens,
            'max_new_tokens': max_new_tokens,
            'metadata': {
                'request_latency_s': latency,
                'verification_calls': calls,
                'accepted_draft_tokens': sum(block['accepted_draft_tokens'] for block in trace),
                'committed_progress': progress,
                'committed_progress_per_call': progress / calls if calls else None,
                'calls_per_1000_output_tokens': 1000 * calls / len(response),
                'time_to_first_token_s': result.time_to_first_token,
                'block_size': self.block_size,
                'context_limit': self.context_limit,
                'trace': trace,
                'cache_policy': 'new_target_and_draft_caches_per_request',
                'feedback_capture_in_timed_request': False,
                'trace_instrumentation_in_timed_request': True,
            },
        }
        if save_to is not None:
            write_json(save_to, record)
        return record

    def prepare_feedback(self, record: dict, path: Path) -> dict:
        """Replay the committed response and save target features for training."""
        from .native_replay import replay_frozen_target

        torch = self.torch
        path = Path(path)
        if record['purpose'] != 'serving':
            raise ValueError('benchmark replays cannot supply training feedback')
        prompt = record['prompt_ids']
        response = record['response_ids']
        if len(response) < 2:
            return {
                'status': 'ineligible_short_response',
                'record_id': record['record_id'],
                'supervised_positions': len(response),
                'teacher_preparation_s': 0.0,
            }
        combined = prompt + response
        if len(combined) > self.context_limit:
            raise ValueError('full feedback exceeds context limit')
        if path.exists():
            raise FileExistsError('immutable feedback features already exist')

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        ids = torch.tensor([combined], dtype=torch.long, device=self.device)
        with preserve_rng(), torch.inference_mode():
            features, last_hidden, teacher_ids, replay = replay_frozen_target(
                self.target, self.draft.target_layer_ids, record, self.device,
                path.with_suffix('.native_replay_failure.json'))
            receipt = {
                'record_id': record['record_id'],
                'prompt_length': len(prompt),
                'response_length': len(response),
                'target_argmax_ids': teacher_ids,
                'label_mismatches': [],
                'target_layer_ids': list(self.draft.target_layer_ids),
                'feature_shape': list(features.shape),
                'capture_backend': 'same_frozen_HF_target_saved_native_block_replay',
                'native_replay': replay,
                'label_binding_scope': replay['label_binding_scope'],
                'response_only_loss': True,
            }

            mask = torch.zeros_like(ids, dtype=torch.float32)
            mask[:, len(prompt):] = 1
            payload = {
                'input_ids': ids.cpu(),
                'loss_mask': mask.cpu(),
                'hidden_states': features.cpu(),
                'target_last_hidden_states': last_hidden.cpu(),
                'record_id': record['record_id'],
                'prompt_length': len(prompt),
                'label_binding': receipt,
            }
            del features, last_hidden
        torch.cuda.synchronize(self.device)
        preparation_s = time.perf_counter() - started

        started = time.perf_counter()
        save_torch(path, payload)
        receipt.update(
            status='prepared', feature_path=str(path),
            teacher_preparation_s=preparation_s,
            feature_persistence_s=time.perf_counter() - started,
            supervised_positions=len(response))
        write_json(path.with_suffix('.json'), receipt)
        return receipt

    def publish(self, checkpoint_path: str | Path) -> dict:
        """Load and validate an export, then replace the active serving draft."""
        from dflash.model import DFlashDraftModel

        torch = self.torch
        path = Path(checkpoint_path)
        started = time.perf_counter()
        with preserve_rng():
            candidate, loading_info = DFlashDraftModel.from_pretrained(
                str(path), dtype=torch.bfloat16, attn_implementation='sdpa',
                output_loading_info=True)
            candidate = candidate.to(self.device).eval()
        candidate.requires_grad_(False)
        validation = validate_publication(
            candidate, self.draft, loading_info,
            tensor_isfinite=lambda tensor: bool(torch.isfinite(tensor).all()))
        torch.cuda.synchronize(self.device)
        activation_s = time.perf_counter() - started

        # Every fallible loading/validation operation precedes this assignment.
        self.draft = candidate
        self.checkpoint_id = str(path)
        return {
            'checkpoint_id': self.checkpoint_id,
            'checkpoint_path': str(path),
            'activation_s': activation_s,
            'native_activation_s': activation_s,
            'publication_validation': validation,
            'cache_maintenance_s': 0.0,
            'cache_policy': 'caches_are_request_local_not_reused_after_publication',
        }
