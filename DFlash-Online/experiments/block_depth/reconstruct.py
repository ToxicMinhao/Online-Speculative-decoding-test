"""Recover fit feedback from retained native blocks without another generation.

The first sixteen READOUT completions were timed with capture disabled. Replay
their exact target prefill and verification inputs, including rejected proposals,
and compare every posterior token before retaining committed feature rows.
"""
from copy import deepcopy
from pathlib import Path
import time
import traceback

from online_feedback.native import validate_generation
from online_feedback.storage import write_json


def validate_source(record):
    prompt, response = record['prompt_ids'], record['response_ids']
    trace = record['metadata']['trace']
    _, finish = validate_generation(prompt, prompt + response, trace,
        max_new_tokens=record['max_new_tokens'], stop_tokens=record['stop_token_ids'])
    if finish != record['finish_reason'] or len(prompt) > 8192:
        raise ValueError('Retained qualification input/termination differs')
    return prompt, response, trace


def replay_request(request_type, sampling_type, record, model_config):
    """Supply the model vocabulary boundary used by native EOS validation."""
    sampling = sampling_type(temperature=0, top_p=1, top_k=1,
        max_new_tokens=record['max_new_tokens'],
        stop_token_ids=record['stop_token_ids'], ignore_eos=False)
    # Native scheduling normally normalizes absent string/regex stops to empty
    # lists before Req.check_finished. This exact-token replay has no text stops.
    sampling.normalize(None)
    return request_type(rid='depth_feedback_reconstruction', origin_input_text='',
        origin_input_ids=record['prompt_ids'], vocab_size=model_config.vocab_size,
        eos_token_ids=set(record['stop_token_ids']),
        sampling_params=sampling)


def reconstruct(self, record, outputfile):
    import torch
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
    from online_feedback.sglang_hooks import RequestCapture, _feature_pair, flush_idle

    prompt, response, blocks = validate_source(record)
    destination = Path(outputfile)
    if destination.exists():
        raise FileExistsError('Retain the previous reconstruction receipt')
    receipt = dict(status='running', record_id=record['record_id'],
        new_generations=0, prefill_forwards=0, verification_forwards=0,
        posterior_positions_checked=0, active_forward=None)
    started = time.perf_counter()
    runner = self.tp_worker.model_runner
    capture = RequestCapture(prompt, destination.with_suffix('.capture.json'), True)
    capture.cache_maintenance_s = flush_idle(self)
    try:
        req = replay_request(Req, SamplingParams, record, runner.model_config)
        req.fill_ids = list(prompt)
        req.extend_input_len = len(prompt)
        cache = RadixCache(CacheInitParams(disable=False,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            page_size=runner.server_args.page_size))
        batch = ScheduleBatch.init_new(reqs=[req],
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=cache, model_config=runner.model_config,
            enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.DFLASH)
        batch.prepare_for_extend()
        model_batch = batch.get_model_worker_batch()
        model_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        receipt['active_forward'] = 'prefill'
        write_json(destination, receipt)
        with torch.inference_mode():
            result = self.tp_worker.forward_batch_generation(model_batch)
        receipt.update(prefill_forwards=1, active_forward=None)
        first = result.next_token_ids.reshape(-1).cpu().tolist()
        if first != response[:1]:
            raise RuntimeError('Retained native prefill prediction differs')
        aux, final = _feature_pair(result.logits_output, len(prompt))
        capture.prefill(0, prompt, first[0], aux=aux, final=final, block_size=16)
        req.output_ids.extend(first)
        req.check_finished()
        _, custom_mask = resolve_dflash_verify_mask_policy(runner.attn_backend)
        for index, block in enumerate(blocks):
            before = list(req.output_ids)
            anchor = len(prompt) + len(before) - 1
            candidates = [block['anchor_token']] + block['draft_tokens']
            if (block['positions'] != list(range(anchor, anchor + 16))
                    or candidates[0] != before[-1] or block['output_start'] != len(before) - 1):
                raise ValueError('Retained native block conditioning differs')
            verify = DFlashVerifyInput(
                draft_token=torch.tensor(candidates, device=runner.device, dtype=torch.int64),
                positions=torch.arange(anchor, anchor + 16, device=runner.device, dtype=torch.int64),
                draft_token_num=16)
            verify.prepare_for_verify(batch, runner.server_args.page_size,
                                      build_custom_mask=custom_mask)
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            batch.spec_info = verify
            batch.return_hidden_states = False
            receipt['active_forward'] = dict(kind='verification', block=index)
            write_json(destination, receipt)
            with torch.inference_mode():
                result = self.tp_worker.forward_batch_generation(batch.get_model_worker_batch(), is_verify=True)
            receipt['verification_forwards'] += 1
            receipt['active_forward'] = None
            posterior = result.logits_output.next_token_logits.argmax(-1).cpu().tolist()
            if posterior != block['target_argmax']:
                receipt['mismatch'] = dict(block=index, expected=block['target_argmax'], actual=posterior)
                raise RuntimeError('Retained native full verification posterior differs')
            receipt['posterior_positions_checked'] += 16
            aux, final = _feature_pair(result.logits_output, 16)
            with torch.inference_mode():
                _, lengths, _, accepted = verify.verify(batch=batch,
                    logits_output=result.logits_output, page_size=runner.server_args.page_size)
            count = int(lengths[0].item())
            if (count != block['committed_progress'] or req.output_ids != response[:len(before) + count]):
                raise RuntimeError('Native reconstruction committed a different continuation')
            capture.verification(before=before, after=list(req.output_ids),
                candidates=candidates, positions=block['positions'], posterior=posterior,
                commit_rows=count, native_accepted=accepted[0], finished=req.finished(),
                aux=aux, final=final)
            batch.forward_mode = ForwardMode.DECODE
        observation = capture.finish(response)
        restored = deepcopy(record)
        restored['purpose'] = 'serving'
        restored['metadata']['capture'] = observation
        restored['metadata']['reconstructed_feedback'] = True
        restored['metadata']['reconstruction_receipt'] = str(destination)
        receipt.update(status='passed', record=restored)
        return receipt
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['cache_cleanup_s'] = flush_idle(self)
        receipt['wall_s'] = time.perf_counter() - started
        write_json(destination, receipt)
