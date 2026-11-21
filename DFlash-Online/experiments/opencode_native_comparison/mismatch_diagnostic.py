"""Two read-only native replays of the retained AR/F qualification mismatch.

This is diagnostic instrumentation, not a serving backend or a gate waiver.
The observer reads native target logits after the existing forward (including
CUDA graph replay), before sampling/verification. It changes no tensor or input.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import time

from online_feedback.storage import write_json
from . import ar_worker
from experiments.persistent_prefix_withholding import worker as f_worker

_AR_SCHEDULER = ar_worker.scheduler_process
_F_SCHEDULER = f_worker.scheduler_process


def logit_positions(input_positions, rows):
    """C1 prefill returns its final row; decode/verification return all rows."""
    if rows == 1 and input_positions:
        return input_positions[-1:]
    if rows == len(input_positions):
        return list(input_positions)
    raise ValueError('Unrecognized native target-logit row mapping')


def replay_check(observed, retained):
    tokens = observed['output_ids']
    stop = observed['meta_info']['finish_reason']
    expected_stop = retained['metadata'].get('native_meta_info', {}).get('finish_reason')
    result = dict(exact_tokens=tokens == retained['response_ids'],
                  output_tokens=len(tokens), native_finish=stop,
                  retained_finish=retained['finish_reason'])
    result['exact_termination'] = (stop == expected_stop if expected_stop is not None
        else stop.get('type') == 'stop' and tokens[-1] in retained['stop_token_ids'])
    if 'trace' in observed:
        result['exact_trace'] = observed['trace']['blocks'] == retained['metadata']['trace']
    result['passed'] = all(value for key, value in result.items() if key.startswith('exact_'))
    return result


def install_observer():
    """Install inside the child scheduler, leaving its native execution intact."""
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner

    root = Path(os.environ['DFLASH_MISMATCH_OUTPUT'])
    position = int(os.environ['DFLASH_MISMATCH_POSITION'])
    rivals = [int(x) for x in os.environ['DFLASH_MISMATCH_RIVALS'].split(',')]
    original = ModelRunner.forward
    counts = dict(target=0, draft=0)
    parameter_versions = None

    @functools.wraps(original)
    def observed(runner, batch, *args, **kwargs):
        nonlocal parameter_versions
        active = (root / 'ACTIVE').exists()
        if active and not runner.is_draft_worker and parameter_versions is None:
            parameter_versions = [(name, id(p), p._version, str(p.dtype), list(p.shape))
                                  for name, p in runner.model.named_parameters()]
        out = original(runner, batch, *args, **kwargs)
        if not active:
            return out
        kind = 'draft' if runner.is_draft_worker else 'target'
        counts[kind] += 1
        write_json(root / 'FORWARD_COUNTS.json', counts)
        if kind == 'draft':
            return out
        logits = out.logits_output.next_token_logits.detach()
        inputs = batch.input_ids.detach().cpu().tolist()
        positions = batch.positions.detach().cpu().tolist()
        prediction_positions = logit_positions(positions, logits.shape[0])
        finite = bool(torch.isfinite(logits).all())
        values, indices = logits.topk(2, dim=-1)
        row = dict(call=counts[kind], forward_mode=str(batch.forward_mode),
            graph=bool(out.can_run_graph), input_ids=inputs, positions=positions,
            seq_lens=batch.seq_lens.detach().cpu().tolist(),
            logit_positions=prediction_positions, logit_dtype=str(logits.dtype),
            argmax=logits.argmax(-1).cpu().tolist(), top2_ids=indices.cpu().tolist(),
            top2_values=values.float().cpu().tolist(), finite=finite,
            rival_ids=rivals, rival_logits=logits[:, rivals].float().cpu().tolist())
        with (root / 'TARGET_LOGITS.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=True) + '\n')
            stream.flush()
        if not finite:
            raise FloatingPointError('Nonfinite native target logits; diagnostic stopped')
        if position in prediction_positions:
            index = prediction_positions.index(position)
            # Full vector only at the disputed prediction; no model-sized dump.
            torch.save(logits[index].cpu().clone(), root / f'LOGITS_{counts[kind]:04d}.pt')
        current = [(name, id(p), p._version, str(p.dtype), list(p.shape))
                   for name, p in runner.model.named_parameters()]
        write_json(root / 'PARAMETER_STABILITY.json', dict(
            unchanged_identity_versions_and_layout=current == parameter_versions,
            parameter_tensors=len(current), bitwise_weight_equality_checked=False))
        if current != parameter_versions:
            raise RuntimeError('Native target parameter identity/version changed')
        return out

    ModelRunner.forward = observed


def ar_scheduler(*args, **kwargs):
    install_observer()
    return _AR_SCHEDULER(*args, **kwargs)


def f_scheduler(*args, **kwargs):
    install_observer()
    return _F_SCHEDULER(*args, **kwargs)


def run_case(plan, arm, destination):
    """One cold-cache generation, same card/settings as the retained failure."""
    from .runtime import WorkLedger, boundary, occupied_card, read

    base = Path(destination)
    base.mkdir(parents=True, exist_ok=False)
    root = Path(plan['root'])
    old = root / 'qualification/B'
    retained = read(old / 'records' / f'B_{arm}_cross_card_0_2048_0.json')
    settings = read(old / f'B_{arm}/native/engine_settings.json')
    other = read(old / 'records/B_F_cross_card_0_2048_0.json')
    ar = read(old / 'records/B_AR_cross_card_0_2048_0.json')
    first = next(i for i, (a, b) in enumerate(zip(ar['response_ids'], other['response_ids'])) if a != b)
    if first != 279 or ar['prompt_ids'] != other['prompt_ids']:
        raise ValueError('Diagnostic is bound to the retained first mismatch')
    os.environ.update(DFLASH_MISMATCH_OUTPUT=str(base),
        DFLASH_MISMATCH_POSITION=str(len(retained['prompt_ids']) + first - 1),
        DFLASH_MISMATCH_RIVALS=f'{ar["response_ids"][first]},{other["response_ids"][first]}')
    write_json(base / 'SETTINGS.json', settings)
    write_json(base / 'ADMISSION.json', boundary(plan))
    write_json(base / 'CONTRACT.json', dict(arm=arm, first_difference=first,
        inherited_execution_commit=plan['execution_commit'],
        diagnostic_commit=os.environ['DFLASH_DIAGNOSTIC_COMMIT'],
        native_generations=1, optimizer_steps=0, extra_model_calls=0,
        precision_kernel_and_graph_settings_changed=False,
        diagnostic_only=True, equality_gate_unchanged=True))
    worker = None
    began = time.perf_counter()
    with occupied_card(root, 'native_mismatch_' + arm, plan['devices']['B']):
        try:
            if arm == 'AR':
                ar_worker.scheduler_process = ar_scheduler
                worker = ar_worker.ARWorker(settings, base)
            else:
                f_worker.scheduler_process = f_scheduler
                worker = f_worker.Worker(settings, base)
            loading = time.perf_counter() - began
            write_json(base / 'READY.json', worker.status())
            (base / 'ACTIVE').touch()
            request = dict(prompt_ids=retained['prompt_ids'], max_new_tokens=2048,
                stop_token_ids=retained['stop_token_ids'], capture=False,
                trace_path=str(base / 'NATIVE.json'))
            ledger = WorkLedger(root, 'native_mismatch_' + arm, 'qualification')
            result = ledger.call({'generation': 1}, 'diagnostic_replay', worker.generate, request)
            write_json(base / 'OUTPUT.json', result)
            check = replay_check(result, retained)
            write_json(base / 'REPRODUCTION.json', check)
            write_json(base / 'WORK.json', worker.work_receipt({}))
            write_json(base / 'COST.json', dict(loading_s=loading,
                total_before_shutdown_s=time.perf_counter() - began,
                timing_is_instrumented_diagnostic_not_benchmark=True))
            if not check['passed']:
                raise ValueError('Instrumented replay does not reproduce retained evidence')
        finally:
            if worker is not None:
                worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arm', choices=('AR', 'F'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_case(json.loads(args.plan.read_text()), args.arm, args.output)


if __name__ == '__main__':
    main()
