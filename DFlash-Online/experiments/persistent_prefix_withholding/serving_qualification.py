"""Two fixed old requests, twice per physical card, at authenticated K6.

These eight calls are runtime parity checks against independently authenticated
references. They are not target-only generation or deployment speed measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

from experiments.fresh_confirmation.protocol import measurement_errors, weight_check_errors
from experiments.o_half_matched_lr.archive import validate_record
from experiments.osd_persistent.run import score_record
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.storage import _temporary_file, write_json
from .evaluation import validate_references
from .qualification_entry import HostMemorySample, install_interrupt_handlers
from .references import record_errors

SESSION = 'persistent_prefix_withholding_serving_qualification'


def gpu_memory(uuid):
    result = subprocess.run(['nvidia-smi', '--id='+uuid,
        '--query-gpu=uuid,memory.used,memory.free', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, timeout=10, check=True)
    device, used, free = [part.strip() for part in result.stdout.strip().split(',')]
    if device != uuid:
        raise ValueError('GPU memory receipt describes another physical device')
    return dict(uuid=device, used_bytes=int(used)*2**20, free_bytes=int(free)*2**20,
                scope='device-wide nvidia-smi sample, not an isolated allocation peak')


def execution_errors(record, request, reference, checkpoint, label):
    errors = validate_record(record, request, checkpoint, reference, label=label, session=SESSION)
    errors.extend(record_errors(record, request))
    return sorted(set(errors))


def evaluate(backend, requests, references, checkpoint, output, *, slot, uuid, deadline=None):
    """Exactly four attempts at most, preserving raw records before checks."""
    if len(requests) != 2 or [r['panel_ordinal'] for r in requests] != [0, 1]:
        raise ValueError('Require the two sealed old-development requests')
    validate_references(requests, references)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, previous = [], {}
    result = dict(status='running', slot=slot, physical_gpu_uuid=uuid, planned_generations=4,
        attempted_generations=0, returned_generations=0, valid_generations=0,
        invalid_generations=0, unknown_generations=0, unattempted_generations=4,
        training_steps=0, teacher_preparations=0, feedback_collections=0,
        extra_generation_warmups=0, capture=False, checkpoint=str(checkpoint), rows=rows)
    pending = None

    def persist():
        result.update(unknown_generations=int(pending is not None),
                      unattempted_generations=4-result['attempted_generations'])
        write_json(output/'RESULT.json', result)

    try:
        for repeat in range(2):
            for request in requests:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Shared qualification deadline reached before the next generation')
                label = f"q{slot}_r{repeat}_w{request['panel_ordinal']}"
                path = output/(label+'.json')
                pending = dict(label=label, repeat=repeat, request_id=request['request_id'])
                result['attempted_generations'] += 1
                persist()
                record = backend.generate(request['prompt_ids'], max_new_tokens=2048,
                    request_id=label, session_id=SESSION, request_index=request['panel_ordinal'],
                    purpose='evaluation', save_to=path)
                result['returned_generations'] += 1
                with _temporary_file(path) as temporary:
                    temporary.write_text(json.dumps(record, allow_nan=True)+'\n')
                pending = None
                try:
                    errors = execution_errors(record, request, references[request['request_id']], checkpoint, label)
                    row = score_record(record, request_id=request['request_id'], repeat=repeat, method='K6')
                    row.update(finish_reason=record['finish_reason'],
                        cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
                    errors.extend(measurement_errors(row, request, references[request['request_id']]))
                    native = sum(block['native_accepted_draft_tokens'] for block in record['metadata']['trace'])
                    signature = dict(response_ids=record['response_ids'], finish_reason=record['finish_reason'],
                        verification_calls=row['verification_calls'], native_accepted_draft_tokens=native,
                        accepted_draft_tokens=row['accepted_draft_tokens'], committed_progress=row['committed_progress'],
                        output_tokens=row['output_tokens'], native_acceptance_histogram={str(k):v for k,v in sorted(
                            Counter(block['native_accepted_draft_tokens'] for block in record['metadata']['trace']).items())})
                    prior = previous.get(request['request_id'])
                    if prior is not None and prior != signature:
                        errors.append('within_card_repeat_count_or_output_mismatch')
                    row.update(signature=signature, source_id=request['source_id'], label=label,
                        record_path=str(path), native_accepted_draft_tokens=native,
                        errors=sorted(set(errors)), valid=not errors)
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    errors = ['malformed_returned_record: '+repr(error)]
                    signature = None
                    row = dict(request_id=request['request_id'], repeat=repeat, method='K6',
                        source_id=request['source_id'], label=label, record_path=str(path),
                        errors=errors, valid=False)
                rows.append(row)
                result['valid_generations'] += int(not errors)
                result['invalid_generations'] += int(bool(errors))
                previous[request['request_id']] = signature
                pending = None
                persist()
                if errors:
                    raise ValueError('Native serving qualification failed: '+repr(errors))
        result.update(status='passed', signatures=previous)
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        raise
    finally:
        persist()
    return result


def run(plan, slot):
    if slot not in ('A', 'B'):
        raise ValueError('Physical slot must be A or B')
    selected = plan['devices'][slot]
    uuid = selected['uuid'] if isinstance(selected, dict) else selected
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('Serving worker must retain its sealed CUDA-visible GPU UUID')
    requests = json.loads(Path(plan['warmups']).read_text())['requests']
    references = json.loads(Path(plan['warmup_references']).read_text())
    validate_references(requests, references)
    root = Path(plan['root'])/'serving_qualification'/slot
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = dict(status='running', slot=slot, physical_gpu_uuid=uuid,
                  execution_commit=plan['execution_commit'], complete_generation_budget=4,
                  occupied_started_unix=time.time(), occupied_started_monotonic=began,
                  scientific_claim='native runtime parity; no deployment speedup')
    backend = None
    monitor = HostMemorySample()
    write_json(root/'STATUS.json', report)
    try:
        with monitor:
            before = time.perf_counter()
            backend = SGLangDFlashBackend(plan['target'], plan['origin_export'],
                worker_python=plan['worker_python'], output_root=root/'runtime', serving_gpu=uuid,
                context_limit=4096, block_size=16, reasoning='off',
                worker_module='experiments.persistent_prefix_withholding.worker')
            report['loading_s'] = time.perf_counter()-before
            report['graph_initialization'] = backend._call('work_receipt')
            report['gpu_memory_after_loading'] = gpu_memory(uuid)
            before = time.perf_counter()
            check = backend.check_weights(plan['origin_export'])
            errors = weight_check_errors(check, str(plan['origin_export']))
            write_json(root/'WEIGHTS_BEFORE.json', dict(check=check, errors=errors))
            report['weight_check_before_s'] = time.perf_counter()-before
            if errors:
                raise ValueError('Loaded K6 weights differ: '+repr(errors))
            report['qualification'] = evaluate(backend, requests, references, str(plan['origin_export']),
                root/'calls', slot=slot, uuid=uuid, deadline=plan.get('gpu_deadline_monotonic'))
            before = time.perf_counter()
            check = backend.check_weights(plan['origin_export'])
            errors = weight_check_errors(check, str(plan['origin_export']))
            write_json(root/'WEIGHTS_AFTER.json', dict(check=check, errors=errors))
            report['weight_check_after_s'] = time.perf_counter()-before
            if errors:
                raise ValueError('K6 weights changed during qualification')
            report['graph_and_physical_work'] = backend._call('work_receipt')
            report['gpu_memory_after_generations'] = gpu_memory(uuid)
            report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        write_json(root/'STATUS.json', report)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                report.update(status='failed', cleanup_error=repr(error))
        report.update(wall_s=time.monotonic()-began, occupied_card_s=time.monotonic()-began,
            max_main_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            sampled_process_tree_peak_rss_bytes=monitor.maximum_bytes, rss_sample_interval_s=.25,
            rss_not_continuous_peak=True)
        write_json(root/'STATUS.json', report)
    if report.get('cleanup_error'):
        raise RuntimeError('Serving cleanup failed; preserved in STATUS.json')
    return report


def compare_serving_cards(first, second):
    if first['status'] != 'passed' or second['status'] != 'passed':
        raise ValueError('Both card serving checks must pass')
    a = first.get('qualification', first)
    b = second.get('qualification', second)
    if (a['valid_generations'] != 4 or b['valid_generations'] != 4
            or a['signatures'] != b['signatures']):
        raise ValueError('Cross-card native counts, tokens or termination differ')
    return dict(status='passed', generations=8, exact_tokens_termination_and_native_counts=True,
                training_steps=0, additional_generation_warmups=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--slot', required=True, choices=('A', 'B'))
    args = parser.parse_args()
    install_interrupt_handlers()
    run(json.loads(args.plan.read_text()), args.slot)


if __name__ == '__main__':
    main()
