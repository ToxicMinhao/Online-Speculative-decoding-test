"""Five frozen endpoints, one AR reference, and no learning during timing."""
from collections import defaultdict
from pathlib import Path
import json
import time

from online_feedback.storage import write_json
from experiments.kw_v1.evaluate import append_json
from experiments.o_half_matched_lr.archive import validate_record
from experiments.osd_persistent.run import score_record
from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
from .contract import METHODS, digest
from .schedule import evaluation_schedule, schedule_summary
from .validity import validate_ar_output, validate_measurement

SESSION = 'opencode_final58'
COUNT_FIELDS = ('output_tokens', 'verification_calls', 'accepted_draft_tokens',
                'native_accepted_draft_tokens', 'committed_progress')


def check_reference(request, reference, provenance):
    if (reference['reference_kind'] != 'target_only_AR' or not reference['verified']
            or reference['provenance'] != provenance
            or reference['binding_digest'] != digest({k:v for k,v in reference.items() if k!='binding_digest'})):
        raise ValueError('AR reference identity changed')
    for key in ('request_id', 'prompt_ids', 'stop_token_ids', 'max_new_tokens'):
        if reference[key] != request[key]:
            raise ValueError('AR reference input differs: '+key)


def label_for(slot):
    return f"a{slot['activation']:03d}_{slot['phase']}_{slot['request_ordinal']:04d}_{slot['method']}"


def measure(record, request, checkpoint, canonical, slot):
    """Validate both protocols; never invent speculative metrics for target-only AR."""
    errors = []
    method, meta = slot['method'], record['metadata']
    if method == 'AR':
        try:
            _, finish = validate_ar_output(record['prompt_ids'], record['response_ids'],
                meta['native_meta_info'], max_new_tokens=request['max_new_tokens'],
                stop_token_ids=request['stop_token_ids'])
            validate_measurement(record, arm=method)
            if finish != record['finish_reason']:
                errors.append('AR_termination')
        except (KeyError, ValueError, TypeError) as error:
            errors.append('AR_structure:'+repr(error))
        for key in ('prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens'):
            if record[key] != canonical[key]:
                errors.append('canonical_'+key)
        if (record['record_id'] != label_for(slot) or record['session_id'] != SESSION
                or record['request_index'] != request['request_index']
                or record['checkpoint_id'] != checkpoint or record['purpose'] != 'evaluation'
                or meta['speculation_disabled'] is not True
                or meta['feedback_capture_in_timed_request'] is not False
                or meta['context_limit'] != 4096 or meta['physical_context_length'] != 4112
                or meta['runtime'] != 'pinned_SGLang_FA3'
                or meta['cache_policy'] != 'idle_full_flush_before_every_request_and_publication'):
            errors.append('AR_execution_contract')
    else:
        native_request = dict(request, panel_ordinal=request['request_index'])
        errors.extend(validate_record(record, native_request, checkpoint, canonical,
            label=label_for(slot), session=SESSION))
        try:
            validate_measurement(record, arm=method)
        except (ValueError, KeyError, TypeError) as error:
            errors.append('native_measurement:'+repr(error))
    row = score_record(record, request_id=request['request_id'], repeat=slot['repeat'], method=method)
    row.update(**{k:v for k,v in slot.items() if k not in ('method','repeat')},
        finish_reason=record['finish_reason'],
        source_id=request['source_id'],
        cache_maintenance_s=(meta['cache_maintenance_s'] if method=='AR' else meta['capture']['cache_maintenance_s']),
        native_accepted_draft_tokens=None if method=='AR' else sum(
            block['native_accepted_draft_tokens'] for block in meta['trace']))
    if method!='AR' and row['native_accepted_draft_tokens'] != row['committed_progress']-row['verification_calls']:
        errors.append('native_acceptance_accounting')
    return row, errors


def execute(plan, root, requests, warmups, canonicals, warmup_canonicals,
            checkpoints, *, activate, boundary, ledger):
    """Activation callback owns AR/DF lifecycle; all work is reserved before calls."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    schedule = evaluation_schedule(requests)
    balance = schedule_summary(schedule)
    if set(checkpoints) != set(METHODS):
        raise ValueError('All five fixed endpoints are required')
    if len(warmups)!=2 or len({r['request_id'] for r in requests+warmups})!=1026:
        raise ValueError('Require disjoint panel and exactly two warmup requests')
    for group, references in ((requests,canonicals),(warmups,warmup_canonicals)):
        if set(references)!={r['request_id'] for r in group}:
            raise ValueError('Incomplete canonical coverage')
        for request in group:
            check_reference(request, references[request['request_id']], plan['canonical_provenance'])
    write_json(root/'PROTOCOL.json', dict(schedule=schedule, balance=balance,
        checkpoints=checkpoints, execution_commit=plan['execution_commit'],
        gpu=plan['devices']['A'], canonical_provenance=plan['canonical_provenance'],
        initial_prefill_token_excluded_from_committed_progress=True,
        native_and_stop_clipped_acceptance_separate=True))
    write_json(root/'REFERENCES.json', dict(panel=canonicals,warmup=warmup_canonicals))
    write_json(root/'INPUTS.json', dict(panel=requests,warmup=warmups))
    by_id = {row['request_id']:row for row in requests}
    grouped, rows, checked = defaultdict(list), [], set()
    progress = dict(status='running', planned=15600, attempted=0, returned=0,
        completed_valid=0, invalid=0, pending=None, automatic_retry=False)
    started = time.perf_counter()

    def persist():
        progress.update(unattempted=progress['planned']-progress['attempted'],
            unknown=int(progress['pending'] is not None), wall_s=time.perf_counter()-started)
        write_json(root/'STATUS.json', progress)

    try:
        for entry in schedule:
            append_json(root/'BOUNDARIES.jsonl', boundary(entry))
            method, checkpoint = entry['method'], checkpoints[entry['method']]
            backend, activation = activate(method, checkpoint, entry)
            append_json(root/'ACTIVATIONS.jsonl', {**entry,**activation})
            if method!='AR' and activation.get('publication') is not None:
                errors = publication_errors(activation['publication'], checkpoint)
                if errors:
                    raise ValueError('Native publication failed: '+repr(errors))
            if method!='AR' and method not in checked:
                receipt = backend.check_weights(checkpoint)
                errors = weight_check_errors(receipt, checkpoint)
                write_json(root/f'WEIGHTS_{method}.json', dict(receipt=receipt,errors=errors))
                if errors:
                    raise ValueError('Loaded endpoint differs: '+repr(errors))
                checked.add(method)
            slots = [('warmup',r) for r in warmups]+[('scored',by_id[key]) for key in entry['request_ids']]
            for phase, request in slots:
                slot = dict(activation=entry['activation'],repeat=entry['repeat'],
                    group=entry['group'],method=method,phase=phase,
                    request_ordinal=request['request_index'])
                path = root/'raw'/(label_for(slot)+'.json')
                progress['attempted'] += 1
                progress['pending'] = slot
                persist()
                record = ledger.call({phase+'_generation':1},label_for(slot),backend.generate,
                    request['prompt_ids'],max_new_tokens=2048,request_id=label_for(slot),
                    session_id=SESSION,request_index=request['request_index'],purpose='evaluation',save_to=path)
                write_json(path,record)
                progress['returned'] += 1
                reference = (warmup_canonicals if phase=='warmup' else canonicals)[request['request_id']]
                progress['pending'] = None
                persist()
                try:
                    row, errors = measure(record,request,checkpoint,reference,slot)
                except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
                    row = dict(slot,request_id=request['request_id'],checkpoint_id=checkpoint)
                    errors = ['malformed_returned_record:'+repr(error)]
                row['physical_gpu_uuid'] = plan['devices']['A']
                previous = grouped[phase,method,request['request_id']]
                if not errors and any(tuple(old[key] for key in COUNT_FIELDS)!=tuple(row[key] for key in COUNT_FIELDS) for old in previous):
                    errors.append('repeat_count_mismatch')
                row.update(valid_measurement=not errors,errors=sorted(set(errors)),raw_path=str(path))
                previous.append(row)
                rows.append(row)
                append_json(root/'MEASUREMENTS.jsonl',row)
                progress['pending'] = None
                progress['invalid' if errors else 'completed_valid'] += 1
                persist()
                if errors:
                    raise ValueError('Final native validity failure: '+repr(errors))
        append_json(root/'BOUNDARIES.jsonl',boundary({'activation':'finished'}))
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed',error=repr(error))
        raise
    finally:
        persist()
    return progress
