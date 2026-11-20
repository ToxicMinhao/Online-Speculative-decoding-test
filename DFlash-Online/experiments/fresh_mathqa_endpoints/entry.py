"""One model-owning qualification, canonical shard, or scored serving phase."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.fresh_confirmation.evaluate import append_json
from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
from experiments.persistent_prefix_withholding.task_runner import create_backend, close_backend
from experiments.persistent_prefix_withholding.references import record_errors
from .runtime import WorkLedger, boundary, occupied_card, read, schedule_rows


def install_interrupt_handlers():
    """Allow the owned worker to preserve evidence and close its engine."""
    def interrupt(signum, _frame):
        raise InterruptedError('Endpoint worker interrupted by '+signal.Signals(signum).name)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupt)


def timing_barrier(owner_pid=None):
    # Delay training-module imports until the native runtime is actually used.
    from experiments.kplus_full_gsm8k.preflight import timing_barrier as native_barrier
    return native_barrier(owner_pid)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def activate(backend, checkpoint, base, label):
    publication = backend.publish(checkpoint)
    errors = publication_errors(publication, checkpoint)
    write_json(base/(label+'_PUBLICATION.json'), dict(receipt=publication, errors=errors))
    if errors:
        raise ValueError('Native publication failed: '+repr(errors))
    reply = backend.check_weights(checkpoint)
    errors = weight_check_errors(reply, checkpoint)
    write_json(base/(label+'_WEIGHTS.json'), dict(receipt=reply, errors=errors))
    if errors:
        raise ValueError('Native checkpoint identity failed: '+repr(errors))
    return publication


def qualify(backend, plan, base, ledger, slot):
    """Two fixed old warmups × three checkpoints × two repeats on each card."""
    from .evaluation import METHODS, measure, _repeat_errors, label_for
    requests = read(plan['warmups'])['requests']
    refs = read(plan['warmup_canonicals'])
    rows, previous = [], {}
    for repeat in range(2):
        for position, method in enumerate(METHODS):
            boundary(plan)
            activate(backend, plan['checkpoints'][method], base, f'{repeat}_{method}')
            for request in requests:
                native_slot = dict(phase='scored', method=method, repeat=repeat,
                    event=58, dataset='mathqa', panel_ordinal=request['panel_ordinal'],
                    request_id=request['request_id'], activation=repeat*3+position,
                    block_ordinal=0, position=position)
                label = label_for(native_slot)
                record = ledger.call('qualification_generation', label, backend.generate,
                    request['prompt_ids'], max_new_tokens=2048, request_id=label,
                    session_id=plan['session'], request_index=request['panel_ordinal'],
                    purpose='evaluation', save_to=base/(label+'.json'))
                # Save the native return before any validity exception.
                write_json(base/(label+'.json'), record)
                row, errors = measure(record, request, plan['checkpoints'][method],
                    refs[request['request_id']], native_slot, plan['devices'][slot], dataset='mathqa')
                prior = previous.setdefault((method, request['request_id']), [])
                errors.extend(_repeat_errors(prior, row))
                row.update(errors=errors, valid_measurement=not errors,
                           validation_status='invalid' if errors else 'valid')
                append_json(base/'MEASUREMENTS.jsonl', row)
                prior.append(row)
                rows.append(row)
                if errors:
                    raise ValueError('Qualification output mismatch: '+repr(errors))
    return dict(status='passed', complete_generations=len(rows), extra_warmups=0,
                measured_latency_claim=False, physical_gpu_uuid=plan['devices'][slot])


def canonical_shard(backend, plan, base, ledger, slot):
    from .evaluation import validate_references
    requests = read(plan['panel'])['requests']
    shard = [r for r in requests if r['panel_ordinal'] % 2 == ('B' == slot)]
    producer = plan['checkpoints']['O58']
    activate(backend, producer, base, 'O58')
    references = {}
    for request in shard:
        boundary(plan)
        label = 'canonical_'+request['request_id']
        path = base/(label+'.json')
        record = ledger.call('canonical_generation', label, backend.generate,
            request['prompt_ids'], max_new_tokens=2048, request_id=label,
            session_id=plan['session']+'_canonicals', request_index=request['panel_ordinal'],
            purpose='evaluation', save_to=path)
        write_json(path, record)
        errors = record_errors(record, request)
        if record.get('checkpoint_id') != producer:
            errors.append('canonical_producer_checkpoint')
        if (record.get('record_id') != label
                or record.get('session_id') != plan['session']+'_canonicals'
                or record.get('request_index') != request['panel_ordinal']):
            errors.append('canonical_generation_identity')
        write_json(base/(label+'_VALIDATION.json'), dict(errors=errors, valid=not errors))
        if errors:
            raise ValueError('Fresh canonical invalid: '+repr(errors))
        raw = path.read_bytes()
        reference = dict(request_id=request['request_id'], source_id=request['source_id'],
            stop_token_ids=request['stop_token_ids'], max_new_tokens=request['max_new_tokens'],
            prompt_ids=request['prompt_ids'], response_ids=record['response_ids'],
            finish_reason=record['finish_reason'], verified=True,
            reference_kind='native_greedy_target_verifier', reference_checkpoint=producer,
            native_checkpoint=producer, source_record_path=str(path),
            source_record_identity=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
            producer_weight_identity=plan['checkpoint_identities']['O58'],
            **plan['canonical_provenance'])
        reference['binding_digest'] = digest(reference)
        references[request['request_id']] = reference
        write_json(base/'CANONICALS.json', references)
    validate_references(shard, references, reference_checkpoint=producer,
        producer_weight_identity=plan['checkpoint_identities']['O58'],
        required_provenance=True, canonical_provenance=plan['canonical_provenance'])
    return dict(status='passed', complete_generations=len(shard),
                untimed=True, producer=producer, canonical_digest=digest(references))


def run(plan, phase, slot):
    root = Path(plan['root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['devices'][slot]:
        raise ValueError('Worker does not own the declared physical GPU')
    base = root/'phases'/f'{phase}_{slot}'
    base.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', phase=phase, slot=slot, started_epoch=time.time(),
                  execution_commit=plan['execution_commit'])
    write_json(base/'STATUS.json', result)
    backend = None
    ledger = WorkLedger(root, f'{phase}_{slot}')
    try:
        with occupied_card(root, f'{phase}_{slot}', plan['devices'][slot]):
            try:
                boundary(plan)
                backend = create_backend(plan, base, plan['checkpoints']['O58'], plan['devices'][slot])
                result['loading_s'] = backend.loading_s
                if phase == 'qualification':
                    result['result'] = qualify(backend, plan, base, ledger, slot)
                elif phase == 'canonical':
                    result['result'] = canonical_shard(backend, plan, base, ledger, slot)
                elif phase == 'scored' and slot == 'A':
                    from .evaluation import evaluate_final, execution_slots, label_for
                    requests = read(plan['panel'])['requests']
                    warmups = read(plan['warmups'])['requests']
                    schedule = schedule_rows(plan['schedule'])
                    categories = {label_for(s):s['phase']+'_generation'
                        for s, _ in execution_slots(schedule, requests, warmups, dataset='mathqa')}
                    original = backend.generate
                    def generate(*args, **kwargs):
                        label = kwargs['request_id']
                        return ledger.call(categories[label], label, original, *args, **kwargs)
                    backend.generate = generate
                    def quiet():
                        resources = boundary(plan)
                        check = timing_barrier(backend.worker.pid)
                        return dict(status='passed', physical_gpu_uuid=plan['devices']['A'],
                            both_cards_checked=True, other_gpu_idle=True, host_quiet=True,
                            storage_admitted=True, time_admitted=True, resources=resources, quiet=check)
                    result['result'] = evaluate_final(backend, requests, plan['checkpoints'], root/'evaluation',
                        dataset='mathqa', canonicals=read(root/'CANONICALS.json'),
                        warmup_requests=warmups, warmup_canonicals=read(plan['warmup_canonicals']),
                        schedule=schedule, reference_checkpoints=plan['reference_checkpoints'],
                        physical_gpu_uuid=plan['devices']['A'], boundary_check=quiet,
                        execution_commit=plan['execution_commit'],
                        checkpoint_identities=plan['checkpoint_identities'],
                        canonical_provenance=plan['canonical_provenance'])
                else:
                    raise ValueError('Unknown phase or scored GPU')
            finally:
                if backend is not None:
                    closing, backend = backend, None
                    close_backend(closing, base)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            close_backend(backend, base)
        result['wall_s'] = time.time()-result['started_epoch']
        write_json(base/'STATUS.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--phase', choices=['qualification','canonical','scored'], required=True)
    parser.add_argument('--slot', choices=['A','B'], required=True)
    args = parser.parse_args()
    install_interrupt_handlers()
    run(read(args.plan), args.phase, args.slot)
