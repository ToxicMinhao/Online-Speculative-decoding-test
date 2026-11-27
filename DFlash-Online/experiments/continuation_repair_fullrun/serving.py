"""Four fixed native endpoints, one fresh F reference, and resumable slots."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
from experiments.kw_v1.evaluate import append_json
from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.contract import validate_requests
from experiments.opencode_native_comparison.evaluation import (
    COUNT_FIELDS, SESSION, check_reference, label_for, measure,
)
from experiments.opencode_native_comparison.serving_gate import make_reference
from online_feedback.storage import write_json

METHODS = ('BASE', 'TAIL', 'AREPO', 'F')
ORDERS = (('BASE', 'TAIL', 'F', 'AREPO'), ('TAIL', 'AREPO', 'BASE', 'F'),
          ('AREPO', 'F', 'TAIL', 'BASE'), ('F', 'BASE', 'AREPO', 'TAIL'))
REQUESTS, REPEATS, GROUP_SIZE = 1024, 3, 256


def read(path):
    return json.loads(Path(path).read_text())


def measured_return(record,request,checkpoint,canonical,slot):
    """A returned malformed record is invalid evidence, not an unknown call."""
    try:
        return measure(record,request,checkpoint,canonical,slot)
    except (KeyError,TypeError,ValueError,IndexError,RuntimeError) as error:
        return dict(slot,request_id=request['request_id'],checkpoint_id=checkpoint), [
            'malformed_returned_record:'+repr(error)]


def schedule(requests):
    validate_requests(requests, count=REQUESTS, role='evaluation')
    entries = []
    for repeat in range(REPEATS):
        for group in range(4):
            for position, method in enumerate(ORDERS[(repeat + group) % 4]):
                entries.append(dict(activation=len(entries), repeat=repeat, group=group,
                    position=position, method=method, warmups=2,
                    request_ids=[r['request_id'] for r in requests[group*GROUP_SIZE:(group+1)*GROUP_SIZE]]))
    return entries


def schedule_summary(entries):
    positions, precedences, coverage = Counter(), Counter(), Counter()
    if len(entries) != 48:
        raise ValueError('The fixed four-arm schedule has48 activations')
    for start in range(0,48,4):
        repeat, group = divmod(start//4,4)
        batch = entries[start:start+4]
        if tuple(r['method'] for r in batch) != ORDERS[(repeat+group)%4]:
            raise ValueError('Balanced endpoint order changed')
        ids = batch[0]['request_ids']
        if len(ids) != GROUP_SIZE or len(set(ids)) != GROUP_SIZE:
            raise ValueError('A timing group requires256 unique requests')
        for position, row in enumerate(batch):
            if (row['activation'],row['repeat'],row['group'],row['position'],row['warmups'],row['request_ids']) != (start+position,repeat,group,position,2,ids):
                raise ValueError('Timing activation identity or request order changed')
            positions[row['method'],position] += 1
            coverage.update((key,repeat,row['method']) for key in ids)
            for later in batch[position+1:]:
                precedences[row['method'],later['method']] += 1
    if (len(coverage) != 12288 or set(coverage.values()) != {1}
            or any(positions[m,p] != 3 for m in METHODS for p in range(4))
            or any(precedences[a,b] != 6 for a in METHODS for b in METHODS if a!=b)):
        raise ValueError('Request coverage or six-six endpoint precedence changed')
    return dict(activations=48, scored_generations=12288, warmup_generations=96,
                groups=12, requests=1024, repeats=3, group_size=256,
                each_endpoint_each_position=3, each_pair_precedence=6)


def slots_for(entry, requests, warmups):
    by_id = {row['request_id']:row for row in requests}
    for phase, request in [('warmup',r) for r in warmups]+[('scored',by_id[k]) for k in entry['request_ids']]:
        yield dict(activation=entry['activation'],repeat=entry['repeat'],group=entry['group'],
                   method=entry['method'],phase=phase,request_ordinal=request['request_index']), request


def _persist_contract(path, value, resume):
    if path.exists():
        if not resume or read(path) != value:
            raise ValueError('Cannot replace the sealed serving contract: '+path.name)
    else:
        write_json(path,value)


def _activation(activate, checkpoint, method, entry, root, *, check_weights=True):
    backend, receipt = activate(method, checkpoint, entry)
    errors = publication_errors(receipt['publication'], checkpoint)
    if check_weights:
        weight = backend.check_weights(checkpoint)
        errors += weight_check_errors(weight,checkpoint)
        write_json(root/f'WEIGHTS_{method}.json',dict(receipt=weight,errors=errors))
    append_json(root/'ACTIVATIONS.jsonl',dict(**entry,**receipt,weight_errors=errors,loaded_weights_checked=check_weights))
    if errors:
        raise ValueError('Native checkpoint publication failed: '+repr(errors))
    return backend


def execute(plan, root, requests, warmups, canonicals, warmup_canonicals, checkpoints,
            *, activate, boundary, ledger, resume=False):
    """Resume validated slots only; uncertain native calls require reconciliation.

    A partial activation receives two explicitly charged retry warmups. They are
    outside the96 planned warmup slots and never enter scored timing reductions.
    """
    root=Path(root); root.mkdir(parents=True,exist_ok=resume)
    if resume and (root/'MEASUREMENTS.jsonl').exists() and not (root/'PROTOCOL.json').exists():
        raise ValueError('Cannot reconstruct a missing historical protocol around existing measurements')
    entries=schedule(requests); balance=schedule_summary(entries)
    validate_requests(warmups,count=2,role='warmup')
    if (set(checkpoints)!=set(METHODS) or len(set(map(str,checkpoints.values())))!=4
            or len({tuple(r['prompt_ids']) for r in requests+warmups})!=1026
            or len({r['request_id'] for r in requests+warmups})!=1026):
        raise ValueError('Require four named exports and disjoint panel/warmups')
    for group,refs in ((requests,canonicals),(warmups,warmup_canonicals)):
        if set(refs)!={r['request_id'] for r in group}:
            raise ValueError('Incomplete native frozen-F references')
        for request in group:
            check_reference(request,refs[request['request_id']],plan['canonical_provenance'],method='F')
    contract=dict(schema='continuation_repair_fullrun_serving_v1',schedule=entries,balance=balance,
        checkpoints=checkpoints,execution_commit=plan['execution_commit'],gpu=plan['devices']['A'],
        canonical_provenance=plan['canonical_provenance'],native_record_session=SESSION,
        canonical_policy='All four endpoints match fresh native frozen F; no target-only AR claim',
        feedback_capture=False,initial_prefill_token_excluded_from_committed_progress=True,
        native_and_stop_clipped_acceptance_separate=True)
    for name,value in [('PROTOCOL',contract),('INPUTS',dict(panel=requests,warmup=warmups)),
                       ('REFERENCES',dict(panel=canonicals,warmup=warmup_canonicals))]:
        _persist_contract(root/(name+'.json'),value,resume)
    planned=[(slot,request) for entry in entries for slot,request in slots_for(entry,requests,warmups)]
    rows=[json.loads(line) for line in (root/'MEASUREMENTS.jsonl').read_text().splitlines()] if (root/'MEASUREMENTS.jsonl').exists() else []
    old=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
    if old.get('pending') is not None or old.get('invalid',0):
        raise ValueError('Preserved unknown/invalid native execution requires explicit reconciliation')
    if len(rows)>len(planned):
        raise ValueError('More rows than the sealed native schedule')
    grouped=defaultdict(list)

    def validate(record,slot,request):
        reference=(warmup_canonicals if slot['phase'] in ('warmup','resume_warmup') else canonicals)[request['request_id']]
        row,errors=measured_return(record,request,checkpoints[slot['method']],reference,slot)
        if not errors:
            for previous in grouped[slot['phase'],slot['method'],request['request_id']]:
                if tuple(previous[k] for k in (*COUNT_FIELDS,'finish_reason'))!=tuple(row[k] for k in (*COUNT_FIELDS,'finish_reason')):
                    errors.append('repeat_count_mismatch')
        row.update(physical_gpu_uuid=plan['devices']['A'],valid_measurement=not errors,
                   errors=sorted(set(errors)),raw_path=str(root/'raw'/(label_for(slot)+'.json')))
        return row,errors

    for row,(slot,request) in zip(rows,planned):
        if any(row[k]!=v for k,v in slot.items()) or not row['valid_measurement']:
            raise ValueError('Retained slot differs from its chronological plan')
        actual,errors=validate(read(root/'raw'/(label_for(slot)+'.json')),slot,request)
        if errors or actual!=row:
            raise ValueError('Retained raw validity or measured evidence changed')
        grouped[slot['phase'],slot['method'],request['request_id']].append(row)
    progress=dict(status='running',planned=12384,attempted=len(rows),returned=len(rows),completed_valid=len(rows),
        invalid=0,unknown=0,pending=None,automatic_retry=False,unattempted=12384-len(rows),
        previous_wall_s=old.get('wall_s',0.),resume_warmup_generations=old.get('resume_warmup_generations',0))
    started=time.perf_counter()
    def persist():
        progress.update(wall_s=progress['previous_wall_s']+time.perf_counter()-started,
            unknown=int(progress['pending'] is not None),unattempted=12384-progress['attempted'])
        write_json(root/'STATUS.json',progress)
    def generate(backend,slot,request,kind,filename):
        progress['pending']=slot; persist()
        path=root/'raw'/filename
        if path.exists():
            raise ValueError('Unreconciled existing raw slot will not be overwritten')
        record=ledger.call({kind:1},filename,backend.generate,request['prompt_ids'],max_new_tokens=2048,
            request_id=label_for(slot),session_id=SESSION,request_index=request['request_index'],purpose='evaluation',save_to=path)
        write_json(path,record)
        if kind!='resume_warmup_generation':progress['returned']+=1
        row,errors=validate(record,slot,request)
        row['raw_path']=str(path)
        append_json(root/('RETRY_MEASUREMENTS.jsonl' if kind=='resume_warmup_generation' else 'MEASUREMENTS.jsonl'),row)
        progress['pending']=None
        if errors:
            progress['invalid']+=1; persist()
            raise ValueError('Native output/termination/count failure: '+repr(errors))
        return row
    persist()
    try:
        completed=len(rows);checked=set()
        for entry in entries:
            begin=entry['activation']*258; end=begin+258
            if completed>=end:continue
            append_json(root/'BOUNDARIES.jsonl',boundary(entry))
            resumed=completed>begin
            backend=_activation(activate,checkpoints[entry['method']],entry['method'],dict(entry,resumed=resumed),root,check_weights=entry['method'] not in checked)
            checked.add(entry['method'])
            if resumed:
                retry_number=len((root/'ACTIVATIONS.jsonl').read_text().splitlines())
                for request in warmups:
                    slot=dict(activation=entry['activation'],repeat=entry['repeat'],group=entry['group'],
                        method=entry['method'],phase='resume_warmup',request_ordinal=request['request_index'])
                    generate(backend,slot,request,'resume_warmup_generation',f'retry{retry_number:03d}_{label_for(slot)}.json')
                    progress['resume_warmup_generations']+=1; persist()
            for slot,request in planned[max(completed,begin):end]:
                progress['attempted']+=1; persist()
                row=generate(backend,slot,request,slot['phase']+'_generation',label_for(slot)+'.json')
                rows.append(row); grouped[slot['phase'],slot['method'],request['request_id']].append(row)
                progress['completed_valid']+=1; completed+=1; persist()
        append_json(root/'BOUNDARIES.jsonl',boundary({'activation':'finished'}))
        progress['status']='completed'
    except BaseException as error:
        progress.update(status='failed',error=repr(error)); raise
    finally:persist()
    return progress


def execute_canonicals(plan, root, requests, checkpoint, *, activate, boundary, ledger, resume=False):
    """Exactly one F canonical per sealed request; never a target-only replay."""
    root=Path(root);root.mkdir(parents=True,exist_ok=resume)
    validate_requests(requests,count=1024,role='evaluation')
    _persist_contract(root/'INPUTS.json',requests,resume)
    protocol=dict(checkpoint=checkpoint,execution_commit=plan['execution_commit'],
                  canonical_provenance=plan['canonical_provenance'],gpu=plan['devices']['A'])
    _persist_contract(root/'PROTOCOL.json',protocol,resume)
    prior=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
    if prior.get('pending') is not None or prior.get('invalid',0):
        raise ValueError('Unresolved canonical attempt must not be regenerated')
    references=read(root/'CANONICALS.json') if (root/'CANONICALS.json').exists() else {}
    for request in requests:
        if request['request_id'] in references:
            check_reference(request,references[request['request_id']],plan['canonical_provenance'],method='F')
            slot=dict(activation=0,repeat=0,group=0,method='F',phase='canonical',request_ordinal=request['request_index'])
            _,errors=measure(read(root/'raw'/(label_for(slot)+'.json')),request,checkpoint,references[request['request_id']],slot)
            if errors:raise ValueError('Retained canonical raw record changed')
    if set(references)-{r['request_id'] for r in requests}:raise ValueError('Unknown canonical request')
    status=dict(status='running',planned=1024,attempted=len(references),returned=len(references),completed=len(references),pending=None,invalid=0)
    def persist():
        status.update(unknown=int(status['pending'] is not None),unattempted=1024-status['attempted'])
        write_json(root/'STATUS.json',status)
    persist()
    try:
        if len(references)<1024:
            boundary({'phase':'canonical'})
            backend=_activation(activate,checkpoint,'F',dict(activation='canonical',phase='canonical'),root)
            for request in requests:
                key=request['request_id']
                if key in references:continue
                slot=dict(activation=0,repeat=0,group=0,method='F',phase='canonical',request_ordinal=request['request_index'])
                label=label_for(slot);path=root/'raw'/(label+'.json')
                if path.exists():raise ValueError('Existing uncommitted canonical raw return requires reconciliation')
                status['pending']=slot;status['attempted']+=1;persist()
                record=ledger.call({'canonical_generation':1},label,backend.generate,request['prompt_ids'],max_new_tokens=2048,
                    request_id=label,session_id=SESSION,request_index=request['request_index'],purpose='evaluation',save_to=path)
                write_json(path,record)
                status['returned']+=1;status['pending']=None;persist()
                try:
                    reference=make_reference(plan,request,record,method='F')
                    _,errors=measured_return(record,request,checkpoint,reference,slot)
                except (KeyError,TypeError,ValueError,IndexError,RuntimeError) as error:
                    errors=['malformed_returned_record:'+repr(error)]
                append_json(root/'VALIDITY.jsonl',dict(slot,valid_measurement=not errors,errors=errors,raw_path=str(path)))
                if errors:
                    status['invalid']+=1;raise ValueError('Invalid frozen F canonical: '+repr(errors))
                references[key]=reference;write_json(root/'CANONICALS.json',references)
                status.update(completed=len(references),pending=None);persist()
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed',error=repr(error));raise
    finally:persist()
    return references


def endpoint_receipts(plan):
    root=Path(plan['root']);receipts={}
    for arm in ('BASE','TAIL'):
        path=plan.get('endpoint_status_paths',{}).get(arm,root/'campaign_001/arms'/arm/'STATUS.json')
        receipt=read(path)
        if (receipt.get('status')!='completed' or receipt.get('events_completed')!=58
                or receipt.get('global_step')!=7424 or receipt.get('presentations')!=14848):
            raise ValueError('Final endpoint must finish all58 events: '+arm)
        receipts[arm]=receipt
    receipts.update(plan['retained_endpoints'])
    if set(receipts)!=set(METHODS):raise ValueError('Require BASE/TAIL/AREPO/F only')
    for method,receipt in receipts.items():
        if identity(Path(receipt['checkpoint'])/'model.safetensors')!=receipt['checkpoint_identity']:
            raise ValueError('Endpoint tensor file changed: '+method)
        for path,expected in receipt.get('companion_files',{}).items():
            if identity(path)!=expected:raise ValueError('Endpoint companion file changed')
    return receipts


def cost_totals(root,rows):
    """Canonical work, planned timing and extra resume work have separate totals."""
    root=Path(root)
    activations={phase:[json.loads(line) for line in (root/phase/'ACTIVATIONS.jsonl').read_text().splitlines()]
                 for phase in ('canonicals','evaluation')}
    canonical_records=[read(path)['metadata'] for path in sorted((root/'canonicals/raw').glob('*.json'))]
    retry_path=root/'evaluation/RETRY_MEASUREMENTS.jsonl'
    retry=[json.loads(line) for line in retry_path.read_text().splitlines()] if retry_path.exists() else []
    return dict(loading_s=sum(r['loading_s'] for group in activations.values() for r in group),
        canonical_activation_s=sum(r['publication']['activation_s'] for r in activations['canonicals']),
        activation_s=sum(r['publication']['activation_s'] for r in activations['evaluation']),
        canonical_request_latency_s=sum(r['request_latency_s'] for r in canonical_records),
        canonical_native_generate_s=sum(r['native_generate_s'] for r in canonical_records),
        canonical_cache_maintenance_s=sum(r['capture']['cache_maintenance_s'] for r in canonical_records),
        warmup_request_latency_s=sum(r['request_latency_s'] for r in rows if r['phase']=='warmup'),
        scored_request_latency_s=sum(r['request_latency_s'] for r in rows if r['phase']=='scored'),
        resume_warmup_request_latency_s=sum(r['request_latency_s'] for r in retry),
        actual_scoring_phase_activations=len(activations['evaluation']),
        actual_canonical_activations=len(activations['canonicals']),
        canonical_generations=len(canonical_records),resume_warmup_generations=len(retry),
        retry_cost_scope='Any additional activation/warmup is charged separately; planned slots are never repeated')


def run(plan,*,ledger,boundary):
    """One native serving worker on physical A; B and transfers remain quiet."""
    from experiments.repair_conditional_kl_pilot.campaign import create_backend
    from experiments.repair_conditional_kl_pilot.production import quiet_evaluation
    from .analysis import summarize,exported_parameters_identity
    root=Path(plan['root']);inputs=read(plan['evaluation_inputs'])
    receipts=endpoint_receipts(plan)
    checkpoints={m:r['checkpoint'] for m,r in receipts.items()}
    parameters={m:exported_parameters_identity(Path(p)/'model.safetensors') for m,p in checkpoints.items()}
    write_json(root/'SERVING_PARAMETERS.json',parameters)
    backend=None
    def guarded(slot):
        return dict(slot=slot,resources=boundary(),quiet=quiet_evaluation(plan,backend))
    def activate(method,checkpoint,slot):
        nonlocal backend
        loading=0.
        if backend is None:
            backend=create_backend(plan,root/'final_native',card='A');loading=backend.loading_s
        publication=ledger.call({'activation':1},'publish_'+str(slot['activation'])+'_'+method,backend.publish,checkpoint)
        return backend,dict(publication=publication,loading_s=loading)
    try:
        references=execute_canonicals(plan,root/'canonicals',inputs['evaluation_requests'],checkpoints['F'],
            activate=activate,boundary=guarded,ledger=ledger,resume=(root/'canonicals').exists())
        warmup_references=read(plan['warmup_canonicals'])
        warmup_references={key:(value['F'] if set(value)=={'AR','F'} else value) for key,value in warmup_references.items()}
        execute(plan,root/'evaluation',inputs['evaluation_requests'],inputs['warmup_requests'],references,
            warmup_references,checkpoints,activate=activate,boundary=guarded,ledger=ledger,
            resume=(root/'evaluation').exists())
    finally:
        if backend is not None:
            try:
                if not backend.failed:write_json(root/'evaluation_physical_work.json',backend._call('work_receipt'))
            finally:backend.close()
    after=endpoint_receipts(plan)
    if {m:r['checkpoint_identity'] for m,r in after.items()}!={m:r['checkpoint_identity'] for m,r in receipts.items()}:
        raise ValueError('Final evaluation changed an endpoint')
    write_json(root/'evaluation/EXPORTS_UNCHANGED.json',{m:r['checkpoint_identity'] for m,r in after.items()})
    rows=[json.loads(line) for line in (root/'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
    result=summarize(rows)
    result['parameter_identities']=parameters
    traces={}
    for row in rows:
        if row['phase']=='scored' and row['repeat']==0 and row['method'] in ('BASE','TAIL'):
            traces[row['request_id'],row['method']]=read(row['raw_path'])['metadata']['trace']
    result['identical_native_trajectories']=all(traces[key,'BASE']==traces[key,'TAIL'] for key in {r['request_id'] for r in rows if r['phase']=='scored'})
    result['serving_costs']=cost_totals(root,rows)
    write_json(root/'evaluation/SUMMARY.json',result)
    return result


def qualify_serving(plan,root,request,checkpoints,*,activate,boundary,ledger):
    """Eight included native generations; a single retry can use the other eight."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    if set(checkpoints)!=set(METHODS):raise ValueError('All four actual endpoint exports are required')
    canonical=None;counts={};completed=attempted=returned=invalid=0
    write_json(root/'STATUS.json',dict(status='running',planned=8,completed=0))
    try:
        for repeat in range(2):
            for method in ('F','BASE','TAIL','AREPO'):
                slot=dict(activation=completed,repeat=repeat,group=0,method=method,phase='qualification',request_ordinal=request['request_index'])
                boundary(slot)
                backend=_activation(activate,checkpoints[method],method,slot,root)
                path=root/'raw'/(label_for(slot)+'.json')
                attempted+=1
                record=ledger.call({'generation':1},label_for(slot),backend.generate,request['prompt_ids'],max_new_tokens=2048,
                    request_id=label_for(slot),session_id=SESSION,request_index=request['request_index'],purpose='evaluation',save_to=path)
                write_json(path,record)
                returned+=1
                try:
                    if canonical is None:
                        canonical=make_reference(plan,request,record,method='F')
                        write_json(root/'CANONICAL.json',canonical)
                    row,errors=measured_return(record,request,checkpoints[method],canonical,slot)
                except (KeyError,TypeError,ValueError,IndexError,RuntimeError) as error:
                    row=dict(slot,request_id=request['request_id'],checkpoint_id=checkpoints[method])
                    errors=['malformed_returned_record:'+repr(error)]
                if not errors:
                    current=tuple(row[k] for k in (*COUNT_FIELDS,'finish_reason'))
                    if method in counts and counts[method]!=current:errors.append('repeat_count_mismatch')
                row.update(valid_measurement=not errors,errors=errors)
                append_json(root/'MEASUREMENTS.jsonl',row)
                if errors:
                    invalid+=1
                    raise ValueError('Native serving qualification failed: '+repr(errors))
                counts[method]=current;completed+=1
                write_json(root/'STATUS.json',dict(status='running',planned=8,completed=completed))
        result=dict(status='passed',generation_attempts=8,completed=8,method_order=['F','BASE','TAIL','AREPO'],
                    repeats=2,canonical_kind='native frozen F',target_only_AR=False)
        write_json(root/'STATUS.json',result);return result
    except BaseException as error:
        write_json(root/'STATUS.json',dict(status='failed',planned=8,completed=completed,
            attempted=attempted,returned=returned,invalid=invalid,unknown=attempted-returned,
            unattempted=8-attempted,error=repr(error)));raise
