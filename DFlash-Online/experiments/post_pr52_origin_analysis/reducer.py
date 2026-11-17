"""Retrospective KPLUS4 count contrasts on separate PR51 and PR52 panels.

Reads authenticated JSON/token records only. Reference latency is intentionally
not read or compared. No model, tensor, training or serving package is imported.
"""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

COUNTS = ('verification_calls', 'native_accepted_draft_tokens',
          'accepted_draft_tokens', 'committed_progress', 'output_tokens')
SETTINGS = dict(block_size=16, context_limit=4096, runtime='pinned_SGLang_FA3',
    cache_policy='idle_full_flush_before_every_request_and_publication')
PANELS = dict(PR51=dict(methods=('N', 'S'), seed=1711,
    final_session='mathqa_fixed_lr_transfer_event6', reference_session='mathqa_lr_origin_references'),
    PR52=dict(methods=('N', 'C', 'S'), seed=1717,
    final_session='unchanged_c_mathqa_transfer_event6', reference_session='mathqa_unchanged_c_origin_references'))


def identity(payload):
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_name = ('RETRIEVAL_MANIFEST.json' if (self.root/'RETRIEVAL_MANIFEST.json').is_file()
                              else 'DELIVERY_MANIFEST.json')
        raw = (self.root/self.manifest_name).read_bytes()
        self.manifest = json.loads(raw)
        self.manifest_identity = identity(raw)
        self.consumed = {}
        self.checks = 0
        self.errors = []
        self.plan = self.read('sealed/PLAN.json')
        self.origin = Path(self.plan['root'])

    def check(self, valid, field):
        self.checks += 1
        if not valid:
            self.errors.append(field)
            raise ValueError(field)

    def relative(self, path):
        value = Path(path)
        return str(value.relative_to(self.origin)) if value.is_absolute() else str(value)

    def read(self, path, *, lines=False, raw_only=False):
        relative = str(path)
        if hasattr(self, 'origin'):
            relative = self.relative(path)
        local = (self.root/relative).resolve()
        self.check(local.is_relative_to(self.root), 'unsafe_source_path:'+relative)
        raw = local.read_bytes()
        actual = identity(raw)
        self.check(actual == self.manifest.get(relative), 'source_identity:'+relative)
        self.consumed[relative] = actual
        if raw_only:
            return raw
        return [json.loads(line) for line in raw.splitlines() if line] if lines else json.loads(raw)


def count_native(record, capture, request, canonical, *, checkpoint, session, label, check):
    """Rebuild progress and both distinct acceptance counters, including EOS/caps."""
    prompt, response = record['prompt_ids'], record['response_ids']
    meta, trace = record['metadata'], record['metadata']['trace']
    check(prompt == request['prompt_ids'] == canonical['prompt_ids'], label+':exact_input_tokens')
    check(record['record_id'] == label and record['request_index'] == request['panel_ordinal'], label+':request_identity')
    check(record['session_id'] == session and record['purpose'] == 'evaluation', label+':producer_session')
    check(record['checkpoint_id'] == checkpoint, label+':checkpoint')
    check(response == canonical['response_ids'] and record['finish_reason'] == canonical['finish_reason'], label+':canonical_output_termination')
    check(bool(response) and all(type(t) is int and t >= 0 for t in prompt+response), label+':native_token_ids')
    stops = request['stop_token_ids']
    check(record['max_new_tokens'] == 2048 and record['stop_token_ids'] == stops, label+':cap_and_stops')
    check(len(prompt)+2048 <= 4096 and len(response) <= 2048, label+':context_cap')
    check(not any(t in stops for t in response[:-1]), label+':no_prior_eos')
    check((record['finish_reason'] == 'eos' and response[-1] in stops) or
          (record['finish_reason'] == 'length' and len(response) == 2048 and response[-1] not in stops), label+':termination')
    check(all(meta.get(k) == v for k, v in SETTINGS.items()), label+':native_settings')
    check(meta['feedback_capture_in_timed_request'] is False and capture['collect_features'] is False
          and capture['feature_bytes'] == 0 and capture['native_capture_validated'] is True, label+':no_feedback')
    check(capture == meta['capture'] and trace == capture['blocks']
          and capture['prompt_ids'] == prompt and capture['response_ids'] == response, label+':separate_trace_binding')
    progress = native = matched = 0
    for ordinal, block in enumerate(trace):
        name = f'{label}:block{ordinal}'
        proposed, target = block['draft_tokens'], block['target_argmax']
        width, committed = block['verify_size'], block['committed_progress']
        check(type(width) is int and type(committed) is int and 1 <= committed <= width <= 16
              and width == len(proposed)+1 == len(target), name+':width')
        check(all(type(t) is int and t >= 0 for t in proposed+target), name+':draft_target_ids')
        match = 0
        while match < len(proposed) and proposed[match] == target[match]:
            match += 1
        check(committed <= match+1, name+':first_rejection')
        check(block['output_start'] == progress and block['anchor_token'] == response[progress], name+':seed')
        check(block['positions'] == list(range(len(prompt)+progress, len(prompt)+progress+width)), name+':absolute_positions')
        check(block['committed_tokens'] == (proposed[:match]+[target[match]])[:committed]
              == target[:committed] == response[progress+1:progress+1+committed], name+':commits')
        check(block['raw_accepted_draft_tokens_before_stop'] == match
              and block['accepted_draft_tokens'] == min(match, committed), name+':matching_acceptance')
        check(block['native_accepted_draft_tokens'] == committed-1, name+':native_acceptance')
        check(block['stop_clipped'] == (committed < match+1), name+':stop_clipping')
        if 'finished' in block:
            check(block['finished'] == (progress+committed == len(response)-1), name+':finished')
        progress += committed
        native += committed-1
        matched += min(match, committed)
    check(progress == len(response)-1 and native == progress-len(trace), label+':first_token_progress')
    counts = dict(verification_calls=len(trace), native_accepted_draft_tokens=native,
        accepted_draft_tokens=matched, committed_progress=progress, output_tokens=len(response))
    for key in ('verification_calls', 'accepted_draft_tokens', 'committed_progress'):
        check(type(meta[key]) is int and meta[key] == counts[key], label+':stored_'+key)
    return dict(**counts, finish_reason=record['finish_reason'], cap_hit=record['finish_reason'] == 'length')


def load_panel(root, panel, compatibility):
    evidence = Evidence(root)
    settings, plan = PANELS[panel], evidence.plan
    evidence.check(compatibility['status'] == 'supported', 'count_compatibility_not_supported')
    evidence.check(compatibility['execution_commit'] == plan['execution_commit'], 'compatibility_execution_commit')
    evidence.check(compatibility['origin_weights_identity'] == plan['origin_weights_identity'], 'origin_export_identity')
    evidence.check(compatibility['reference_gpu_uuid'] == plan['devices']['B']
                   and compatibility['final_gpu_uuid'] == plan['devices']['A'], 'qualified_card_binding')
    for relative, expected in compatibility['consumed_files'].items():
        evidence.read(relative, raw_only=True)
        evidence.check(evidence.consumed[relative] == expected, 'compatibility_evidence_identity:'+relative)
    final = Path('tasks/final/evaluation')
    protocol = evidence.read(final/'PROTOCOL.json')
    requests = evidence.read(final/'REQUESTS.json')
    canonical = evidence.read(final/'CANONICAL.json')
    summary = evidence.read(final/'SUMMARY.json')
    sealed = evidence.read(plan['data_panel'])
    refs = evidence.read(plan['references'])
    seal = evidence.read('references/SEAL.json')
    evidence.check(requests == sealed['requests'] and canonical == refs, 'exact_panel_reference_join')
    evidence.check(len(requests) == 256 and len({r['source_id'] for r in requests}) == 256
        and len({r['request_id'] for r in requests}) == 256
        and [r['panel_ordinal'] for r in requests] == list(range(256)), 'complete_unique_scored_panel')
    evidence.check(protocol['methods'] == list(settings['methods']) and protocol['event'] == 6
        and protocol['execution_commit'] == plan['execution_commit']
        and protocol['origin_checkpoint'] == plan['origin_export']
        and protocol['physical_gpu_uuid'] == plan['devices']['A']
        and protocol['session'] == settings['final_session'] and protocol['capture'] is False, 'final_protocol')
    endpoints = {}
    for method in settings['methods']:
        if panel == 'PR52' and method in ('N', 'S'):
            endpoint = plan['retained_endpoints'][method+'6']['endpoint']
        else:
            result = evidence.read(Path('tasks')/(method+'6')/'RESULT.json')
            evidence.check(result['status'] == 'completed' and result['arm'] == method
                           and result['event'] == 6, 'endpoint_receipt:'+method)
            endpoint = result['endpoint']
        evidence.check(protocol['checkpoints'][method] == endpoint['export'], 'endpoint_export:'+method)
        endpoints[method] = endpoint
    evidence.check(seal['status'] == 'passed' and seal['binding'] == plan['binding']
        and seal['checkpoint'] == plan['origin_export'] and seal['checkpoint_identity'] == plan['origin_weights_identity'], 'origin_seal')
    evidence.check(seal['files'][plan['references']] == evidence.consumed[evidence.relative(plan['references'])], 'canonical_file_identity')
    evidence.check(summary['status'] == 'completed' and summary['request_count'] == 256, 'published_summary_complete')
    published = {(r['source_id'], r['request_id']): r for r in summary['paired_requests']}
    evidence.check(len(published) == 256, 'published_complete_identity_join')
    status = evidence.read(final/'STATUS.json')
    evidence.check(status['status'] == 'completed' and status['scored_generations'] == 256*3*len(settings['methods'])
        and status['invalid_generations'] == status['unknown_generation_attempts'] == status['unattempted_generations'] == 0, 'final_no_missing_work')
    paired, bindings = [], []
    for request in sorted(requests, key=lambda r:(r['source_id'], r['request_id'])):
        source, request_id = request['source_id'], request['request_id']
        reference = canonical[request_id]
        evidence.check(reference['binding_digest'] == digest_json({k:v for k,v in reference.items() if k!='binding_digest'}), request_id+':reference_digest')
        evidence.check(reference['verified'] is True and reference['source_id'] == source
            and reference['reference_checkpoint'] == plan['origin_export']
            and reference['reference_kind'] == 'native_greedy_target_verifier', request_id+':reference_producer')
        raw_path = evidence.relative(reference['source_record_path'])
        raw = evidence.read(raw_path)
        evidence.check(evidence.consumed[raw_path] == reference['source_record_identity'], request_id+':reference_record_identity')
        capture_path = evidence.relative(raw['metadata']['native_trace_path'])
        capture = evidence.read(capture_path)
        origin = count_native(raw,capture,request,reference,checkpoint=plan['origin_export'],
            session=settings['reference_session'],label=request_id,check=evidence.check)
        row = dict(source_id=source, request_id=request_id, panel_ordinal=request['panel_ordinal'], KPLUS4=origin)
        binding = dict(source_id=source,request_id=request_id,input_token_identity=digest_json(request['prompt_ids']),
            output_token_identity=digest_json(reference['response_ids']),finish_reason=reference['finish_reason'],
            origin_record=raw_path,origin_trace=capture_path,final_records={})
        for method in settings['methods']:
            repeated=[];binding['final_records'][method]=[]
            for repeat in range(3):
                name=f"{request['panel_ordinal']:04d}_r{repeat}_{method}"
                path=str(final/(name+'.json'));record=evidence.read(path)
                trace_path=evidence.relative(record['metadata']['native_trace_path'])
                trace=evidence.read(trace_path)
                measured=count_native(record,trace,request,reference,checkpoint=protocol['checkpoints'][method],
                    session=settings['final_session'],label=name,check=evidence.check)
                validation=evidence.read(final/(name+'_VALIDATION.json'))
                evidence.check(validation == dict(status='valid',errors=[]), name+':execution_validation')
                repeated.append(measured)
                binding['final_records'][method].append(dict(record=path,trace=trace_path))
            evidence.check(repeated[0] == repeated[1] == repeated[2], request_id+':'+method+':repeat_consistency')
            row[method]=repeated[0]
            retained=published[(source,request_id)][method]
            evidence.check(all(retained[k] == row[method][k] for k in (*COUNTS,'finish_reason','cap_hit')), request_id+':'+method+':published_reduction')
        paired.append(row);bindings.append(binding)
    evidence.check(summary['bootstrap']['seed'] == settings['seed'] and summary['bootstrap']['draws'] == 10000
                   and summary['bootstrap']['generator'] == 'PCG64' and summary['bootstrap']['method'] == 'linear', 'original_bootstrap_contract')
    return evidence, paired, dict(execution_commit=plan['execution_commit'],origin_export=plan['origin_export'],
        origin_weights_identity=plan['origin_weights_identity'],endpoint_exports=protocol['checkpoints'],
        endpoint_identities=endpoints,
        original_bootstrap=summary['bootstrap'],
        original_source_order=[dict(source_id=r['source_id'],request_id=r['request_id']) for r in summary['paired_requests']],
        request_bindings=bindings,
        compatibility=compatibility,reference_latency_used=False)


def ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator/denominator)


def interval(point, draws):
    finite=np.isfinite(draws)
    return dict(point=point,ci95=np.quantile(draws,[.025,.975],method='linear').tolist() if finite.all() else None,
                undefined_draws=int((~finite).sum()))


def summarize(paired, methods, seed):
    paired=sorted(paired,key=lambda r:(r['source_id'],r['request_id']))
    if len(paired)!=256 or len({r['source_id'] for r in paired})!=256:
        raise ValueError('Require the entire original 256-request panel')
    methods=('KPLUS4',*methods)
    totals={m:{k:sum(r[m][k] for r in paired) for k in COUNTS} for m in methods}
    for method,total in totals.items():
        total.update(native_accepted_proposals_per_call=ratio(total['native_accepted_draft_tokens'],total['verification_calls']),
            stop_clipped_matching_per_call=ratio(total['accepted_draft_tokens'],total['verification_calls']),
            progress_per_call=ratio(total['committed_progress'],total['verification_calls']),
            cap_hit_requests=sum(r[method]['cap_hit'] for r in paired),
            zero_call_requests=sum(r[method]['verification_calls']==0 for r in paired),
            termination_counts=dict(Counter(r[method]['finish_reason'] for r in paired)))
    draws=np.random.Generator(np.random.PCG64(seed)).integers(0,256,size=(10000,256))
    comparisons={}
    array=lambda method,key:np.asarray([r[method][key] for r in paired],dtype=np.float64)
    rv,ra=array('KPLUS4','verification_calls'),array('KPLUS4','native_accepted_draft_tokens')
    for method in methods[1:]:
        cv,ca=array(method,'verification_calls'),array(method,'native_accepted_draft_tokens')
        candidate=totals[method]['native_accepted_proposals_per_call'];reference=totals['KPLUS4']['native_accepted_proposals_per_call']
        with np.errstate(divide='ignore',invalid='ignore'):
            ac=ca[draws].sum(1)/cv[draws].sum(1);ar=ra[draws].sum(1)/rv[draws].sum(1)
            gain=ac/ar-1;difference=ac-ar;call_reduction=1-cv[draws].sum(1)/rv[draws].sum(1)
        saved=rv-cv
        point=None if candidate is None or reference is None or reference==0 else candidate/reference-1
        diff=None if candidate is None or reference is None else candidate-reference
        call_point=None if not rv.sum() else float(1-cv.sum()/rv.sum())
        comparisons[method+'_vs_KPLUS4']=dict(candidate=method,reference='KPLUS4',retrospective=True,
            acceptance_gain=interval(point,gain),acceptance_difference=interval(diff,difference),
            call_reduction=interval(call_point,call_reduction),
            requests_improved=int((saved>0).sum()),requests_regressed=int((saved<0).sum()),requests_tied=int((saved==0).sum()),
            gross_calls_saved=int(saved[saved>0].sum()),gross_calls_added=int(-saved[saved<0].sum()),net_calls_saved=int(saved.sum()))
    return dict(request_count=256,method_totals=totals,comparisons=comparisons,
        bootstrap=dict(seed=seed,draws=10000,generator='PCG64',method='linear',
            unit='whole paired request with origin and all endpoints; panels separate',
            source_order=[dict(source_id=r['source_id'],request_id=r['request_id']) for r in paired],
            draw_matrix_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()),
        reference_latency_used=False,model_work=0)


def csv_table(path, fields, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def run(pr51_root, pr52_root, compatibility_path, output_dir):
    start=time.perf_counter();output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    compatibility_raw=Path(compatibility_path).read_bytes();compatibility=json.loads(compatibility_raw)
    result=dict(status='BLOCKED_RECORD_AVAILABILITY',retrospective=True,panels={},new_model_work=0,
        compatibility_receipt_identity=identity(compatibility_raw),scope='Two separate original MathQA panels; counts only',
        candidate_status='candidate retained for contribution assessment; performance preservation unresolved')
    for panel,root in (('PR51',pr51_root),('PR52',pr52_root)):
        evidence=None
        try:
            evidence,paired,bindings=load_panel(root,panel,compatibility['panels'][panel])
            stats=summarize(paired,PANELS[panel]['methods'],PANELS[panel]['seed'])
            evidence.check(stats['bootstrap']['draw_matrix_sha256']==bindings['original_bootstrap']['draw_matrix_sha256'], 'original_draw_identity')
            evidence.check(stats['bootstrap']['source_order']==bindings['original_source_order'], 'original_draw_source_order')
            bindings.update(delivery_manifest=evidence.manifest_name,delivery_manifest_identity=evidence.manifest_identity,
                consumed_files=evidence.consumed,source_root=str(Path(root).resolve()))
            write_json(output/(panel+'_BINDINGS.json'),bindings)
            write_json(output/(panel+'_PAIRED_REQUESTS.json'),paired)
            write_json(output/(panel+'_SUMMARY.json'),stats)
            table=[]
            for method,total in stats['method_totals'].items():
                comparison=stats['comparisons'].get(method+'_vs_KPLUS4')
                row=dict(panel=panel,method=method,**{k:total[k] for k in COUNTS},
                    native_accepted_proposals_per_call=total['native_accepted_proposals_per_call'],
                    cap_hit_requests=total['cap_hit_requests'],zero_call_requests=total['zero_call_requests'])
                for key in ('requests_improved','requests_regressed','requests_tied','gross_calls_saved','gross_calls_added','net_calls_saved'):
                    row[key]=comparison[key] if comparison else None
                for key in ('acceptance_gain','acceptance_difference','call_reduction'):
                    metric=comparison[key] if comparison else dict(point=None,ci95=None,undefined_draws=None)
                    row.update({key:metric['point'],key+'_ci_low':metric['ci95'][0] if metric['ci95'] else None,
                        key+'_ci_high':metric['ci95'][1] if metric['ci95'] else None,key+'_undefined_draws':metric['undefined_draws']})
                table.append(row)
            csv_table(output/(panel+'_TABLE.csv'),list(table[0]),table)
            result['panels'][panel]=dict(status='available',checks=evidence.checks,errors=[],
                available_contrasts=list(stats['comparisons']),summary=panel+'_SUMMARY.json',bindings=panel+'_BINDINGS.json')
        except (OSError,KeyError,ValueError,TypeError,IndexError) as error:
            result['panels'][panel]=dict(status='unavailable',error=repr(error),path=getattr(error,'filename',None),
                checks=evidence.checks if evidence else 0,errors=evidence.errors if evidence else [],available_contrasts=[])
        write_json(output/'STATUS.json',result)
    available=sum(r['status']=='available' for r in result['panels'].values())
    result.update(status=('COMPLETED_EXISTING_RECORD_ANALYSIS' if available==2 else
        'PARTIAL_RECORD_AVAILABILITY' if available else 'BLOCKED_RECORD_AVAILABILITY'),wall_s=time.perf_counter()-start)
    write_json(output/'STATUS.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('pr51-root','pr52-root','compatibility','output'):parser.add_argument('--'+arg,required=True)
    args=parser.parse_args();result=run(args.pr51_root,args.pr52_root,args.compatibility,args.output)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='COMPLETED_EXISTING_RECORD_ANALYSIS' else 1)
