"""Independent raw-audit tests: invalid evidence must not become a gain claim."""
from copy import deepcopy
import json

import pytest

from . import raw_audit, report
from .analysis import summarize
from .contract import METHODS, PRODUCTION_WORK, QUALIFICATION_CEILINGS, digest
from .schedule import evaluation_schedule
from .test_analysis import fixture


def raw_fixture(*, ar=False, zero=False):
    request=dict(request_id='r',source_id='source-r',request_index=0,
                 prompt_ids=[10,11],max_new_tokens=2048,stop_token_ids=[99])
    slot=dict(activation=0,phase='scored',request_ordinal=0,method='AR' if ar else 'F',repeat=0,group=0)
    response=[99] if zero else [20,21,99]
    canonical=dict(request,response_ids=response,finish_reason='eos')
    meta=dict(request_latency_s=.2,native_generate_s=.1,context_limit=4096,
              cache_policy='idle_full_flush_before_every_request_and_publication',
              runtime='pinned_SGLang_FA3',feedback_capture_in_timed_request=False)
    if ar:
        meta.update(speculation_disabled=True,physical_context_length=4112,
            cache_maintenance_s=.01,native_meta_info=dict(prompt_tokens=2,completion_tokens=len(response),
                cached_tokens=0,finish_reason=dict(type='stop',matched=99)),
            verification_calls=None,accepted_draft_tokens=None,committed_progress=None)
    else:
        trace=[] if zero else [dict(draft_tokens=[21,99],target_argmax=[21,99,42],verify_size=3,
            committed_progress=2,output_start=0,anchor_token=20,committed_tokens=[21,99],positions=[2,3,4],
            accepted_draft_tokens=2,native_accepted_draft_tokens=1,
            raw_accepted_draft_tokens_before_stop=2,stop_clipped=True)]
        meta.update(block_size=16,verification_calls=len(trace),accepted_draft_tokens=0 if zero else 2,
            committed_progress=len(response)-1,trace=trace,
            capture=dict(collect_features=False,feature_bytes=0,native_capture_validated=True,
                         prompt_ids=request['prompt_ids'],response_ids=response,blocks=trace,cache_maintenance_s=.01))
    record=dict(record_id=raw_audit.label(slot),session_id='opencode_final58',request_index=0,
        checkpoint_id='endpoint',purpose='evaluation',prompt_ids=request['prompt_ids'],response_ids=response,
        finish_reason='eos',max_new_tokens=2048,stop_token_ids=[99],metadata=meta)
    return record,request,canonical,slot


@pytest.mark.parametrize('ar,zero',[(False,False),(False,True),(True,False),(True,True)])
def test_native_and_ar_counts_including_zero_rounds(ar,zero):
    record,request,canonical,slot=raw_fixture(ar=ar,zero=zero)
    metrics=raw_audit.raw_metrics(record,request,canonical,method=slot['method'],checkpoint='endpoint',slot=slot)
    assert metrics['output_tokens']==(1 if zero else 3)
    assert metrics['verification_calls']==(None if ar else 0 if zero else 1)
    if not ar and not zero:
        assert metrics['native_accepted_draft_tokens']==1
        assert metrics['accepted_draft_tokens']==2


@pytest.mark.parametrize('change', ['canonical','trace','count','position','checkpoint','after_eos','timing','capture'])
def test_raw_corruption_rejected(change):
    record,request,canonical,slot=deepcopy(raw_fixture())
    if change=='canonical': canonical['response_ids']=[20,22,99]
    elif change=='trace': record['metadata']['trace'][0]['committed_tokens']=[22,99]
    elif change=='count': record['metadata']['verification_calls']=2
    elif change=='position': record['metadata']['trace'][0]['output_start']=1
    elif change=='checkpoint': record['checkpoint_id']='later'
    elif change=='after_eos': record['response_ids']=[99,20,99]
    elif change=='timing': record['metadata']['native_generate_s']=.3
    else: record['metadata']['capture']['collect_features']=True
    with pytest.raises(ValueError):
        raw_audit.raw_metrics(record,request,canonical,method='F',checkpoint='endpoint',slot=slot)


def test_ar_zero_counts_are_not_fabricated():
    record,request,canonical,slot=raw_fixture(ar=True)
    record['metadata']['verification_calls']=0
    with pytest.raises(ValueError,match='N/A'):
        raw_audit.raw_metrics(record,request,canonical,method='AR',checkpoint='endpoint',slot=slot)


def test_raw_fixture_matches_the_qualified_native_evaluator():
    from .evaluation import measure
    record,request,canonical,slot=raw_fixture()
    measured,errors=measure(record,request,'endpoint',canonical,slot)
    assert errors==[]
    independent=raw_audit.raw_metrics(record,request,canonical,method='F',checkpoint='endpoint',slot=slot)
    assert all(measured[k]==value for k,value in independent.items())


def test_independent_schedule_covers_every_slot():
    panel=[dict(request_id=f'r{i:04}',request_index=i) for i in range(1024)]
    warmups=[dict(request_id=f'w{i}',request_index=i) for i in range(2)]
    assert raw_audit.planned_schedule(panel)==evaluation_schedule(panel)
    slots=raw_audit.planned_slots(panel,warmups)
    assert len(slots)==15600 and len({raw_audit.label(s) for s in slots})==15600
    assert sum(s['phase']=='warmup' for s in slots)==240


def test_independent_bootstrap_matches_production_reducer():
    rows=fixture()
    actual=raw_audit.reduce_endpoint(rows,expected_requests=2)
    raw_audit.compare_reductions(actual,summarize(rows,expected_requests=2))
    assert actual['totals']['AR']['verification_calls'] is None
    assert actual['totals']['F']['output_tokens']==42


@pytest.mark.parametrize('change',['missing','duplicate','invalid','repeat','source','endpoint','gpu','ar_count'])
def test_independent_reduction_rejects_invalid_pairs(change):
    rows=deepcopy(fixture())
    if change=='missing': rows.pop()
    elif change=='duplicate': rows.append(rows[0])
    elif change=='invalid': rows[0]['valid_measurement']=False
    elif change=='repeat': rows[0]['committed_progress']=1
    elif change=='source': rows[0]['source_id']='other'
    elif change=='endpoint': rows[0]['checkpoint_id']='other'
    elif change=='gpu': rows[0]['physical_gpu_uuid']='other'
    else:
        for row in rows:
            if row['method']=='AR': row['verification_calls']=0
    with pytest.raises(ValueError):
        raw_audit.reduce_endpoint(rows,expected_requests=2)


def test_work_unknown_is_separate_from_unattempted():
    limits={**{'production.'+k:v for k,v in PRODUCTION_WORK.items()},
            **{'qualification.'+k:v for k,v in QUALIFICATION_CEILINGS.items()}}
    state=dict(limits=limits,attempted={k:0 for k in limits},completed={k:0 for k in limits},
               pending={'0':dict(counts={'production.adam':1})})
    state['attempted']['production.adam']=2
    state['completed']['production.adam']=1
    result={r['kind']:r for r in raw_audit.work_accounting(state)}
    assert result['production.adam']['unknown']==1
    assert result['production.adam']['unattempted']==22270
    assert result['qualification.adam']['unattempted'] is None
    assert result['qualification.adam']['unused_allowance']==48
    state['completed']['production.adam']=2
    with pytest.raises(ValueError): raw_audit.work_accounting(state)


def test_canonical_must_bind_source_and_contents():
    _,request,reference,_=raw_fixture()
    reference.update(reference_kind='target_only_AR',verified=True,provenance={'target':'fixed'})
    reference['binding_digest']=digest(reference)
    raw_audit.check_canonical(reference,request,{'target':'fixed'})
    reference['response_ids']=[98]
    with pytest.raises(ValueError): raw_audit.check_canonical(reference,request,{'target':'fixed'})


def test_partial_report_never_fabricates_results_or_exports_private_text(tmp_path,monkeypatch):
    rows=fixture()[:1]
    rows[0].update(raw_input='PRIVATE PROMPT',response_ids=[123456],raw_path='/private/runtime.json')
    partial=dict(status='incomplete',scope='test',errors=[],evaluation=dict(status='incomplete',rows=rows,errors=[]),
                 training=dict(arms={},event_curves=[],costs=[]),work=[])
    monkeypatch.setattr(raw_audit,'audit_run',lambda *args:partial)
    result=report.build_report(dict(execution_commit='fixture'),tmp_path)
    assert result['status']=='incomplete'
    assert not (tmp_path/'public/RESULTS.json').exists()
    public=''.join(p.read_text() for p in (tmp_path/'public').iterdir())
    assert 'PRIVATE PROMPT' not in public and 'response_ids' not in public and '/private/runtime.json' not in public
    assert 'No full-panel effect estimate' in public


def test_mismatched_second_reducer_stops_publication(tmp_path,monkeypatch):
    rows=fixture()
    independent=raw_audit.reduce_endpoint(rows,expected_requests=2)
    independent['totals']['F']['output_tokens']+=1
    audit=dict(status='passed',scope='test',errors=[],evaluation=dict(status='passed',rows=rows,errors=[],
        independent_summary=independent),training=dict(arms={},event_curves=[],costs=[]),work=[])
    monkeypatch.setattr(raw_audit,'audit_run',lambda *args:audit)
    monkeypatch.setattr(report.analysis,'summarize',lambda value:summarize(value,expected_requests=2))
    result=report.build_report(dict(execution_commit='fixture'),tmp_path)
    assert result['status']=='failed'
    assert json.loads((tmp_path/'public/AUDIT_SUMMARY.json').read_text())['status']=='failed'
    assert 'numeric' in (tmp_path/'private/REDUCTION_FAILURE.json').read_text()
    assert not (tmp_path/'public/RESULTS.json').exists()


def test_chronological_training_work_rejects_future_feedback(tmp_path):
    path=tmp_path/'WORK.jsonl'
    rows=[dict(kind='online_generations',ordinal=i,phase=phase)
          for i in range(1,130) for phase in ('started','completed')]
    path.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='wrong pre-update'):
        raw_audit.audit_arm_work(path)
    path.write_text('\n'.join(map(json.dumps,rows[:256])))
    result=raw_audit.audit_arm_work(path)
    assert result['completed']['online_generations']==128
    assert result['completed']['optimizer_steps']==0


def test_archives_independently_verify_exact_original_members(tmp_path):
    from .test_compact import event_fixture
    from .compact import archive_completed_event
    arm,records,_,_=event_fixture(tmp_path)
    receipt=archive_completed_event(tmp_path,arm,1,records)
    seen={}
    audit=raw_audit.stream_archive(receipt['archive_path'],lambda name,payload:seen.update({name:len(payload)}))
    assert audit['members']==131 and len(seen)==131
    with open(receipt['archive_path'],'ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError,match='archive bytes'):
        raw_audit.stream_archive(receipt['archive_path'],lambda *args:None)


def test_report_refuses_to_overwrite_an_earlier_result_directory(tmp_path):
    (tmp_path/'public').mkdir()
    with pytest.raises(FileExistsError):
        report.build_report(dict(execution_commit='fixture'),tmp_path)


@pytest.mark.parametrize('supported',[True,False])
def test_relative_C_win_does_not_hide_all_arms_losing_to_frozen(supported):
    rows=[row for row in fixture() if row['request_id']=='a']
    calls={'O_CODE58':13,'AREPO_CODE58':12,'C_CODE58':11,'F':10}
    latencies={'O_CODE58':4.,'AREPO_CODE58':3.,'C_CODE58':2.,'F':1.,'AR':5.}
    for row in rows:
        method=row['method']
        row['request_latency_s']=latencies[method]*(row['repeat']+1)
        row['native_generate_s']=row['request_latency_s']/2
        if method!='AR':
            row['verification_calls']=calls[method]
            row['native_accepted_draft_tokens']=row['accepted_draft_tokens']=40-calls[method]
    summary=summarize(rows,expected_requests=1)
    if not supported:
        summary['contrasts']['O_CODE58_vs_F']['request_latency_s_reduction_pct']['ci95']=[-300.,1.]
    text=report.conclusion(summary)
    assert 'C reduces complete-request latency against AREPO' in text
    assert 'All three adapted endpoints use more verification calls and take longer than frozen F' in text
    assert ('Adaptation is harmful' in text)==supported
    assert ('numerical regressions' in text)==(not supported)


def test_archived_phase_and_engine_costs_remain_separate_nonadditive_receipts(tmp_path):
    old=tmp_path/'attempts/output_policy_resume_001'
    phases=[tmp_path/'phases/qualification_serving_A/STATUS.json',
            old/'phases/qualification_serving_A/STATUS.json']
    for index,path in enumerate(phases):
        raw_audit.write(path,dict(status='completed' if index==0 else 'failed',
            wall_s=70. if index==0 else None,started_epoch=100-index*80))
    engine_paths=[tmp_path/'evaluation/engines/AR/LIFECYCLE.json',
                  tmp_path/'qualification/A/A_AR/LIFECYCLE.json',
                  old/'qualification/A/A_AR/LIFECYCLE.json']
    for index,path in enumerate(engine_paths):
        raw_audit.write(path,dict(loading_s=10.+index))
    raw_audit.write(old/'qualification/A/A_AR/CLOSE.json',dict(teardown_s=2.))
    unknown=dict(startup_forwards=None,uninstrumented_piecewise_initialization=True)
    raw_audit.write(old/'qualification/A/A_AR/PHYSICAL_WORK.json',unknown)
    raw_audit.write(tmp_path/'CLOCK.json',dict(completed_card_s=237.903,active={}))
    raw_audit.write(old/'CLOCK.json',dict(completed_card_s=237.903,active={}))
    before=(tmp_path/'CLOCK.json').read_bytes()
    result=raw_audit.overhead_accounting(tmp_path)
    assert len(result['phases'])==2
    assert len({row['source'] for row in result['phases']})==2
    assert sum(row['archived'] for row in result['phases'])==1
    archived=next(row for row in result['phases'] if row['archived'])
    assert archived['status']=='failed' and archived['wall_s'] is None
    assert len(result['costs'])==4
    assert len({(row['source'],row['metric']) for row in result['costs']})==4
    assert sum(row['archived'] for row in result['costs'])==2
    assert result['physical_work'][0]['receipt']==unknown
    assert all('CLOCK' not in row['source'] for row in result['costs'])
    assert 'not additive totals' in result['overlap']
    assert (tmp_path/'CLOCK.json').read_bytes()==before


def test_policy_resume_report_preserves_original_unknown_startup_cost(tmp_path,monkeypatch):
    audit=dict(status='incomplete',scope='test',errors=[],
        historical_startup_retry=dict(failed_startup_card_s=56.0318),
        clock=dict(completed_card_s=237.903,first_model_load_epoch=1),audit_observed_epoch=100,
        evaluation=dict(status='incomplete',rows=[],errors=[]),
        overhead=dict(phases=[dict(source='attempts/old/phases/start/STATUS.json',
            archived=True,phase='start',status='failed',wall_s=None,started_epoch=1)]))
    monkeypatch.setattr(raw_audit,'audit_run',lambda *args:audit)
    report.build_report(dict(execution_commit='fixture',policy_resume={'receipt':'retained'}),tmp_path)
    costs=json.loads((tmp_path/'public/CAMPAIGN_COSTS.json').read_text())
    assert costs['occupied_card_s']==237.903
    assert costs['startup_retry']['failed_startup_card_s']==56.0318
    text=(tmp_path/'public/REPORT.md').read_text()
    assert 'exact count remains unknown' in text
    assert '56.0318 prior occupied-card seconds' in text
    assert 'attempts/old/phases/start/STATUS.json' in (tmp_path/'public/PHASE_COSTS.csv').read_text()


def test_unauthenticated_historical_startup_cost_is_not_exposed(tmp_path,monkeypatch):
    from . import policy_resume
    receipt=tmp_path/'OUTPUT_POLICY_RESUME.json'
    raw_audit.write(receipt,dict(historical_startup_retry=dict(failed_startup_card_s=999.)))
    plan=dict(root=str(tmp_path),execution_commit='fixture',input_identities={},
              policy_resume={'receipt':str(receipt)})
    plan['binding']=digest(plan)
    def reject_receipt(root,actual):
        assert actual==plan
        raise ValueError('Corrupted policy receipt')
    monkeypatch.setattr(policy_resume,'_receipt',reject_receipt)
    result=raw_audit.audit_run(plan,tmp_path/'audit')
    assert result['status']=='failed'
    assert 'Corrupted policy receipt' in result['errors']
    assert 'historical_startup_retry' not in result
