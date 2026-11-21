"""AR/speculative differences are observations; within-path validity stays strict."""
import pytest

from . import analysis, evaluation, raw_audit, report
from .contract import digest
from .test_analysis import fixture
from .test_raw_audit import raw_fixture
from .validity import canonical_for_method, output_comparison, validate_canonical


def references():
    _, request, frozen, _ = raw_fixture()
    ar = dict(frozen, response_ids=[20, 22, 23, 99])
    bundle = {}
    for method, value in [('AR', ar), ('F', frozen)]:
        reference = dict(value, reference_kind='target_only_AR' if method=='AR' else 'frozen_DFlash',
                         verified=True, provenance={'target':'fixed'})
        reference['binding_digest'] = digest(reference)
        bundle[method] = reference
    return request, bundle


def test_output_difference_does_not_change_family_canonical():
    request, bundle = references()
    witness = output_comparison(bundle['AR'], bundle['F'])
    assert witness == raw_audit.cross_path_comparison(bundle['AR'], bundle['F'])
    assert witness['first_difference'] == 1
    assert not witness['output_equal'] and witness['termination_equal']
    assert (witness['record_output_tokens'], witness['reference_output_tokens']) == (4, 3)
    for method in ('AR', 'F', 'O_CODE58', 'AREPO_CODE58', 'C_CODE58'):
        reference = canonical_for_method(bundle, method)
        assert reference is bundle['AR' if method=='AR' else 'F']
    with pytest.raises(ValueError, match='Canonical mismatch'):
        validate_canonical(bundle['AR'], bundle['F'])
    for method in ('AR', 'F'):
        evaluation.check_reference(request, bundle[method], {'target':'fixed'}, method=method)
    assert raw_audit.check_bundle(bundle, request, {'target':'fixed'}) == witness


@pytest.mark.parametrize('key,value', [('prompt_ids',[12]), ('max_new_tokens',1024), ('stop_token_ids',[98])])
def test_cross_path_allowance_never_waives_input_bindings(key,value):
    _, bundle = references()
    bundle['F'][key] = value
    for compare in (output_comparison, raw_audit.cross_path_comparison):
        with pytest.raises(ValueError):
            compare(bundle['AR'],bundle['F'])


@pytest.mark.parametrize('mutation', ['kind','digest','provenance','missing_family'])
def test_separate_reference_identity_still_checked(mutation):
    request,bundle = references()
    if mutation == 'missing_family':
        del bundle['F']
        with pytest.raises(ValueError): canonical_for_method(bundle,'AR')
        with pytest.raises(ValueError): raw_audit.check_bundle(bundle,request,{'target':'fixed'})
        return
    if mutation == 'kind': bundle['F']['reference_kind']='target_only_AR'
    elif mutation == 'digest': bundle['F']['response_ids']=[99]
    else: bundle['F']['provenance']={'target':'wrong'}
    with pytest.raises(ValueError): evaluation.check_reference(request,bundle['F'],{'target':'fixed'},method='F')
    with pytest.raises(ValueError): raw_audit.check_bundle(bundle,request,{'target':'fixed'})


def test_actual_ar_and_spec_records_validate_against_own_families():
    spec,request,frozen,slot = raw_fixture()
    ar,_,_,ar_slot = raw_fixture(ar=True)
    ar['response_ids']=[20,22,99]
    ar['metadata'].update(committed_progress_per_call=None,calls_per_1000_output_tokens=None)
    ar_canonical=dict(frozen,response_ids=ar['response_ids'])
    witness=output_comparison(ar_canonical,frozen)
    row,errors=evaluation.measure(ar,request,'endpoint',ar_canonical,ar_slot,comparison=witness)
    assert not errors and row['ar_spec_comparison']['output_equal'] is False
    row,errors=evaluation.measure(spec,request,'endpoint',frozen,slot,comparison=witness)
    assert not errors
    assert evaluation.measure(ar,request,'endpoint',frozen,ar_slot)[1]
    assert evaluation.measure(spec,request,'endpoint',ar_canonical,slot)[1]
    raw_audit.raw_metrics(ar,request,ar_canonical,method='AR',checkpoint='endpoint',slot=ar_slot)
    raw_audit.raw_metrics(spec,request,frozen,method='F',checkpoint='endpoint',slot=slot)
    with pytest.raises(ValueError,match='canonical'):
        raw_audit.raw_metrics(spec,request,ar_canonical,method='F',checkpoint='endpoint',slot=slot)


def different_output_rows(*, every_request=False, termination_difference=True):
    rows=fixture()
    for row in rows:
        if not every_request and row['request_id']!='a':
            continue
        witness=row['ar_spec_comparison']
        witness.update(token_ids_equal=False, output_equal=False, first_difference=0)
        if termination_difference:
            witness.update(termination_equal=False, record_finish_reason='length',
                           record_output_tokens=row['output_tokens']+2)
        if row['method']=='AR':
            row['request_latency_s']*=2
            if termination_difference:
                row['output_tokens']+=2
                row['finish_reason']='length'
    return rows


@pytest.mark.parametrize('termination_difference',[True,False])
def test_cross_path_rows_preserved_but_excluded_from_identical_output_gain(termination_difference):
    rows=different_output_rows(termination_difference=termination_difference)
    result=analysis.summarize(rows,expected_requests=2)
    raw_audit.compare_reductions(result,raw_audit.reduce_endpoint(rows,expected_requests=2))
    assert result['ar_spec_output_agreement']['differing_outputs']==1
    contrast=result['contrasts']['C_CODE58_vs_AR']
    assert contrast['identical_output_requests']==contrast['different_output_requests']==1
    assert contrast['cache_maintenance_s_reduction_pct']['status']=='undefined'
    assert contrast['absolute_speedup']['value']==2
    assert contrast['all_request_time_ratio']['value']==3
    assert 'no_identical_output_gain_claim' in contrast['request_latency_s_reduction_pct']['status']
    assert result['bootstrap']['identical_output_subset']['input_order']==['b']
    assert result['requests']==2
    assert result['contrasts']['C_CODE58_vs_AREPO_CODE58']['wins']==1


def test_no_matching_outputs_has_no_identical_output_speedup():
    rows=different_output_rows(every_request=True)
    result=analysis.summarize(rows,expected_requests=2)
    raw_audit.compare_reductions(result,raw_audit.reduce_endpoint(rows,expected_requests=2))
    assert result['contrasts']['F_vs_AR']['absolute_speedup']['value'] is None
    assert result['contrasts']['F_vs_AR']['identical_output_requests']==0


@pytest.mark.parametrize('mutation',['repeat','observation','speculative'])
def test_cross_path_exception_cannot_admit_repeat_or_speculative_disagreement(mutation):
    rows=different_output_rows()
    if mutation=='repeat':
        next(row for row in rows if row['method']=='AR')['output_tokens']+=1
    elif mutation=='observation':
        rows[0]['ar_spec_comparison']['output_equal']=False
    else:
        for row in rows:
            if row['method']=='C_CODE58' and row['request_id']=='a': row['finish_reason']='length'
    for reducer in (analysis.reduce_requests,raw_audit.reduce_endpoint):
        with pytest.raises(ValueError): reducer(rows,expected_requests=2)


def test_final_report_discloses_cross_path_mismatches_and_subset(tmp_path,monkeypatch):
    rows=different_output_rows()
    independent=raw_audit.reduce_endpoint(rows,expected_requests=2)
    audit=dict(status='passed',scope='test',errors=[],evaluation=dict(status='passed',rows=rows,errors=[],
        independent_summary=independent),training=dict(arms={},event_curves=[],costs=[]),work=[])
    monkeypatch.setattr(raw_audit,'audit_run',lambda *args:audit)
    original=analysis.summarize
    monkeypatch.setattr(report.analysis,'summarize',lambda value:original(value,expected_requests=2))
    result=report.build_report(dict(execution_commit='fixture'),tmp_path)
    assert result['status']=='passed'
    text=(tmp_path/'public/REPORT.md').read_text()
    assert 'Differences remain for 1 requests' in text
    assert 'not identical-output speedup claims' in text
    assert 'Every AR repeat matches its own AR canonical' in text
    assert (tmp_path/'public/AR_SPEC_OUTPUT_COMPARISONS.csv').exists()
    assert 'All five endpoints preserve the target-only AR canonical' not in text
