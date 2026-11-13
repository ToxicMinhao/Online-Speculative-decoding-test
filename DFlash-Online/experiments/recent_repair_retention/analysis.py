"""Precommitted request-paired endpoints for the one-event retention pilot."""
from collections import defaultdict
import math
import numpy as np

ARMS = ('NATIVE', 'RECENT', 'CONTROL')
ANALYSIS = dict(seed=1707, bootstrap_draws=10000, percentile_method='linear',
    minimum_relative_accepted_length_gain_pct=0.50,
    matching_minimum_slots=32, matching_minimum_requests=16,
    matching_minimum_switched_mass_fraction=0.10, matching_maximum_absolute_smd=0.10,
    count_definition='clipped accepted draft proposals divided by verification calls',
    sample_unit='whole request; counts once, serving time mean over two sweeps',
    primary_comparisons=['RECENT_vs_NATIVE', 'RECENT_vs_CONTROL'])


def paired_requests(rows):
    grouped = defaultdict(list)
    for row in rows:
        if (not row.get('valid_measurement') or row.get('errors')
                or row['method'] not in ARMS):
            raise ValueError('Invalid or unbound measurement cannot enter gains')
        grouped[row['request_id'], row['method']].append(row)
    requests = sorted({key[0] for key in grouped})
    output = []
    for rid in requests:
        for arm in ARMS:
            repeats = sorted(grouped[rid, arm], key=lambda row: row['repeat'])
            if [r['repeat'] for r in repeats] != [0, 1]:
                raise ValueError('Require two distinct validated sweeps for every arm/request')
            fields = ('verification_calls', 'accepted_draft_tokens', 'output_tokens',
                      'committed_progress', 'finish_reason')
            if any(repeats[0][key] != repeats[1][key] for key in fields):
                raise ValueError('Counts or termination differ between timing repeats')
            output.append(dict(request_id=rid, method=arm,
                **{key:repeats[0][key] for key in fields},
                **{key:sum(r[key] for r in repeats)/2 for key in
                   ('request_latency_s','native_generate_s','cache_maintenance_s')}))
    return output


def _interval(samples):
    finite = np.asarray(samples, dtype=np.float64)
    valid = np.isfinite(finite)
    values = finite[valid]
    return dict(lower=float(np.quantile(values, .025, method='linear')) if len(values) else None,
                upper=float(np.quantile(values, .975, method='linear')) if len(values) else None,
                valid_draws=int(valid.sum()), undefined_draws=int((~valid).sum()))


def _ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    return np.divide(numerator, denominator, out=np.full(np.broadcast_shapes(
        numerator.shape, denominator.shape), np.nan), where=denominator != 0)


def compare(rows, other, *, draws=10000):
    own = {r['request_id']:r for r in rows if r['method']=='RECENT'}
    ref = {r['request_id']:r for r in rows if r['method']==other}
    if not own or own.keys() != ref.keys():
        raise ValueError('Require identical nonempty paired request populations')
    ids = sorted(own)
    def vector(arm, key): return np.asarray([arm[rid][key] for rid in ids], dtype=np.float64)
    ca, cb = vector(own,'verification_calls'), vector(ref,'verification_calls')
    aa, ab = vector(own,'accepted_draft_tokens'), vector(ref,'accepted_draft_tokens')
    ta, tb = vector(own,'request_latency_s'), vector(ref,'request_latency_s')
    rng = np.random.Generator(np.random.PCG64(1707))
    index = rng.integers(0,len(ids),size=(draws,len(ids)))
    accepted_gain = 100*(_ratio(_ratio(aa[index].sum(1),ca[index].sum(1)),
                               _ratio(ab[index].sum(1),cb[index].sum(1)))-1)
    time_gain = 100*(1-_ratio(ta[index].sum(1),tb[index].sum(1)))
    calls_gain = 100*(1-_ratio(ca[index].sum(1),cb[index].sum(1)))
    def number(value):
        return float(value) if np.isfinite(value) else None
    return dict(reference=other, requests=len(ids), recent_calls=int(ca.sum()),
        reference_calls=int(cb.sum()), recent_accepted=int(aa.sum()), reference_accepted=int(ab.sum()),
        recent_accepted_per_call=number(_ratio(aa.sum(),ca.sum())),
        reference_accepted_per_call=number(_ratio(ab.sum(),cb.sum())),
        accepted_length_relative_gain_pct=number(100*(_ratio(_ratio(aa.sum(),ca.sum()),
                                                    _ratio(ab.sum(),cb.sum()))-1)),
        accepted_length_interval=_interval(accepted_gain),
        call_reduction_pct=number(100*(1-_ratio(ca.sum(),cb.sum()))),
        call_reduction_interval=_interval(calls_gain),
        gross_calls_saved=int(np.maximum(cb-ca,0).sum()), gross_calls_added=int(np.maximum(ca-cb,0).sum()),
        requests_improved=int((ca<cb).sum()),requests_regressed=int((ca>cb).sum()),requests_tied=int((ca==cb).sum()),
        recent_request_s=float(ta.sum()), reference_request_s=float(tb.sum()),
        speedup=number(_ratio(tb.sum(),ta.sum())), time_reduction_pct=number(100*(1-_ratio(ta.sum(),tb.sum()))),
        time_reduction_interval=_interval(time_gain))


def fixed_start_summary(starts, outcomes, *, draws=10000, request_ids=None):
    """Resample whole requests; earlier breaks remain in all repair denominators."""
    index = {(row['request_id'],row['canonical_start'],row['method']):row for row in outcomes}
    if len(index)!=len(outcomes) or len(outcomes)!=3*len(starts) or not all(row.get('valid') for row in outcomes):
        raise ValueError('Incomplete or duplicated fixed-start endpoints')
    ids = sorted(set(request_ids) if request_ids is not None else {row['request_id'] for row in starts})
    if not {row['request_id'] for row in starts} <= set(ids):
        raise ValueError('A fixed start lies outside the sealed request population')
    fields = ('exact_slot_relapse','earlier_break','accepts_through','direct_canonical_agreement',
              'passes_parent_first_blocker','acquired_prefix_occurrences','lost_prefix_occurrences','net_prefix_occurrences')
    per_request=[]
    for rid in ids:
        parent=[s for s in starts if s['request_id']==rid]
        for arm in ARMS:
            rows=[index[rid,s['canonical_start'],arm] for s in parent]
            per_request.append(dict(request_id=rid,method=arm,starts=len(parent),
                repairs=sum(s['repair_slot'] is not None for s in parent),
                blockers=sum(s['genuine_k6_rejection'] for s in parent),
                **{field:sum(row[field] for row in rows if row[field] is not None) for field in fields}))
    totals={arm:{field:sum(r[field] for r in per_request if r['method']==arm)
                 for field in ('starts','repairs','blockers',*fields)} for arm in ARMS}
    rng=np.random.Generator(np.random.PCG64(1707))
    resample=rng.integers(0,len(ids),size=(draws,len(ids))) if ids else np.empty((draws,0),dtype=int)
    def vector(arm,key):return np.asarray([next(r[key] for r in per_request if r['method']==arm and r['request_id']==rid) for rid in ids],dtype=float)
    contrasts={}
    for other in ('NATIVE','CONTROL'):
        own={}
        for key in fields:
            denom='repairs' if key in fields[:4] else 'blockers' if key=='passes_parent_first_blocker' else 'starts'
            a,b=vector('RECENT',key),vector(other,key)
            n=vector('RECENT',denom)
            diff=_ratio((a[resample]-b[resample]).sum(1),n[resample].sum(1))
            point=_ratio(a.sum()-b.sum(),n.sum())
            own[key]=dict(difference_per_eligible_start=float(point) if np.isfinite(point) else None,
                          interval=_interval(diff),denominator=int(n.sum()))
        contrasts[other]=own
    return dict(totals=totals,contrasts=contrasts,per_request=per_request,
                interpretation='Local native prediction changes; never a fraction of time saved.')


def decision(comparisons, fixed, placement_passed):
    native, control = comparisons['NATIVE'], comparisons['CONTROL']
    lower=native['accepted_length_interval']['lower']
    upper=native['accepted_length_interval']['upper']
    point=native['accepted_length_relative_gain_pct']
    time_upper=native['time_reduction_interval']['upper']
    totals=fixed['totals']
    retention=totals['RECENT']['repairs']>0 and all(
        totals['RECENT']['exact_slot_relapse']<totals[m]['exact_slot_relapse']
        and totals['RECENT']['accepts_through']>totals[m]['accepts_through'] for m in ('NATIVE','CONTROL'))
    checks=dict(native_gain_at_least_half_percent=point is not None and point>=.50,
        native_lower_bound_positive=lower is not None and lower>0,
        placement_screen=bool(placement_passed),
        control_lower_bound_positive=control['accepted_length_interval']['lower'] is not None and control['accepted_length_interval']['lower']>0,
        held_out_repair_direction=retention,
        no_demonstrated_slowdown=time_upper is not None and time_upper>=0)
    if all(checks.values()): value='advance_to_separately_designed_confirmation'
    elif time_upper is not None and time_upper<0:value='fails_serving_criterion_slower'
    elif upper is not None and upper<.50:value='prespecified_useful_size_unsupported_on_this_fork'
    elif retention and point is not None and point<=0:value='local_retention_without_net_serving_benefit_closes_version'
    elif checks['native_gain_at_least_half_percent'] and checks['native_lower_bound_positive'] and not (checks['placement_screen'] and checks['control_lower_bound_positive']):value='performance_observation_history_specific_explanation_unresolved'
    else:value='inconclusive_or_failed_joint_pilot_screen'
    return dict(decision=value,checks=checks,automatic_followup=False,
        limits='Single trained seed and shared-feedback fork; no own-feedback robustness or training-inclusive speedup established.')
