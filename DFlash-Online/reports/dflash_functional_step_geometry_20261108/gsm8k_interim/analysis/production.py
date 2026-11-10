"""CPU-only audit of completed GSM8K FSG R1 production receipts.

This reads compact immutable evidence; it imports no model code and executes
no scientific forwards, optimization, or serving.
"""
from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import time

COMMIT = '6440e8ff5af1a845f830d65c884f6711d7e4a687'
ARMS = ('KPLUS', 'FSG', 'SCALAR')
EVENTS = (5, 6)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def distribution(values, population):
    finite = sorted(value for value in values if value is not None)
    if any(not math.isfinite(value) for value in finite):
        raise ValueError('Nonfinite recorded scalar')
    result = dict(n=len(finite), missing=population-len(finite), population=population)
    if not finite:
        return dict(result, minimum=None, q25=None, median=None, q75=None, maximum=None, mean=None)
    def quantile(q):
        position = (len(finite)-1)*q
        lo = math.floor(position)
        return finite[lo] + (position-lo)*(finite[min(lo+1, len(finite)-1)]-finite[lo])
    return dict(result, minimum=finite[0], q25=quantile(.25), median=quantile(.5),
                q75=quantile(.75), maximum=finite[-1], mean=statistics.fmean(finite))


def ratio(value, denominator):
    return value / denominator if value is not None and denominator else None


def independent_gate(row):
    """Re-evaluate the declared realized endpoint gate from recorded scalars."""
    candidate, native = row.get('candidate'), row['native']
    if native['native_norm'] == 0:
        return False, 'zero_native_displacement'
    if row.get('strength') == 0:
        return False, 'lambda_zero'
    if row.get('sketch', {}).get('sigma') == 0:
        return False, 'empty_sketch'
    if row.get('scalar_target_norm') == 0:
        return False, 'zero_hypothetical_displacement'
    if candidate['identical_native']:
        return False, 'identical_native_endpoint'
    if candidate['identical_prestate']:
        return False, 'rounded_to_prestate'
    if candidate['norm'] > native['native_norm']:
        return False, 'rounded_norm_exceeds_native'
    a, b = row['candidate_risk'], row['native_risk']
    if a['value']+a['eta'] <= b['value']-b['eta']:
        return True, 'actual_risk_decreased'
    overlap = a['value']-a['eta'] <= b['value']+b['eta'] and b['value']-b['eta'] <= a['value']+a['eta']
    return False, 'numerical_ambiguity' if overlap else 'actual_risk_increased'


def step_row(payload, event, arm):
    native = payload['native']
    denominator = native['native_norm']
    committed = payload['committed']
    accepted, reason = independent_gate(payload)
    assert payload['decision'] == dict(accepted=accepted, reason=reason)
    for key in ('identical_native', 'identical_prestate'):
        assert type(committed[key]) is bool
    nontrivial = not committed['identical_native'] and not committed['identical_prestate']
    assert accepted == nontrivial
    if not accepted:
        assert committed['identical_native']
    realization = payload.get('realization', {})
    if realization:
        assert realization['status'] == 'completed'
        assert realization['candidate'] == payload['feasible_fsg']
        assert realization['raw_candidate'] == payload['raw_fsg']
        if realization['search_ran']:
            assert realization['iterations'] == realization['probe_count'] == realization['completed_probes'] == 24
            assert realization['alpha'] == realization['bracket'][0]
            assert realization['candidate']['norm'] == realization['bracket_norms'][0] <= denominator
    protection = payload.get('responses', [])
    result = dict(event=event, arm=arm, global_step=payload['global_step'],
        window=payload['global_step']-(event-1)*128, status=payload['status'],
        native_nonzero=denominator > 0, protected_response_count=sum(r['protected_occurrences'] > 0 for r in protection),
        protected_occurrences=sum(r['protected_occurrences'] for r in protection),
        geometry_columns=payload.get('sketch', {}).get('usable_columns'),
        nonempty_sketch=payload.get('sketch', {}).get('sigma', 0) > 0,
        accepted=accepted, genuine_write=nontrivial, committed_identical_native=committed['identical_native'],
        committed_identical_prestate=committed['identical_prestate'], reason=reason,
        realization_bypass=realization.get('bypass'), search_ran=realization.get('search_ran', False),
        search_alpha=realization.get('alpha'), search_iterations=realization.get('iterations'),
        native_norm=denominator, ordinary_preclip_gradient_norm=payload['preclip_gradient_norm'],
        final_candidate_check=payload.get('final_candidate_check'),
        candidate_replayed=payload.get('final_candidate_check') == 'actual_bf16_replay',
        scalar_matching_exercised=payload.get('scalar_matching_exercised'),
        scalar_target_norm=payload.get('scalar_target_norm'),
        protected_weight_sum=sum(r['weight_sum'] for r in protection),
        margin_floor_occurrences=sum(r['margin_floor_occurrences'] for r in protection),
        margin_floor_weight=sum(r['margin_floor_weight'] for r in protection),
        tie_occurrences=sum(r['tie_occurrences'] for r in protection), tie_weight=sum(r['tie_weight'] for r in protection),
        hook_total_s=payload['hook_total_s'])
    for name, source, norm_key in (
        ('continuous_raw', 'raw_fsg', 'continuous_norm'), ('raw_bf16', 'raw_fsg', 'norm'),
        ('feasible', 'feasible_fsg', 'norm'), ('actual_candidate', 'candidate', 'norm'), ('committed', 'committed', 'norm')):
        data = payload.get(source, {})
        result[name+'_norm'] = data.get(norm_key)
        result[name+'_norm_ratio'] = ratio(data.get(norm_key), denominator)
        if name != 'continuous_raw':
            result[name+'_direction_cosine'] = data.get('direction_cosine')
    scalar = payload.get('scalar', {})
    for key in ('alpha', 'absolute_norm_mismatch', 'relative_norm_mismatch'):
        result['scalar_'+key] = scalar.get(key)
    for name in ('native', 'candidate'):
        risk = payload.get(name+'_risk')
        result[name+'_kl'] = risk.get('value') if risk is not None else None
        result[name+'_kl_eta'] = risk.get('eta') if risk is not None else None
        response_risks = payload.get(name+'_response_risks')
        if risk is not None and response_risks is not None:
            assert len(response_risks) == 2
            assert .5*math.fsum(r['risk_unaveraged'] for r in response_risks) == risk['raw']
        for index in range(2):
            result[f'{name}_response_{index}_kl'] = response_risks[index]['risk_unaveraged'] if response_risks else None
    result['candidate_native_kl_ratio'] = ratio(result['candidate_kl'], result['native_kl'])
    for key, value in payload.get('costs', {}).items():
        result['nested_cost_'+key] = value
    for key, value in realization.get('costs', {}).items():
        result['nested_cost_realization_'+key] = value
    for key in ('probe_count', 'norm_scan_passes', 'construction_scan_passes', 'identity_scan_passes', 'write_passes'):
        result['feasibility_'+key] = realization.get(key)
    return result


def summarize(rows, arm, event):
    subset = [row for row in rows if row['arm'] == arm and (event == 'combined' or row['event'] == event)]
    count = len(subset)
    expected = 256 if event == 'combined' else 128
    assert count == expected
    def number(key):
        return sum(bool(row[key]) for row in subset)
    result = dict(arm=arm, event=event, scheduled_windows=expected, present_windows=count,
        native_nonzero=number('native_nonzero'), protected_windows=sum(r['protected_response_count'] > 0 for r in subset),
        nonempty_sketch=number('nonempty_sketch'), candidate_replayed=number('candidate_replayed'),
        genuine_accepted_writes=number('genuine_write'), native_endpoint_windows=number('committed_identical_native'),
        prestate_endpoint_windows=number('committed_identical_prestate'),
        genuine_write_fraction=number('genuine_write')/count, searches=number('search_ran'),
        search_fraction=number('search_ran')/count,
        reasons=dict(Counter(r['reason'] for r in subset)),
        realization_bypasses=dict(Counter(r['realization_bypass'] or 'none' for r in subset)),
        final_candidate_checks=dict(Counter(r['final_candidate_check'] or 'unavailable' for r in subset)),
        scalar_matching_exercised=number('scalar_matching_exercised') if arm == 'SCALAR' else None,
        protected_occurrences=sum(r['protected_occurrences'] for r in subset),
        margin_floor_occurrences=sum(r['margin_floor_occurrences'] for r in subset),
        margin_floor_weight_sum=sum(r['margin_floor_weight'] for r in subset),
        tie_occurrences=sum(r['tie_occurrences'] for r in subset),
        tie_weight_sum=sum(r['tie_weight'] for r in subset),
        protected_weight_sum=sum(r['protected_weight_sum'] for r in subset))
    result['margin_floor_weight_fraction'] = ratio(result['margin_floor_weight_sum'], result['protected_weight_sum'])
    result['tie_weight_fraction'] = ratio(result['tie_weight_sum'], result['protected_weight_sum'])
    columns = [key for key in subset[0] if key.endswith(('_norm_ratio', '_direction_cosine'))]
    columns += ['search_alpha', 'scalar_alpha', 'scalar_absolute_norm_mismatch', 'scalar_relative_norm_mismatch',
                'native_kl', 'candidate_kl', 'candidate_native_kl_ratio', 'ordinary_preclip_gradient_norm']
    result['distributions'] = {key: distribution([r.get(key) for r in subset], count) for key in columns}
    searched = [r['search_alpha'] for r in subset if r['search_ran']]
    result['search_only_alpha'] = distribution(searched, len(searched))
    result['timing_components_s'] = {key: math.fsum(r.get(key, 0.) for r in subset)
        for key in set().union(*(r.keys() for r in subset)) if key.startswith('nested_cost_')}
    result['hook_total_s_includes_native_optimizer_and_diagnostics'] = math.fsum(r['hook_total_s'] for r in subset)
    return result


def main(root, output):
    started = time.perf_counter()
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    def read(relative):
        path = root/relative
        data = path.read_bytes()
        inputs[str(relative)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return json.loads(data)
    plan = read('sealed/PLAN.json')
    assert plan['execution_source']['commit'] == COMMIT
    rows, costs, receipts = [], [], []
    schedules = {}
    for event in EVENTS:
        for arm in ARMS:
            base = Path(f'campaign_001/gsm8k/event_{event:03d}/{arm}')
            result = read(base/'RESULT.json')
            status = read(base/'STATUS.json')
            resource = read(base/'RESOURCE_COSTS.json')
            startup = read(base/'startup/RESULT.json')
            assert result['status'] == status['status'] == startup['status'] == 'completed'
            assert result['start_step'] == (event-1)*128 and result['end_step'] == event*128
            assert result['optimizer_steps'] == 128 and result['response_presentations'] == 256
            assert result['method'] == arm and result['dataset'] == 'gsm8k'
            schedule = [{k: row[k] for k in ('record_id', 'pass', 'sampled_anchors')} for row in result['records']]
            if arm == 'KPLUS':
                schedules[event] = schedule
            else:
                assert schedule == schedules[event]
            cost = dict(event=event, arm=arm, **result['costs'], loading_s=status['loading_s'],
                startup_wall_s=startup['wall_s'], worker_wall_s=status['wall_s'],
                peak_allocated_bytes=resource['training_peak_allocated_bytes'],
                peak_reserved_bytes=resource['training_peak_reserved_bytes'])
            assert math.isclose(cost['training_s']-cost['diagnostic_observer_s'],
                                cost['training_excluding_diagnostic_observer_s'], rel_tol=0., abs_tol=1e-9)
            costs.append(cost)
            receipts.append(dict(event=event, arm=arm, checkpoint=result['checkpoint_path'],
                weights_identity=result['weights_identity'], state_identity=result['state_identity']))
            if arm == 'KPLUS':
                continue
            geometry = result['objective_work']['functional_geometry']
            embedded = geometry['steps']
            assert len(embedded) == 128
            assert [r['global_step'] for r in embedded] == list(range((event-1)*128+1, event*128+1))
            for item in embedded:
                saved = read(base/f"geometry/step_{item['global_step']:06d}.json")
                assert saved == item
                assert saved['status'] == 'completed' and saved['arm'] == arm
                assert saved['realization_revision'] == 'norm_feasible_bf16_r1'
                rows.append(step_row(saved, event, arm))
    summaries = [summarize(rows, arm, event) for arm in ('FSG', 'SCALAR') for event in (*EVENTS, 'combined')]
    for cost in costs:
        baseline = next(r for r in costs if r['event'] == cost['event'] and r['arm'] == 'KPLUS')
        cost['intrinsic_training_ratio_vs_KPLUS'] = cost['training_excluding_diagnostic_observer_s']/baseline['training_excluding_diagnostic_observer_s']
    cost_summary = []
    for arm in ARMS:
        selected = [r for r in costs if r['arm'] == arm]
        summary = dict(arm=arm, events=[5, 6], event_count=len(selected))
        for key in ('training_s', 'diagnostic_observer_s', 'training_excluding_diagnostic_observer_s',
                    'loading_s', 'startup_wall_s', 'worker_wall_s', 'checkpoint_save_s', 'export_s'):
            summary[key] = math.fsum(r[key] for r in selected)
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            summary[key] = max(r[key] for r in selected)
        summary['timing_scope'] = 'Each field is its own envelope or nested component; do not sum all fields.'
        cost_summary.append(summary)
    baseline = cost_summary[0]['training_excluding_diagnostic_observer_s']
    for value in cost_summary:
        value['intrinsic_training_ratio_vs_KPLUS'] = value['training_excluding_diagnostic_observer_s']/baseline
    write_csv(output/'PRODUCTION_WINDOWS.csv', rows)
    write_json(output/'PRODUCTION_SUMMARY.json', summaries)
    write_csv(output/'ACTIVITY_SUMMARY.csv', [{key: value for key, value in item.items()
        if key not in ('distributions', 'search_only_alpha', 'timing_components_s')} for item in summaries])
    write_csv(output/'SCALAR_DISTRIBUTIONS.csv', [dict(arm=item['arm'], event=item['event'], variable=name, **value)
        for item in summaries for name, value in {**item['distributions'], 'search_only_alpha': item['search_only_alpha']}.items()])
    write_csv(output/'EVENT_COSTS.csv', costs)
    write_json(output/'TRAINING_COST_SUMMARY.json', cost_summary)
    write_json(output/'CHECKPOINT_IDENTITIES.json', receipts)
    write_json(output/'INPUTS.json', inputs)
    write_json(output/'ANALYSIS_RECEIPT.json', dict(status='completed', execution_commit=COMMIT,
        binding=plan['binding'], filtered_windows=len(rows), event_receipts=len(receipts),
        exact_embedded_step_matches=len(rows), independent_gate_matches=len(rows),
        new_model_work=0, gpu_work=0, analysis_wall_s=time.perf_counter()-started))
    print(json.dumps(dict(filtered_windows=len(rows), summaries=summaries, costs=costs), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.evidence, args.output)
