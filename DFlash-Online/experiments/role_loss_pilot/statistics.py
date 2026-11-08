"""Per-dataset paired-request statistics for the fixed early-adaptation pilot."""
from collections import defaultdict
import numpy as np

from .evaluation import METHODS, DATASETS, PANEL_SIZE
from experiments.full_k_gsm8k.statistics import comparison, method_totals as base_method_totals

COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens', 'native_accepted_draft_tokens',
                'committed_progress', 'output_tokens')
TIMINGS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
BOOTSTRAP_SEED = 20261107
RESAMPLES = 10000

def reduce_stage(rows, *, dataset, event):
    """Average repeats within request; keep deterministic native counts once."""
    if dataset not in DATASETS or event not in (2, 4):
        raise ValueError("Unknown dataset or evaluation event")
    repeats = 1 if event == 2 else 2
    grouped = defaultdict(list)
    identities = {}
    for row in rows:
        if (row['phase'] != ('final' if event == 4 else 'intermediate') or row['event'] != event or row['dataset'] != dataset or row['seed'] != 17
                or row['method'] not in METHODS or row['validation_status'] != 'valid'
                or row['valid_measurement'] is not True or row['errors']):
            raise ValueError('Only valid same-dataset/stage executions enter a comparison')
        ordinal, request_id = row['panel_ordinal'], row['request_id']
        if ordinal in identities and identities[ordinal] != request_id:
            raise ValueError('A panel ordinal has multiple request identities')
        identities[ordinal] = request_id
        grouped[ordinal, row['method']].append(row)
    if (len(identities) != PANEL_SIZE or sorted(identities) != list(range(PANEL_SIZE))
            or len(set(identities.values())) != len(identities)):
        raise ValueError('Missing or duplicated paired request identities')
    paired = []
    for ordinal, request_id in sorted(identities.items()):
        value = dict(seed=17, dataset=dataset, event=event, panel_ordinal=ordinal, request_id=request_id)
        for method in METHODS:
            records = sorted(grouped[ordinal, method], key=lambda row: row['repeat'])
            if [row['repeat'] for row in records] != list(range(repeats)):
                raise ValueError('Comparison requires every distinct planned repeat for every arm')
            first = records[0]
            for field in COUNT_FIELDS:
                if (type(first[field]) is not int or first[field] < 0
                        or any(row[field] != first[field] for row in records)):
                    raise ValueError('Invalid or inconsistent repeated counts')
            if first['verification_calls'] <= 0:
                raise ValueError('Verification denominator must be positive')
            if (first['accepted_draft_tokens'] > first['committed_progress']
                    or first['native_accepted_draft_tokens'] != first['committed_progress'] - first['verification_calls']
                    or first['committed_progress'] != first['output_tokens'] - 1):
                raise ValueError('Native acceptance or committed-progress accounting differs')
            if (any(row['checkpoint_id'] != first['checkpoint_id'] or row['finish_reason'] != first['finish_reason']
                    or row['cap_hit'] != first['cap_hit'] for row in records)
                    or type(first['cap_hit']) is not bool or first['cap_hit'] != (first['finish_reason']=='length')
                    or first['finish_reason'] not in ('eos','length')
                    or (first['cap_hit'] and first['output_tokens'] != 2048)):
                raise ValueError('Checkpoint or termination changes within a method')
            times = {field: float(np.mean([row[field] for row in records])) for field in TIMINGS}
            if any(not np.isfinite(row[field]) or row[field] < 0
                    or (field != 'cache_maintenance_s' and row[field] == 0)
                    for row in records for field in TIMINGS):
                raise ValueError('Invalid latency measurement')
            value[method] = {field: first[field] for field in COUNT_FIELDS}
            value[method].update(**times, cap_hit=first['cap_hit'], finish_reason=first['finish_reason'],
                                 checkpoint_id=first['checkpoint_id'], count_trajectory_repeat=0, timing_repeats=repeats)
        if len({(value[m]['output_tokens'], value[m]['finish_reason']) for m in METHODS}) != 1:
            raise ValueError('Paired output length or termination differs')
        paired.append(value)
    return paired


def method_totals(values):
    """Keep the native counter distinct from stop-clipped matching acceptance."""
    total = base_method_totals(values)
    total['native_accepted_draft_tokens'] = int(values['native_accepted_draft_tokens'].sum())
    total['native_accepted_tokens_per_call'] = total['native_accepted_draft_tokens'] / total['verification_calls']
    total['matched_accepted_tokens_per_call'] = total['accepted_tokens_per_call']
    return total


def summarize(rows, *, dataset, event):
    paired = reduce_stage(rows, dataset=dataset, event=event)
    repeats = 1 if event == 2 else 2
    arrays = {method: {field: np.array([row[method][field] for row in paired], dtype=np.float64)
              for field in (*COUNT_FIELDS, *TIMINGS)} for method in METHODS}
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(RESAMPLES, len(paired)))
    effects = {a + '_vs_' + b: comparison(arrays[a], arrays[b], indices)
               for a, b in (('ROLE', 'KPLUS'), ('MIX', 'KPLUS'), ('ROLE', 'MIX'))}
    for name, effect in effects.items():
        candidate, reference = name.split('_vs_')
        delta = arrays[reference]['request_latency_s']-arrays[candidate]['request_latency_s']
        effect.update(request_latency_wins=int((delta>0).sum()),
                      request_latency_regressions=int((delta<0).sum()), request_latency_ties=int((delta==0).sum()))
    totals = {method: method_totals(values) for method, values in arrays.items()}
    for method in METHODS:
        totals[method]['cap_hit_requests'] = sum(row[method]['cap_hit'] for row in paired)
    repeat_totals = {str(repeat): {method: {field: sum(row[field] for row in rows
        if row['repeat'] == repeat and row['method'] == method) for field in (*COUNT_FIELDS, *TIMINGS)}
        for method in METHODS} for repeat in range(repeats)}
    return dict(dataset=dataset, event=event, request_count=len(paired), bootstrap=dict(resamples=RESAMPLES, seed=BOOTSTRAP_SEED,
        generator='PCG64', percentiles=[2.5, 97.5], method='linear',
        unit='One paired request carrying every arm; repeats form one mean time.'),
        method_totals=totals,
        comparisons=effects, repeat_totals=repeat_totals,
        primary='ROLE_vs_KPLUS' if event == 4 else None,
        intervals='Unadjusted descriptive intervals; event 2 does not select checkpoints.',
        timing_scope='Fresh contemporaneous checkpoint-ready serving; not training-inclusive speedup.',
        comparison_scope='Own-feedback recipe comparison, not identical-feature replay.')


def recommend(endpoints):
    """One conservative action from both endpoints, never from a pooled mean."""
    if set(endpoints) != set(DATASETS):
        return dict(outcome="invalid_or_incomplete", next_action="Resolve the concrete evidence failure before interpreting efficacy.", automatic_followup_authorized=False)
    effects = {d: endpoints[d]["comparisons"] for d in DATASETS}
    def favorable(dataset, contrast):
        e = effects[dataset][contrast]
        return all(e[key]["point_pct"] > 0 for key in ("call_reduction", "request_latency_s_reduction"))
    role = [favorable(d, "ROLE_vs_KPLUS") for d in DATASETS]
    mix = [favorable(d, "MIX_vs_KPLUS") for d in DATASETS]
    split = [favorable(d, "ROLE_vs_MIX") for d in DATASETS]
    clear_regression = all(effects[d][c]["request_latency_s_reduction"]["ci95_pct"][1] < 0
                           for d in DATASETS for c in ("ROLE_vs_KPLUS", "MIX_vs_KPLUS"))
    reliable_role = all(effects[d]["ROLE_vs_KPLUS"]["request_latency_s_reduction"]["ci95_pct"][0] > 0 for d in DATASETS)
    if clear_regression:
        outcome, action = "both_candidates_regress", "Keep KPLUS; do not launch a coefficient sweep."
    elif all(role) and all(split):
        outcome = "ROLE_point_estimates_favor_both_datasets"
        action = ("Prioritize a separately agreed longer ROLE confirmation; "
                  + ("primary latency intervals favor ROLE here." if reliable_role else "the intervals still limit reliability."))
    elif any(role) != all(role) or any(mix) != all(mix):
        outcome, action = "workload_dependent", "Inspect the fixed role/gradient diagnostics to locate the dataset difference before a separately agreed revision."
    elif all(mix) and all(effects[d]["ROLE_vs_MIX"]["request_latency_s_reduction"]["ci95_pct"][1] < 0 for d in DATASETS):
        outcome, action = "MIX_exceeds_ROLE", "Prefer MIX for review; role specialization is not supported."
    elif all(mix) and all(role):
        outcome, action = "both_selected_losses_improve_without_established_role_benefit", "Review changing the selected-position loss; this pilot does not establish role specialization."
    else:
        outcome, action = "inconclusive_early_adaptation_pilot", "Keep KPLUS pending a separately agreed confirmation; this is not proof of no full-run benefit."
    return dict(outcome=outcome, next_action=action, automatic_followup_authorized=False,
                primary_by_dataset={d: effects[d]["ROLE_vs_KPLUS"]["request_latency_s_reduction"] for d in DATASETS})
