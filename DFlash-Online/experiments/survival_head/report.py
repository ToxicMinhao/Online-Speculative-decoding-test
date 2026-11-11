"""CPU-only survival pilot report: paired complete requests and shared first rounds."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from experiments.full_k_gsm8k.statistics import comparison, method_totals as base_method_totals
from .contract import ARMS, DATASETS, PANEL_SIZE, ANALYSIS, budgets, expected_work
from .evaluation import audit_stage, evaluation_schedule, schedule_balance
METHODS = ARMS

def file_identity(path):
    payload=Path(path).read_bytes()
    return dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


class Evidence:
    def __init__(self,root,plan):
        self.root,self.plan=Path(root),plan
        self.files,self.missing,self.errors={},{},[]

    def remember(self,path,observed):
        name=str(path)
        if name in self.files and self.files[name]!=observed:
            self.errors.append(dict(path=name,error='Input changed between reads'))
        self.files.setdefault(name,observed)

    def local(self,path):
        path=Path(path)
        if path.is_absolute():
            try: return self.root/path.relative_to(self.plan['root'])
            except ValueError: return path
        return self.root/path

    def read(self,path,*,lines=False):
        path=self.local(path)
        if not path.is_file():
            self.missing[str(path)]='No retained receipt; not imputed as zero or success.'
            return None
        raw=path.read_bytes()
        self.remember(path,dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        try:
            return [json.loads(row) for row in raw.splitlines() if row.strip()] if lines else json.loads(raw)
        except (ValueError,TypeError) as error:
            self.errors.append(dict(path=str(path),error=repr(error)))
            return None

    def remember_tree(self,path):
        """Hash compact records only; never load weights or tensor diagnostics."""
        path=self.local(path)
        for file in sorted(path.rglob('*')) if path.exists() else []:
            if file.is_file() and file.suffix in ('.json','.jsonl','.csv'):
                self.remember(file,file_identity(file))

    def stability(self):
        changed=[]
        for name,expected in self.files.items():
            path=Path(name)
            if not path.is_file() or file_identity(path)!=expected: changed.append(name)
        return dict(status='passed' if not changed else 'changed_during_analysis',changed=changed)


def dump_json(path,value):
    def safe(item):
        if isinstance(item,float) and not math.isfinite(item):
            return {'nonfinite_saved_value':repr(item)}
        if isinstance(item,dict): return {key:safe(v) for key,v in item.items()}
        if isinstance(item,list): return [safe(v) for v in item]
        return item
    Path(path).write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n')


def dump_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or ['status'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})



COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens', 'native_accepted_draft_tokens',
                'committed_progress', 'output_tokens')
TIMINGS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
BOOTSTRAP_SEED = ANALYSIS['seed']
RESAMPLES = ANALYSIS['resamples']

def reduce_stage(rows, *, dataset, event, panel_size=PANEL_SIZE, allow_seed_eos=False):
    """Average repeats within request; keep deterministic native counts once."""
    if dataset not in DATASETS or event != 5:
        raise ValueError("Unknown dataset or evaluation event")
    repeats = 2
    grouped = defaultdict(list)
    identities = {}
    for row in rows:
        if (row['phase'] != 'final' or row['event'] != event or row['dataset'] != dataset or row['seed'] != 17
                or row['method'] not in METHODS or row['validation_status'] != 'valid'
                or row['valid_measurement'] is not True or row['errors']):
            raise ValueError('Only valid same-dataset/stage executions enter a comparison')
        ordinal, request_id = row['panel_ordinal'], row['request_id']
        if ordinal in identities and identities[ordinal] != request_id:
            raise ValueError('A panel ordinal has multiple request identities')
        identities[ordinal] = request_id
        grouped[ordinal, row['method']].append(row)
    if (len(identities) != panel_size or sorted(identities) != list(range(panel_size))
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
            seed_eos = (allow_seed_eos and first['verification_calls'] == 0
                and first['output_tokens'] == 1 and first['committed_progress'] == 0
                and first.get('finish_reason') == 'eos' and first.get('first_round_present') is False)
            if first['verification_calls'] <= 0 and not seed_eos:
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
            first_fields = ('first_round_present', 'first_round_start_digest',
                'first_round_accepted_prefix_tokens', 'first_round_raw_prefix_tokens',
                'first_round_output_start')
            if any(row.get(field) != first.get(field) for row in records for field in first_fields):
                raise ValueError('First native round disagrees across timing repeats')
            if type(first.get('first_round_present')) is not bool:
                raise ValueError('Missing shared first-round evidence')
            for field in ('first_round_accepted_prefix_tokens', 'first_round_raw_prefix_tokens'):
                if type(first.get(field)) is not int or not 0 <= first[field] <= 15:
                    raise ValueError('Invalid first-round acceptance')
            if first['first_round_accepted_prefix_tokens'] > first['first_round_raw_prefix_tokens']:
                raise ValueError('Stop-clipped acceptance exceeds raw prefix')
            if first['first_round_present']:
                if first['first_round_output_start'] != 0 or not isinstance(first['first_round_start_digest'], str) or len(first['first_round_start_digest']) != 64:
                    raise ValueError('Invalid shared first-round prefix identity')
            elif any(first[field] for field in first_fields[1:]):
                raise ValueError('Absent first round has invented acceptance')
            value[method].update({field: first[field] for field in first_fields})
        if len({(value[m]['output_tokens'], value[m]['finish_reason']) for m in METHODS}) != 1:
            raise ValueError('Paired output length or termination differs')
        if len({(value[m]['first_round_present'], value[m]['first_round_start_digest']) for m in METHODS}) != 1:
            raise ValueError('First rounds do not share the same starting prefix')
        paired.append(value)
    return paired


def method_totals(values):
    """Keep the native counter distinct from stop-clipped matching acceptance."""
    total = base_method_totals(values)
    total['native_accepted_draft_tokens'] = int(values['native_accepted_draft_tokens'].sum())
    total['native_accepted_tokens_per_call'] = total['native_accepted_draft_tokens'] / total['verification_calls']
    total['matched_accepted_tokens_per_call'] = total['accepted_tokens_per_call']
    return total


def summarize(rows, *, dataset, event, panel_size=PANEL_SIZE, allow_seed_eos=False):
    paired = reduce_stage(rows, dataset=dataset, event=event, panel_size=panel_size,
                         allow_seed_eos=allow_seed_eos)
    repeats = 2
    arrays = {method: {field: np.array([row[method][field] for row in paired], dtype=np.float64)
              for field in (*COUNT_FIELDS, *TIMINGS)} for method in METHODS}
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(RESAMPLES, len(paired)))
    effects = {a + '_vs_' + b: comparison(arrays[a], arrays[b], indices)
               for a, b in (('SURVIVAL', 'KPLUS'), ('SURVIVAL', 'LOCAL'), ('LOCAL', 'KPLUS'))}
    for name, effect in effects.items():
        candidate, reference = name.split('_vs_')
        delta = arrays[reference]['request_latency_s']-arrays[candidate]['request_latency_s']
        effect.update(request_latency_wins=int((delta>0).sum()),
                      request_latency_regressions=int((delta<0).sum()), request_latency_ties=int((delta==0).sum()))
        prefix = np.array([row[candidate]['first_round_accepted_prefix_tokens'] -
                           row[reference]['first_round_accepted_prefix_tokens'] for row in paired])
        effect['first_native_round'] = dict(requests=len(prefix),
            present_requests=sum(row[candidate]['first_round_present'] for row in paired),
            seed_eos_requests=sum(not row[candidate]['first_round_present'] for row in paired),
            wins=int((prefix>0).sum()), losses=int((prefix<0).sum()), ties=int((prefix==0).sum()),
            gross_prefix_tokens_gained=int(np.maximum(prefix, 0).sum()),
            gross_prefix_tokens_lost=int(np.maximum(-prefix, 0).sum()),
            net_prefix_tokens_gained=int(prefix.sum()),
            scope='First native round at authenticated shared starting prefix; stop-clipped acceptance')
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
        primary=['SURVIVAL_vs_KPLUS', 'SURVIVAL_vs_LOCAL'], controls=['LOCAL_vs_KPLUS'],
        intervals='Unadjusted paired-request intervals; no familywise confirmation claim.',
        timing_scope='Fresh contemporaneous checkpoint-ready serving; not training-inclusive speedup.',
        comparison_scope='Shared saved KPLUS4 feedback; one continuing training seed and an input-selected reused development subset.')


def recommend(summary):
    """Apply the supplied decision rule; never start a confirmation automatically."""
    if summary is None:
        return dict(outcome='invalid_or_incomplete', automatic_followup_authorized=False,
                    fresh_confirmation_eligible=False)
    primary = ('SURVIVAL_vs_KPLUS', 'SURVIVAL_vs_LOCAL')
    def improves(name):
        effect = summary['comparisons'][name]
        return all(effect[key]['point_pct'] > 0 for key in
                   ('call_reduction', 'request_latency_s_reduction'))
    def sweep_support(name):
        candidate, reference = name.split('_vs_')
        return all(repeat[candidate]['request_latency_s'] < repeat[reference]['request_latency_s']
                   for repeat in summary['repeat_totals'].values())
    eligible = all(improves(name) and sweep_support(name) for name in primary)
    unresolved = any(summary['comparisons'][name][key]['ci95_pct'][0] <= 0 for name in primary
                     for key in ('call_reduction', 'request_latency_s_reduction'))
    both_heads = improves('SURVIVAL_vs_KPLUS') and improves('LOCAL_vs_KPLUS')
    head_effect = summary['comparisons']['SURVIVAL_vs_LOCAL']
    head_unresolved = any(value['ci95_pct'][0] <= 0 <= value['ci95_pct'][1] for value in
        (head_effect['call_reduction'], head_effect['request_latency_s_reduction']))
    nonpositive = all(summary['comparisons'][name][key]['point_pct'] <= 0 for name in primary
        for key in ('call_reduction', 'request_latency_s_reduction'))
    outcome = ('provisional_signal_eligible_for_fresh_confirmation' if unresolved else
               'positive_screen_eligible_for_fresh_confirmation') if eligible else (
               'negative_screen' if nonpositive else 'inconclusive_screen')
    return dict(outcome=outcome, fresh_confirmation_eligible=eligible,
        automatic_followup_authorized=False,
        primary_both_sweeps_support_latency={name: sweep_support(name) for name in primary},
        prefix_specific_explanation='open_both_heads_improve_unresolved_head_to_head'
            if both_heads and head_unresolved else 'not_established_by_this_screen',
        intervals='Paired95% unadjusted intervals; crossing zero is uncertainty, not equivalence.',
        scope='One event, training seed17, historically used development subset; stop after event five.')


def validate_training_result(result, arm, baseline=None):
    expected = dict(status='completed', method=arm, dataset='gsm8k', event_id='event_005',
                    start_step=512, end_step=640, optimizer_steps=128,
                    response_presentations=256, passes=2)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('Incomplete or changed survival training boundary')
    records = result.get('records', [])
    if len(records) != 256 or [r.get('pass') for r in records] != [p for p in range(2) for _ in range(128)]:
        raise ValueError('Require two complete128-response passes')
    if (len({r.get('record_id') for r in records[:128]}) != 128
            or len({r.get('record_id') for r in records[128:]}) != 128
            or {r.get('record_id') for r in records[:128]} != {r.get('record_id') for r in records[128:]}):
        raise ValueError('Feedback presentations/order changed')
    if any(not isinstance(r.get('sampled_anchors'), list) for r in records):
        raise ValueError('Missing sampled anchor evidence')
    if baseline is not None:
        fields = ('record_id', 'pass', 'sampled_anchors')
        if [{key:r[key] for key in fields} for r in records] != [{key:r[key] for key in fields} for r in baseline['records']]:
            raise ValueError('Arms differ in feedback order or anchor draws')
    if (result.get('boundary', {}).get('status') != 'passed'
            or result.get('boundary', {}).get('export_exact') is not True):
        raise ValueError('Missing head-free exact native export boundary')
    if result.get('survival_work') != expected_work(arm):
        raise ValueError('Main work differs from the fixed native forward/backward/head budget')
    head = result.get('survival_head')
    if arm == 'KPLUS':
        if head is not None:
            raise ValueError('Ordinary KPLUS unexpectedly has an auxiliary head')
    elif (not isinstance(head, dict) or head.get('steps') != 128 or not head.get('state_path')
            or len(head.get('state_identity', {}).get('sha256', '')) != 64
            or type(head.get('state_identity', {}).get('bytes')) is not int
            or head['state_identity']['bytes'] <= 0):
        raise ValueError('Missing separate active head endpoint and optimizer state')
    return dict(status='passed', main_optimizer_steps=128, response_presentations=256)


def training(evidence):
    results, valid, costs, statuses = {}, {}, [], []
    for arm in ARMS:
        base = Path('campaign_001/gsm8k/event_005') / arm
        value, status = evidence.read(base/'RESULT.json'), evidence.read(base/'STATUS.json')
        results[arm] = value
        errors = []
        try:
            if value is None:
                raise ValueError('No retained training result')
            validate_training_result(value, arm, results.get('KPLUS') if arm != 'KPLUS' else None)
            from .training import validate_event_receipt
            validate_event_receipt(value, arm=arm, dataset='gsm8k', event_number=5)
        except (ValueError, TypeError, KeyError) as error:
            errors.append(repr(error))
        valid[arm] = not errors
        statuses.append(dict(scope='training', arm=arm,
            status='passed' if not errors else 'missing_or_invalid', errors=errors,
            worker_status=status, observed_result_status=(value or {}).get('status')))
        # Preserve exact trainer components/units rather than deriving resource
        # costs from historical arms. Missing fields remain unavailable.
        for scope, owner in (('trainer', value or {}), ('worker', status or {})):
            for name, observed in owner.get('costs', {}).items():
                costs.append(dict(arm=arm, scope=scope, component=name, value=observed))
            for name in ('wall_s', 'loading_s', 'peak_allocated_bytes', 'peak_reserved_bytes',
                         'peak_allocated_gib', 'peak_reserved_gib', 'head_bce',
                         'backbone_clipping_frequency', 'head_clipping_frequency', 'startup'):
                if name in owner:
                    costs.append(dict(arm=arm, scope=scope, component=name, value=owner[name]))
        for name, observed in (value or {}).get('survival_costs', {}).items():
            costs.append(dict(arm=arm, scope='survival_trainer', component=name, value=observed))
    return results, valid, costs, statuses


def fallacy_checks():
    return dict(coverage='11/11', scope='Report design and retained-evidence interpretation; not proof of bias absence',
        checks=[dict(category=name, disposition=detail) for name, detail in (
            ('Simpson paradox', 'One GSM8K panel; no cross-dataset pooling.'),
            ('Ecological fallacy', 'Paired requests are the uncertainty unit; round counts remain descriptive.'),
            ('Berkson selection bias', 'Input-only SHA256selection avoids outcome filtering; historically used development data still limit generalization.'),
            ('Collider bias', 'Invalid or incomplete arm evidence suppresses endpoint inference without dropping individual bad requests.'),
            ('Base rate neglect', 'Calls saved/added and first-round wins/losses/ties retain every request denominator.'),
            ('Regression to mean', 'All arms start at the same prescribed KPLUS4origin; no selection of worst observed responses.'),
            ('Survivorship bias', 'Missing, failed, unknown and unattempted work remain visible.'),
            ('Look-elsewhere effect', 'Two prespecified primary contrasts and one matched-control contrast; unadjusted intervals, no familywise confirmation claim.'),
            ('Forking paths', 'Sealed seed20261110/10000PCG64draws and request-level statistic; no outcome-driven alternate analysis.'),
            ('Correlation versus causation', 'First-round changes and head BCE do not uniquely mediate full-request latency; LOCAL tests prefix-specific attribution.'),
            ('Reverse causality', 'Training precedes serving measurement; no claim that auxiliary loss retrospectively causes an observed endpoint contrast.'))])


def endpoint(evidence, results, valid, admission):
    folder = Path('campaign_001/evaluation/gsm8k')
    repeats = evidence.read(folder/'MEASUREMENTS.jsonl', lines=True) or []
    if not all(valid.get(arm) for arm in ARMS) or (admission or {}).get('status') != 'passed':
        return dict(status='invalid_or_incomplete', inference_permitted=False,
                    errors=['Require all three valid training boundaries and native qualification']), [], repeats
    plan = evidence.plan
    expected_checks = ('complete_state','disabled_native_exact','same_feedback_and_anchors','same_native_rng',
        'identical_initial_heads','finite_nonzero_auxiliary_backbone_gradients',
        'native_gradients_unchanged_by_observer','separate_finite_head_optimizer',
        'one_native_step_per_window','head_free_native_serving','raw_serving_revalidation')
    if (admission.get('binding') != plan.get('binding')
            or admission.get('execution_commit') != plan.get('execution_source', {}).get('commit')
            or any(admission.get('checks', {}).get(name) is not True for name in expected_checks)):
        return dict(status='invalid_or_incomplete', inference_permitted=False,
                    errors=['Qualification admission source/binding/checks differ']), [], repeats
    panel = plan.get('evaluation_requests')
    if panel is None:
        selected = evidence.read(plan.get('panel_manifest', 'sealed/PANEL.json'))
        if selected is None:
            return dict(status='invalid_or_incomplete', inference_permitted=False,
                        errors=['Missing sealed evaluation panel']), [], repeats
        panel = selected['requests']
    canonicals = evidence.read(plan.get('references', 'sealed/CANONICAL.json'))
    checkpoints = {arm: results[arm]['checkpoint_path'] for arm in ARMS}
    evidence.remember_tree(folder)
    audit = audit_stage(evidence.local(folder), panel, checkpoints, canonicals=canonicals,
        dataset='gsm8k', event=5, schedule=plan.get('evaluation_schedule'), evidence_root=evidence.root)
    if audit['status'] != 'passed':
        return dict(status=audit['status'], inference_permitted=False, errors=audit['errors']), [], repeats
    try:
        summary = summarize(audit['rows'], dataset='gsm8k', event=5)
        paired = reduce_stage(audit['rows'], dataset='gsm8k', event=5)
    except (ValueError, KeyError, TypeError) as error:
        return dict(status='invalid_or_incomplete', inference_permitted=False, errors=[repr(error)]), [], repeats
    return dict(status='passed', inference_permitted=True, summary=summary), paired, repeats


def contrast_rows(endpoint_result):
    if endpoint_result.get('status') != 'passed':
        return []
    rows = []
    for name, effect in endpoint_result['summary']['comparisons'].items():
        calls, timing = effect['call_reduction'], effect['request_latency_s_reduction']
        prefix = effect['first_native_round']
        rows.append(dict(contrast=name, candidate_calls=int(calls['candidate_sum']),
            reference_calls=int(calls['reference_sum']), calls_saved=effect['gross_calls_saved'],
            calls_added=effect['gross_calls_added'], call_reduction_pct=calls['point_pct'],
            call_ci95_pct=calls['ci95_pct'], request_latency_reduction_pct=timing['point_pct'],
            request_latency_ci95_pct=timing['ci95_pct'],
            candidate_latency_total_s=timing['candidate_sum'], reference_latency_total_s=timing['reference_sum'],
            call_wins=effect['request_wins'], call_losses=effect['request_regressions'],
            call_ties=effect['request_ties'], prefix_wins=prefix['wins'], prefix_losses=prefix['losses'],
            prefix_ties=prefix['ties'], prefix_tokens_gained=prefix['gross_prefix_tokens_gained'],
            prefix_tokens_lost=prefix['gross_prefix_tokens_lost']))
    return rows


def render(summary, comparisons, costs):
    lines = ['# Prefix-survival head feasibility screen', '', '## Material Passport', '',
        'ARS experiment-agent CPU evidence report. One GSM8K event, three arms, one continuing seed17. '
        'Stage1/projectNone unchanged; no model work performed by this report.', '',
        f"Status: **{summary['status']}**. Decision: **{summary['decision']['outcome']}**.", '',
        'The question is whether prefix-survival supervision improves complete-request serving beyond ordinary K+ '
        'and the matched token-correctness head. LOCAL is the strongest matched control for the prefix-specific explanation.', '']
    if comparisons:
        lines.extend(['| Contrast | Calls (candidate/reference) | Calls saved/added | Call reduction%,95%CI | Complete latency reduction%,95%CI | First-round wins/losses/ties | Prefix tokens gained/lost |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        def fmt(point, interval):
            return f'{point:.4f} [{interval[0]:.4f},{interval[1]:.4f}]'
        for row in comparisons:
            lines.append(f"| {row['contrast']} | {row['candidate_calls']}/{row['reference_calls']} | "
                f"{row['calls_saved']}/{row['calls_added']} | {fmt(row['call_reduction_pct'], row['call_ci95_pct'])} | "
                f"{fmt(row['request_latency_reduction_pct'], row['request_latency_ci95_pct'])} | "
                f"{row['prefix_wins']}/{row['prefix_losses']}/{row['prefix_ties']} | "
                f"{row['prefix_tokens_gained']}/{row['prefix_tokens_lost']} |")
    else:
        lines.append('No complete valid three-arm endpoint is available. Missing, invalid and unattempted work remains visible; no efficacy inference is permitted.')
    lines.extend(['', 'Own-run incremental training resources:'])
    for arm in ARMS:
        selected = {row['component']:row['value'] for row in costs if row['arm'] == arm}
        def value(name):
            observed = selected.get(name)
            return 'unavailable' if observed is None else str(observed)
        reserved = selected.get('peak_reserved_bytes')
        peak = 'unavailable' if reserved is None else f'{reserved/2**30:.4f}GiB'
        clipped = (f"{value('backbone_clipped_windows')}/{value('backbone_optimizer_windows')}"
            if 'backbone_clipped_windows' in selected else 'unavailable')
        bce = 'not applicable' if arm == 'KPLUS' else value('head_bce_response_mean')
        lines.append(f"- {arm}: training {value('training_s')} s; peak reserved VRAM {peak}; "
                     f"response-mean head BCE {bce}; backbone clipping {clipped} windows.")
    lines.extend(['', 'Latency averages the two repeats within each request before comparing paired totals. '
        'Counts use one repeat after agreement is checked. Paired bootstrap resamples128requests with all arms together, '
        '10,000PCG64 draws, seed20261110, linear2.5/97.5percentiles; intervals are unadjusted. '
        'Only the first native round is aligned across arms. Later divergent rounds contribute solely to full-request endpoints.', '',
        'The fixed input-only schedule has16eight-request blocks with cyclic arm orders6/5/5; sweep two reverses each arm order. '
        'Pair precedence is balanced; position frequencies are approximately balanced. Zero extra serving warmup generations. '
        'The first8requests are correctness checks inside the768timed-completion budget.', '',
        'Trainer and worker resource receipts are retained verbatim in TRAINING_COSTS.csv and TRAINING_STATUS.json. '
        'Report training time, peak VRAM, head BCE and backbone clipping using those own-run receipts; missing values are unavailable. '
        'Qualification, startup, packet preparation and nested worker timers remain separate; these are serving-only comparisons, not adaptation-inclusive speedup.', '',
        'The panel is selected by SHA256("survival-v1|"+UTF8requestID) from a historically used256-request development panel '
        'and is disjoint from feedback. It is not a fresh confirmation panel. A single training seed does not measure seed robustness. '
        'Both primary latency and call point estimates must improve and both sweeps must support latency before fresh confirmation is eligible. '
        'Intervals crossing zero leave a provisional signal; unresolved LOCAL/SURVIVAL differences do not establish equivalence or a prefix-specific cause.', '',
        'Run/seed/input bindings, integrity checks and incremental resources: SUMMARY.json, INPUT_MANIFEST.json, '
        'TRAINING_COSTS.csv, TRAINING_STATUS.json, TIMING_REPEATS.csv, PAIRED_REQUESTS.json and COMPARISONS.csv. '
        'Stop after event five; no automatic tuning, confirmation or stage advance.'])
    return '\n'.join(lines)+'\n'


def build(root, output):
    root, output = Path(root), Path(output)
    plan = json.loads((root/'sealed/PLAN.json').read_text())
    if plan.get('budgets') != budgets() or plan.get('analysis') != ANALYSIS:
        raise ValueError('Report requires sealed unchanged survival budgets and analysis')
    if output.exists():
        raise FileExistsError('A report is immutable; choose a separate output directory')
    output.mkdir(parents=True)
    evidence = Evidence(root, plan)
    evidence.read('sealed/PLAN.json')
    results, valid, costs, statuses = training(evidence)
    admission = evidence.read('qualification/ADMISSION.json')
    result, paired, repeats = endpoint(evidence, results, valid, admission)
    controller = evidence.read('campaign_001/STATUS.json')
    postflight = evidence.read('POSTFLIGHT_PILOT.json')
    evaluation_status = evidence.read('campaign_001/evaluation/gsm8k/STATUS.json')
    # Independent work receipts stay separate and retain exact observed counters.
    resources = {name: evidence.read(name) for name in
        ('qualification/STATUS.json', 'qualification/ADMISSION.json', 'feedback/PREPARATION.json',
         'campaign_001/evaluation/gsm8k_WORKER.json')}
    stability = evidence.stability()
    complete = (all(valid.values()) and result['status'] == 'passed'
        and (controller or {}).get('status') == 'completed'
        and (postflight or {}).get('status') == 'passed'
        and stability['status'] == 'passed' and not evidence.errors)
    if stability['status'] != 'passed' or evidence.errors:
        result = dict(status='changed_or_malformed_input', inference_permitted=False,
                      errors=evidence.errors, input_stability=stability)
        paired = []
    decision = recommend(result['summary'] if complete else None)
    summary = dict(status='complete' if complete else 'partial_or_incomplete',
        binding=plan.get('binding'), execution_source=plan.get('execution_source'),
        seed=17, head_seed=17, analysis=ANALYSIS, budgets=budgets(),
        valid_arm_events=sum(valid.values()), endpoint=result, decision=decision,
        controller=controller, postflight=postflight, evaluation_status=evaluation_status,
        independent_resource_receipts=resources, input_stability=stability,
        model_work=0, automatic_followup_authorized=False)
    if result.get('status') == 'passed':
        requests = plan.get('evaluation_requests')
        if requests is None:
            requests = evidence.read(plan.get('panel_manifest', 'sealed/PANEL.json'))['requests']
        summary['schedule_balance'] = schedule_balance(evaluation_schedule(requests))
    comparisons = contrast_rows(result)
    dump_json(output/'SUMMARY.json', summary)
    dump_json(output/'PAIRED_REQUESTS.json', paired)
    dump_json(output/'TRAINING_STATUS.json', statuses)
    dump_json(output/'FALLACY_CHECKS.json', fallacy_checks())
    dump_json(output/'INPUT_MANIFEST.json', dict(files=evidence.files, missing=evidence.missing,
        parse_errors=evidence.errors, stability=stability, model_work=0))
    for name, values in (('COMPARISONS.csv', comparisons), ('TRAINING_COSTS.csv', costs),
                         ('TIMING_REPEATS.csv', repeats)):
        dump_csv(output/name, values)
    (output/'REPORT.md').write_text(render(summary, comparisons, costs))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build(args.root, args.output)
    print(json.dumps(dict(status=result['status'], model_work=0)))
