"""Serving-only confirmation report; native raw audit gates all inference."""
import argparse
import math
from pathlib import Path

from experiments.survival_head.contract import ANALYSIS, ARMS
from experiments.survival_head.report import (
    Evidence, contrast_rows, dump_csv, dump_json, reduce_stage, summarize,
)


def validate_panel_size(value):
    if type(value) is not int or not 136 <= value <= 768 or value % 8:
        raise ValueError('Confirmation requires sealed N=136..768 in multiples of eight')
    return value


def decision(summary):
    """Apply all four primary bounds and both sweeps, with strict positivity."""
    if summary is None:
        return dict(outcome='invalid_or_incomplete', failures=['No complete valid three-arm endpoint'],
                    automatic_followup_authorized=False)
    failures, unresolved, sweeps = [], [], {}
    for control in ('KPLUS', 'LOCAL'):
        name = 'SURVIVAL_vs_' + control
        effect = summary['comparisons'][name]
        for metric in ('call_reduction', 'request_latency_s_reduction'):
            estimate = effect[metric]
            if not math.isfinite(estimate['point_pct']) or estimate['point_pct'] <= 0:
                failures.append(name + ':' + metric + ':nonpositive_point')
            if not all(math.isfinite(x) for x in estimate['ci95_pct']):
                failures.append(name + ':' + metric + ':nonfinite_interval')
            elif estimate['ci95_pct'][0] <= 0:
                unresolved.append(name + ':' + metric + ':lower_bound_not_positive')
        sweeps[name] = []
        for repeat in ('0', '1'):
            totals = summary['repeat_totals'][repeat]
            candidate = totals['SURVIVAL']['request_latency_s']
            reference = totals[control]['request_latency_s']
            reduction = 100 * (1 - candidate / reference)
            sweeps[name].append(reduction)
            if not math.isfinite(reduction) or reduction <= 0:
                failures.append(name + ':sweep_' + repeat + ':latency_not_improved')
    outcome = ('mixed_or_unfavorable' if failures else
               'directionally_replicated_unresolved' if unresolved else
               'confirmed_on_study_heldout_gsm8k')
    return dict(outcome=outcome, failures=failures, unresolved_intervals=unresolved,
                sweep_latency_reductions_pct=sweeps, automatic_followup_authorized=False,
                scope='Frozen step640 checkpoints, one continuing training seed; no automatic second seed or training.')


def endpoint_from_audit(audit, panel_size):
    validate_panel_size(panel_size)
    if (not isinstance(audit, dict) or audit.get('status') != 'passed'
            or audit.get('errors') or len(audit.get('rows', [])) != 6 * panel_size):
        return dict(status='invalid_or_incomplete', inference_permitted=False,
                    errors=['Require passed native raw audit and complete sealed 6N coverage']), []
    try:
        pairs = reduce_stage(audit['rows'], dataset='gsm8k', event=5, panel_size=panel_size, allow_seed_eos=True)
        if any(sum(r[arm]['verification_calls'] for r in pairs) <= 0 for arm in ARMS):
            raise ValueError('Relative call reductions require positive pooled reference denominators')
        result = summarize(audit['rows'], dataset='gsm8k', event=5, panel_size=panel_size, allow_seed_eos=True)
        result['comparison_scope'] = ('SURVIVAL-study-held-out GSM8K requests; historical unrelated '
            'project execution is annotated, not described as project-unused; frozen step640 weights.')
        return dict(status='passed', inference_permitted=True, summary=result), pairs
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return dict(status='invalid_or_incomplete', inference_permitted=False, errors=[repr(error)]), []


def fallacy_checks():
    notes = [
        ('Simpson paradox', 'One GSM8K panel; each timing sweep and each control is shown separately.'),
        ('Ecological fallacy', 'Whole requests carrying all arms are resampled; repeat timings are not independent samples.'),
        ('Berkson selection bias', 'Input-only official-row hash order; study exclusions and prior exposure annotations are sealed before serving.'),
        ('Collider bias', 'Reference availability, response length and prior outcomes never select or replace questions.'),
        ('Base rate neglect', 'All N requests, cap hits, gross calls saved/added and first-round ties remain visible.'),
        ('Regression to mean', 'Pilot selected the checkpoints; confirmation estimates remain separate on study-held-out questions.'),
        ('Survivorship bias', 'Any missing or invalid execution suppresses whole-panel inference; no complete-case filtering.'),
        ('Look-elsewhere effect', 'Four primary intervals across two controls; descriptive LOCAL/KPLUS; intervals remain unadjusted, not familywise guarantees.'),
        ('Forking paths', 'Sealed N, order, budget, 10000 PCG64 draws/seed20261110 and fixed three-way decision; stop after this panel.'),
        ('Correlation versus causation', 'LOCAL is the matched control; first-round prefix changes do not uniquely explain complete-request latency.'),
        ('Reverse causality', 'Frozen weights predate this confirmation; no post-outcome tuning or training occurs.'),
    ]
    return dict(coverage='11/11 checked', checks=[dict(name=name, disposition=note) for name, note in notes])


def resource_rows(receipts):
    """Expose timers without adding nested clocks or inventing missing values."""
    rows = []
    fields = {
        'references': ('wall_s', 'loading_s', 'generation_wall_s', 'valid_generations', 'reused',
                       'planned_generations', 'attempted_generations', 'returned_generations',
                       'invalid_generations', 'unknown_generations', 'unattempted_generations'),
        'worker': ('wall_s', 'loading_s'),
        'timing': ('evaluation_wall_s', 'activation_s', 'weight_validation_s', 'publications',
                   'planned_generations', 'completed_valid_generations', 'invalid_generations',
                   'unknown_generation_attempts', 'unattempted_generations'),
        'run': ('wall_s', 'optimizer_steps'),
    }
    for phase, names in fields.items():
        receipt = receipts.get(phase) or {}
        for name in names:
            rows.append(dict(phase=phase, metric=name, value=receipt.get(name, 'unavailable'),
                             timing_note='Nested component or enclosing phase; do not sum indiscriminately.'))
    return rows


def render(report, comparisons, costs):
    lines = ['# Frozen SURVIVAL study-held-out GSM8K confirmation', '', '## Material Passport', '',
        'ARS experiment-agent validate; ANALYZED. CPU reconstruction of native serving records; '
        'one continuing training seed17, no training in this confirmation.', '',
        f"Status: **{report['status']}**. Decision: **{report['decision']['outcome']}**.", '',
        f"Execution commit: `{report.get('execution_source', {}).get('commit', 'unavailable')}`.", '',
        f"Sealed N={report['panel_size']}; {6 * report['panel_size']} timed completions. "
        f"Planning half-width: ±{report['projected_halfwidth_pp']:.4f} percentage points "
        '(1.225×sqrt(128/N), an assumption-based projection, not measured precision).', '',
        'The panel is held out from SURVIVAL design, development and checkpoint-lineage feedback. '
        'It is not claimed project-unused. Exact exclusions, deduplication and prior exposure are retained '
        'in SEALED_PANEL.json and EXPOSURE_INVENTORY.json. Older serving records supply only identities '
        'or authenticated canonical outputs; all three arms are timed anew.', '']
    if report['status'] != 'complete':
        lines.extend(['No complete valid three-arm endpoint. Raw partial repeats and all integrity failures '
                      'are retained; no serving comparison is inferred.', '',
                      'Errors: ' + repr(report.get('completion_errors', [])), ''])
    else:
        lines.extend(['| Contrast | Calls candidate/reference | Gross saved/added | Call reduction % [95% CI] | Complete-time reduction % [95% CI] |',
                      '|---|---:|---:|---:|---:|'])
        for row in comparisons:
            lines.append(f"| {row['contrast']} | {row['candidate_calls']}/{row['reference_calls']} | "
                f"{row['calls_saved']}/{row['calls_added']} | {row['call_reduction_pct']:.4f} "
                f"[{row['call_ci95_pct'][0]:.4f}, {row['call_ci95_pct'][1]:.4f}] | "
                f"{row['request_latency_reduction_pct']:.4f} "
                f"[{row['request_latency_ci95_pct'][0]:.4f}, {row['request_latency_ci95_pct'][1]:.4f}] |")
        endpoint = report['endpoint']['summary']
        lines.extend(['', '| Arm | Calls (one repeat) | Summed mean request seconds | Native seconds | Cache seconds | Cap-hit requests |',
                      '|---|---:|---:|---:|---:|---:|'])
        for arm in ARMS:
            total = endpoint['method_totals'][arm]
            lines.append(f"| {arm} | {total['verification_calls']} | {total['request_latency_s']:.6f} | "
                         f"{total['native_generate_s']:.6f} | {total['cache_maintenance_s']:.6f} | {total['cap_hit_requests']} |")
        lines.extend(['', '| Contrast | Sweep 1 complete-time reduction % | Sweep 2 complete-time reduction % |', '|---|---:|---:|'])
        for row in comparisons:
            a, b = row['contrast'].split('_vs_')
            values = [100 * (1 - endpoint['repeat_totals'][str(r)][a]['request_latency_s'] /
                            endpoint['repeat_totals'][str(r)][b]['request_latency_s']) for r in (0, 1)]
            lines.append(f"| {row['contrast']} | {values[0]:.6f} | {values[1]:.6f} |")
        lines.extend(['', 'Failed directional conditions: ' + repr(report['decision'].get('failures', [])),
                      'Primary lower bounds not above zero: ' + repr(report['decision'].get('unresolved_intervals', [])), '',
                      '| Shared first-round diagnostic | Wins/losses/ties | Prefix tokens gained/lost |', '|---|---:|---:|'])
        for row in comparisons:
            lines.append(f"| {row['contrast']} | {row['prefix_wins']}/{row['prefix_losses']}/{row['prefix_ties']} | "
                         f"{row['prefix_tokens_gained']}/{row['prefix_tokens_lost']} |")
    lines.extend(['', '| A800 phase/component | Recorded value |', '|---|---:|'])
    displayed = {('references', 'wall_s'), ('references', 'loading_s'), ('references', 'generation_wall_s'),
                 ('references', 'valid_generations'), ('references', 'reused'), ('worker', 'wall_s'),
                 ('worker', 'loading_s'), ('timing', 'evaluation_wall_s'), ('timing', 'activation_s'),
                 ('timing', 'weight_validation_s'), ('run', 'wall_s'), ('run', 'optimizer_steps')}
    for cost in costs:
        if (cost['phase'], cost['metric']) in displayed:
            lines.append(f"| {cost['phase']}/{cost['metric']} | {cost['value']} |")
    lines.extend(['', 'Timing averages the two repeats inside each request. Calls use one repeat after exact '
        'agreement. Bootstrap resamples whole requests jointly across arms, 10,000 PCG64 draws, seed20261110, '
        'linear 2.5/97.5 percentiles; intervals are unadjusted. Pilot and confirmation estimates are separate. '
        'Only the first native round shares a conditioning prefix; later rounds are not aligned mechanism evidence.', '',
        'Reference preparation, startup and timed serving receipts are separate in RESOURCES.json and '
        'RESOURCE_TIMERS.csv. Native generation/cache components sit inside complete-request time; activation '
        'and checks sit inside evaluation wall time; loading/evaluation sit inside worker wall time. '
        'Missing timers remain unavailable. No training-inclusive speedup is claimed.', '',
        'Confirmation requires positive calls and latency against both controls, all four lower interval '
        'bounds strictly above zero, and both latency sweeps favorable against both controls. Favorable '
        'points/sweeps with an interval including zero are directionally replicated but unresolved. '
        'Any failed point or sweep is mixed or unfavorable. A KPLUS-only advantage leaves the prefix-specific '
        'benefit unresolved. This is a decision for these checkpoints, not seed-robustness or unique mediation.', '',
        'All 11 fallacy checks are retained. Stop after this panel; no additional training seed, tuning or '
        'follow-up experiment is automatically authorized.'])
    return '\n'.join(lines) + '\n'


def build(root, output):
    from .evaluation import audit_stage
    root, output = Path(root), Path(output)
    import json
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    size = validate_panel_size(plan['panel_size'])
    if plan.get('analysis') != ANALYSIS:
        raise ValueError('Require the prespecified unchanged bootstrap/contrast contract')
    if output.exists():
        raise FileExistsError('A report is immutable; choose a separate output directory')
    evidence = Evidence(root, plan)
    evidence.read('sealed/PLAN.json')
    panel = evidence.read(plan.get('panel_manifest', 'sealed/PANEL.json'))
    exposure = evidence.read(plan.get('exposure_inventory', 'sealed/EXPOSURE_INVENTORY.json'))
    canonicals = evidence.read(plan.get('references', 'references/CANONICAL.json'))
    seal = evidence.read('sealed/SEAL.json')
    run = evidence.read('STATUS.json')
    postflight = evidence.read('POSTFLIGHT.json')
    receipts = dict(references=evidence.read('references/STATUS.json'),
                    worker=evidence.read('evaluation/WORKER.json'),
                    timing=evidence.read('evaluation/STATUS.json'), run=run)
    repeats = evidence.read('evaluation/MEASUREMENTS.jsonl', lines=True) or []
    evidence.remember_tree('evaluation')
    evidence.remember_tree('evaluation_native')
    evidence.remember_tree('references')
    audit = dict(status='invalid_or_incomplete', errors=['Missing sealed inputs'], rows=[])
    if panel is not None and canonicals is not None:
        try:
            checkpoints = {arm: plan['checkpoints'][arm]['checkpoint_path'] for arm in ARMS}
            audit = audit_stage(root / 'evaluation', panel['requests'], checkpoints,
                canonicals=canonicals, dataset='gsm8k', event=5,
                schedule=plan['evaluation_schedule'], evidence_root=root, sealed_n=size,
                physical_gpu=plan['serving_gpu_identity'])
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            audit = dict(status='invalid_or_incomplete', errors=[repr(error)], rows=[])
    endpoint, paired = endpoint_from_audit(audit, size)
    errors = []
    for name, receipt, expected in [('run', run, 'completed'), ('postflight', postflight, 'passed'),
                                    ('references', receipts['references'], 'completed'),
                                    ('worker', receipts['worker'], 'completed')]:
        if (receipt or {}).get('status') != expected:
            errors.append(name + '_not_' + expected)
    for name, receipt in [('run', run), ('references', receipts['references'])]:
        if (receipt or {}).get('optimizer_steps') != 0:
            errors.append(name + '_zero_optimizer_work_not_confirmed')
    if seal is None or exposure is None:
        errors.append('missing_seal_or_exposure_inventory')
    if endpoint['status'] != 'passed':
        errors.extend(endpoint['errors'])
    stability = evidence.stability()
    if stability['status'] != 'passed' or evidence.errors:
        errors.append('changed_or_malformed_input')
    complete = not errors
    if not complete:
        endpoint = dict(status='invalid_or_incomplete', inference_permitted=False, errors=errors)
        paired = []
    result = dict(status='complete' if complete else 'partial_or_incomplete', panel_size=size,
        projected_halfwidth_pp=1.225 * math.sqrt(128 / size), binding=plan.get('binding'),
        execution_source=plan.get('execution_source'), analysis=ANALYSIS,
        physical_gpu=plan.get('serving_gpu_identity'),
        checkpoints=plan.get('checkpoints'), endpoint=endpoint, completion_errors=errors,
        decision=decision(endpoint['summary'] if complete else None), input_stability=stability,
        postflight=postflight, resources=receipts, model_work=0,
        model_work_scope='CPU report only; actual serving/reference work is recorded in resources',
        automatic_followup_authorized=False)
    comparisons = contrast_rows(endpoint)
    costs = resource_rows(receipts)
    output.mkdir(parents=True)
    for name, value in [('SUMMARY.json', result), ('PAIRED_REQUESTS.json', paired), ('INTEGRITY_AUDIT.json', audit),
                        ('SEALED_PANEL.json', panel), ('EXPOSURE_INVENTORY.json', exposure),
                        ('CHECKPOINT_IDENTITIES.json', plan.get('checkpoints')), ('RESOURCES.json', receipts),
                        ('FALLACY_CHECKS.json', fallacy_checks()),
                        ('INPUT_MANIFEST.json', dict(files=evidence.files, missing=evidence.missing,
                            parse_errors=evidence.errors, stability=stability, model_work=0))]:
        dump_json(output / name, value)
    for name, values in [('COMPARISONS.csv', comparisons), ('TIMING_REPEATS.csv', repeats), ('RESOURCE_TIMERS.csv', costs)]:
        dump_csv(output / name, values)
    (output / 'REPORT.md').write_text(render(result, comparisons, costs))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build(args.root, args.output)
    raise SystemExit(0 if result['status'] == 'complete' else 1)
