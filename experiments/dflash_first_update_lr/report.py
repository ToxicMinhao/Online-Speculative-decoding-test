"""Descriptive fixed-input analysis; no historical timings are pooled."""
import csv
import json
from pathlib import Path


def ratio_reduction(base, value):
    return 100 * (1 - value / base) if base else None


def classify_calls(frozen, high, low):
    if low <= frozen:
        return 'eliminates' if low == frozen else 'reverses'
    if low < high:
        return 'mitigates'
    return 'unchanged' if low == high else 'worsens'


def write_reports(results, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw, summary, costs = [], [], []
    for result in results:
        for u in result['updates']:
            costs.append(dict(episode_id=result['episode_id'], label=u['checkpoint']['checkpoint_id'],
                recipe=u['accounting']['recipe'], costs=u.get('costs'),
                observed_branch_call_s=u.get('observed_branch_call_s'),
                trainer_startup_s=u.get('trainer_startup_s'),
                anchor_exposure_sha256=u['accounting']['anchor_exposure_sha256'],
                optimizer_steps=u['accounting']['optimizer_steps_this_event'],
                eligible_positions=u['accounting']['eligible_position_presentations_this_event'],
                eligible_blocks=u['accounting']['eligible_block_presentations_this_event']))
        for r in result['scores']:
            raw.append({k: r.get(k) for k in ('episode_id', 'source', 'turn', 'probe', 'checkpoint_id',
                'repeat', 'context_tokens', 'output_tokens', 'cap_hit', 'verification_calls',
                'accepted_draft_tokens', 'verification_committed_tokens', 'request_s', 'native_generation_s',
                'activation_s', 'native_activation_s', 'cache_maintenance_s', 'same_generation_valid',
                'input_match', 'canonical_output_match', 'checkpoint_identity_match',
                'checkpoint_identity_sha256', 'original_verification_count_match')})
        for turn in (2, 1):
            groups = {l: [r for r in result['scores'] if r['turn'] == turn and r['checkpoint_id'] == l]
                      for l in ('F', 'D1', 'L1')}
            valid = result['status'] == 'complete' and result['exposure_checks']['status'] == 'passed' and all(len(rows) == 3 and all(r['same_generation_valid'] and r.get('original_verification_count_match', True) for r in rows)
                        for rows in groups.values()) and result['reproduction_check']['status'] == 'passed'
            if not valid:
                summary.append(dict(episode_id=result['episode_id'], turn=turn, valid=False,
                                    reason='Incomplete or invalid matrix; no gain claim'))
                continue
            values = {}
            for label, rows in groups.items():
                calls = sum(r['verification_calls'] for r in rows)
                output = sum(r['output_tokens'] for r in rows)
                values[label] = dict(request_s=sum(r['request_s'] for r in rows)/3,
                    verification_calls=calls/3, calls_per_1000_tokens=calls/output*1000,
                    accepted_per_verification=sum(r['accepted_draft_tokens'] for r in rows)/calls,
                    progress_per_verification=sum(r['verification_committed_tokens'] for r in rows)/calls,
                    timings=[r['request_s'] for r in sorted(rows, key=lambda r: r['repeat'])])
            row = dict(episode_id=result['episode_id'], source=result['source'], synthetic=result['synthetic'],
                turn=turn, probe='next_request_transfer' if turn == 2 else 'in_sample_fit', valid=True,
                frozen_calls=values['F']['verification_calls'], high_calls=values['D1']['verification_calls'],
                low_calls=values['L1']['verification_calls'], frozen_request_s=values['F']['request_s'],
                high_request_s=values['D1']['request_s'], low_request_s=values['L1']['request_s'])
            for label, prefix in (('D1', 'high'), ('L1', 'low')):
                row[prefix+'_time_reduction_pct'] = ratio_reduction(values['F']['request_s'], values[label]['request_s'])
                row[prefix+'_call_reduction_pct'] = ratio_reduction(values['F']['verification_calls'], values[label]['verification_calls'])
            row['smaller_update_call_effect'] = classify_calls(row['frozen_calls'], row['high_calls'], row['low_calls'])
            row['details'] = values
            summary.append(row)
    def csv_out(name, rows):
        if not rows: return
        columns = list(dict.fromkeys(k for row in rows for k in row if k != 'details'))
        with (out/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
    csv_out('raw_repeats.csv', raw); csv_out('per_request.csv', summary)
    payload = dict(status='complete' if len(results) == 6 and all(r['status']=='complete' for r in results) else 'partial',
        streams=len(results), updates=sum(len(r['updates']) for r in results),
        optimizer_steps=sum(c['optimizer_steps'] for c in costs), timed_executions=len(raw), rows=summary,
        actual_experiment_wall_s=sum(r['actual_wall_s'] for r in results),
        interpretation='First-update sensitivity only; request 1 is in-sample; not a full-trajectory low-LR/warmup test.')
    (out/'SUMMARY.json').write_text(json.dumps(payload, indent=2)+'\n')
    (out/'training_costs.json').write_text(json.dumps(costs, indent=2)+'\n')
    text = ['# First-update learning-rate investigation', '', '## Material Passport', '',
        'Mode: run/validate. Verification: native reproduction gates and exact-input/output checks. '
        'Stage 1 / project None unchanged. Descriptive six-conversation sensitivity test.', '',
        f"Status: {payload['status']}; {payload['updates']} isolated updates, {payload['optimizer_steps']} steps, "
        f"{payload['timed_executions']} fresh timed executions.", '',
        '## Request 2: next-request transfer', '',
        'Positive reductions mean improvement over released DFlash. Request time includes prefill, drafting and '
        'verification with an available checkpoint; it excludes training, publication and external cache maintenance.', '',
        '| Stream | Calls F / 3e-4 / 1e-4 | Time reduction 3e-4 / 1e-4 | Smaller-update call effect |',
        '|---|---:|---:|---|']
    for row in summary:
        if row['turn'] != 2 or not row['valid']: continue
        text.append(f"| {row['episode_id']} | {row['frozen_calls']:g} / {row['high_calls']:g} / {row['low_calls']:g} | "
                    f"{row['high_time_reduction_pct']:+.2f}% / {row['low_time_reduction_pct']:+.2f}% | {row['smaller_update_call_effect']} |")
    text += ['', '## Request 1: in-sample fit only', '',
        '| Stream | Calls F / 3e-4 / 1e-4 | Time reduction 3e-4 / 1e-4 |', '|---|---:|---:|']
    for row in summary:
        if row['turn'] != 1 or not row['valid']: continue
        text.append(f"| {row['episode_id']} | {row['frozen_calls']:g} / {row['high_calls']:g} / {row['low_calls']:g} | "
                    f"{row['high_time_reduction_pct']:+.2f}% / {row['low_time_reduction_pct']:+.2f}% |")
    text += ['', 'Training uses only the saved first response. Improved fit to it does not establish improvement '
        'on request 2. All timing repeats, gains and regressions remain in raw_repeats.csv. Timing-only changes '
        'do not establish changed drafting agreement. Training costs and exposure remain separate in training_costs.json.', '',
        'The prior beneficial update-2 diagnostic in wildchat_00 showed 156→136 calls at 3e-4 versus 141 at 1e-4. '
        'That is separate prior evidence that smaller updates can sacrifice gains; its timings are not pooled here.', '',
        'This experiment does not establish that a smaller learning rate or warmup improves a complete conversation. '
        'Repeated timings reuse twelve fixed contexts, not 108 independent examples. Task/cap correlations are descriptive '
        'and do not identify a causal explanation. No significance or general superiority claim is made.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    return payload
