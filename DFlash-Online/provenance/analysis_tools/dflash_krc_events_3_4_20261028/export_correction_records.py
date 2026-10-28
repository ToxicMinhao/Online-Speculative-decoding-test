#!/usr/bin/env python3
"""Export audited KRC3/4 scalar window records; never export response details.

Post-K reference totals precede direct correction. They are not post-correction
prefix measurements. Blank CSV values preserve absent/null source values.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path

EXECUTION = '11b7718a04416a89025a6df69c0c33e558e5fe1b'
ANALYSIS = 'ea4f289c6e18383b7560c8c4103a9d91ca3943bf'
CORRECTION = (
    'V', 'global_protected_weight', 'gradient_norm_squared_fp32',
    'coefficient_fp32', 'proposed_correction_norm', 'realized_correction_norm',
    'K_update_norm', 'proposed_correction_to_K_norm_ratio',
    'realized_correction_to_K_norm_ratio',
)
POST_K = (
    'blocks', 'old_matching_prefix_tokens', 'new_matching_prefix_tokens',
    'net_matching_prefix_change', 'old_first_errors',
    'first_errors_repaired_with_prefix_intact', 'new_earlier_errors',
    'old_prefix_tokens_lost', 'protected_positions', 'protected_weight',
    'strict_margin_violations', 'weighted_strict_margin_violations',
    'wrong_at_tie_positions', 'weighted_wrong_at_tie_positions',
    'protected_argmax_errors',
)
WORK = (
    'original_response_forwards', 'pre_reference_response_forwards',
    'post_k_reference_response_forwards', 'post_c_reference_response_forwards',
    'post_k_learning_response_forwards', 'post_c_learning_response_forwards',
    'violation_backward_attempted', 'violation_backward_completed',
    'violation_backward_failed', 'direct_corrections', 'pre_reference_attempted',
    'post_k_reference_attempted', 'post_c_reference_attempted',
    'pre_reference_chunks', 'post_k_reference_chunks', 'post_c_reference_chunks',
    'learning_objective_chunks_at_probe_steps', 'post_k_learning_chunks',
    'post_c_learning_chunks',
)
COSTS = (
    'pre_reference_s', 'post_k_s', 'correction_s', 'post_c_reference_s',
    'post_k_learning_s', 'post_c_learning_s', 'pre_reference_integrity_s',
    'first_learning_gate_s',
)
INTEGRITY_PHASES = ('pre_reference_capture', 'post_K_replay', 'correction')
INTEGRITY_FLAGS = ('optimizer', 'gradients', 'runtime', 'parameter_identity',
                   'parameter_versions')
NORM_PRECISION = 'float64; correction decisions and vectors are float32'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(value, *, nullable=False):
    require((value is None and nullable) or
            (type(value) in (int, float) and math.isfinite(value)),
            'Expected finite numeric scalar')
    return value


def export(root):
    root = Path(root).resolve(strict=True)
    audit_bytes = (root / 'FINAL_AUDIT.json').read_bytes()
    audit = json.loads(audit_bytes)
    summary = json.loads((root / 'SUMMARY.json').read_bytes())
    require(audit['status'] == 'passed' and not audit['issues'], 'Audit must pass')
    require(summary['source']['execution_git_commit'] == EXECUTION, 'Execution identity')
    require(summary['postprocessing']['git_commit'] == ANALYSIS, 'Analysis identity')
    rows, identities, event_counts = [], [], {}
    for event in (3, 4):
        name = f'event_{event:03d}'
        directory = root / 'raw/campaign_001/events' / name / 'correction'
        paths = sorted(directory.glob('window_*.json'))
        require([p.name for p in paths] == [f'window_{n:03d}.json' for n in range(1, 129)],
                f'{name}: exactly 128 consecutive windows required')
        event_rows = []
        for local, path in enumerate(paths, 1):
            data = path.read_bytes()
            identities.append(f'{path.relative_to(root).as_posix()} {sha(data)}\n')
            s = json.loads(data)['summary']
            coordinates = s['integrity']['step_coordinates']
            before, after = (event - 1) * 128 + local - 1, (event - 1) * 128 + local
            require(s['step'] == local == coordinates['event_local_step'] ==
                    s['actual_Adam_steps_completed'], f'{name}/{local}: local coordinate')
            require(s['trainer_step_before_return'] == before == coordinates['global_step_before']
                    and coordinates['global_step_after_native_Adam'] == after,
                    f'{name}/{local}: global coordinate')
            correction = s['correction']
            require(s['status'] == 'completed', 'Incomplete window')
            require(correction['status'] in ('applied', 'zero_violation'), 'Unexpected branch')
            require(correction['norm_reporting_precision'] == NORM_PRECISION, 'Precision changed')
            row = {'event': event, 'event_local_step': local, 'global_step_before': before,
                   'global_step_after_native_Adam': after, 'window_status': s['status'],
                   'correction_status': correction['status'],
                   'reference_results_phase': 'after_K_before_correction',
                   'norm_reporting_precision': NORM_PRECISION}
            for key in CORRECTION:
                row['correction_' + key] = number(correction[key], nullable=key == 'coefficient_fp32')
            for group, fields, prefix in ((s['post_K_totals'], POST_K, 'post_K_pre_correction_reference_'),
                                          (s['work'], WORK, 'work_'), (s['costs'], COSTS, 'cost_')):
                for key in fields:
                    row[prefix + key] = number(group[key])
            row['window_wall_s'] = number(s['window_wall_s'])
            row['integrity_s'] = number(s['integrity_s'])
            for phase in INTEGRITY_PHASES:
                for flag in INTEGRITY_FLAGS:
                    value = s['integrity'][phase].get(flag)
                    optional = phase == 'correction' and flag == 'parameter_versions'
                    require(value is True or (optional and value is None), 'Integrity flag failed/missing')
                    row[f'integrity_{phase}_{flag}'] = value
            event_rows.append(row)
        observed = Counter(x['correction_status'] for x in event_rows)
        expected = audit['training']['events'][name]
        require(dict(observed) == expected['correction_branches'], f'{name}: branches differ')
        realized = sum(x['correction_realized_correction_norm'] > 0 for x in event_rows)
        require(realized == expected['realized_nonzero_corrections'], 'Realized correction count')
        for key in WORK:
            require(sum(x['work_' + key] for x in event_rows) == expected['work'][key],
                    f'{name}: work mismatch: {key}')
        for key in COSTS:
            require(math.isclose(sum(x['cost_' + key] for x in event_rows),
                                 expected['correction_costs'][key], rel_tol=1e-12, abs_tol=1e-9),
                    f'{name}: cost mismatch: {key}')
        require(sum(x['post_K_pre_correction_reference_protected_positions'] for x in event_rows)
                == expected['reference_protected_occurrences'], 'Protected exposure mismatch')
        event_counts[name] = {'rows': len(event_rows), 'branches': dict(observed),
                              'BF16_nonzero_corrections': realized}
        rows.extend(event_rows)
    require([x['global_step_after_native_Adam'] for x in rows] == list(range(257, 513)),
            'Global range must be 257..512 exactly')
    branches = dict(Counter(x['correction_status'] for x in rows))
    require(branches == {'applied': 252, 'zero_violation': 4}, 'Unexpected actual branch totals')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode('utf-8')
    receipt = {
        'schema': 'krc34_scalar_corrections_v1', 'status': 'passed',
        'execution_git_commit': EXECUTION, 'report_analysis_git_commit': ANALYSIS,
        'delivery_scope': 'Separate scalar extraction; report builder and analysis identity unchanged.',
        'script_sha256': sha(Path(__file__).read_bytes()), 'audit_sha256': sha(audit_bytes),
        'csv': {'file': 'CORRECTIONS.csv', 'sha256': sha(csv_bytes), 'bytes': len(csv_bytes),
                'rows': len(rows), 'columns': list(rows[0])},
        'input_window_count': len(identities),
        'ordered_input_identities_sha256': sha(''.join(identities).encode('utf-8')),
        'input_identity_format': 'Ordered event 3 then 4; each UTF-8 line is relative_path SPACE full_file_sha256 LF.',
        'global_steps_after_native_Adam': {'first': 257, 'last': 512, 'consecutive': True},
        'correction_branches': branches, 'events': event_counts,
        'semantics': {
            'post_K_pre_correction_reference': 'After ordinary K, before direct correction; no final post-correction prefix effect measured.',
            'blank_values': 'Source null coefficient or absent optional integrity flag; never replaced with zero/false.',
            'costs': 'Window costs and integrity are nested timers; do not add them to inclusive window/event wall.',
            'privacy': 'Explicit scalar allowlist only; record_ids, raw IDs/tokens, response details and arrays excluded.',
        },
        'scientific_optimizer_steps_added': 0, 'model_passes_added': 0,
    }
    (root / 'CORRECTIONS.csv').write_bytes(csv_bytes)
    (root / 'CORRECTIONS_EXTRACTION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = export(args.root)
    print(json.dumps({key: result[key] for key in ('status', 'script_sha256', 'correction_branches')}))
