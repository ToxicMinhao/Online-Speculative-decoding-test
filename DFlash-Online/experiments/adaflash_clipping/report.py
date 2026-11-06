"""Compile compact descriptive results without publishing private tokens."""
import argparse
import csv
import json
from pathlib import Path


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def show(value, *, percent=False):
    if value is None:
        return 'n/a (zero norm)'
    return f'{100 * value:.2f}%' if percent else f'{value:.6f}'


def build(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads((root / 'diagnostic/RESULT.json').read_text())
    status = json.loads((root / 'STATUS.json').read_text())
    prepared = json.loads((root / 'preparation/PREPARED.json').read_text())
    if result['status'] != 'passed' or status['status'] != 'passed':
        raise ValueError('Completed-run report requires successful accounting and validity')
    rows, prefixes, losses = [], [], []
    for comparison in result['comparisons']:
        batch = comparison['batch']
        row = dict(batch=batch, identical_preupdate_blocks=comparison['identical_preupdate_blocks'])
        for name in ('gradients', 'displacements'):
            stats = comparison[name]['global_statistics']
            row.update({name + '_' + key: stats[key] for key in
                        ('left_norm', 'right_norm', 'difference_norm', 'cosine', 'coordinates', 'different')})
            row[name + '_relative_difference'] = ratio(stats['difference_norm'], stats['left_norm'])
            row[name + '_different_fraction'] = ratio(stats['different'], stats['coordinates'])
        for mode in ('paper_entry', 'repo_position'):
            branch = json.loads((root / f'diagnostic/batch_{batch:02d}_{mode}/RESULT.json').read_text())
            for position, source in enumerate(branch['source_fit']):
                prefixes.append(dict(batch=batch, mode=mode, **source))
                for phase in ('before', 'after'):
                    metrics = branch['responses_' + phase][position]['metrics']
                    denominator = metrics['adaflash_loss_denominator']
                    losses.append(dict(batch=batch, mode=mode, phase=phase, record_id=source['record_id'],
                        denominator=denominator,
                        mixed_loss=metrics['adaflash_loss_numerator'] / denominator,
                        teacher_ce=metrics['adaflash_ce_numerator'] / denominator,
                        effective_rkl=metrics['adaflash_rkl_numerator'] / denominator,
                        raw_rkl=metrics['clipping_raw_reverse_kl_numerator'] / denominator,
                        paper_clipped_entries=metrics['clipping_paper_entry_clipped_count'],
                        paper_clipped_positions=metrics['clipping_paper_entry_clipped_positions'],
                        repo_clipped_positions=metrics['clipping_repo_position_clipped_count'],
                        eligible_entries=metrics['clipping_eligible_entries'],
                        available_positions=metrics['clipping_available_positions']))
            row[mode + '_prefix_gain'] = sum(x['net_matching_prefix_change'] for x in branch['source_fit'])
            row[mode + '_old_prefix_lost'] = sum(x['old_prefix_tokens_lost'] for x in branch['source_fit'])
            row[mode + '_useful_repairs'] = sum(x['first_errors_repaired_with_prefix_intact'] for x in branch['source_fit'])
        rows.append(row)
    for filename, table in (('PAIRED_BATCHES.csv', rows), ('SOURCE_PREFIXES.csv', prefixes), ('LOSSES.csv', losses)):
        with (output / filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    capsule = dict(status='analyzed', execution_commit=status['execution_commit'], batches=rows,
                   work=result['work'], reconstruction=prepared['work'],
                   preparation_costs=prepared['costs'], diagnostic_wall_s=result['wall_s'],
                   total_wall_s=status['wall_s'])
    (output / 'SUMMARY.json').write_text(json.dumps(capsule, indent=2) + '\n')
    lines = ['# AdaFlash clipping-location diagnostic', '', '## Material Passport', '',
        '- Status: completed bounded execution; descriptive analysis, not an endpoint replication.',
        f"- Scientific execution commit: `{status['execution_commit']}`.",
        '- Six fixed saved responses in three pairs; two independent one-step branches per pair.',
        '- Six disposable AdamW steps; no complete-response generation or serving checkpoint.', '',
        '## Paired changes', '',
        '| Pair | Gradient cosine | Gradient difference / paper norm | BF16 displacement cosine | Displacement difference / paper norm | Prefix gain: paper / repo |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['batch']} | {show(row['gradients_cosine'])} | {show(row['gradients_relative_difference'], percent=True)} | "
                     f"{show(row['displacements_cosine'])} | {show(row['displacements_relative_difference'], percent=True)} | "
                     f"{row['paper_entry_prefix_gain']:+d} / {row['repo_position_prefix_gain']:+d} |")
    lines += ['', 'The displacement comparison uses the actual rounded BF16 weights after native AdamW. '
        'Prefix counts are occurrences across overlapping sampled blocks, not independent output tokens or native verification calls.', '',
        '## Scope and arithmetic', '',
        'Both arms share BF16 projection, log-softmax/exp, teacher-argmax CE and mixture arithmetic. '
        'The paper arm caps individual vocabulary contributions. The repository-rule arm caps whole-position reverse KL '
        'after normalization by the full response denominator, using FP32 cap/rescaling and the common BF16 mixture. '
        'This tests clipping placement under common arithmetic; it is not a bitwise reproduction of the official kernels.', '',
        'The teacher features were reconstructed from exact saved native verification inputs. Full posterior IDs and '
        'committed tokens were checked. Retired original feature tensors cannot be compared byte for byte.', '',
        'These are source-fitting and update-sensitivity measurements. They do not establish which loss wins after '
        '58 events, held-out transfer, native latency, or the cause of the prior A58 result.', '',
        '## Costs and stop', '',
        f"Total wrapper time: {status['wall_s']:.3f} s. Preparation: {prepared['costs']['wall_s']:.3f} s. "
        f"Diagnostic: {result['wall_s']:.3f} s. Startup, reconstruction, forward/chunk/recomputation and step counts "
        'are retained in SUMMARY.json and raw receipts.',
        'No additional event, tuning or serving run was started.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return capsule


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('output')
    args = parser.parse_args()
    build(args.root, args.output)
