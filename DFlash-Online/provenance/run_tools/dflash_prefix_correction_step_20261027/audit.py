"""Independent compact-evidence audit: no Torch, model execution or remote access."""
import csv
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def identity(path):
    with Path(path).open('rb') as stream:
        return dict(bytes=Path(path).stat().st_size,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def valid_length(labels, weights, eos):
    if labels[0] in eos:
        return 0
    length = 0
    for label, weight in zip(labels[1:], weights[1:]):
        if weight <= 0:
            break
        length += 1
        if label in eos:
            break
    return length


def prefix_length(labels, predictions, available):
    length = 0
    for offset in range(1, available + 1):
        if labels[offset] != predictions[offset]:
            break
        length += 1
    return length


def audit(root=ROOT):
    root = Path(root)
    raw = root / 'raw'
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    receipt = load(raw / 'RETRIEVAL_MANIFEST.json')
    for name, expected in receipt.items():
        actual = identity(raw / name)
        check(actual['sha256'] == expected['sha256'] and
              actual['bytes'] == expected.get('bytes', expected.get('size')), 'Retrieval: ' + name)
    result = load(raw / 'execution_001/RESULT.json')
    if result['status'] != 'completed':
        summary = dict(status='incomplete', experiment_status=result['status'], issues=issues,
                       phase=result.get('phase'), error=result.get('error'),
                       completed_optimizer_steps=result.get('completed_optimizer_steps', 0),
                       work=result.get('core', {}).get('work', {}),
                       core_work_counts=result.get('core_work_counts', {}),
                       startup_work_counts=result.get('startup_work_counts', {}))
        (root / 'FINAL_AUDIT.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary

    generation = load(root / 'model_metadata/target_generation_config.json')
    tokenizer = load(root / 'model_metadata/target_tokenizer_config.json')
    eos = generation['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    tokenizer_eos = tokenizer['eos_token']
    tokenizer_eos = tokenizer_eos.get('content') if isinstance(tokenizer_eos, dict) else tokenizer_eos
    eos.update(int(index) for index, token in tokenizer['added_tokens_decoder'].items()
               if token['content'] == tokenizer_eos)
    pin = load(raw / 'sealed/MODEL_PIN.json')['target']['files']
    for name in ('generation_config.json', 'tokenizer_config.json'):
        actual = identity(root / ('model_metadata/target_' + name))
        expected = pin[name]
        check(actual['sha256'] == expected['sha256'], 'Pinned target metadata: ' + name)

    core = result['core']
    posts = [result['post_k'], result['post_c']]
    table = []
    response_rows = []
    all_totals = [dict(), dict()]
    for response_index, source in enumerate(core['responses']):
        response_totals = [dict(), dict()]
        weighted_hinges = [0.0, 0.0]
        repair_counts = dict(original_first_errors=0, K_useful_repairs=0, C_useful_repairs=0,
                             useful_repairs_retained=0, useful_repairs_lost=0,
                             useful_repairs_newly_obtained=0)
        for block, labels in enumerate(source['labels'][0]):
            validity = source['validity_weights'][0][block]
            available = valid_length(labels, validity, eos)
            old = prefix_length(labels, source['original_proposals'][0][block], available)
            weights = [0] + [max(old - offset + 1, 0) for offset in range(1, len(labels))]
            check(weights == source['protected_weights'][0][block], 'Original triangular weights')
            lengths = []
            wrong_ties = []
            for index, post in enumerate(posts):
                row = post['responses'][response_index]
                predictions = row['proposals'][0][block]
                margins = row['protected_margins'][0][block]
                new = prefix_length(labels, predictions, available)
                lengths.append(new)
                ties = sum(weight > 0 and predictions[i] != labels[i] and margins[i] == 0
                           for i, weight in enumerate(weights))
                wrong_ties.append(ties)
                strict = sum(weight > 0 and margins[i] < 0 for i, weight in enumerate(weights))
                wrong = sum(weight > 0 and predictions[i] != labels[i] for i, weight in enumerate(weights))
                check(wrong == strict + ties, 'Negative-margin errors plus wrong ties exhaust protected errors')
                metrics = dict(blocks=1, old_matching_prefix_tokens=old, new_matching_prefix_tokens=new,
                    net_matching_prefix_change=new-old, old_first_errors=int(old < available),
                    first_errors_repaired_with_prefix_intact=int(old < available and new > old),
                    new_earlier_errors=int(new < old), old_prefix_tokens_lost=max(old-new, 0),
                    protected_positions=old, protected_weight=sum(weights),
                    strict_margin_violations=strict,
                    weighted_strict_margin_violations=sum(w for i,w in enumerate(weights) if margins[i] < 0),
                    wrong_at_tie_positions=ties,
                    weighted_wrong_at_tie_positions=sum(w for i,w in enumerate(weights)
                        if predictions[i] != labels[i] and margins[i] == 0),
                    protected_argmax_errors=wrong)
                for key, value in metrics.items():
                    response_totals[index][key] = response_totals[index].get(key, 0) + value
                weighted_hinges[index] += math.fsum(w*max(0, -m) for w,m in zip(weights,margins))
                check(row['matching_prefix_lengths'][0][block] == new, 'Recorded prefix length')
            repaired_k = old < available and lengths[0] > old
            repaired_c = old < available and lengths[1] > old
            transitions = dict(original_first_errors=int(old < available), K_useful_repairs=int(repaired_k),
                C_useful_repairs=int(repaired_c), useful_repairs_retained=int(repaired_k and repaired_c),
                useful_repairs_lost=int(repaired_k and not repaired_c),
                useful_repairs_newly_obtained=int(not repaired_k and repaired_c))
            for key, value in transitions.items():
                repair_counts[key] += value
            table.append(dict(response_id=source['record_id'], block=block,
                anchor=source['anchors'][0][block], anchor_kept=source['keep'][0][block],
                available_positions=available, old_prefix_length=old, K_prefix_length=lengths[0],
                C_prefix_length=lengths[1], K_lost_old_prefix_tokens=max(old-lengths[0],0),
                C_lost_old_prefix_tokens=max(old-lengths[1],0), K_wrong_ties=wrong_ties[0],
                C_wrong_ties=wrong_ties[1], **transitions))
        check(repair_counts == posts[1]['repair_transitions'][response_index], 'Per-response repair transitions')
        for index, totals in enumerate(response_totals):
            row = posts[index]['responses'][response_index]
            check(all(row[key] == value for key,value in totals.items()), 'Per-response native prefix metrics')
            for key, value in totals.items():
                all_totals[index][key] = all_totals[index].get(key,0) + value
            response_rows.append(dict(response_id=source['record_id'], method=('K','C')[index],
                numerator_recorded_fp32=row['violation_numerator'],
                numerator_reconstructed_fp64=weighted_hinges[index],
                numerator_absolute_difference=abs(row['violation_numerator']-weighted_hinges[index]),
                **totals))
    for index, post in enumerate(posts):
        check(all_totals[index] == post['totals'], 'Complete two-response totals')
        check(post['denominator'] == all_totals[index]['protected_weight'], 'One global denominator')
        numerator = f32(sum(row['violation_numerator'] for row in post['responses']))
        check(numerator == post['numerator'], 'FP32 sum of response numerators')
        expected_v = f32(numerator/post['denominator']) if post['denominator'] else numerator
        check(post['V'] == expected_v, 'FP32 global V normalization')

    prefix_metrics = ('blocks','old_matching_prefix_tokens','new_matching_prefix_tokens',
        'net_matching_prefix_change','old_first_errors','first_errors_repaired_with_prefix_intact',
        'new_earlier_errors','old_prefix_tokens_lost')
    retained = load(root.parent / 'dflash_prefix_preservation_guard_20261027/main_003/raw/campaign_001/guard/probe_001_ordinary_K.json')
    check([[r[key] for key in prefix_metrics] for r in retained['responses']] ==
          [[r[key] for key in prefix_metrics] for r in posts[0]['responses']], 'Original K per-response reproduction')
    corrected = result['correction']
    if corrected['status'] == 'applied':
        check(corrected['coefficient_fp32'] == f32(-posts[0]['V']/corrected['gradient_norm_squared_fp32']),
              'Prespecified FP32 correction coefficient')
    check(result['completed_optimizer_steps'] == 1 and result['completed_presentations'] == 2,
          'Single ordinary optimizer window')
    check(result['complete_response_generations'] == 0 and not result['checkpoint_exported'], 'No generation/export')
    check(result['budget']['core_response_forwards'] == 6 and
          result['budget']['core_backward_equivalents'] <= 4 and
          result['budget']['direct_corrections'] <= 1, 'Bounded core work')
    check(result['startup']['optimizer_steps'] == 0 and result['startup']['response_evaluations'] == 2 and
          result['startup']['backwards'] == 2, 'Separate startup with zero Adam')
    for name in ('POST_K_STATE_UNCHANGED','POST_K_WEIGHTS_UNCHANGED','CORRECTION_STATE_UNCHANGED',
                 'POST_C_STATE_UNCHANGED','POST_C_WEIGHTS_UNCHANGED','REPRODUCTION','PRESENTATIONS'):
        check(load(raw / f'execution_001/{name}.json')['exact_equal'], 'Exact receipt: '+name)
    for name in ('INITIAL_VS_PIN','FRESH_CORE'):
        check(load(raw / f'execution_001/initialization/{name}.json')['exact_equal'], 'Exact initialization: '+name)
    preflight, postflight = load(raw/'PREFLIGHT.json'), load(raw/'POSTFLIGHT.json')
    check(preflight['status'] == postflight['status'] == 'passed', 'Pre/postflight')
    check(preflight['files_checked'] == postflight['files_checked'] ==
          len(load(raw/'sealed/SEAL.json')['files']), 'Complete unchanged source/input seal')
    advancement = (all_totals[1]['old_prefix_tokens_lost'] < all_totals[0]['old_prefix_tokens_lost'] and
                   all_totals[1]['net_matching_prefix_change'] > all_totals[0]['net_matching_prefix_change'])
    check(result['comparison']['local_advancement'] == advancement, 'Prespecified local advancement')
    differences = [row['C_prefix_length']-row['K_prefix_length'] for row in table]
    block_comparison = dict(improved=sum(value>0 for value in differences),
        regressed=sum(value<0 for value in differences), tied=sum(value==0 for value in differences),
        added_matching_prefix_tokens=sum(max(value,0) for value in differences),
        lost_post_K_matching_prefix_tokens=sum(max(-value,0) for value in differences),
        K_damaged_old_prefixes_restored=sum(row['K_prefix_length'] < row['old_prefix_length'] <=
            row['C_prefix_length'] for row in table),
        new_old_prefix_damage_from_correction=sum(row['C_prefix_length'] < row['old_prefix_length'] <=
            row['K_prefix_length'] for row in table))
    summary = dict(status='passed' if not issues else 'failed', issues=issues,
        verified_retrieved_files=len(receipt), eos_token_ids=sorted(eos),
        sealed_files_verified=postflight['files_checked'], C_vs_K_block_comparison=block_comparison,
        blocks=len(table), responses=2, K=all_totals[0], C=all_totals[1],
        repair_retention=posts[1]['repair_totals'], response_checks=response_rows,
        correction=corrected, budget=result['budget'], startup=result['startup'],
        work=core['work'], costs=core['costs'], wall_s=result['wall_s'],
        peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes'],
        peak_cuda_reserved_bytes=result['peak_cuda_reserved_bytes'],
        ordinary_K_peak_cuda=result.get('ordinary_K_peak_cuda'),
        correction_peak_cuda=result.get('correction_peak_cuda'), local_advancement=advancement,
        limitation='Local training-conditioned diagnostic; no held-out or serving improvement established.')
    (root/'analysis').mkdir(exist_ok=True)
    with (root/'analysis/PER_BLOCK.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (root/'FINAL_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    result=audit()
    print(json.dumps({key:result[key] for key in ('status','issues')}))
    raise SystemExit(result['status'] != 'passed')
