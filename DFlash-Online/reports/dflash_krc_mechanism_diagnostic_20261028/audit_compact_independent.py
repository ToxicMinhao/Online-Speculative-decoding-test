"""Independent standard-library audit of retained mechanism block evidence.

No production imports, model passes, network operations, or source writes.
Valid/EOS masks are checked for consistency; their native creation is covered
by the separately reviewed execution path, not reconstructed from full logits.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

STEPS = (1, 16, 32, 48, 64, 65, 80, 96, 112, 128)
CANDIDATES = ('R_FULL', 'R_HALF', 'D_FULL', 'K_ONLY')
ROLES = ('A', 'R', 'D')


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def same(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) if isinstance(b, float) else a == b


def prefix(predictions, labels, valid):
    result = 0
    for offset in range(1, len(labels)):
        if not valid[offset] or predictions[offset] != labels[offset]:
            break
        result += 1
    return result


def reconstruct(row):
    y = row['canonical_tokens']
    valid = row['valid_positions']
    before, ordinary, candidate = [row[k] for k in ('pre_K_predictions', 'K_ONLY_predictions', 'candidate_predictions')]
    w = row['protected_weights']
    km, cm = row['K_ONLY_canonical_margins'], row['candidate_canonical_margins']
    assert all(len(v) == len(y) for v in (valid, before, ordinary, candidate, w, km, cm))
    available = sum(valid)
    assert valid == [False] + [i <= available for i in range(1, len(y))]
    pre, k, c = [prefix(p, y, valid) for p in (before, ordinary, candidate)]
    assert w == [0] + [pre - i + 1 if i <= pre else 0 for i in range(1, len(y))]
    assert available == 0 or row['kept_anchor']
    for p, margins in ((ordinary, km), (candidate, cm)):
        assert all(math.isfinite(m) for m in margins)
        assert all(not valid[i] or ((p[i] == y[i] and margins[i] >= 0) or (p[i] != y[i] and margins[i] <= 0)) for i in range(len(y)))
    old_errors = {i for i in range(1, pre + 1) if ordinary[i] != y[i]}
    new_errors = {i for i in range(1, pre + 1) if candidate[i] != y[i]}
    change, gain = c - k, max(0, k - pre)
    retained = min(gain, max(0, c - pre))
    error = pre < available
    kr, cr = error and k > pre, error and c > pre
    counts = dict(valid_blocks=int(available > 0), pre_K_prefix_tokens=pre, K_ONLY_prefix_tokens=k,
        candidate_prefix_tokens=c, prefix_change_sum=change, gross_prefix_gains=max(0, change),
        gross_prefix_losses=max(0, -change), improved_blocks=int(available > 0 and change > 0),
        regressed_blocks=int(available > 0 and change < 0), tied_blocks=int(available > 0 and change == 0),
        old_prefix_tokens_lost=max(0, pre - c), K_new_prefix_gain=gain,
        K_prefix_gain_retained=retained, K_prefix_gain_lost=gain - retained,
        original_first_errors=int(error), K_useful_repairs=int(kr), C_useful_repairs=int(cr),
        useful_repairs_retained=int(kr and cr), useful_repairs_lost=int(kr and not cr),
        useful_repairs_newly_obtained=int(cr and not kr), protected_positions=pre,
        K_protected_errors=len(old_errors), candidate_protected_errors=len(new_errors),
        protected_errors_repaired=len(old_errors - new_errors), new_protected_errors=len(new_errors - old_errors),
        K_strict_negative_margin_errors=sum(km[i] < 0 for i in old_errors),
        K_wrong_native_tie_errors=sum(km[i] == 0 for i in old_errors),
        candidate_strict_negative_margin_errors=sum(cm[i] < 0 for i in new_errors),
        candidate_wrong_native_tie_errors=sum(cm[i] == 0 for i in new_errors),
        protected_weight=sum(w), K_hinge_numerator_fp64=sum(a * max(0, -b) for a, b in zip(w, km)),
        candidate_hinge_numerator_fp64=sum(a * max(0, -b) for a, b in zip(w, cm)))
    for key, value in counts.items():
        assert same(row[key], value), (key, row[key], value)
    first = min(old_errors) if old_errors else None
    one, all_fixed = list(ordinary), list(ordinary)
    if first is not None:
        one[first] = y[first]
    for offset in old_errors:
        all_fixed[offset] = y[offset]
    oracle = dict(first=prefix(one, y, valid) - k, all=prefix(all_fixed, y, valid) - k,
                  first_kind=None if first is None else ('wrong_native_tie' if km[first] == 0 else 'strict_negative_margin'))
    assert row['oracle_first_protected_error_offset'] == first
    assert row['oracle_first_protected_error_kind'] == oracle['first_kind']
    assert row['oracle_first_prefix_gain'] == oracle['first']
    assert row['oracle_all_prefix_gain'] == oracle['all']
    return counts, oracle


def audit(root, analysis=None):
    root = Path(root)
    plan = read(root / 'sealed/PLAN.json')
    result = dict(status='passed', execution_source=plan['execution_source'], issues=[],
        windows=0, candidate_slots=0, block_rows=0, endpoint_gates={}, oracle_totals={},
        review_scope='independent scalar reconstruction, bindings and receipts; no GPU replication')
    aggregate = defaultdict(lambda: defaultdict(float))
    oracle_totals = defaultdict(lambda: defaultdict(int))
    for event in (2, 4):
        folder = root / 'campaign_001' / f'event_{event:03d}'
        equality = read(folder / 'ENDPOINT_EQUALITY.json')
        result['endpoint_gates'][str(event)] = equality
        assert equality['status'] == 'passed' and equality['exact_decoded_scientific_state'] and equality['scientific_mismatch_count'] == 0
        sealed_windows = {w['step']: w for w in read(root / 'sealed' / f'WINDOW_PLAN_{event:03d}.json')}
        for step in STEPS:
            record = read(folder / 'mechanism' / f'window_{step:03d}.json')
            assert record['plan'] == sealed_windows[step]
            assert record['status'] == 'completed' and record['restoration_passed'] and record['k_only_witness_passed']
            assert record['candidate_order'] == list(CANDIDATES)
            assert set(record['candidates']) == set(CANDIDATES)
            original = record['original_R_post_K']
            witness = record['candidates']['K_ONLY']['roles']['R']
            assert all(original[key] == witness[key] for key in ('V', 'numerator', 'denominator', 'responses'))
            result['windows'] += 1
            result['candidate_slots'] += 4
            groups, bindings = defaultdict(lambda: defaultdict(float)), defaultdict(dict)
            block_path = folder / 'mechanism' / f'window_{step:03d}_BLOCKS.jsonl'
            for line in block_path.read_text().splitlines():
                row = json.loads(line)
                assert (row['event'], row['step'], row['pass']) == (event, step, (step - 1) // 64)
                candidate, role = row['candidate'], row['role']
                assert candidate in CANDIDATES and role in ROLES
                role_records = {b['record_id']: b for b in record['plan'][role]}
                binding = role_records[row['record_id']]['sampled_anchors'][0]
                index = row['block_index']
                assert row['anchor'] == binding['positions'][0][index]
                assert row['kept_anchor'] == binding['keep_mask'][0][index]
                key = (row['record_id'], index)
                assert key not in bindings[candidate, role]
                fields = ('anchor', 'kept_anchor', 'canonical_tokens', 'valid_positions', 'protected_weights',
                          'pre_K_predictions', 'K_ONLY_predictions', 'K_ONLY_canonical_margins')
                bindings[candidate, role][key] = {k: row[k] for k in fields}
                counts, oracle = reconstruct(row)
                for name, value in counts.items():
                    groups[candidate, role][name] += value
                    aggregate[event, (step - 1) // 64, candidate, role][name] += value
                if candidate == 'K_ONLY':
                    oracle_totals[event, role]['first'] += oracle['first']
                    oracle_totals[event, role]['all'] += oracle['all']
                    if oracle['first_kind'] is not None:
                        oracle_totals[event, role][oracle['first_kind'] + '_gain'] += oracle['first']
                result['block_rows'] += 1
            for candidate in CANDIDATES:
                entry = record['candidates'][candidate]
                if entry['status'] != 'valid':
                    assert entry['status'] == 'invalid' and all(entry['restoration_after_failure'].values())
                    continue
                for role in ROLES:
                    assert bindings[candidate, role] == bindings['K_ONLY', role]
                    expected_rows = sum(len(b['sampled_anchors'][0]['positions'][0]) for b in record['plan'][role])
                    assert len(bindings[candidate, role]) == expected_rows
                    saved = entry['roles'][role]['metrics']['summary']
                    assert all(same(saved[name], value) for name, value in groups[candidate, role].items())
    if analysis:
        with (Path(analysis) / 'CANDIDATE_ROLE.csv').open() as stream:
            for row in csv.DictReader(stream):
                key = int(row['event']), int(row['pass_index']), row['candidate'], row['role']
                assert key in aggregate
                assert all(same(float(row[name]), value) for name, value in aggregate[key].items())
        result['published_candidate_role_sums_match'] = True
    result['oracle_totals'] = {f'{e}/{role}': dict(values) for (e, role), values in oracle_totals.items()}
    result['aggregate'] = [dict(event=e, pass_index=p, candidate=c, role=r, **values)
                           for (e, p, c, r), values in aggregate.items()]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--analysis')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.root, args.analysis)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'windows', 'candidate_slots', 'block_rows')}))
