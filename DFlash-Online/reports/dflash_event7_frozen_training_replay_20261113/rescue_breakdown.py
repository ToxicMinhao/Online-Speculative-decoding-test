"""Descriptive endpoint breakdown, independently derived from public scalar rows."""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path


def reduce(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    key = lambda row: (row['request_id'], row['canonical_start'], row['proposal_slot'])
    multiplicity = Counter(key(row) for row in rows)
    counts = Counter()
    weighted = defaultdict(Fraction)
    for row in rows:
        n, r = int(row['native_accepts_through']), int(row['recent_accepts_through'])
        if n == r:
            continue
        direction = 'rescue' if r else 'loss'
        missing_arm = 'native' if r else 'recent'
        outcome = row[missing_arm + '_outcome']
        direct = int(row[missing_arm + '_direct_correct'])
        if outcome == 'first_mismatch_at_tag':
            assert not direct
            category = 'exact_tag_mismatch'
        else:
            assert outcome == 'earlier_break'
            category = 'earlier_break_tag_already_correct' if direct else 'earlier_break_tag_also_wrong'
        name = direction + '_' + category
        counts[name] += 1
        weighted[name] += Fraction(1, multiplicity[key(row)])
    for direction in ('rescue', 'loss'):
        expected = sum(int(row[direction]) for row in rows)
        actual = sum(value for name, value in counts.items() if name.startswith(direction + '_'))
        assert expected == actual
    result = dict(status='complete', paired_occurrences=len(rows), slots=len(multiplicity),
        interpretation='Descriptions of paired endpoint predictions, not update-level causal mediation.',
        categories=[])
    for direction in ('rescue', 'loss'):
        for category in ('exact_tag_mismatch', 'earlier_break_tag_already_correct', 'earlier_break_tag_also_wrong'):
            name = direction + '_' + category
            mass = weighted[name]
            result['categories'].append(dict(category=name, occurrences=counts[name],
                slot_equivalents=float(mass), slot_equivalents_exact=f'{mass.numerator}/{mass.denominator}',
                percentage_points=float(100 * mass / len(multiplicity))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paired_occurrences')
    parser.add_argument('output')
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(reduce(args.paired_occurrences), indent=2) + '\n')
