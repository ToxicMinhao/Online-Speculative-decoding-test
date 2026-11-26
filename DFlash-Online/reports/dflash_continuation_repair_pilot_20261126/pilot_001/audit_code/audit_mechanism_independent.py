#!/usr/bin/env python3
"""Reconstruct continuation-repair outcomes directly from retained decisions.

This auditor imports no project, model, Torch, or production metric code. Its
independence is computational, not organizational: the author also implemented
native training plumbing and inspected the production metric schema. It cannot
recover full-vocabulary rival optimality or rederive validity without features.
"""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

FIELDS = ('raw_repair', 'useful_repair', 'reachable_origin_followers', 'lost_prefix',
          'damaged_prefix_positions', 'prefix_damaged', 'surviving_origin_followers',
          'positive_progress', 'Y', 'uncapped_progress')
ROLE_FIELDS = ('labels', 'valid', 'prefix', 'repair', 'L', 'e', 'H', 'K', 'm')
RETAINED_MANIFEST = dict(bytes=49428, sha256='760ca8ce26c7132d9abeab71e16700d429353ecd890da13de3262754e404e683')


def file_identity(path):
    checksum = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        while data := stream.read(1024 * 1024):
            size += len(data)
            checksum.update(data)
    return dict(bytes=size, sha256=checksum.hexdigest())


def roles(block):
    """Native token equality, contiguous validity, and a fixed old repair horizon."""
    labels = block['labels']
    valid = block['valid']
    proposals = block['proposals']
    if not (len(labels) == len(valid) == len(proposals) == 16) or valid[0]:
        raise ValueError('Expected sixteen slots with the known seed excluded')
    available = sum(valid)
    if valid != [False] + [True] * available + [False] * (15 - available):
        raise ValueError('Validity does not form a contiguous response suffix')
    length = 0
    while length < available and proposals[length + 1] == labels[length + 1]:
        length += 1
    error = length + 1 if length < available else -1
    horizon = length
    if error >= 1:
        horizon = error
        while horizon < available and proposals[horizon + 1] == labels[horizon + 1]:
            horizon += 1
    return dict(labels=labels, valid=valid,
                prefix=[False] + [True] * length + [False] * (15 - length),
                repair=[slot == error for slot in range(16)], L=length, e=error,
                H=horizon-length if error >= 1 else 0, K=horizon, m=available)


def compact_legacy(block):
    return dict(labels=block['current']['labels'], valid=block['current']['valid'],
                proposals=block['proposals'], roles=block['current'], positions=block['positions'])


def outcomes(parent, child):
    """Direct position counts, independently of the production paired helper."""
    if parent['labels'] != child['labels'] or parent['valid'] != child['valid']:
        raise ValueError('Matched block labels or validity changed')
    origin = roles(parent)
    current = roles(child)
    old_length, error, horizon = origin['L'], origin['e'], origin['K']
    new_length = current['L']
    labels, predictions = parent['labels'], child['proposals']
    raw = error >= 1 and predictions[error] == labels[error]
    useful = raw and all(predictions[slot] == labels[slot] for slot in range(1, error))
    followers = range(error+1, horizon+1) if error >= 1 else ()
    reachable = sum(new_length >= slot for slot in followers)
    loss = sum(new_length < slot for slot in range(1, old_length+1))
    signed = min(new_length, horizon) - old_length
    values = dict(raw_repair=int(raw), useful_repair=int(useful),
        reachable_origin_followers=reachable, lost_prefix=loss,
        damaged_prefix_positions=sum(predictions[slot] != labels[slot] for slot in range(1, old_length+1)),
        prefix_damaged=int(loss > 0),
        surviving_origin_followers=sum(predictions[slot] == labels[slot] for slot in followers),
        positive_progress=max(signed, 0), Y=signed, uncapped_progress=new_length-old_length)
    if signed != int(useful) + reachable - loss:
        raise ValueError('Direct signed-progress decomposition failed')
    return origin, current, values


class Audit:
    def __init__(self, raw, retained, output):
        self.raw, self.retained, self.output = Path(raw), Path(retained), Path(output)
        self.count = 0
        self.inputs = {}
        self.retained_entries = None
        self.production_rows = None
        self.response_rows = []
        self.results = {}

    def check(self, condition, label):
        self.count += 1
        if not condition:
            raise ValueError(label)

    def read(self, path, *, historical=False):
        path = Path(path)
        actual = file_identity(path)
        if historical:
            expected = self.retained_entries[path.relative_to(self.retained).as_posix()]
            self.check(actual == expected, 'Retained authentication changed: ' + str(path))
        self.inputs[str(path)] = actual
        return json.loads(path.read_text())

    def validate_block(self, block, label):
        direct = roles(block)
        self.check(all(block['roles'][key] == direct[key] for key in ROLE_FIELDS), label+': direct roles')
        positions = block['positions']
        self.check(len({row['position'] for row in positions}) == len(positions), label+': position uniqueness')
        for row in positions:
            slot = row['position']
            self.check(1 <= slot <= direct['m'], label+': recorded valid position')
            self.check(row['canonical'] == block['labels'][slot], label+': canonical binding')
            self.check(row['prediction'] == block['proposals'][slot], label+': prediction binding')
            self.check(row['correct'] == (row['prediction'] == row['canonical']), label+': native correctness')
        return direct

    def validate_receipt(self, receipt, binding, *, expected_endpoint):
        self.check(receipt['status'] == 'passed' and receipt['endpoint'] == expected_endpoint,
                   'Native endpoint/status binding')
        self.check(receipt['record_id'] == binding['record_id'], 'Native response identity')
        self.check(receipt['anchors'] == binding['anchors'], 'Native anchor order')
        self.check(len(receipt['blocks']) == len(binding['anchors']), 'Native complete block count')
        for key in ('parameters_unchanged', 'gradients_unchanged', 'runtime_restored', 'rng_preserved'):
            self.check(receipt[key] is True, 'Native receipt failed ' + key)

    def production_index(self, name):
        rows = self.production_rows[name]
        result = {(row['record_id'], row['anchor']): row for row in rows}
        self.check(len(result) == len(rows), 'Production row keys duplicated: ' + name)
        return result

    def cohort(self, name, bindings, parents, children, production_names):
        expected_blocks = 8192 if name == 'primary' else 64
        expected_responses = 128 if name == 'primary' else 8
        self.check(len(bindings) == expected_responses, name+': response denominator')
        self.check(len({b['record_id'] for b in bindings}) == len(bindings), name+': unique responses')
        indexes = {arm: self.production_index(production_names[arm]) for arm in ('BASE', 'TAIL')}
        totals = {arm: defaultdict(int) for arm in ('BASE', 'TAIL')}
        visited = set()
        by_response = {}
        for ordinal, binding in enumerate(bindings):
            record_id = binding['record_id']
            parent = parents[record_id]
            per_response = {arm: defaultdict(int) for arm in ('BASE', 'TAIL')}
            self.check(parent['anchors'] == binding['anchors'], name+': parent anchor binding')
            self.check(len(set(binding['anchors'])) == len(binding['anchors']), name+': unique anchors')
            for block_index, anchor in enumerate(binding['anchors']):
                parent_block = parent['blocks'][block_index]
                old_roles = self.validate_block(parent_block, name+': parent')
                visited.add((record_id, anchor))
                for arm in ('BASE', 'TAIL'):
                    current_response = children[arm][record_id]
                    self.check(current_response['anchors'] == binding['anchors'], name+': child anchor binding')
                    child = current_response['blocks'][block_index]
                    self.validate_block(child, name+': '+arm)
                    origin, current, values = outcomes(parent_block, child)
                    reported = indexes[arm][record_id, anchor]
                    self.check(reported['parent']['labels'] == parent_block['labels']
                               and reported['parent']['valid'] == parent_block['valid']
                               and reported['parent']['proposals'] == parent_block['proposals'],
                               name+': frozen parent decisions')
                    self.check(reported['child']['proposals'] == child['proposals'], name+': child decisions')
                    for key in ('L', 'e', 'K', 'm'):
                        self.check(reported[key] == origin[key], name+': frozen '+key)
                    self.check(reported['high'] == (origin['e'] >= 1 and origin['K'] > origin['e']), name+': HIGH role')
                    self.check(reported['child_L'] == current['L'], name+': child prefix')
                    self.check(old_roles == origin, name+': immutable origin')
                    self.check(values['Y'] == values['useful_repair'] + values['reachable_origin_followers']
                               - values['lost_prefix'], name+': Y identity')
                    per_response[arm]['blocks'] += 1
                    per_response[arm]['origin_errors'] += int(origin['e'] >= 1)
                    for field, value in values.items():
                        self.check(reported[field] == value, name+': block '+field)
                        per_response[arm][field] += value
            row = dict(response_ordinal=ordinal, cohort=name)
            original_row = dict(record_id=record_id, cohort=name)
            for arm in ('BASE', 'TAIL'):
                for field, value in per_response[arm].items():
                    totals[arm][field] += value
                    row[arm+'_'+field] = value
                    original_row[arm+'_'+field] = value
            for field in FIELDS:
                value = per_response['TAIL'][field] - per_response['BASE'][field]
                row['TAIL_minus_BASE_'+field] = value
                original_row['TAIL_minus_BASE_'+field] = value
            self.response_rows.append(row)
            by_response[record_id] = original_row
        self.check(len(visited) == expected_blocks, name+': block denominator')
        self.check(all(set(index) == visited for index in indexes.values()), name+': complete production block coverage')
        summary = self.summary[name]
        self.check(summary['cohort'] == name and summary['responses'] == expected_responses
                   and summary['matched_blocks'] == expected_blocks, name+': summary denominators')
        for arm in ('BASE', 'TAIL'):
            self.check(dict(totals[arm]) == summary['totals'][arm], name+': pooled '+arm)
        difference = {field: totals['TAIL'][field]-totals['BASE'][field] for field in FIELDS}
        self.check(difference == summary['differences'], name+': pooled differences')
        self.check(len(summary['per_response']) == len(by_response), name+': per-response coverage')
        for row in summary['per_response']:
            self.check(row == by_response[row['record_id']], name+': per-response totals')
        return dict(responses=expected_responses, blocks=expected_blocks,
                    totals={arm: dict(values) for arm, values in totals.items()}, differences=difference)

    def execute(self):
        manifest_path = self.retained/'RETRIEVAL_MANIFEST.json'
        self.check(file_identity(manifest_path) == RETAINED_MANIFEST, 'Original PR64 retrieval manifest authentication')
        manifest = self.read(manifest_path)
        self.retained_entries = {row['path']: dict(bytes=row['bytes'], sha256=row['sha256'])
                                 for row in manifest['files']}
        self.check(len(self.retained_entries) == len(manifest['files']), 'Retained manifest paths unique')
        contract = self.read(self.retained/'CONTRACT.json', historical=True)
        self.check(self.read(self.retained/'STATUS.json', historical=True)['status'] == 'completed',
                   'Retained diagnostic completed')
        mechanism_status = self.read(self.raw/'mechanism/STATUS.json')
        self.check(mechanism_status['status'] == 'completed', 'New mechanism phase completed')
        self.production_rows = self.read(self.raw/'mechanism/BLOCKS_PRIVATE.json')
        self.summary = self.read(self.raw/'mechanism/SUMMARY.json')
        bindings = contract['features']
        parent, base, tail = {}, {}, {}
        for binding in bindings:
            ordinal, record_id = binding['ordinal'], binding['record_id']
            shard = self.retained/'primary'/f'gpu{binding["physical_shard"]}'
            parent[record_id] = self.read(shard/'parent'/f'{ordinal:03d}.json', historical=True)
            base[record_id] = self.read(shard/'child'/f'{ordinal:03d}.json', historical=True)
            tail[record_id] = self.read(self.raw/'mechanism/primary'/f'{ordinal:03d}.json')
            for receipt, endpoint in ((parent[record_id], 'parent'), (base[record_id], 'child'), (tail[record_id], 'TAIL')):
                self.validate_receipt(receipt, binding, expected_endpoint=endpoint)
        self.results['primary'] = self.cohort('primary', bindings, parent, {'BASE':base, 'TAIL':tail},
                                             {'BASE':'BASE', 'TAIL':'TAIL'})
        retention = contract['legacy_binding']['sets']['parent_training']
        old_parent = self.read(self.retained/'legacy/parent.json', historical=True)['sets']['parent_training']
        old_base = self.read(self.retained/'legacy/child.json', historical=True)['sets']['parent_training']
        def convert(rows):
            converted = {row['record_id']: dict(record_id=row['record_id'], anchors=row['anchors'],
                                                blocks=[compact_legacy(b) for b in row['blocks']]) for row in rows}
            self.check(len(converted) == len(rows), 'Legacy unique response identities')
            return converted
        tail = {}
        for ordinal, binding in enumerate(retention):
            receipt = self.read(self.raw/'mechanism/parent_training'/f'{ordinal:03d}.json')
            self.validate_receipt(receipt, binding, expected_endpoint='TAIL')
            tail[binding['record_id']] = receipt
        self.results['parent_training'] = self.cohort('parent_training', retention, convert(old_parent),
            {'BASE':convert(old_base), 'TAIL':tail}, {'BASE':'retention_BASE', 'TAIL':'retention_TAIL'})
        return self.results

    def save(self, result):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'AUDIT_MECHANISM.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        (self.output/'INPUTS_PRIVATE.json').write_text(json.dumps(self.inputs, indent=2)+'\n')
        if self.response_rows:
            with (self.output/'PER_RESPONSE_MECHANISM.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.response_rows[0]))
                writer.writeheader()
                writer.writerows(self.response_rows)


def self_test():
    tested = 0
    # Exhaustive paired decision states for small blocks, including empty/no-error.
    for size in range(6):
        labels = list(range(16))
        valid = [False]+[True]*size+[False]*(15-size)
        for previous in range(2**size):
            for current in range(2**size):
                def block(bits):
                    proposals = labels.copy()
                    for slot in range(1,size+1):
                        if bits & (1 << (slot-1)):
                            proposals[slot] = -1
                    return dict(labels=labels, valid=valid, proposals=proposals)
                parent, child = block(previous), block(current)
                old, new, metrics = outcomes(parent, child)
                assert metrics['useful_repair'] <= metrics['raw_repair']
                assert metrics['reachable_origin_followers'] <= metrics['surviving_origin_followers']
                assert metrics['damaged_prefix_positions'] <= metrics['lost_prefix']
                assert metrics['Y'] == metrics['useful_repair']+metrics['reachable_origin_followers']-metrics['lost_prefix']
                tested += 1
    print(json.dumps(dict(status='passed', exhaustive_pairs=tested)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root')
    parser.add_argument('--retained-root')
    parser.add_argument('--output')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.raw_root, args.retained_root, args.output)):
        parser.error('Audit requires --raw-root, --retained-root and --output')
    audit = Audit(args.raw_root, args.retained_root, args.output)
    result = dict(status='running', checks=0, independence='Independent standard-library reconstruction; shared-context/native-plumbing author',
                  limitations=['Validity and native proposals are authenticated retained observations, not independently rerun.',
                               'Full-vocabulary rival optimality and model-forward correctness cannot be reconstructed from compact records.',
                               'TRAIN/retention outcomes are descriptive and do not establish serving gains; no regression or block-level inference.'])
    started = time.perf_counter()
    try:
        result['cohorts'] = audit.execute()
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(checks=audit.count, wall_s=time.perf_counter()-started,
                      audited_input_files=len(audit.inputs), partial_cohorts=audit.results)
        audit.save(result)
    print(json.dumps(dict(status=result['status'], checks=result['checks'], wall_s=result['wall_s'])))


if __name__ == '__main__':
    main()
