"""Offline audit of retrieved main U/P receipts; no model or server execution.

The earlier independent auditor reconstructs native commits and source exposure.
This wrapper adds controlled startup, complete costs, direction counts and a
second comparison against the production validators/report on relocated files.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ARMS = ('F', 'O', 'U', 'P')
PAIRS = (('P', 'O'), ('P', 'U'), ('P', 'F'), ('O', 'F'))
BASE = Path(__file__).resolve().parents[1] / 'dflash_prefix_ranking_one_event_20261025'


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def check(ok, issues, message):
    if not ok:
        issues.append(message)


def close(a, b):
    return a is None and b is None or (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
        and math.isfinite(a) and math.isfinite(b)
        and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9))


def load_prior(path):
    spec = importlib.util.spec_from_file_location('prior_independent_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualification_bindings(reuse, seal, issues):
    dependencies = seal['dependency_sha256']
    receipt = Path(reuse['qualification_receipt'])
    for relative, digest in reuse['exact_gate_sha256'].items():
        check(dependencies.get(str(receipt.parent / relative)) == digest,
              issues, 'Reused qualification gate differs from seal: ' + relative)
    for path, field in [(receipt, 'qualification_receipt_sha256'),
                        (receipt.parent.parent / 'SEAL.json', 'qualification_seal_sha256'),
                        (receipt.parent.parent / 'runner.py', 'qualification_runner_sha256')]:
        check(dependencies.get(str(path)) == reuse[field], issues, 'Reused qualification binding: ' + field)
    pinned = [digest for path, digest in dependencies.items() if path.endswith('/trials/O2/initial.pt')]
    check(pinned == [reuse['pinned_initial_sha256']], issues, 'Reused pinned initial state differs from seal')


def startup(root, ordinary, seal, issues):
    reuse = read(root / 'QUALIFICATION_REUSE.json')
    check(reuse['status'] == 'passed' and reuse['reused'] is True, issues, 'Passed qualification not reused')
    check(len(reuse['exact_gate_sha256']) == 27, issues, 'Qualification binding lacks 27 gates')
    qualification_bindings(reuse, seal, issues)
    result = read(root / 'process_warmup/RESULT.json')
    work = lines(root / 'process_warmup/WORK.jsonl')
    counts = {}
    for kind, expected in [('backward', 2), ('temporary_optimizer_step', 1), ('current_training_forward', 2)]:
        counts[kind] = {state: sum(row['kind'] == kind and row.get('status') == state for row in work)
                        for state in ('started', 'completed')}
        check(counts[kind] == dict(started=expected, completed=expected), issues, 'Warmup work: ' + kind)
        for state, suffix in [('started', 'attempted'), ('completed', 'completed')]:
            check(result['work_counts'].get(kind + '_' + suffix) == counts[kind][state], issues,
                  'Warmup counter: ' + kind + '_' + suffix)
    check(result['status'] == 'passed' and result['main_optimizer_steps'] == result['generations'] == 0,
          issues, 'Warmup status or scope')
    check(result['scientific_state_inherited'] is False, issues, 'Warmup state inherited')
    initial_gates = {}
    for label, directory in [('warmup', 'process_warmup'), ('U', 'U/initialization'), ('P', 'P/initialization')]:
        gate = read(root / directory / 'initial_vs_pinned.json')
        initial_gates[label] = bool(gate['exact_equal'] and not any(gate[k] for k in
            ('missing_left', 'missing_right', 'metadata_differences', 'tensor_differences')))
        check(initial_gates[label], issues, label + ': initial exact gate')
    check(len(result['records']) == result['work_counts']['response_presentations'] == 2,
          issues, 'Warmup presentation accounting')
    shuffles = [row for row in work if row['kind'] == 'full_first_pass_shuffle']
    check(len(shuffles) == 1 and shuffles[0]['n'] == 128 and shuffles[0]['exact_equal']
          and shuffles[0]['record_ids'] == [r['record_id'] for r in ordinary['records'][:128]],
          issues, 'Warmup full original shuffle')
    for row, original in zip(result['records'], ordinary['records'][:2]):
        check(all(row[k] == original[k] for k in ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after')),
              issues, 'Warmup original first two presentations/anchors')
    for arm in ('U', 'P'):
        initial = read(root / (arm + '_INITIAL.json'))
        initialization = initial['controlled_initialization']
        check(initial['global_step'] == initial['presentations'] == initial['accumulation_remainder'] == 0
              and initial['completed_events'] == [] and initial['failed'] is False,
              issues, arm + ': fresh initial counters')
        check(initialization['exact_pinned_initial_state'] and initialization['full_rng_restored_after_construction']
              and initialization['inherited_warmup_state'] is False and initialization['reference_created'] is False,
              issues, arm + ': controlled initialization')
        final = read(root / (arm + '_CLOSED.json'))
        check(final['closed'] is True and final['failed'] is False and final['global_step'] == 128
              and final['presentations'] == 256 and final['accumulation_remainder'] == 0,
              issues, arm + ': trainer not cleanly closed before evaluation')
    return dict(status=result['status'], work=counts, wall_s=result['wall_s'],
                initial_gates=initial_gates, prior_qualification_reused=True,
                historical_O_limit=reuse['historical_O_limit'])


def independent_comparisons(rows, valid_ids):
    """Independent arithmetic: each request counts once; latency averages repeats."""
    grouped = {(request, method): [r for r in rows if r['request_id'] == request and r['method'] == method]
               for request in valid_ids for method in ARMS}
    measurements = {}
    for key, repeats in grouped.items():
        primary = next(r for r in repeats if r['repeat'] == 0)
        measurements[key] = dict(calls=primary['verification_calls'],
                                 latency=sum(r['request_latency_s'] for r in repeats) / 3)
    comparisons = {}
    for candidate, reference in PAIRS:
        values = {}
        for metric, scalar, reduction_name in [('call', 'calls', 'call_reduction_pct'),
                                               ('latency', 'latency', 'request_time_reduction_pct')]:
            before = [measurements[request, reference][scalar] for request in valid_ids]
            after = [measurements[request, candidate][scalar] for request in valid_ids]
            differences = [a - b for a, b in zip(before, after)]
            values[reduction_name] = 100 * (1 - sum(after) / sum(before)) if sum(before) else None
            values[metric + '_request_directions'] = dict(improved=sum(d > 0 for d in differences),
                                                         regressed=sum(d < 0 for d in differences),
                                                         tied=sum(d == 0 for d in differences))
            values[metric + '_gross_saved'] = sum(max(d, 0) for d in differences)
            values[metric + '_gross_extra'] = sum(max(-d, 0) for d in differences)
            values[metric + '_net_saved'] = sum(differences)
        comparisons[candidate + '_vs_' + reference] = values
    return comparisons


def native_and_report(root, requests, ordinary_measurements, prior, issues):
    from experiments.prefix_ranking import protocol, report

    result = prior.evaluation(root, requests, ordinary_measurements, issues)
    check(result['completed'] is True and result['status'] == 'completed'
          and result['raw_measurements'] == 768 and result['valid_common_requests'] == 64,
          issues, 'Evaluation is not complete with all 768 valid slots and 64 paired requests')
    directory = root / 'evaluation'
    rows = lines(directory / 'MEASUREMENTS.jsonl')
    metadata = read(directory / 'PROTOCOL.json')
    expected_order = [(r['request_id'], repeat, method) for r in requests for repeat in range(3)
                      for method in protocol.method_order(r['panel_ordinal'], repeat)]
    actual_order = [(r['request_id'], r['repeat'], r['method']) for r in rows]
    check(actual_order == expected_order[:len(rows)], issues, 'Actual generation order differs')
    relocated = [dict(row, record_path=str(directory / Path(row['record_path']).name)) for row in rows]
    shared, per_request, _, _ = report.summarize(relocated, requests, checkpoints=metadata['checkpoints'],
                                               ordinary_counts=metadata['ordinary_counts'])
    saved = read(directory / 'report/SUMMARY.json')
    for key in ('valid_requests', 'invalid_or_incomplete_requests', 'returned_measurements', 'valid_timed_generations'):
        check(shared[key] == saved[key], issues, 'Revalidated report count: ' + key)
    for method in ARMS:
        check(all(close(value, saved[method][key]) for key, value in shared[method].items()),
              issues, 'Revalidated method totals: ' + method)
        weights = read(directory / ('WEIGHTS_' + method + '.json'))
        check(not weights['errors'] and not protocol.weight_check_errors(weights['native_reply'], metadata['checkpoints'][method]),
              issues, 'Native published weight equality: ' + method)
    valid_ids = [row['request_id'] for row in per_request if row['valid']]
    independent = independent_comparisons(rows, valid_ids)
    for pair, values in independent.items():
        for key, value in values.items():
            expected = saved['comparisons'][pair][key]
            check(value == expected if isinstance(value, dict) else close(value, expected),
                  issues, 'Independent comparison arithmetic: ' + pair + '.' + key)
    activations = lines(directory / 'ACTIVATIONS.jsonl')
    for activation in activations:
        check(not activation['errors'] and not protocol.publication_errors(activation, metadata['checkpoints'][activation['method']]),
              issues, 'Native publication/cache contract')
    execution = read(directory / 'STATUS.json')
    check(close(sum(row['activation_s'] for row in activations), execution['activation_s']),
          issues, 'Activation accounting differs')
    check(execution['optimizer_steps'] == execution['feedback_collections'] == 0, issues, 'Evaluation supplied learning work')
    result.update(independent_comparisons=independent, production_validator_valid_requests=shared['valid_requests'],
                  publication_receipts=len(activations), activation_s=execution['activation_s'])
    return result


def cost_accounting(root, training, issues):
    costs = read(root / 'COSTS.json')
    adaptation = {}
    for arm in ('U', 'P'):
        entry = costs[arm]
        event = training[arm]
        check(entry == read(root / (arm + '_COSTS.json')), issues, arm + ': duplicate cost receipt differs')
        initialization = read(root / (arm + '_INITIAL.json'))['controlled_initialization']
        check(initialization == read(root / arm / 'initialization/RESULT.json')
              and close(entry['initialization_control_s'], initialization['wall_s']),
              issues, arm + ': initialization cost receipt differs')
        check(entry['objective_work'] == event['objective_work'], issues, arm + ': reference costs differ')
        for key, value in event['costs'].items():
            check(close(entry[key], value), issues, arm + ': event cost ' + key)
        check(close(entry['training_s'], sum(r['training_s'] for r in read(root / arm / 'events/event_001/RESULT.json')['records'])),
              issues, arm + ': summed presentation training cost')
        check(entry['reference_cost_is_nested_in_event_wall'] is True
              and entry['teacher_preparation_s'] == entry['new_feedback_generations'] == 0,
              issues, arm + ': reused-feedback/nested cost declaration')
        check(entry['event_wall_s'] + 1e-6 >= sum(entry[k] for k in ('training_s', 'checkpoint_save_s', 'export_s')),
              issues, arm + ': event wall excludes declared work')
        adaptation[arm] = dict(event_wall_s=entry['event_wall_s'], loading_s=entry['loading_s'],
                               initialization_control_s=entry['initialization_control_s'],
                               reference_nested_s=sum(v for k, v in entry['objective_work'].items() if k.startswith('reference_')),
                               peak_allocated_bytes=entry['peak_allocated_bytes'], peak_reserved_bytes=entry['peak_reserved_bytes'])
    warmup = read(root / 'process_warmup/RESULT.json')
    check(costs['process_warmup'] == warmup, issues, 'Warmup cost receipt differs')
    evaluation = read(root / 'evaluation/STATUS.json')
    counted_once = sum(v[k] for v in adaptation.values() for k in ('event_wall_s', 'loading_s', 'initialization_control_s'))
    counted_once += warmup['wall_s'] + costs['backend_loading_s'] + evaluation['evaluation_wall_s']
    check(costs['actual_campaign_wall_s'] + 1e-6 >= counted_once, issues, 'Distinct phase costs exceed campaign wall')
    return dict(arms=adaptation, warmup_s=warmup['wall_s'], backend_loading_s=costs['backend_loading_s'],
                evaluation_wall_s=evaluation['evaluation_wall_s'], nonoverlapping_measured_phase_sum_s=counted_once,
                actual_campaign_wall_s=costs['actual_campaign_wall_s'],
                note='Reference time is nested, never added to event wall; evaluation publication is diagnostic, not deployment adaptation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'manifest', 'ordinary-receipt', 'ordinary-measurements', 'retention-audit', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--worktree', type=Path, default=BASE / 'worktree')
    parser.add_argument('--prior-auditor', type=Path, default=BASE / 'audit_results.py')
    args = parser.parse_args()
    sys.path.insert(0, str(args.worktree))
    prior = load_prior(args.prior_auditor)
    issues, result = [], {}
    ordinary = read(args.ordinary_receipt)
    requests = read(args.manifest)['eval_requests']
    status = read(args.campaign / 'STATUS.json')
    result['campaign_status'] = status['status']
    try:
        seal = read(args.retention_audit.parent / 'SEAL.json')
        result['startup'] = startup(args.campaign, ordinary, seal, issues)
        training = {arm: prior.train_receipt(args.campaign, arm, ordinary, issues) for arm in ('U', 'P')}
        result['training'] = training
        source_rows = {arm: read(args.campaign / arm / 'events/event_001/RESULT.json')['records']
                       for arm in ('U', 'P')}
        reference_differences = []
        for index, (u, p) in enumerate(zip(source_rows['U'], source_rows['P'])):
            for key in ('reference_prefix_positions', 'reference_prefix_weight'):
                if u['metrics'].get(key) != p['metrics'].get(key):
                    reference_differences.append(dict(presentation=index, metric=key,
                                                      U=u['metrics'].get(key), P=p['metrics'].get(key)))
        check(not reference_differences, issues, 'U/P fixed-reference prefix summaries differ')
        result['reference_summary_differences'] = reference_differences
        for key, expected in [('main_optimizer_steps', 256), ('main_response_presentations', 512),
                              ('new_feedback_generations', 0), ('new_teacher_preparation_forwards', 0)]:
            check(status.get(key) == expected, issues, 'Campaign accounting: ' + key)
        check(len(requests) == 64, issues, 'Development panel does not have 64 requests')
        result['evaluation'] = native_and_report(args.campaign, requests, lines(args.ordinary_measurements), prior, issues)
        result['costs'] = cost_accounting(args.campaign, training, issues)
        retention = read(args.retention_audit)
        check(retention.get('status') == 'passed' and retention.get('retained_artifacts') == 132
              and retention.get('source_files') == len(seal['source_sha256'])
              and retention.get('qualification_dependencies') == len(seal['dependency_sha256'])
              and retention.get('changes') == [] and retention.get('exit_code') == 0,
              issues, 'Final immutable-artifact audit')
        result['retention'] = retention
    except (KeyError, ValueError, TypeError, OSError, IndexError) as error:
        issues.append('Unavailable or malformed evidence: ' + repr(error))
    result.update(audit_status='passed' if not issues and status['status'] == 'completed' else 'incomplete_or_inconsistent',
                  issues=issues, historical_O_full_trajectory_equivalence=False,
                  scope='Offline compact receipts and native trajectories; no model tensor replay or new execution.',
                  prior_auditor_sha256=hashlib.sha256(args.prior_auditor.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(audit_status=result['audit_status'], issues=issues, output=str(args.output)), indent=2))
    return int(result['audit_status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
