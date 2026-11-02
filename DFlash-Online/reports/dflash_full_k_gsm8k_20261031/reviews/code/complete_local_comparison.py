"""Complete the type-normalized comparison from the preserved CPU audit JSON.

The remote audit completed every raw-record check and all statistics, then
rejected numpy.float64 in its own strict built-in-number comparison guard. JSON
normalizes those values to Python floats. This continuation preserves that
failed audit and makes no model/remote calls or modifications to old receipts.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
WORK = TASK / 'review_20261102'
DEST = TASK / 'final_delivery_001/reviews'
REMOTE = '/data/sources/minhao/runs/dflash_full_k_gsm8k_20261031/production_002'
raw_path = WORK / 'RAW_STATISTICS_AUDIT_FULL.json'
raw = json.loads(raw_path.read_text())
counts = Counter()
max_difference = 0.0


def check(ok, category):
    counts[category] += 1
    if not ok:
        raise AssertionError(category)


def same(a, b, context):
    global max_difference
    if isinstance(a, dict):
        for k, v in a.items():
            same(v, b[k], context + '/' + k)
    elif isinstance(a, list):
        check(len(a) == len(b), 'matching_array_length')
        for i, v in enumerate(a):
            same(v, b[i], context + '/' + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        check(isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b), 'finite_numeric_comparison')
        max_difference = max(max_difference, abs(a-b))
        check(math.isclose(a, b, rel_tol=2e-13, abs_tol=2e-11), 'numeric_match:' + context)
    else:
        check(a == b, 'exact_match:' + context)


def matched_local(remote, local):
    payload = local.read_bytes()
    check(hashlib.sha256(payload).hexdigest() == raw['source_file_identities'][remote]['sha256'], 'local_matches_remote_audited_hash')
    return json.loads(payload)


check(raw['status'] == 'failed' and raw['errors'] == ["AssertionError('finite_statistics: comparisons/K_HALF_vs_O/call_reduction/point_pct')"], 'recognized_audit_only_failure')
check(raw['audit_totals']['new_evaluation_records'] == 13056 and raw['audit_totals']['historical_records'] == 1024, 'entire_raw_scan_completed')
check(raw['check_counts']['measurement_reconstruction'] == 13056 * 11, 'every_raw_measurement_checked')
check(raw['check_counts']['repeat_trajectory_consistency'] == 1024 * 4 * 7, 'every_repeat_trajectory_checked')
check(raw['source_manifest_sha256'] == hashlib.sha256(json.dumps(raw['source_file_identities'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'source_index_digest')
summary_path = TASK / 'completion_check_20261102/final_report/SUMMARY.json'
summary = matched_local(REMOTE + '/final_report/SUMMARY.json', summary_path)
canonical = matched_local(REMOTE + '/sealed/CANONICALS.json', TASK / 'production_002/sealed/CANONICALS.json')
check(summary['status'] == 'valid' and summary['errors'] == [], 'reported_valid_status')
check(summary['execution_source']['commit'] == raw['execution_commit'], 'execution_commit_identity')
for name in ('comparisons', 'method_totals', 'gain_harm_vs_F', 'repeat_totals'):
    same(raw[name], summary['analysis'][name], name)
for method, total in raw['method_totals'].items():
    derived = dict(calls_per_1000_output_tokens=1000*total['verification_calls']/total['output_tokens'],
        accepted_tokens_per_call=total['accepted_draft_tokens']/total['verification_calls'],
        committed_progress_per_call=total['committed_progress']/total['verification_calls'])
    same(derived, summary['analysis']['method_totals'][method], 'derived_' + method)
same(raw['schedule_balance'], summary['schedule_balance'], 'schedule')
check(len(raw['learning_curve']) == len(summary['learning_curve']) == 8, 'eight_fixed_curve_cells')
same(raw['learning_curve'], summary['learning_curve'], 'learning_curve')
same({k:v for k,v in raw['bootstrap'].items() if k not in ('unit', 'indices_sha256')}, summary['analysis']['bootstrap'], 'bootstrap_configuration')
lengths = [len(c['response_ids']) for c in canonical.values()]
length_summary = dict(requests=len(canonical), minimum_output_tokens=min(lengths), maximum_output_tokens=max(lengths),
    mean_output_tokens=sum(lengths)/len(lengths), cap_hits=sum(c['finish_reason']=='length' for c in canonical.values()),
    termination_counts=dict(Counter(c['finish_reason'] for c in canonical.values())))
for method in raw['method_totals']:
    same(length_summary, summary['lengths_and_termination'][method], 'lengths_and_termination_' + method)
check(sum(len(c['response_ids']) == 2048 for c in canonical.values()) == length_summary['cap_hits'], 'no_eos_at_cap_semantic_ambiguity')
report = {k:v for k,v in raw.items() if k not in ('source_file_identities', 'errors', 'traceback', 'status', 'total_checks', 'check_counts')}
report.update(status='passed', verification_status='ANALYZED', completed_at=datetime.now(timezone.utc).isoformat(),
    independent_raw_reconstruction=True, imported_campaign_or_report_modules=False, gpu_rerun=False,
    raw_scan_successful_checks=raw['total_checks']-1, local_comparison_successful_checks=sum(counts.values()),
    total_successful_checks=raw['total_checks']-1+sum(counts.values()), maximum_report_numeric_abs_difference=max_difference,
    errors=[], lengths_and_termination=length_summary,
    local_comparison_check_counts=dict(counts), remote_raw_check_counts=raw['check_counts'],
    audit_only_failure=dict(remote_exit_code=1, preserved_receipt=str(raw_path),
        receipt_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        error=raw['errors'][0], cause='Audit guard rejected its independently computed numpy.float64; all raw records and statistical arrays were already preserved.',
        resolution='Local comparison of JSON-normalized floats against hash-identical original SUMMARY.json; no remote/model rerun.'),
    code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (WORK/'independent_raw_statistics.py', Path(__file__))},
    limitations=['This is a separately implemented same-family, shared-context record/statistics review; the original report and source schemas were available to the reviewer.',
                 'No independent GPU reproduction, logits recomputation, re-decoding, or full tensor-state equality recheck was performed.',
                 'Weight/publication checks verify retained live-check receipts, not present-day native parameter equality.',
                 'Inference is conditional on seed 17, the fixed trained trajectories, and the previously used 1024-request benchmark.',
                 'The primary endpoint is K_HALF/O; secondary intervals are descriptive and unadjusted for multiplicity.',
                 'Three timing repeats were averaged within requests; they do not provide independent training seeds or independent bootstrap units.',
                 'Timing covers checkpoint-ready serving and includes required cache handling; it does not establish training-inclusive or asynchronous speedup.',
                 'Learning curves use first 128 requests and retained F counts; no matched intermediate O learning-rate comparison is available.'])
DEST.mkdir(parents=True, exist_ok=True)
(DEST/'RAW_STATISTICS_REVIEW.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(status=report['status'], successful_checks=report['total_successful_checks'],
                     maximum_numeric_difference=max_difference, source_file_count=raw['audited_file_count'],
                     native_blocks=raw['audit_totals'], output=str(DEST/'RAW_STATISTICS_REVIEW.json')), indent=2))
