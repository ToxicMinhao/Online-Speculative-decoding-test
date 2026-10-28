"""Assemble the fixed-panel report from retained, independently audited records."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT/'raw'


def read(path):
    return json.loads(path.read_text())


def build():
    evaluation = RAW/'campaign_001/evaluation'
    summary = read(evaluation/'report/SUMMARY.json')
    execution = read(evaluation/'STATUS.json')
    audit = read(ROOT/'FINAL_AUDIT.json')
    comparison = summary['comparisons']['KRC_vs_K1']
    costs = read(RAW/'campaign_001/COSTS.json')
    selection = read(ROOT/'inventory_audit/SELECTION_AUDIT.json')
    repeats = list(csv.DictReader((evaluation/'report/REPEAT_TOTALS.csv').open()))
    paired = []
    for repeat in range(3):
        times = {row['method']:float(row['request_latency_s']) for row in repeats if int(row['repeat']) == repeat}
        paired.append(dict(repeat=repeat, **times, reduction=100*(1-times['KRC']/times['K1'])))
    direction = comparison['call_request_directions']
    starts = (RAW/'STARTED.utc').read_text().strip()
    ends = (RAW/'FINISHED.utc').read_text().strip()
    from datetime import datetime
    wall = (datetime.fromisoformat(ends)-datetime.fromisoformat(starts)).total_seconds()
    reduction = comparison['call_reduction_pct']
    time_reduction = comparison['request_time_reduction_pct']
    if reduction > 0 and time_reduction > 0 and all(row['reduction'] > 0 for row in paired):
        answer = 'KRC has a small measured drafting and request-time advantage on this additional panel for these fixed checkpoints.'
    elif reduction > 0:
        answer = 'KRC uses fewer verification calls on this additional panel, but timing evidence is mixed; a consistent serving-time benefit is not established.'
    elif reduction == 0:
        answer = 'KRC has no aggregate verification-call advantage on this additional panel.'
    else:
        answer = 'KRC uses more verification calls on this additional panel. The development-panel drafting advantage did not extend to this panel.'
    text = ['# K1 versus KRC: additional-request confirmation', '',
        '**'+answer+'**', '',
        f"Exactly {execution['returned_generations']}/384 generations completed; {summary['valid_requests']}/64 requests are valid. "
        f"Independent audit: {audit['status']}. No training, new feedback, corrections, KL evaluation, local probes or new checkpoints were performed.", '',
        '| Metric | Retained K1 | Retained KRC |', '|---|---:|---:|']
    for label,key,fmt in [('Verification calls','verification_calls','d'),
        ('Accepted draft tokens','accepted_draft_tokens','d'),('Output tokens','output_tokens','d'),
        ('Committed progress / call','committed_progress_per_call','.6f'),
        ('Calls / 1,000 output tokens','calls_per_1000_output_tokens','.6f'),
        ('Sum of per-request mean latency (s)','summed_mean_request_latency_s','.9f')]:
        text.append(f"| {label} | {summary['K1'][key]:{fmt}} | {summary['KRC'][key]:{fmt}} |")
    text += ['', f"Primary pooled call reduction: **{reduction:+.6f}%**, computed as `1 − ΣKRC/ΣK1`. "
        f"Requests improved/regressed/tied: **{direction['improved']}/{direction['regressed']}/{direction['tied']}**. "
        f"Gross calls saved/added: **{comparison['call_gross_saved']}/{comparison['call_gross_extra']}**; net saved: **{comparison['call_net_saved']}**.", '',
        f"Request-time reduction from summed three-repeat request means: **{time_reduction:+.6f}%**. "
        'Counts use one validated repeat-0 trajectory per request; timing repeats are not extra request samples.', '',
        '| Repeat | K1 request time (s) | KRC request time (s) | Time reduction |', '|---:|---:|---:|---:|']
    for row in paired:
        text.append(f"| {row['repeat']} | {row['K1']:.9f} | {row['KRC']:.9f} | {row['reduction']:+.6f}% |")
    loo = summary['leave_one_out']
    text += ['', f"Leave-one-request-out call reduction ranges **{loo['minimum_call_reduction_pct']:+.6f}% to "
        f"{loo['maximum_call_reduction_pct']:+.6f}%**. {loo['nonpositive_call_reduction_omissions']}/64 omissions produce a nonpositive reduction. "
        'This is descriptive sensitivity only: **all 64 requests remain in the primary result**. All omission-level count and timing results are retained.', '',
        f"Output lengths are {summary['K1']['output_length_min']}–{summary['K1']['output_length_max']} tokens (mean {summary['K1']['output_length_mean']:.6f}), identical across checkpoints. EOS/cap terminations: {summary['K1']['terminations']['eos']}/{summary['K1']['terminations']['length']} requests. Accepted tokens and committed progress are reported separately; committed progress excludes the initial target-generated token.", '',
        '## Inputs and serving contract', '',
        f"Seed **20261028** selected 64 requests from {selection['eligible_count']} eligible pinned GSM8K test questions. "
        'Current-study exclusions cover 512 feedback questions, the 64-request development panel, the 64-request O/K confirmation panel, and all recorded qualification/probe inputs. '
        'Source IDs and NFKC/whitespace-normalized questions were checked; test prompts were deduplicated. '
        'All 640 inherited prompt sequences and all 64 new prompt sequences were reconstructed exactly. '
        'Selection used no answers, generated lengths or performance. All 64 selected questions have disclosed older unrelated-project exposure; the panel is new to this study, not globally unused.', '',
        'Dataset revision: `740312add88f781978c0658806c59bc2815b9866`. Target/tokenizer: Qwen3-8B '
        '`b968826d9c46dd6066d109eabc6255188de91218`; released DFlash family: '
        '`9b41424b7109f9c5413454f481b09a82b85333f4`. Exact K1/KRC export identities and rendered token IDs were sealed before generation.', '',
        'Qualified SGLang/FA3 runtime, BF16, greedy decoding, block 16, 2,048 tokens or EOS, context 4,096, concurrency 1, capture disabled and established cache flushing/checkpoint refresh. '
        'The order alternates by `(request_ordinal + repeat) % 2`, with 32 requests starting each checkpoint in every repeat. '
        'No concurrent training or artifact transfers occurred during timed serving. Normal native initialization was measured separately; no generation warmup was added.', '',
        '## Correctness and reproducibility', '',
        'The first 8 requests supplied the included 48-execution correctness gate; remaining requests continued unchanged. '
        "Each request's first fresh K1 execution supplied canonical tokens and termination. KRC-first records were kept pending and validated once K1 was available. "
        'This verifies agreement under the qualified native verifier, not an independent target-only reference. '
        'Every returned record was preserved before validation; verification counts must match across repeats. '
        'Native checkpoint tensors were checked before/after, and file identities/source/runtime were rechecked by postflight.', '',
        f"Valid/invalid/pending returned generations: {execution['completed_valid_generations']}/{execution['invalid_generations']}/{execution['pending_canonical_generations']}. "
        f"Unknown/unattempted: {execution['unknown_generation_attempts']}/{execution['unattempted_generations']}. "
        f"Audit issues: {len(audit['issues'])}; reconstructed native blocks: {audit.get('evaluation',{}).get('reconstructed_native_blocks','unavailable')}.", '',
        '18 evaluator/driver CPU tests and 3 selection tests passed; the evaluator/driver checks were independently repeated. '
        'Three focused independent audit-accounting tests also passed. These are implementation checks, separate from the included native gate.', '',
        '## Costs', '',
        '| Cost | Seconds |', '|---|---:|',
        f"| Backend loading | {costs['backend_loading_s']:.6f} |",
        f"| Checkpoint publications ({execution['activation_count']}) | {execution['activation_s']:.6f} |",
        f"| Native before/after weight validation | {execution['weight_validation_s']:.6f} |",
        f"| All 384 measured request latencies | {sum(summary[a]['all_repeats_request_latency_s'] for a in ('K1','KRC')):.6f} |",
        f"| Evaluation controller wall | {execution['evaluation_wall_s']:.6f} |",
        f"| Campaign process wall | {costs['campaign_wall_s']:.6f} |",
        f"| Supervisor wall including pre/postflight and CPU checks | {wall:.0f} |", '',
        f"Supervisor: {starts}–{ends}; exit {(RAW/'EXIT_CODE').read_text().strip()}. "
        'Timers are nested and must not be added again. Loading, checkpoint switching, checks and data retrieval are not included in serving latency. '
        'New training/correction cost is zero; historical preparation costs are not erased and no training-inclusive speedup is claimed.', '',
        '## Interpretation and deliverables', '',
        'This panel is reported separately from development. The test concerns these fixed checkpoints and additional requests. '
        'It does not establish training-seed or feedback-window robustness, isolate preservation from extra reference-response exposure, '
        'or establish a training-inclusive speedup. No statistical-significance claim is made. No algorithm change, new panel, retraining or further event follows automatically.', '',
        '- [Sealed request/checkpoint manifest](raw/sealed/PLAN.json)',
        '- [Exclusions and selection](inventory_audit/SELECTION_AUDIT.json)',
        '- [Per-request table](raw/campaign_001/evaluation/report/PER_REQUEST.csv)',
        '- [All 384 repeats](raw/campaign_001/evaluation/report/ALL_REPEATS.csv)',
        '- [Repeat totals](raw/campaign_001/evaluation/report/REPEAT_TOTALS.csv)',
        '- [Leave-one-out table](raw/campaign_001/evaluation/report/LEAVE_ONE_OUT.csv)',
        '- [Raw validity events](raw/campaign_001/evaluation/VALIDATION_EVENTS.jsonl)',
        '- [Independent audit](FINAL_AUDIT.md)',
        '- [Source review](REVIEW.md)', '',
        'ARS experiment-agent run/validate; separate same-team reviewers, not blinded external review. Stage 1 / project None unchanged.']
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
    return summary


if __name__=='__main__':
    build()
