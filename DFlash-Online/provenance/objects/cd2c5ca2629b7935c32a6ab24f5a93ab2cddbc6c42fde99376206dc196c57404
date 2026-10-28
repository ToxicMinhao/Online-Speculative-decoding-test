"""Summarize the fixed KR comparison without pooling its two request panels."""
import csv
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).parent


def read(path):
    return json.loads(Path(path).read_text())


def make_report():
    raw = ROOT/'raw/campaign_001'
    training = read(ROOT/'analysis/TRAINING_SUMMARY.json')
    costs = read(raw/'COSTS.json')
    status = read(raw/'STATUS.json')
    panels = {name: read(raw/'evaluation'/name/'report/SUMMARY.json')
              for name in ('development','additional')}
    historical = read(ROOT.parent/'dflash_krc_event_20261027/raw/campaign_001/COSTS.json')
    historical_diagnostics = sum(historical['correction']['costs'][key] for key in
        ('post_c_reference_s', 'post_k_learning_s', 'post_c_learning_s'))
    audit = read(ROOT/'FINAL_AUDIT.json') if (ROOT/'FINAL_AUDIT.json').exists() else {'status':'pending'}
    results = {name: value['comparisons']['KR_vs_KRC'] for name,value in panels.items()}
    if all(v['call_reduction_pct'] > 0 for v in results.values()):
        opening = 'KR used fewer verification calls than retained KRC on both fixed panels.'
    elif all(v['call_reduction_pct'] < 0 for v in results.values()):
        opening = 'KR used more verification calls than retained KRC on both fixed panels.'
    else:
        opening = 'KR versus retained KRC produced mixed or tied drafting results across the two fixed panels.'
    lines = ['# KR: ordinary reference replay versus retained KRC', '', '**'+opening+'**', '',
        f"Completed training: **{training['presentations']}/512 presentations, {training['optimizer_steps']}/256 AdamW steps**. "
        f"Native execution: **{status['evaluation_totals']['completed_valid_generations']}/768 valid generations**. "
        f"Independent audit: **{audit.get('status','pending')}**.", '',
        'KR starts from released DFlash with fresh seed-17 AdamW and alternates ordinary K updates on each original A pair and its same-pass next-pair R. '
        'Both updates advance the same optimizer and training RNG. The response-level K normalization, current first-error selection, BF16 path, '
        'constant learning rate 3e-4, clipping and model/runtime pins are unchanged. There is no hinge, protected-prefix capture or direct correction.', '',
        'This is a practical replay control: KR has twice the Adam steps and twice the ordinary K presentations of KRC. '
        'The comparison changes optimizer-state evolution and training exposure; it does not isolate correction direction or match computation exactly.', '',
        '| Panel | KRC calls | KR calls | KR call reduction | Requests improved / regressed / tied | Gross calls saved / added |',
        '|---|---:|---:|---:|---:|---:|']
    for name,p in panels.items():
        c=results[name];d=c['call_request_directions']
        lines.append(f"| {name} | {p['KRC']['verification_calls']} | {p['KR']['verification_calls']} | {c['call_reduction_pct']:+.6f}% | "
                     f"{d['improved']} / {d['regressed']} / {d['tied']} | {c['call_gross_saved']} / {c['call_gross_extra']} |")
    lines += ['', 'Primary call reduction is `1 − ΣKR_calls / ΣKRC_calls`, with one validated trajectory per request after repeat-consistency checks. '
        'Every request and regression remains included. The two panels are reported separately.', '',
        '| Panel | KRC complete-request time (s) | KR complete-request time (s) | Reduction |',
        '|---|---:|---:|---:|']
    for name,p in panels.items():
        lines.append(f"| {name} | {p['KRC']['summed_mean_request_latency_s']:.6f} | {p['KR']['summed_mean_request_latency_s']:.6f} | {results[name]['request_time_reduction_pct']:+.6f}% |")
    lines += ['', 'Times above sum each request’s three-repeat mean. These are contemporaneous KR/KRC measurements with available checkpoints, '
        'after training closed. Native generation and cache maintenance below are components of complete-request latency; they must not be added again.', '',
        '| Panel | Arm | Native-generation time (sum of means, s) | Cache-maintenance time (sum of means, s) | Accepted draft tokens | Progress / call |',
        '|---|---|---:|---:|---:|---:|']
    for name,p in panels.items():
        for arm in ('KRC','KR'):
            v=p[arm]
            lines.append(f"| {name} | {arm} | {v['summed_mean_native_generate_s']:.6f} | {v['summed_mean_cache_maintenance_s']:.6f} | {v['accepted_draft_tokens']} | {v['committed_progress']/v['verification_calls']:.6f} |")
    lines += ['', '| Panel | Repeat | KRC request time (s) | KR request time (s) | Reduction |', '|---|---:|---:|---:|---:|']
    activation = {}
    for name in panels:
        with (raw/'evaluation'/name/'report/REPEAT_TOTALS.csv').open() as f: repeat_rows=list(csv.DictReader(f))
        for repeat in range(3):
            rows={r['method']:r for r in repeat_rows if int(r['repeat'])==repeat}
            base=float(rows['KRC']['request_latency_s']);cand=float(rows['KR']['request_latency_s'])
            lines.append(f'| {name} | {repeat} | {base:.6f} | {cand:.6f} | {100*(1-cand/base):+.6f}% |')
        acts=[json.loads(x) for x in (raw/'evaluation'/name/'ACTIVATIONS.jsonl').read_text().splitlines()]
        activation[name]=dict(publications=len(acts),activation_s=sum(r['activation_s'] for r in acts))
    lines += ['', '## Training exposure, correctness and cost', '',
        f"The sealed schedule contains **{training['sampled_anchor_occurrences']:,} sampled-anchor occurrences**. Each of the 128 responses appears twice as A and twice as R, "
        'using its own original per-pass anchors. R is training exposure, not held-out data. No evaluation request enters training.', '',
        'The first A update exactly matched retained ordinary K training records (loss/exposure metrics, anchors, learning rate and counters). '
        'Fresh released model/Adam/RNG state was checked separately. This does not claim historical post-update tensor equality: that first-step export was not retained. '
        'The first A/R window used four actual presentations and two Adam steps, with no duplicate qualification work.', '',
        '| KR cost | Seconds | Scope |', '|---|---:|---|',
        f"| A updates | {training['roles']['A']['update_wall_s']:.3f} | 128 ordinary updates; includes feature loading and bookkeeping |",
        f"| R updates | {training['roles']['R']['update_wall_s']:.3f} | 128 ordinary updates with continuous Adam/RNG |",
        f"| Full KR event | {costs['event_wall_s']:.3f} | Training, checks, persistence and export |",
        f"| Checkpoint persistence | {costs['event']['checkpoint_save_s']:.3f} | Nested in event |",
        f"| Final export | {costs['event']['export_s']:.3f} | Nested in event |",
        f"| Disposable startup | {costs['startup']['wall_s']:.3f} | 2 forwards/backwards, 0 Adam steps |",
        f"| Fresh initialization | {costs['loading_and_initialization_s']:.3f} | Before main event |",
        f"| Native backend loading | {costs['backend_loading_s']:.3f} | Benchmark setup |",
        f"| KRC full event, historical | {historical['event_wall_s']:.3f} | Prior measurement, not fresh matched training time |",
        f"| Campaign process wall | {costs['actual_campaign_wall_s']:.3f} | Includes benchmarks and switching; excludes supervisor pre/postflight |", '',
        f"Peak KR allocated/reserved GPU memory: **{costs['peak_allocated_bytes']/1e9:.3f}/{costs['peak_reserved_bytes']/1e9:.3f} GB**. "
        f"Historical KRC peaks were **{historical['peak_allocated_bytes']/1e9:.3f}/{historical['peak_reserved_bytes']/1e9:.3f} GB**. "
        'All recorded training head/layer invocations, including recomputation, remain in TRAINING_SUMMARY.json; recomputation is not counted as another response presentation.', '',
        f"The historical KRC event includes {historical_diagnostics:.3f} s of 18 diagnostic forwards; excluding those measured forwards gives "
        f"{historical['event_wall_s']-historical_diagnostics:.3f} s. KR has no corresponding local probe cost. "
        'These are historical training-cost observations, not fresh paired timing or evidence of equal compute.', '',
        'Component costs are nested, so do not add them to the event total again. Feedback collection and teacher preparation are reused from retained artifacts; '
        'there is no new cost for those phases here, but their historical cost is not erased. No fresh KRC training run was performed.', '',
        '| Panel | Checkpoint publications | Publication time (s) | Native weight validation (s) | Evaluation wall (s) |',
        '|---|---:|---:|---:|---:|']
    for name,p in panels.items():
        e=p['execution'];a=activation[name]
        lines.append(f"| {name} | {a['publications']} | {a['activation_s']:.3f} | {e['weight_validation_s']:.3f} | {e['evaluation_wall_s']:.3f} |")
    lines += ['', 'Publication is benchmark switching overhead, not a charge once per deployed request. All activations are retained. '
        'The first eight development requests supplied the included 48-generation correctness gate; no extra generation warmups or optimizer qualifications were added. '
        'The supervisor ran from 2026-10-28 00:42:52 to 01:20:45 UTC (2,273 seconds), including pre/postflight, and exited 0.', '',
        '## Validity and interpretation', '', '| Panel | Valid / invalid / unknown / unattempted executions | Canonical output tokens | EOS / cap |',
        '|---|---|---:|---:|']
    for name,p in panels.items():
        e=p['execution'];o=p['canonical_outputs']
        lines.append(f"| {name} | {e['completed_valid_generations']} / {e['invalid_generations']} / {e['unknown_generation_attempts']} / {e['unattempted_generations']} | "
                     f"{o['output_tokens']} | {o['eos_requests']} / {o['output_cap_hits']} |")
    lines += ['', 'Development outputs are checked against retained F; additional-panel outputs are checked against the retained fresh K1 canonicals from the seed20261028 confirmation. '
        'Both panels reproduce their retained per-request KRC counts and require repeat-consistent counts for both methods. The additional panel has permitted, disclosed older-project exposure. '
        'Neither panel is newly selected for this test.', '',
        'Historical K1 call totals are 3,027 on development and 3,150 on the additional panel; they are context only, with no fresh K1 latency claim.', '',
        '**This ordinary replay recipe did not replace KRC successfully.** KR requires more verification calls and more serving time on both panels; '
        'all six panel-repeat timing comparisons favor KRC. KR uses less peak training memory, but shows no event-time saving against the historical KRC measurement. '
        'KRC therefore gains support against this specific replay control. This does not isolate the correction’s causal role: optimizer-step count, '
        'optimizer-state evolution and training exposure also change.', '',
        'The supported conclusion concerns these fixed feedback responses, checkpoints and two panels. It does not establish training-seed/window robustness, '
        'training-inclusive acceleration, overfitting as the cause of the regression, or a general advantage of correction over ordinary replay. '
        'All results stop at this prespecified comparison; no tuning, extra arms or subsequent event was launched.', '',
        '## Evidence', '',
        '- `IMPLEMENTATION.diff`, `REVIEW.md`, `CPU_TESTS.log`: additive code and independent review; 23 focused CPU tests (independent rerun overlaps).',
        '- `raw/sealed/PLAN.json`, `REPLAY_SCHEDULE.json`, `REFERENCE_SCHEDULE.json`, `SEAL.json`: fixed sources, checkpoint identities, panels and bindings.',
        '- `analysis/TRAINING_SUMMARY.json`, `PRESENTATIONS.csv`, `UPDATES.csv`, `ROLE_EXPOSURE_COSTS.csv`, `RESPONSE_EXPOSURE.csv`: all training exposure/cost accounting.',
        '- `analysis/development/` and `analysis/additional/`: separate per-request, all-repeat and repeat-total tables; activations remain in each raw evaluation directory.',
        '- `raw/campaign_001/evaluation/`: every raw native output, trace and validity record.',
        '- `FINAL_AUDIT.md`, `RETRIEVAL_VERIFIED.json`, `COMPLETION.json`: independent audit, byte-verified retrieval and final status.', '',
        'Final KR export: `/data/sources/minhao/runs/dflash_reference_replay_20261028/campaign_001/KR/events/event_001/export`. '
        'Recovery state: `/data/sources/minhao/runs/dflash_reference_replay_20261028/campaign_001/KR/training_state.pt`. '
        'All existing checkpoints and evidence remain retained.', '',
        'Material Passport: computational experiment; ARS executor/monitor and independent audit; retained private requests/outputs; no stage advancement.', '']
    for name in panels:
        dest=ROOT/'analysis'/name;dest.mkdir(exist_ok=True)
        for p in (raw/'evaluation'/name/'report').iterdir():
            if p.is_file():shutil.copy2(p,dest/p.name)
    summary=dict(status=status['status'],training=training,panels=panels,activation=activation,
        audit_status=audit.get('status','pending'),interpretation=opening,historical_KRC_event_s=historical['event_wall_s'])
    (ROOT/'analysis/SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    make_report()
