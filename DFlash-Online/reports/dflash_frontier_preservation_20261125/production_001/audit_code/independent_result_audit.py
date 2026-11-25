"""Independent offline arithmetic; no exporter, experiment or model imports."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'
ARMS = ('CONTROL', 'FRONTIER_P')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNTS = ('output_tokens', 'verification_calls', 'accepted_draft_tokens',
          'native_accepted_draft_tokens', 'committed_progress')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b)


def main():
    retrieval, manifest = read(ROOT / 'RETRIEVAL.json'), read(ROOT / 'COMPACT_MANIFEST.json')
    assert retrieval['status'] == 'verified' and manifest['missing_required'] == []
    for row in manifest['files']:
        path = RAW / row['path']
        assert path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    assert len(manifest['files']) == retrieval['files'] == 1134
    assert sum(row['bytes'] for row in manifest['files']) == retrieval['bytes'] == 442061947
    assert sha(ROOT / 'compact_results.tar.gz') == retrieval['archive_sha256']
    rows = [json.loads(line) for line in (RAW / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
    assert len(rows) == 792 and all(r['valid_measurement'] is True and r['errors'] == [] for r in rows)
    assert Counter(r['phase'] for r in rows) == {'scored': 768, 'warmup': 24}
    groups = defaultdict(list)
    for row in rows:
        if row['phase'] == 'scored':
            groups[row['request_id'], row['method']].append(row)
    ids = sorted({key[0] for key in groups})
    assert len(ids) == 128 and len(groups) == 256
    values, totals = {}, {}
    for arm in ARMS:
        values[arm] = {name: [] for name in TIMES + COUNTS}
        for key in ids:
            sample = sorted(groups[key, arm], key=lambda r: r['repeat'])
            assert [r['repeat'] for r in sample] == [0, 1, 2]
            for metric in COUNTS:
                assert len({r[metric] for r in sample}) == 1
                values[arm][metric].append(sample[0][metric])
            for metric in TIMES:
                assert all(type(r[metric]) in (int, float) and math.isfinite(r[metric]) and r[metric] > 0 for r in sample)
                values[arm][metric].append(math.fsum(r[metric] for r in sample) / 3)
            assert all(r['finish_reason'] == 'eos' for r in sample)
        totals[arm] = {name: math.fsum(data) for name, data in values[arm].items()}
        assert totals[arm]['output_tokens'] == 50226
        assert totals[arm]['committed_progress'] == totals[arm]['output_tokens'] - 128
        assert totals[arm]['native_accepted_draft_tokens'] == totals[arm]['committed_progress'] - totals[arm]['verification_calls']
    saved = read(RAW / 'analysis/SERVING.json')
    for arm, metrics in totals.items():
        for name, value in metrics.items():
            close(value, saved['totals'][arm][name])
    draws = np.random.Generator(np.random.PCG64(20261125)).integers(0, 128, size=(10000, 128))
    draw_sha = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    assert draw_sha == saved['bootstrap']['draw_sha256']
    assert ids == saved['bootstrap']['input_order']
    contrasts = {}
    for metric, label in [(name, name + '_reduction_pct') for name in TIMES] + [('verification_calls', 'verification_call_reduction_pct')]:
        control, candidate = values['CONTROL'][metric], values['FRONTIER_P'][metric]
        # Python compensated summation is separate from the exporter's vectorized sum.
        distribution = [100 * (1 - math.fsum(candidate[i] for i in draw) / math.fsum(control[i] for i in draw)) for draw in draws]
        contrast = dict(value=100*(1-math.fsum(candidate)/math.fsum(control)), ci95=np.percentile(distribution, [2.5, 97.5], method='linear').tolist())
        expected = saved['contrasts']['FRONTIER_P_vs_CONTROL'][label]
        close(contrast['value'], expected['value'])
        for a, b in zip(contrast['ci95'], expected['ci95']):
            close(a, b)
        contrasts[label] = contrast
    differences = [a-b for a, b in zip(values['CONTROL']['verification_calls'], values['FRONTIER_P']['verification_calls'])]
    changes = dict(wins=sum(d>0 for d in differences), regressions=sum(d<0 for d in differences), ties=sum(d==0 for d in differences), gross_calls_saved=sum(max(d,0) for d in differences), gross_calls_added=sum(max(-d,0) for d in differences), net_calls_saved=sum(differences))
    for name, value in changes.items():
        assert saved['contrasts']['FRONTIER_P_vs_CONTROL'][name] == value
    repeat_effects = []
    for repeat in range(3):
        sums = {arm: math.fsum(row['request_latency_s'] for row in rows if row['phase']=='scored' and row['repeat']==repeat and row['method']==arm) for arm in ARMS}
        repeat_effects.append(100 * (1-sums['FRONTIER_P']/sums['CONTROL']))
    witness_saved = read(RAW / 'analysis/WITNESSES.json')
    witness = {endpoint: read(RAW / 'witness' / (endpoint+'.json')) for endpoint in ('parent', *ARMS)}
    witness_result = {}
    metric_names = ('raw_repair', 'useful_repair', 'damaged_prefix_positions', 'damaged_prefix_blocks', 'surviving_followers', 'reachable_followers', 'matching_prefix_length', 'fixed_K_shortfall')
    for name in ('new_buffer', 'parent_training'):
        result = {}
        for endpoint, receipt in witness.items():
            calculated = []
            for parent_response, response in zip(witness['parent']['sets'][name], receipt['sets'][name]):
                assert parent_response['record_id'] == response['record_id'] and parent_response['anchors'] == response['anchors']
                for original, block in zip(parent_response['blocks'], response['blocks']):
                    origin = original['origin']
                    assert block['origin'] == origin
                    valid = origin['valid']
                    q, y = block['proposals'], origin['labels']
                    prefix = 0
                    for j in range(1,16):
                        if not valid[j] or q[j] != y[j]:
                            break
                        prefix += 1
                    damage = sum(bool(origin['prefix'][j]) and q[j] != y[j] for j in range(1,16))
                    repair = origin['e'] >= 1 and q[origin['e']] == y[origin['e']]
                    followers = range(origin['e']+1,origin['K']+1) if origin['e'] >= 1 else []
                    metrics = dict(raw_repair=int(repair),useful_repair=int(repair and not damage),damaged_prefix_positions=damage,damaged_prefix_blocks=int(damage>0),surviving_followers=sum(q[j]==y[j] for j in followers),reachable_followers=sum(j<=prefix for j in followers),matching_prefix_length=prefix,fixed_K_shortfall=max(0,origin['K']-prefix))
                    for key, value in metrics.items():
                        assert block['metrics'][key] == value
                    # Assessment values remain native FP32 readouts; check their real-valued formula independently within rounding.
                    assessment = math.fsum((origin['K']-j+1) * float(np.logaddexp(0., -point['margin'])) / math.log(2) for j,point in enumerate(block['positions'],1))
                    assert math.isclose(assessment, block['metrics']['assessment'], rel_tol=5e-6, abs_tol=5e-5)
                    metrics['assessment'] = block['metrics']['assessment']
                    calculated.append(metrics)
            assert len(calculated) == 64
            result[endpoint] = {key: math.fsum(r[key] for r in calculated) for key in metric_names + ('assessment',)}
            result[endpoint]['response_mean_assessment'] = math.fsum(r['response_assessment'] for r in receipt['sets'][name])/8
            for key in metric_names + ('assessment',):
                if endpoint != 'parent':
                    close(result[endpoint][key], witness_saved['sets'][name]['parent_to_'+endpoint][key]['second_sum'])
            close(result[endpoint]['response_mean_assessment'], witness_saved['sets'][name]['endpoints'][endpoint]['response_mean_assessment'])
        witness_result[name] = dict(responses=8, blocks=64, original_errors=witness_saved['sets'][name]['original_errors'], original_prefix_positions=witness_saved['sets'][name]['original_prefix_positions'], endpoints=result)
    exporter = read(ROOT / 'export_001/PRIVATE_AUDIT.json')
    assert exporter['status']=='passed' and exporter['raw_generations_checked']==792 and exporter['native_blocks_checked']==47640
    result = dict(status='passed', verification_status='ANALYZED', scope='Independent local raw-file authentication and scalar arithmetic; no model rerun. No exporter or scientific module imports.', execution_commit=manifest['execution_commit'], extra_model_work=0,
                  evidence=dict(files_verified=1134, bytes_verified=442061947, archive_sha256=retrieval['archive_sha256'], manifest_sha256=sha(ROOT/'COMPACT_MANIFEST.json'), measurements_sha256=sha(RAW/'evaluation/MEASUREMENTS.jsonl'), source_audit_script_sha256=sha(Path(__file__)), exporter_raw_generations=792, exporter_native_blocks=47640),
                  serving=dict(requests=128, repeats_per_request=3, scored_generations=768, separate_warmups=24, totals=totals, contrasts=contrasts, call_changes=changes, repeat_latency_reduction_pct=repeat_effects, bootstrap=dict(draws=10000,seed=20261125,generator='PCG64',draw_sha256=draw_sha, unit='paired whole request after averaging its three timing repeats'), conclusion='completed_but_efficacy_unresolved'),
                  witnesses=witness_result,
                  limitations=['One warm-parent fork and one training seed, with a reused development panel.', 'The request bootstrap conditions on fixed endpoints; three timing repeats are not independent trained models or 384 independent requests.', 'Witness sets contain overlapping training blocks, not independent held-out evidence.', 'Stored margins permit formula checks, but strongest-wrong vocabulary rivals cannot be reprojected without model work.', 'Production selected-update actual/shadow parameter displacement norms were not recorded; integration BETA displacement is not a production substitute.', 'No equivalence margin was declared; intervals crossing zero do not establish equal performance.', 'Secondary intervals are unadjusted; a cache-maintenance interval does not establish complete-request benefit.'])
    result['fallacy_scan'] = dict(checked=11, findings=[
        dict(name='Simpson paradox', disposition='No demonstrated paradox; repeat signs vary around zero. Keep new-buffer and parent-training witness sets separate because assessment changes have opposite directions.'),
        dict(name='Ecological fallacy', disposition='Request-level bootstrap and wins are distinct from aggregate ratios. Do not infer universal request benefit or independent sample size from overlapping blocks.'),
        dict(name='Berkson selection', disposition='Fixed reused first-128 panel and first-eight training witnesses limit representativeness. No outcome-conditioned filtering is applied.'),
        dict(name='Collider bias', disposition='Panel, witnesses and common feedback were fixed before branch outcomes. Do not select only repaired or faster requests for the primary comparison.'),
        dict(name='Base-rate neglect', disposition='Witness denominators retained: 44/31 original-error blocks and 488/676 original-prefix positions across new/parent sets.'),
        dict(name='Regression to mean', disposition='Compare contemporaneous CONTROL and candidate from the same warm parent; before/after fitting alone does not show the candidate adds benefit.'),
        dict(name='Survivorship bias', disposition='All 792 planned generations retained and valid; zero excluded invalids. Historical failed qualification remains a failed diagnostic.'),
        dict(name='Look-elsewhere effect', disposition='Primary latency interval crosses zero. Secondary intervals are unadjusted; cache-maintenance change and favorable witness metrics cannot replace the primary outcome.'),
        dict(name='Garden of forking paths', disposition='Prespecified counts, endpoints, seed and paired estimator retained. Disclose the prospective numerical-contract revision and unchanged earlier gate failure.'),
        dict(name='Correlation versus causation', disposition='Controlled fork supports this fixed-run contrast, not general mechanism efficacy. Witness changes and derivative fidelity are not independent proof of serving benefit.'),
        dict(name='Reverse causality', disposition='Training precedes serving and witness origins are frozen; no reverse-time selection route found. Post-update margin associations do not identify a causal mediator.'),
    ])
    (ROOT/'INDEPENDENT_RESULT_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], primary=contrasts['request_latency_s_reduction_pct'], calls=changes, repeat_latency_reduction_pct=repeat_effects, witness=witness_result),indent=2))


if __name__ == '__main__':
    main()
