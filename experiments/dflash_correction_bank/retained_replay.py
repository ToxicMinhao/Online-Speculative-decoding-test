"""Exactly five retained singleton cases, ten generations, no learning/campaign."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.dflash_correction_bank import native
from experiments.dflash_correction_bank.artifacts import atomic_json, binding, checked, load_snapshot
from experiments.dflash_correction_bank.retained_hooks import selected_matrix

base = native.base
CASES = (('wildchat_00', 1, 0), ('wildchat_01', 2, 1), ('wildchat_02', 6, 3),
         ('wildchat_19', 15, 2), ('personamem_persona42', 3, 2))
INTERPRETATION = ('This is a same-response hindsight diagnostic: the correction has already learned from the '
    'answer being replayed. Success demonstrates native usefulness on these fitted examples, not transfer to '
    'subsequent requests. Failure supports deferring the current learned corrections, but does not prove '
    'that every correction within this space would fail.')


def storage_preflight(out):
    """No models, optimizer states or full feature tensors are exported/copied."""
    location = Path(out).resolve().parent
    while not location.exists():
        location = location.parent
    plan = dict(schema='retained_replay_storage_v1', native_generations=10, maximum_feature_reconstructions=5,
        compact_outputs_and_diagnostics_bytes=2*2**30, runtime_scratch_and_source_staging_bytes=2**30,
        reserve_bytes=8*2**30, model_exports=0, optimizer_states=0, historical_artifacts_copied=False,
        reconstruction_features='Transient memory only; retained metadata provides exact reconstruction.')
    plan['required_free_bytes'] = 11*2**30
    plan['free_before_bytes'] = shutil.disk_usage(location).free
    base.require(plan['free_before_bytes'] >= plan['required_free_bytes'], 'Retained replay storage allowance plus reserve exceeds free space')
    return plan


def prepare_cases(campaign_root):
    """Read main-campaign evidence only; pin source before starting a worker."""
    import numpy as np
    root = Path(campaign_root).resolve()
    cases = []
    for stream, turn, module in CASES:
        directory = root/'campaign'/stream
        episode = directory/'native'/stream
        bank = episode/'training_worker'/'bank'
        case = dict(case_id=f'{stream}:t{turn}:m{module}', stream=stream, turn=turn, module=module,
            after_snapshot=binding(bank/f'after_turn_{turn:02d}.json'),
            feedback=binding(bank/f'feedback_{turn:02d}.json'), projections=binding(bank/'fixed_projections.npz'),
            source_native_config=binding(episode/'NATIVE_CONFIG.json'), source_result=binding(directory/'RESULT.json'),
            solver=binding(bank/f'solver_{turn:02d}.json'))
        feedback = load_snapshot(case['feedback']); response = feedback['response']
        base.require(response['episode_id'] == stream and response['turn'] == turn and response['arm'] == 'BANK',
                     'Wrong main-campaign source feedback')
        case.update(input_ids=response['input_ids'], output_ids=response['output_ids'],
            input_ids_sha256=base.object_hash(response['input_ids']), output_ids_sha256=base.object_hash(response['output_ids']))
        base.require(case['input_ids_sha256'] == response['input_ids_sha256'] and
                     case['output_ids_sha256'] == base.object_hash(response['output_ids']), 'Source token hashes differ')
        result = json.loads(checked(case['source_result']).read_text())
        base.require(result['status'] == 'complete' and result['qualification'] is False, 'Use completed main campaign, not qualification')
        canonical = next(r for r in result['canonical'] if r['arm'] == 'BANK' and r['turn'] == turn)
        base.require(canonical['input_ids'] == case['input_ids'] and canonical['output_ids'] == case['output_ids'],
                     'Saved feedback differs from canonical history')
        update = next(r for r in result['updates'] if r['arm'] == 'BANK' and r['event'] == turn)
        base.require(update['bank_snapshot'] == case['after_snapshot'] and update['scheduled_steps'] == 2 and
                     update['owner'] == module, 'Use final state after both scheduled steps for the specified owner')
        scored = [r for r in result['scores'] if r['arm'] == 'F' and r['turn'] == turn]
        base.require(len(scored) == 3 and all(r['same_generation_valid'] and r['input_ids'] == case['input_ids'] and
                     r['output_ids'] == case['output_ids'] for r in scored), 'Retained frozen baseline is not exact/valid')
        metrics = ('verification_calls', 'accepted_draft_tokens', 'verification_committed_tokens')
        base.require(all(len({r[k] for r in scored}) == 1 for k in metrics), 'Retained frozen counts disagree across repeats')
        case['frozen_metrics'] = {k: scored[0][k] for k in metrics}
        case['scheduled_steps'], case['accepted_steps'] = 2, update['accepted_steps']
        case['recorded_source_gain'] = update['singleton_fit_gain']
        matrix = selected_matrix(case)
        case['matrix_sha256'] = hashlib.sha256(matrix.tobytes()).hexdigest()
        case['matrix_norm'] = float(np.linalg.norm(matrix))
        case['sampling_and_weights_sha256'] = feedback['summary']['metadata']['sampling_and_weights_sha256']
        case['supervised_positions'] = feedback['summary']['metadata']['eligible_positions']
        del result, feedback
        cases.append(case)
    return cases


def matching_precision(case, campaign_root):
    """Qualification evidence is reusable only after exact case identity checks."""
    import numpy as np
    if case['stream'] != 'wildchat_00':
        return None
    root = Path(campaign_root)/'qualification'/'wildchat_00'
    bank = root/'native'/'wildchat_00'/'training_worker'/'bank'
    try:
        prior = dict(case, after_snapshot=binding(bank/f'after_turn_{case["turn"]:02d}.json'))
        old_matrix, matrix = selected_matrix(prior), selected_matrix(case)
        if not np.array_equal(old_matrix, matrix):
            return None
        old_feedback_bound = binding(bank/f'feedback_{case["turn"]:02d}.json')
        old_feedback, feedback = load_snapshot(old_feedback_bound), load_snapshot(case['feedback'])
        if any(old_feedback['response'][k] != feedback['response'][k] for k in
               ('input_ids', 'output_ids', 'target_identity_sha256', 'checkpoint_identity_sha256')):
            return None
        if old_feedback['summary']['metadata'] != feedback['summary']['metadata']:
            return None
        if any(not np.array_equal(old_feedback['summary'][k], feedback['summary'][k]) for k in
               ('omega', 'phi', 'G', 'S', 'T')) or not np.array_equal(old_feedback['M'], feedback['M']):
            return None
        projections = binding(bank/'fixed_projections.npz')
        with np.load(checked(projections), allow_pickle=False) as old, np.load(checked(case['projections']), allow_pickle=False) as new:
            if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
                return None
        result_bound = binding(root/'RESULT.json')
        result = json.loads(checked(result_bound).read_text())
        feedback_row = next(r for r in result['feedback_summaries'] if r['event'] == case['turn'])
        precision = feedback_row['qualification_checks']['native_precision']
        chosen = [c for c in precision['cases'] if c['label'] in ('zero', 'fitted')]
        base.require(len(chosen) == 2 and precision['status'] == 'measured', 'Missing matching fitted/zero precision')
        return dict(status='measured', method='exact_matching_qualification_evidence',
            precision={**precision, 'cases': chosen}, reconstruction={'teacher_reconstructions': 0, 'draft_feature_forwards': 0},
            provenance=dict(result=result_bound, after_snapshot=prior['after_snapshot'], feedback=old_feedback_bound,
                            projections=projections, matched_matrix_source_tokens_objective_and_projection_arrays=True))
    except (FileNotFoundError, KeyError, ValueError, RuntimeError):
        return None


def trace_offsets(bound):
    """Expose already recorded offsets only, with their original field names."""
    document = json.loads(checked(bound).read_text())
    found = []
    def visit(value, path=''):
        if isinstance(value, dict):
            for key, item in value.items():
                location = path+'/'+str(key)
                if 'reject' in key.lower() and any(part in key.lower() for part in ('offset', 'position', 'index')):
                    found.append(dict(path=location, value=item))
                elif isinstance(item, (dict, list)):
                    visit(item, location)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    visit(item, path+'/'+str(i))
    visit(document)
    return dict(status='available' if found else 'not_exposed_by_existing_fields', fields=found)


def summarize(case, rows, precision):
    f, c = rows['F'], rows['BANK']
    valid = all(r['output_ids'] == case['output_ids'] and r['input_ids'] == case['input_ids'] for r in rows.values())
    baseline = f['verification_calls'] == case['frozen_metrics']['verification_calls']
    read = c.get('read') or {}
    alpha = [0.]*4; alpha[case['module']] = 1.
    applied = (read.get('routing_bypassed') is True and read.get('fixed_for_entire_request') is True and
               read.get('matrix_sha256') == case['matrix_sha256'] and read.get('coefficients') == alpha and
               read.get('corrected_head_calls', 0) > 0)
    measured = precision.get('precision') or {}
    selected = next((r for r in measured.get('cases', []) if r['label'] in ('selected', 'fitted')), {})
    return dict(case_id=case['case_id'], stream=case['stream'], turn=case['turn'], module=case['module'],
        exact_outputs_and_inputs=valid, frozen_count_reproduced=baseline, fixed_singleton_applied=applied,
        gain_comparison_valid=valid and baseline and applied,
        frozen_calls=f['verification_calls'], singleton_calls=c['verification_calls'],
        calls_saved=f['verification_calls']-c['verification_calls'] if valid and baseline and applied else None,
        frozen_accepted=f['accepted_draft_tokens'], singleton_accepted=c['accepted_draft_tokens'],
        frozen_progress=f['verification_committed_tokens']/f['verification_calls'],
        singleton_progress=c['verification_committed_tokens']/c['verification_calls'],
        output_tokens=len(case['output_ids']), precision_status=precision.get('status'),
        reference_loss_change=selected.get('reference_loss_change'), native_loss_change=selected.get('native_loss_change'),
        precision_details=selected, timing_scope='One routing-bypassed diagnostic execution; not deployment speedup.')


def write_report(out, cases, results, precision):
    rows = [summarize(case, results[case['case_id']], precision[case['case_id']]) for case in cases]
    atomic_json(out/'COMPARISON.json', dict(rows=rows, interpretation=INTERPRETATION,
        native_generations=sum(len(v) for v in results.values()),
        precision_distinction='Surviving BF16 rounding does not itself establish improved acceptance or fewer verification calls.'))
    lines = ['# Five-case retained-correction replay', '', INTERPRETATION, '',
        'Frozen → singleton; progress means committed tokens per verification. Loss changes are versus each precision path’s zero-correction baseline.', '',
        '| Case (turn/module) | Verification calls | Accepted draft tokens | Progress/verification | Valid | Δloss reference / native | BF16 changed positions |',
        '|---|---:|---:|---:|---|---:|---:|']
    for row in rows:
        detail = row['precision_details']
        loss = ('unavailable' if row['reference_loss_change'] is None else
                f'{row["reference_loss_change"]:+.6g} / {row["native_loss_change"]:+.6g}')
        positions = ('unavailable' if not detail else f'{detail["changed_positions"]}/{detail["anchor_count"]}')
        lines.append(f'| {row["stream"]} ({row["turn"]}/{row["module"]}) | {row["frozen_calls"]} → {row["singleton_calls"]} | '
                     f'{row["frozen_accepted"]} → {row["singleton_accepted"]} | {row["frozen_progress"]:.3f} → {row["singleton_progress"]:.3f} | '
                     f'{"yes" if row["gain_comparison_valid"] else "NO—see raw discrepancy"} | {loss} | {positions} |')
    lines += ['', '**Precision evidence is separate:** surviving BF16 rounding does not itself establish improved acceptance or fewer verification calls.', '',
        'Exactly five fixed cases, one frozen and one singleton generation each. No training or state updates. The singleton uses the unmodified final after-turn matrix at coefficient 1 with routing bypassed. Full input histories, greedy decoding, block10 and 1,024-or-EOS remain pinned. Timings are retained as diagnostic measurements and are not deployment-speedup claims.', '',
        'All raw generation records, original baseline counts, identity checks, available rejection offsets and precision evidence are retained in this directory. No cases are replaced or silently omitted.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    base.require(not args.out.exists(), 'Fresh diagnostic output directory required')
    started = time.monotonic()
    budget = storage_preflight(args.out)
    cases = prepare_cases(args.campaign_root)
    args.out.mkdir(parents=True)
    atomic_json(args.out/'STORAGE_PREFLIGHT.json', budget)
    atomic_json(args.out/'CASES.json', cases)
    if not args.execute:
        print(json.dumps(dict(status='prepared', cases=5, planned_generations=10, gpu_started=False))); return 0
    config = json.loads(checked(cases[0]['source_native_config']).read_text())
    config.update(output_root=str(args.out.resolve()), episode_out=str(args.out.resolve()), episode_id='retained_replay',
        retained_cases=cases, correction_bank_source_sha256=native.source_pins(), startup_timeout_s=1800)
    native.check_sources(config); native.check_numerical_environment(); base.check_environment(config)
    base.check_dependencies(config['workspace_root'])
    atomic_json(args.out/'CONFIG.json', config)
    original_command = native._original_worker_command
    def command(cfg):
        value = original_command(cfg)
        base.require(value[-1] == str(base.HERE/'native.py'), 'Worker command changed')
        value[-1] = str(Path(__file__).with_name('retained_worker.py'))
        return value
    base.worker_command = command
    worker = None; results = {}; precision = {}; state = dict(status='running', generations=0, feature_reconstructions=0)
    try:
        worker = base._WorkerClient(config, 'sd', args.out/'serving_worker')
        for case in cases:
            case_id = case['case_id']; results[case_id] = {}
            directory = args.out/case['stream']; directory.mkdir()
            for arm in ('F', 'BANK'):
                snapshot = dict(arm=arm, turn=case['turn'], before_read=arm=='BANK',
                    state_sha256=case['after_snapshot']['sha256'] if arm=='BANK' else config['released']['checkpoint_identity_sha256'],
                    snapshot=case['after_snapshot'] if arm=='BANK' else None)
                if arm == 'BANK':
                    activation = worker.call('bank_activate', {'state':snapshot, 'projections':case['projections']})
                else:
                    activation = None
                cache = worker.call('flush', {})
                tick = time.perf_counter()
                result = worker.call('generate', dict(case_id=case_id, input_ids=case['input_ids'],
                    checkpoint_id='F', capture=False, arm=arm, state=snapshot, projections=case['projections'],
                    canonical=False, turn=case['turn'], repeat=0))
                result.update(diagnostic_request_s=time.perf_counter()-tick, activation=activation, cache_reset=cache,
                    exact_canonical_output=result['output_ids']==case['output_ids'], exact_input=result['input_ids']==case['input_ids'],
                    original_baseline=case['frozen_metrics'], first_rejection_offsets=trace_offsets(result['trajectory']))
                atomic_json(directory/f'GENERATION_{arm}.json', result)
                results[case_id][arm] = result
                state['generations'] += 1
                base.write_json(args.out/'STATUS.json', state)
                print(json.dumps(dict(case=case_id, arm=arm, generations=state['generations'],
                    calls=result['verification_calls'], output_match=result['exact_canonical_output'])), flush=True)
        worker.close(); worker = None
        for case in cases:
            reused = matching_precision(case, args.campaign_root)
            if reused is not None:
                precision[case['case_id']] = reused
                atomic_json(args.out/case['stream']/'PRECISION.json', reused)
        needed = [c for c in cases if c['case_id'] not in precision]
        if needed:
            try:
                worker = base._WorkerClient(config, 'train', args.out/'precision_worker')
                for case in needed:
                    evidence = worker.call('precision', {'case_id':case['case_id']})['evidence']
                    precision[case['case_id']] = evidence
                    state['feature_reconstructions'] += evidence.get('reconstruction', {}).get('teacher_reconstructions', 0)
                    atomic_json(args.out/case['stream']/'PRECISION.json', evidence)
                    base.write_json(args.out/'STATUS.json', state)
                    print(json.dumps(dict(case=case['case_id'], precision_status=evidence['status'])), flush=True)
            except Exception as error:
                # Preserve completed native results. Never restart generation or
                # retry a feature reconstruction merely to fill a precision cell.
                failure = dict(status='unavailable', reason=repr(error), traceback=traceback.format_exc(),
                    no_retry=True, no_resampling=True, precision=None)
                atomic_json(args.out/'PRECISION_WORKER_FAILURE.json', failure)
                for case in needed:
                    if case['case_id'] not in precision:
                        precision[case['case_id']] = dict(failure, case_id=case['case_id'])
                        atomic_json(args.out/case['stream']/'PRECISION.json', precision[case['case_id']])
        rows = write_report(args.out, cases, results, precision)
        for case in cases:
            for key in ('after_snapshot', 'feedback', 'projections', 'solver', 'source_native_config', 'source_result'):
                checked(case[key])
            selected_matrix(case)
        state.update(status='complete', original_artifact_bindings_unchanged=True,
            all_comparisons_valid=all(r['gain_comparison_valid'] for r in rows), training_steps=0)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        if worker is not None:
            worker.close()
        state['actual_wall_s'] = time.monotonic()-started
        base.write_json(args.out/'STATUS.json', state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
