"""Export compact Point1 evidence from retained receipts; no model execution.

Raw readouts contain private identities and token content. This exporter selects
public scalar fields explicitly, recomputes paired reductions, and authenticates
its source files. It cannot reconstruct unsaved tensor contractions.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .point1_readout import compare_readouts

CONTRASTS = (('F', 'C'), ('F', 'BETA'), ('C', 'BETA'))
ENDPOINTS = ('parent', 'Q_BETA', 'Q_C', 'Q_F', 'N_BETA', 'N_C', 'N_F')
METRICS = ('raw_repair', 'useful_repair', 'damaged_prefix_positions',
           'damaged_prefix_blocks', 'surviving_followers', 'reachable_followers',
           'matching_prefix_length', 'fixed_K_shortfall', 'assessment')
CONTRACT_FIELDS = (
    'schema', 'status', 'execution_commit', 'cases', 'primary_contrast', 'secondary_contrast',
    'native_steps_independently_restored', 'parent_counters', 'case_counters',
    'live_precision', 'diagnostic_product_reduction', 'shadow_precision', 'objective_changes',
    'full_events', 'generation_calls', 'serial_token_oracle_calls', 'coefficients',
    'clip_jvp_test_atol', 'clip_jvp_test_rtol', 'native_beta_g0_equality', 'native_C_repeatability',
    'vector_zero_policy', 'prior_occupied_card_s', 'limits', 'devices', 'local_allocated_bytes',
    'remaining_write_budget_bytes', 'created_epoch', 'old_hidden_gate_status',
)


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def selected(value, names):
    return {key: value[key] for key in names if key in value}


def _vector_scalar(row):
    return {key: value for key, value in row.items() if key != 'per_layer'}


def _prediction(prediction, benefit):
    sign = lambda value: 'positive' if value > 0 else 'negative' if value < 0 else 'zero'
    error = prediction - benefit
    return dict(prediction=prediction, finite_native_benefit=benefit,
                prediction_sign=sign(prediction), finite_native_benefit_sign=sign(benefit),
                signs_match=sign(prediction) == sign(benefit),
                signed_prediction_error=error, absolute_prediction_error=abs(error),
                relative_absolute_prediction_error=abs(error) / abs(benefit) if benefit else None,
                relative_error_denominator='abs(finite_native_assessment_benefit)',
                no_fidelity_gate=True)


def _score_summary(scoring, contrast):
    if contrast in scoring.get('predictions', {}):
        rows = scoring['predictions'][contrast]
        return dict(
            raw=math.fsum(row['raw']['total'] for row in rows),
            beta_centered=math.fsum(row['beta_centered']['total'] for row in rows),
            raw_positive=math.fsum(row['raw']['positive'] for row in rows),
            raw_negative=math.fsum(row['raw']['negative'] for row in rows),
            centered_positive=math.fsum(row['beta_centered']['positive'] for row in rows),
            centered_negative=math.fsum(row['beta_centered']['negative'] for row in rows),
            raw_absolute=math.fsum(row['raw']['absolute'] for row in rows),
            centered_absolute=math.fsum(row['beta_centered']['absolute'] for row in rows),
            raw_minus_centered=math.fsum(row['raw_minus_centered'] for row in rows),
            mass_offset_contribution=math.fsum(row['mass_offset_contribution'] for row in rows),
            max_absolute_pool_mass_residual=max(row['max_absolute_pool_mass_residual'] for row in rows),
            prefix_occurrences=sum(row['prefix_occurrences'] for row in rows),
            nonempty_pools=sum(row['nonempty_pools'] for row in rows),
            construction='sum response score contrasts; normalization already included in scores',
        )
    if contrast == 'C-BETA' and all(key in scoring.get('predictions', {}) for key in ('F-C', 'F-BETA')):
        fc, fb = _score_summary(scoring, 'F-C'), _score_summary(scoring, 'F-BETA')
        return dict(raw=fb['raw'] - fc['raw'], beta_centered=fb['beta_centered'] - fc['beta_centered'],
                    construction='derived as (F-BETA)-(F-C); signed occurrence contributions unavailable')
    return None


def summarize_readout(scope, result):
    response_rows, block_rows = [], []
    for ordinal, response in enumerate(result['responses']):
        blocks = response['blocks']
        row = dict(scope=scope, endpoint=result['endpoint'], response_ordinal=ordinal,
                   block_occurrences=len(blocks), nonempty_blocks=response['nonempty_blocks'],
                   assessment_fp32=response['response_assessment_fp32'],
                   assessment_fp64=response['response_assessment_fp64'],
                   existing_block_reduced_assessment=response['existing_block_reduced_assessment'],
                   original_first_errors=sum(block['origin']['e'] >= 1 for block in blocks),
                   original_prefix_occurrences=sum(block['origin']['L'] for block in blocks),
                   original_usable_followers=sum(max(0, block['origin']['K'] - block['origin']['e'])
                                                for block in blocks if block['origin']['e'] >= 1),
                   assessed_occurrences=sum(len(block['positions']) for block in blocks),
                   margin_ties=sum(position['margin_tie'] for block in blocks for position in block['positions']),
                   wrong_rival_ties=sum(position['wrong_rival_tie'] for block in blocks for position in block['positions']))
        for metric in METRICS:
            row[metric] = math.fsum(block['metrics'][metric] for block in blocks)
        response_rows.append(row)
        for index, block in enumerate(blocks):
            block_rows.append(dict(scope=scope, endpoint=result['endpoint'], response_ordinal=ordinal,
                                   block_ordinal=index, **{key: block['origin'][key] for key in ('L', 'e', 'H', 'K', 'm')},
                                   **block['metrics'], assessment_sum_fp64=block['assessment_sum_fp64']))
    summary = dict(scope=scope, endpoint=result['endpoint'], responses=len(response_rows),
                   block_occurrences=sum(row['block_occurrences'] for row in response_rows),
                   mean_response_assessment_fp32=result['mean_response_assessment_fp32'],
                   mean_response_assessment=result['mean_response_assessment'],
                   mean_response_assessment_fp64=result['mean_response_assessment_fp64'],
                   state_preserved=result['state_preserved'], status=result['status'],
                   draft_forwards=result['draft_forwards'], head_calls=result['head_calls'])
    for key in ('original_first_errors', 'original_prefix_occurrences', 'original_usable_followers',
                'assessed_occurrences', 'margin_ties', 'wrong_rival_ties', *METRICS):
        summary[key] = math.fsum(row[key] for row in response_rows)
    return summary, response_rows, block_rows


def export(root, output, attempt=None):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Use a new Point1 publication directory')
    attempts = sorted(root.glob('attempt_[0-9][0-9][0-9]'))
    attempt_root = root / f'attempt_{attempt:03d}' if attempt is not None else attempts[-1]
    native = attempt_root / 'native'
    consumed, missing = {}, []

    def read(path, *, required=False):
        path = Path(path)
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            missing.append(str(path.relative_to(root)))
            return None
        consumed[str(path.relative_to(root))] = identity(path)
        return json.loads(path.read_text())

    plan = read(root / 'CONTRACT.json', required=True)
    result = read(native / 'RESULT.json', required=True)
    scoring, vectors = read(native / 'SCORING.json'), read(native / 'VECTOR_CHAIN.json')
    work = read(native / 'WORK.json')
    ledger, clock = read(root / 'STEP_LEDGER.json'), read(root / 'CLOCK.json')
    process, admission = read(attempt_root / 'PROCESS.json'), read(attempt_root / 'ADMISSION.json')
    native_probe = read(native / 'NATIVE_POINT_PROBE.json')
    cases, state = {}, {}
    for name in ('BETA', 'C', 'F', 'C_REPEAT'):
        case = read(native / f'{name}.json')
        if case:
            cases[name] = selected(case, ('status', 'adam_attempts', 'adam_completed', 'gradient_identity',
                                         'beta_g0_exact', 'native_preclip_norm', 'endpoint', 'exact_C_repeat', 'restoration'))
        for suffix in ('INITIAL_STATE', 'RESTORATION'):
            receipt = read(native / f'{name}_{suffix}.json')
            if receipt is not None:
                state[f'{name}_{suffix}'] = selected(receipt, ('status','exact_resume','global_step','presentations',
                    'no_rng_reset','extra_model_passes','extra_optimizer_steps'))
                state[f'{name}_{suffix}']['mismatch_count'] = len(receipt.get('mismatches', []))
    for name in ('AFTER_STARTUP', 'FINAL_RESTORATION'):
        receipt = read(native / f'{name}.json')
        if receipt is not None:
            state[name] = selected(receipt, ('status','exact_resume','global_step','presentations',
                'no_rng_reset','extra_model_passes','extra_optimizer_steps'))
            state[name]['mismatch_count'] = len(receipt.get('mismatches', []))
    readouts = {'window': {}, 'new_buffer': {}}
    summary_rows, response_rows, block_rows = [], [], []
    for scope, prefix in (('window', 'window'), ('new_buffer', 'witness')):
        for endpoint in ENDPOINTS:
            if scope == 'new_buffer' and endpoint.startswith('Q_'):
                continue
            receipt = read(native / f'{prefix}_{endpoint}.json')
            if receipt is None:
                continue
            if receipt['endpoint'] != endpoint or receipt['status'] != 'passed':
                raise ValueError('Point1 readout label or status differs from its file')
            readouts[scope][endpoint] = receipt
            summary, responses, blocks = summarize_readout(scope, receipt)
            summary_rows.append(summary)
            response_rows.extend(responses)
            block_rows.extend(blocks)
    comparisons = {}
    for scope, rows in readouts.items():
        comparisons[scope] = {}
        pairs = [(reference, candidate) for reference, candidate in (
            ('Q_C','Q_F'), ('Q_BETA','Q_F'), ('Q_BETA','Q_C'),
            ('N_C','N_F'), ('N_BETA','N_F'), ('N_BETA','N_C'))
            if reference in rows and candidate in rows]
        pairs.extend(('parent', endpoint) for endpoint in rows if endpoint != 'parent' and 'parent' in rows)
        for reference, candidate in pairs:
            comparisons[scope][candidate + '-' + reference] = compare_readouts(rows[reference], rows[candidate])
    contrast_rows, vector_layers, prediction_rows = [], [], []
    for candidate, reference in CONTRASTS:
        contrast = candidate + '-' + reference
        common = dict(contrast=contrast, role='primary' if contrast=='F-C' else 'secondary' if contrast=='F-BETA' else 'derived_context')
        score = _score_summary(scoring, contrast) if scoring else None
        if score:
            contrast_rows.append(dict(**common, stage='score_prediction', value=score['raw'], **score))
        chain = vectors.get(contrast) if vectors else None
        if not chain:
            continue
        for stage, metric in chain.items():
            if stage in ('native_gradient', 'native_point_prediction'):
                contrast_rows.append(dict(**common, stage=stage,
                                          value=metric['reference_signed_projection'], **_vector_scalar(metric)))
            else:
                contrast_rows.append(dict(**common, stage=stage, **_vector_scalar(metric)))
            for layer, values in metric.get('per_layer', {}).items():
                vector_layers.append(dict(**common, stage=stage, layer=layer, **values))
        for family, stage in (('Q','cast_shadow_functional_benefit'), ('N','native_functional_benefit')):
            paired = comparisons['window'].get(f'{family}_{candidate}-{family}_{reference}')
            if paired:
                contrast_rows.append(dict(**common, stage=stage, value=paired['assessment_benefit_fp64'],
                    original_fp32_value=paired['assessment_benefit_original_fp32'],
                    positive=paired['positive_contribution_sum_fp64'], negative=paired['negative_contribution_sum_fp64'],
                    positive_occurrences=paired['positive_contributions'], negative_occurrences=paired['negative_contributions'],
                    zero_occurrences=paired['zero_contributions']))
        finite = comparisons['window'].get(f'N_{candidate}-N_{reference}')
        if finite:
            benefit = finite['assessment_benefit_fp64']
            for key in ('native_gradient', 'native_point_prediction'):
                prediction_rows.append(dict(**common, evaluation_point='Q_BETA' if key=='native_gradient' else 'N_BETA',
                    **_prediction(chain[key]['reference_signed_projection'], benefit)))
        gradient_projection = chain['native_gradient']['reference_signed_projection']
        linear_projection = chain['linear_to_finite']['reference_signed_projection']
        contrast_rows.append(dict(**common, stage='jvp_vjp_reconciliation', value=linear_projection-gradient_projection,
            gradient_projection=gradient_projection, linear_projection=linear_projection,
            relative_difference=(linear_projection-gradient_projection)/abs(gradient_projection) if gradient_projection else None,
            relative_difference_denominator='abs(native_gradient_projection)', zero_reference=gradient_projection==0))
    output.mkdir(parents=True)
    public_contract = selected(plan, CONTRACT_FIELDS)
    public_contract['witness'] = selected(plan['witness'], ('status','scope','set_name','responses',
        'qualification_record_ids_disjoint','qualification_feature_paths_disjoint'))
    identifiers = {key: value for key, value in plan.items() if key.endswith('_identity') or key=='parent_binding'}
    identifiers.update(execution_commit=result['execution_commit'],
        parent_feature_identities=[selected(row, ('identity','bytes','sha256')) for row in plan['parent_features']],
        witness_identities=[selected(row, ('feature_identity',)) for row in plan['witness'].get('items', [])],
        witness_bindings=selected(plan['witness'], ('binding_identity','parent_identity','feature_manifest_identity','archive_manifest_identity')),
        cases=cases, raw_input_receipts=consumed)
    write_json(output/'CONTRACT.json', public_contract)
    write_json(output/'IDENTITIES.json', identifiers)
    write_json(output/'STATE_AUDIT.json', dict(receipts=state, qualification_status=result['status'],
        beta_gradient_exact=cases.get('BETA',{}).get('beta_g0_exact'),
        C_repeat_exact=cases.get('C_REPEAT',{}).get('exact_C_repeat'),
        parent_restored=selected(result.get('final_restoration') or {}, ('status','exact_resume','global_step','presentations')),
        immutable_vectors_retained=False,
        audit_limit='Scalar arithmetic and receipt consistency only; full-vector contractions cannot be recomputed offline.'))
    if ledger:
        safe_ledger = selected(ledger, ('schema','contract_identity','planned_cases','maximum_attempted_calls','maximum_attempts','calls'))
        safe_ledger['attempts'] = [dict(**selected(item, ('number','status','started_epoch','finished_epoch')),
            correction_note_present=bool(item.get('correction_note'))) for item in ledger['attempts']]
        write_json(output/'STEP_LEDGER.json', safe_ledger)
    write_json(output/'WORK.json', dict(work=work, startup=selected(result.get('startup') or {},
        ('status','optimizer_steps','physical_work','wall_s','work')),
        interpretation='Module-operation counters; recomputation retained separately where instrumented.'))
    write_json(output/'RESOURCES.json', dict(process=process, admission=admission,
        clock=clock, native=selected(result, ('wall_s','peak_allocated_bytes','peak_reserved_bytes'))))
    write_json(output/'SCORING_SUMMARY.json', selected(scoring or {}, ('coefficients','reference_gradient','scores',
        'assessment','direction_policy','direction_identity','effective_direction_identity',
        'response_denominators','clipping','frozen_before_native_updates')))
    write_json(output/'VECTOR_SUMMARY.json', vectors)
    write_json(output/'NATIVE_POINT_PROBE.json', native_probe)
    write_json(output/'READOUT_SUMMARY.json', summary_rows)
    write_json(output/'READOUT_COMPARISONS.json', comparisons)
    write_json(output/'PREDICTION_ERRORS.json', prediction_rows)
    write_csv(output/'CONTRASTS.csv', contrast_rows)
    write_csv(output/'ENDPOINT_READOUTS.csv', response_rows)
    write_csv(output/'ENDPOINT_BLOCKS.csv', block_rows)
    write_csv(output/'VECTOR_LAYERS.csv', vector_layers)
    completed = result['status']=='completed' and process and process['status']=='completed'
    write_json(output/'EXPORT_AUDIT.json', dict(
        status='completed_receipts_exported' if completed else 'partial_receipts_exported',
        native_status=result['status'], process_status=None if process is None else process['status'],
        consumed_files=len(consumed), missing_files=missing, scope=public_contract['witness'],
        readout_count=sum(len(rows) for rows in readouts.values()), endpoint_response_rows=len(response_rows),
        endpoint_block_rows=len(block_rows), contrast_rows=len(contrast_rows), private_content_excluded=True,
        limitations=['No full vectors saved: offline audit cannot reconstruct tensor products.',
                    'TRAIN witness anchors overlap; no population confidence interval is produced.'],
    ))
    write_json(output/'PUBLIC_FILES.json', dict(execution_commit=result['execution_commit'],
        files={str(path.relative_to(output)): identity(path) for path in sorted(output.iterdir()) if path.is_file()}))
    return dict(status='exported', output=str(output), complete=bool(completed), files=len(list(output.iterdir())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attempt', type=int)
    print(json.dumps(export(**vars(parser.parse_args()))))
