"""Publish Point2A compact receipts and paired arithmetic without model execution.

Private response IDs, anchors, labels, proposals, paths and error text are never
publication fields. Missing or failed receipts remain explicitly partial.
"""

import argparse
import json
from pathlib import Path
import re

from .point1_export import METRICS, identity, selected, summarize_readout, write_csv, write_json
from .point1_readout import compare_readouts
from .point2a_readout import CASES, CONTRASTS, ENDPOINTS, paired_contrasts


CONTRACT_FIELDS = (
    'schema', 'status', 'execution_commit', 'cases', 'primary_contrast', 'secondary_contrast',
    'context_contrast', 'native_steps_independently_restored', 'parent_counters', 'case_counters',
    'live_precision', 'diagnostic_product_reduction', 'shadow_precision', 'objective_changes',
    'full_events', 'generation_calls', 'serial_token_oracle_calls', 'coefficients',
    'prior_occupied_card_s', 'limits', 'local_allocated_bytes', 'remaining_write_budget_bytes',
    'created_epoch', 'old_hidden_gate_status',
    'expected_oracle_pool_lengths', 'planned_oracle_backwards', 'numerical_criteria',
    'numerical_contract', 'oracle_selection',
)
STATE_FIELDS = ('status', 'exact_resume', 'global_step', 'presentations', 'no_rng_reset',
                'extra_model_passes', 'extra_optimizer_steps', 'runtime_exact', 'ancillary_exact')
SCORING_FIELDS = (
    'coefficients', 'reference_gradient', 'scores', 'assessment', 'direction_policy',
    'direction_identity', 'effective_direction_identity', 'response_denominators', 'clipping',
    'frozen_before_native_updates', 'sealed_before_native_update', 'reference_case',
    'allocation_diagnostics', 'score_movement', 'centered_score_movement',
    'reference', 'accumulation_factor', 'relative_cap', 'reference_binding',
    'coefficients_sealed_before_own_native_update', 'score_changes_from_beta',
    'shadow_identity', 'cotangent_identity', 'qualification_identity',
)
PRIVATE_FIELDS = frozenset((
    'record_id', 'record_ids', 'response_ids', 'anchors', 'labels', 'canonical', 'rival', 'proposals',
    'feature_path', 'path', 'root', 'parent_path', 'rows', 'tokens', 'input_ids', 'text',
    'prompt', 'response', 'error', 'traceback', 'command', 'processes', 'correction_note',
    'stopped_reason', 'final_storage_error', 'mismatches', 'positions', 'prefix', 'repair',
    'valid', 'teacher_hidden', 'score_arrays', 'coefficient_arrays', 'tensors',
))
PUBLIC_WORDS = frozenset((
    'parent', 'C', 'F_BETA', 'F_C', 'BETA', 'Q_C', 'Q_BETA', *ENDPOINTS,
    'F_C-F_BETA', 'F_C-C', 'F_BETA-C', 'new_buffer', 'TRAIN',
    'admitted', 'authenticated', 'completed', 'passed', 'failed', 'running', 'reserved', 'selected',
    'reused', 'skipped', 'unavailable', 'partial', 'coverage_blocked', 'measured', 'exited',
    'FP32', 'FP64', 'BF16', 'torch.float32', 'torch.float64', 'torch.bfloat16',
    'positive', 'negative', 'zero', 'both_zero', 'both_nonzero', 'reference_zero', 'candidate_zero',
    'point2a_native_step_ledger_v1', 'point2a_clock_v1',
))


def compact(value):
    """Keep scalar evidence under selected receipts, dropping free-form content.

    Shape/token arrays and free text are not trusted publication metadata. Hashes,
    fixed status names and numeric diagnostic lists remain auditable.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if value in PUBLIC_WORDS or re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', value) else None
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PRIVATE_FIELDS or key.endswith(('_path', '_root')) or '/' in key:
                continue
            converted = compact(item)
            if converted is not None or item is None:
                result[key] = converted
        return result
    raise TypeError('Unsupported receipt value')


def _state(receipt):
    result = compact(selected(receipt, STATE_FIELDS))
    result['mismatch_count'] = len(receipt.get('mismatches', []))
    result['additional_mismatch_count'] = len(receipt.get('additional_mismatches', []))
    return result


def _case(receipt):
    result = compact(selected(receipt, (
        'name', 'status', 'adam_attempts', 'adam_completed', 'coefficient_identity',
        'gradient_identity', 'native_preclip_norm', 'endpoint', 'historical_reproduction',
        'reused_from', 'coefficient_equality', 'complete_state_equality', 'physical_work_delta',
    )))
    if receipt.get('restoration'):
        result['restoration'] = _state(receipt['restoration'])
    result['error_present'] = bool(receipt.get('error'))
    return result


def _contract(plan):
    # Contract prose is fixed by source; never copy private free-form instructions.
    result = compact(selected(plan, CONTRACT_FIELDS))
    result['schema'] = 'point2a_c_reference_public_v1'
    result['cases'] = list(CASES)
    result['primary_contrast'] = 'F_C-F_BETA'
    result['working_method_contrast'] = 'F_C-C'
    result['historical_context_contrast'] = 'F_BETA-C'
    from .point2a_prepare import CONVERGENCE
    if 'prospective_convergence' in plan and plan['prospective_convergence'] != CONVERGENCE:
        raise ValueError('Point2A prospective convergence differs from its source')
    result['prospective_convergence'] = dict(CONVERGENCE)
    result['oracle_selection'] = 'Lowest-index complete prefix pool with at least two positions in each response; selected before scores.'
    result['old_hidden_gate_status'] = 'failed_retained_diagnostic'
    result['live_precision'] = 'native BF16'
    result['witness'] = compact(selected(plan.get('witness', {}), (
        'status', 'set_name', 'responses', 'qualification_record_ids_disjoint',
        'qualification_feature_paths_disjoint',
    )))
    return result


def export(root, output, attempt=None):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Use a new Point2A publication directory')
    attempts = sorted(root.glob('attempt_[0-9][0-9][0-9]'))
    if attempt is None and not attempts:
        raise FileNotFoundError('No Point2A attempt receipts exist')
    attempt_root = root / f'attempt_{attempt:03d}' if attempt is not None else attempts[-1]
    native = attempt_root / 'native'
    consumed, missing = {}, []

    def read(path, required=False):
        path = Path(path)
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            missing.append(str(path.relative_to(root)))
            return None
        consumed[str(path.relative_to(root))] = identity(path)
        return json.loads(path.read_text())

    plan = read(root / 'CONTRACT.json', required=True)
    if plan.get('cases') != list(CASES):
        raise ValueError('Point2A contract cases differ')
    result = read(native / 'RESULT.json') or {'status': 'unavailable'}
    scoring = {name: read(native / ('SCORING_' + name + '.json')) for name in ('BETA', 'C')}
    contrasts = read(native / 'CONTRASTS.json')
    qualification = read(native / 'QUALIFICATION.json')
    reference_progress = read(native / 'REFERENCE_PROGRESS.json')
    work = read(native / 'WORK.json')
    ledger, clock = read(root / 'STEP_LEDGER.json'), read(root / 'CLOCK.json')
    process = read(attempt_root / 'PROCESS.json')
    admission = read(attempt_root / 'ADMISSION.json')
    resource_samples = []
    samples_path = attempt_root / 'RESOURCE_SAMPLES.jsonl'
    if samples_path.exists():
        consumed[str(samples_path.relative_to(root))] = identity(samples_path)
        resource_samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    cases, state = {}, {}
    for name in CASES:
        receipt = read(native / (name + '.json'))
        if receipt is not None:
            if receipt.get('name', name) != name:
                raise ValueError('Point2A case label differs from its file')
            cases[name] = _case(receipt)
        for suffix in ('INITIAL_STATE', 'RESTORATION'):
            receipt = read(native / (name + '_' + suffix + '.json'))
            if receipt is not None:
                state[name + '_' + suffix] = _state(receipt)
    for name in ('AFTER_STARTUP', 'AFTER_QUALIFICATION', 'FINAL_RESTORATION'):
        receipt = read(native / (name + '.json'))
        if receipt is not None:
            state[name] = _state(receipt)

    readouts = {'window': {}, 'new_buffer': {}}
    summaries, response_rows, block_rows = [], [], []
    authenticated_witness = plan.get('witness', {}).get('status') == 'authenticated'
    for scope, prefix in (('window', 'window'), ('new_buffer', 'witness')):
        if scope == 'new_buffer' and not authenticated_witness:
            continue
        for endpoint in ENDPOINTS:
            receipt = read(native / (prefix + '_' + endpoint + '.json'))
            if receipt is None:
                continue
            if receipt.get('endpoint') != endpoint or receipt.get('status') != 'passed':
                raise ValueError('Point2A readout label or status differs from its file')
            if len(receipt.get('responses', [])) != (2 if scope == 'window' else 8):
                raise ValueError('Point2A readout response count differs from its frozen set')
            readouts[scope][endpoint] = receipt
            summary, responses, blocks = summarize_readout(scope, receipt)
            summaries.append(summary)
            response_rows.extend(responses)
            # PR61's formatter expands metrics; retain only its declared fields.
            block_rows.extend(selected(row, ('scope', 'endpoint', 'response_ordinal', 'block_ordinal',
                'L', 'e', 'H', 'K', 'm', *METRICS, 'assessment_sum_fp64')) for row in blocks)
    comparisons = {scope: paired_contrasts(rows) for scope, rows in readouts.items()}
    parent_comparisons = {}
    for scope, rows in readouts.items():
        parent_comparisons[scope] = {
            endpoint + '-parent': compare_readouts(rows['parent'], row)
            for endpoint, row in rows.items() if endpoint != 'parent' and 'parent' in rows
        }

    contrast_rows = []
    roles = {'F_C-F_BETA': 'primary', 'F_C-C': 'working_method', 'F_BETA-C': 'historical_context'}
    for candidate, reference in CONTRASTS:
        name = candidate + '-' + reference
        common = dict(contrast=name, role=roles[name])
        scalar = (contrasts or {}).get(name, {})
        prediction = scalar.get('prediction')
        if prediction:
            contrast_rows.append(dict(**common, scope='window', stage='C_score_prediction',
                value=prediction['raw']['total'], **compact(prediction['raw']),
                beta_centered_value=prediction['beta_centered']['total']))
        actual = scalar.get('actual_gradient_contraction')
        if actual is not None:
            contrast_rows.append(dict(**common, scope='window', stage='C_native_gradient_contraction',
                                      value=actual['total'], **compact(actual)))
        for scope, pairs in comparisons.items():
            paired = pairs.get(name)
            if paired is not None:
                contrast_rows.append(dict(**common, scope=scope, stage='native_assessment_benefit',
                    value=paired['assessment_benefit_fp64'],
                    original_fp32_value=paired['assessment_benefit_original_fp32'],
                    positive=paired['positive_contribution_sum_fp64'],
                    negative=paired['negative_contribution_sum_fp64'],
                    positive_occurrences=paired['positive_contributions'],
                    negative_occurrences=paired['negative_contributions'],
                    zero_occurrences=paired['zero_contributions']))

    output.mkdir(parents=True)
    public_contract = _contract(plan)
    write_json(output / 'CONTRACT.json', public_contract)
    identifiers = {key: compact(value) for key, value in plan.items()
                   if key.endswith('_identity') or key == 'parent_binding'}
    identifiers.update(execution_commit=result.get('execution_commit', plan.get('execution_commit')),
        cases=cases, raw_input_receipts=consumed,
        parent_feature_identities=[compact(selected(row, ('identity', 'bytes', 'sha256')))
                                   for row in plan.get('parent_features', [])],
        witness_identities=[compact(selected(row, ('feature_identity',)))
                            for row in plan.get('witness', {}).get('items', [])])
    write_json(output / 'IDENTITIES.json', identifiers)
    write_json(output / 'STATE_AUDIT.json', dict(
        receipts=state, native_status=result['status'], cases=cases,
        parent_restored=_state(result.get('final_restoration') or {}),
        immutable_vectors_retained=False,
        audit_limit='Only scalar arithmetic and receipt consistency are independently recoverable; full vectors were not saved.',
    ))
    if ledger is not None:
        safe_ledger = compact(selected(ledger, ('schema', 'contract_identity', 'planned_cases',
            'maximum_attempted_calls', 'maximum_attempts', 'calls', 'oracle_calls', 'maximum_oracle_calls',
            'token_backwards', 'reused_cases', 'skips')))
        safe_ledger['attempts'] = [dict(
            **compact(selected(row, ('number', 'status', 'started_epoch', 'finished_epoch'))),
            correction_note_present=bool(row.get('correction_note')),
        ) for row in ledger.get('attempts', [])]
        write_json(output / 'STEP_LEDGER.json', safe_ledger)
    write_json(output / 'WORK.json', dict(work=compact(work), startup=compact(selected(
        result.get('startup') or {}, ('status', 'optimizer_steps', 'physical_work', 'wall_s', 'work')))))
    write_json(output / 'RESOURCES.json', dict(
        process=compact(process), admission=compact(admission), clock=compact(clock),
        samples=compact(resource_samples),
        sampled_child_peak_resident_bytes=max(
            ((sample.get('child_memory') or {}).get('peak_resident_bytes') or 0
             for sample in resource_samples), default=None),
        native=compact(selected(result, ('wall_s', 'peak_allocated_bytes', 'peak_reserved_bytes'))),
        stopped_reason_present=bool((process or {}).get('stopped_reason')),
        storage_error_present=bool((process or {}).get('final_storage_error')),
    ))
    write_json(output / 'SCORING_SUMMARY.json', {
        name: compact(selected(receipt or {}, SCORING_FIELDS)) for name, receipt in scoring.items()
    })
    write_json(output / 'NUMERICAL_QUALIFICATION.json', compact(qualification))
    token_rows = [selected(row, ('response_index', 'block_index', 'position', 'unweighted_native_fkl',
        'native_fkl_dtype', 'primary_dot_fp64', 'effective_dot_fp64', 'denominator', 'accumulation_factor',
        'reference_score_fp64', 'reference_score_fp32', 'effective_reference_score_fp64'))
        for row in (reference_progress or {}).get('rows', [])]
    write_csv(output / 'REFERENCE_TOKENS.csv', token_rows)
    write_json(output / 'ALLOCATION_CONTRASTS.json', compact(contrasts))
    write_json(output / 'READOUT_SUMMARY.json', summaries)
    write_json(output / 'READOUT_COMPARISONS.json', comparisons)
    write_json(output / 'PARENT_COMPARISONS.json', parent_comparisons)
    write_json(output / 'DENOMINATORS.json', dict(
        assessment='Fixed parent-origin K; FP32 native margin softplus contributions, divided by each response nonempty-block count, then mean over responses.',
        paired_assessment='Subtract matched represented FP32 contributions in FP64; same frozen response denominators.',
        discrete='Sums and paired gains/losses/ties over overlapping TRAIN block occurrences and response totals.',
        useful_repair='Original first error repaired with every preceding original prefix match surviving.',
        inference='Development diagnostics only; overlapping anchors are not independent samples or serving calls.',
        score_prediction='Sum (a_X-a_Y)*s_C once across responses; s_C already includes 1/(2*Z_response).',
        native_gradient_contraction='-dot(w_C,g_X-g_Y); native gradients are already normalized; no extra 1/(2*Z).',
        cancellation='1-abs(sum signed terms)/sum absolute terms; null if the exact denominator is zero.',
        coefficient_movement='Relative movement uses frozen beta_i; half-L1 mass movement uses sum_prefix beta_i.',
        diagnostic_precision='Represented operands promoted to FP32 before subtraction; diagnostic products and reductions FP64.',
    ))
    write_csv(output / 'CONTRASTS.csv', contrast_rows)
    write_csv(output / 'ENDPOINT_READOUTS.csv', response_rows)
    write_csv(output / 'ENDPOINT_BLOCKS.csv', block_rows)
    gate_passes = dict(
        numerical=(qualification or {}).get('passed') is True,
        oracle=((reference_progress or {}).get('status') == 'completed'
                and (reference_progress or {}).get('completed') == 28 and len(token_rows) == 28),
        state=bool(state) and all(row.get('status') == 'passed' and row.get('exact_resume') is True
                                 and not row['mismatch_count'] and not row['additional_mismatch_count']
                                 for row in state.values()),
        cases=all(cases.get(name, {}).get('status') == 'completed' for name in CASES),
        historical_controls=all(
            cases.get(name, {}).get('historical_reproduction', {}).get('status') == 'passed'
            and cases[name]['historical_reproduction'].get('complete_recorded_scientific_identity') is True
            for name in ('C', 'F_BETA')),
        scalar_contrasts=all(
            (contrasts or {}).get(candidate + '-' + reference, {}).get('prediction') is not None
            and (contrasts or {}).get(candidate + '-' + reference, {}).get('actual_gradient_contraction') is not None
            for candidate, reference in CONTRASTS),
        readout_state=all(row.get('state_preserved') is True for rows in readouts.values() for row in rows.values()),
    )
    complete = (result['status'] == 'completed' and (process or {}).get('status') == 'completed'
                and not missing and authenticated_witness
                and all(gate_passes.values())
                and all(len(pairs) == 3 for pairs in comparisons.values()))
    write_json(output / 'EXPORT_AUDIT.json', dict(
        status='completed_receipts_exported' if complete else 'partial_receipts_exported',
        native_status=result['status'], process_status=(process or {}).get('status'),
        consumed_files=len(consumed), missing_files=missing, witness=public_contract['witness'],
        readout_count=sum(len(rows) for rows in readouts.values()),
        endpoint_response_rows=len(response_rows), endpoint_block_rows=len(block_rows),
        contrasts_by_scope={scope: len(rows) for scope, rows in comparisons.items()},
        contrast_rows=len(contrast_rows), private_content_excluded=True,
        required_gate_passes=gate_passes,
        no_scientific_promotion=not complete,
        no_automatic_promotion=True,
        limitations=['No full vectors saved: tensor contractions cannot be independently reconstructed.',
                     'Repeatedly inspected TRAIN witnesses are development data; no population inference.'],
    ))
    write_json(output / 'PUBLIC_FILES.json', dict(
        execution_commit=result.get('execution_commit', plan.get('execution_commit')),
        files={path.name: identity(path) for path in sorted(output.iterdir()) if path.is_file()},
    ))
    return dict(status='exported', output=str(output), complete=bool(complete), files=len(list(output.iterdir())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attempt', type=int)
    print(json.dumps(export(**vars(parser.parse_args()))))
