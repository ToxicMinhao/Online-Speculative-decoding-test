"""Reconstruct public scalar summaries from private mechanism diagnostic records.

No model is loaded. Raw token, margin and anchor arrays remain in the run folder.
An endpoint or restoration failure excludes the affected episode/window from
scientific comparisons but remains visible in the coverage and work accounting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct


WINDOWS = (1, 16, 32, 48, 64, 65, 80, 96, 112, 128)
CANDIDATES = ('R_FULL', 'R_HALF', 'D_FULL', 'K_ONLY')
ROLES = ('A', 'R', 'D')
SUM_FIELDS = (
    'valid_blocks', 'pre_K_prefix_tokens', 'K_ONLY_prefix_tokens',
    'candidate_prefix_tokens', 'prefix_change_sum', 'gross_prefix_gains',
    'gross_prefix_losses', 'improved_blocks', 'regressed_blocks', 'tied_blocks',
    'old_prefix_tokens_lost', 'K_new_prefix_gain', 'K_prefix_gain_retained',
    'K_prefix_gain_lost', 'original_first_errors', 'K_useful_repairs',
    'C_useful_repairs', 'useful_repairs_retained', 'useful_repairs_lost',
    'useful_repairs_newly_obtained', 'protected_positions', 'K_protected_errors',
    'candidate_protected_errors', 'protected_errors_repaired', 'new_protected_errors',
    'K_strict_negative_margin_errors', 'K_wrong_native_tie_errors',
    'candidate_strict_negative_margin_errors', 'candidate_wrong_native_tie_errors',
    'protected_weight', 'K_hinge_numerator_fp64', 'candidate_hinge_numerator_fp64',
)
PREDICTION_FIELDS = (
    'V_before', 'alpha', 'branch_status', 'formal_ideal', 'ideal', 'ideal_null_reason',
    'g_dot_proposed_fp32_displacement', 'g_dot_realized_bf16_displacement',
    'proposed_linearized_V', 'realized_linearized_V', 'actual_V',
    'proposed_minus_formal_ideal', 'realized_minus_proposed',
    'actual_minus_realized_linearized',
)
LIMITS = (
    'Windows and overlapping anchors share trajectories; they are not independent training replications.',
    'D is buffered training data, not unseen or statistically independent data; D is fitted data for D_FULL.',
    'Changing R to D changes both direction and natural step length; this is not a norm-matched direction test.',
    'Different parent states and feedback prevent an isolated training-age interpretation of event 2 versus event 4.',
    'Token-replacement oracles describe local opportunity; they do not bound total algorithm or serving improvement.',
    'Actual-minus-linearized residuals combine possible nonsmooth/nonlinear and finite-precision effects; causes are not uniquely identified.',
    'No significance tests, serving-speed claims, optimal-dose claims or training-inclusive acceleration claims are made.',
    'Earlier KRC1 probes already showed local non-monotonicity, including increased R hinge at window 64.',
)


def read(path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def require_equal(actual, expected, label):
    if isinstance(expected, float):
        equal = isinstance(actual, (float, int)) and close(actual, expected)
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(f'{label}: compact record differs from reconstructed value')


def exact(actual, expected, label):
    if actual != expected:
        raise ValueError(label + ': recorded scalar differs from exact reconstructed value')


def fp32(value):
    """IEEE binary32 round-to-nearest/even for the recorded scalar operations."""
    result = struct.unpack('f', struct.pack('f', value))[0]
    if not math.isfinite(result):
        raise ValueError('Nonfinite native FP32 scalar')
    return result


def reduction_bound(reference, positions):
    """Pre-result bound for nonnegative FP32 products and arbitrary sum trees.

    There are at most N products, N within-chunk additions and N chunk/pair
    additions. A single contribution traverses at most 2N+4 rounding operations.
    gamma_(2N+4) bounds relative rounding; (2N+4)*2^-126 also allows flushing
    subnormal contributions. This deliberately conservative bound is fixed by
    vector size, never fitted to observed discrepancies. Exact pair arithmetic
    below is checked separately, without this bound.
    """
    operations = 2 * positions + 4
    product = operations * 2**-24
    if product >= 1:
        raise ValueError('Response too large for the declared FP32 reduction bound')
    return product / (1-product) * abs(reference) + operations * 2**-126


def prefix_length(predictions, labels, valid):
    length = 0
    for offset in range(1, len(labels)):
        if not valid[offset] or predictions[offset] != labels[offset]:
            break
        length += 1
    return length


def reconstruct_block(block):
    """Independent scalar reconstruction; no torch or original metric helper."""
    labels = block['canonical_tokens']
    valid = block['valid_positions']
    old, k, current = (block[key] for key in
                       ('pre_K_predictions', 'K_ONLY_predictions', 'candidate_predictions'))
    weights = block['protected_weights']
    km, cm = block['K_ONLY_canonical_margins'], block['candidate_canonical_margins']
    if not labels or any(len(values) != len(labels) for values in (valid, old, k, current, weights, km, cm)):
        raise ValueError('Block vectors have different lengths')
    if valid[0] or weights[0] != 0:
        raise ValueError('Known seed entered valid/protected accounting')
    pre, ordinary, candidate = [prefix_length(p, labels, valid) for p in (old, k, current)]
    if any((weight > 0) != (1 <= offset <= pre) for offset, weight in enumerate(weights)):
        raise ValueError('Protected membership differs from the original prefix')
    expected_weights = [pre-offset+1 if 1 <= offset <= pre else 0 for offset in range(len(weights))]
    if weights != expected_weights:
        raise ValueError('Protected weights differ from the qualified triangular magnitudes')
    available = sum(valid)
    if valid != [False] + [offset <= available for offset in range(1, len(valid))]:
        raise ValueError('Valid positions must be one contiguous prefix after the seed')
    for predictions, margins in ((k, km), (current, cm)):
        if any(valid[i] and ((predictions[i] != labels[i] and margins[i] > 0)
                            or (predictions[i] == labels[i] and margins[i] < 0)) for i in range(len(labels))):
            raise ValueError('Native predictions and margins disagree')
    protected = [i for i, weight in enumerate(weights) if weight > 0]
    k_errors = {i for i in protected if k[i] != labels[i]}
    c_errors = {i for i in protected if current[i] != labels[i]}
    difference = candidate - ordinary
    gain = max(0, ordinary - pre)
    retained = min(gain, max(0, candidate - pre))
    has_error = pre < available
    repaired_k, repaired_c = has_error and ordinary > pre, has_error and candidate > pre
    totals = dict(
        valid_blocks=int(available > 0), pre_K_prefix_tokens=pre, K_ONLY_prefix_tokens=ordinary,
        candidate_prefix_tokens=candidate, prefix_change_sum=difference,
        gross_prefix_gains=max(0, difference), gross_prefix_losses=max(0, -difference),
        improved_blocks=int(available > 0 and difference > 0),
        regressed_blocks=int(available > 0 and difference < 0), tied_blocks=int(available > 0 and difference == 0),
        old_prefix_tokens_lost=max(0, pre-candidate), K_new_prefix_gain=gain,
        K_prefix_gain_retained=retained, K_prefix_gain_lost=gain-retained,
        original_first_errors=int(has_error), K_useful_repairs=int(repaired_k), C_useful_repairs=int(repaired_c),
        useful_repairs_retained=int(repaired_k and repaired_c), useful_repairs_lost=int(repaired_k and not repaired_c),
        useful_repairs_newly_obtained=int(not repaired_k and repaired_c), protected_positions=len(protected),
        K_protected_errors=len(k_errors), candidate_protected_errors=len(c_errors),
        protected_errors_repaired=len(k_errors-c_errors), new_protected_errors=len(c_errors-k_errors),
        K_strict_negative_margin_errors=sum(km[i] < 0 for i in k_errors),
        K_wrong_native_tie_errors=sum(km[i] == 0 for i in k_errors),
        candidate_strict_negative_margin_errors=sum(cm[i] < 0 for i in c_errors),
        candidate_wrong_native_tie_errors=sum(cm[i] == 0 for i in c_errors),
        protected_weight=sum(weights), K_hinge_numerator_fp64=sum(w*max(0, -m) for w,m in zip(weights, km)),
        candidate_hinge_numerator_fp64=sum(w*max(0, -m) for w,m in zip(weights, cm)),
    )
    first = min(k_errors) if k_errors else None
    first_predictions, all_predictions = list(k), list(k)
    if first is not None:
        first_predictions[first] = labels[first]
    for offset in k_errors:
        all_predictions[offset] = labels[offset]
    first_length = prefix_length(first_predictions, labels, valid)
    all_length = prefix_length(all_predictions, labels, valid)
    oracle = dict(oracle_first_protected_error_offset=first,
        oracle_first_protected_error_kind=None if first is None else
            ('wrong_native_tie' if km[first] == 0 else 'strict_negative_margin'),
        oracle_first_prefix_length=first_length, oracle_all_prefix_length=all_length,
        oracle_first_prefix_gain=first_length-ordinary, oracle_all_prefix_gain=all_length-ordinary)
    for key, value in {**totals, **oracle}.items():
        require_equal(block.get(key), value, key)
    binding = {key: block[key] for key in ('record_id', 'block_index', 'anchor', 'kept_anchor', 'canonical_tokens',
        'valid_positions', 'protected_weights', 'pre_K_predictions', 'K_ONLY_predictions', 'K_ONLY_canonical_margins')}
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return totals, oracle, fingerprint


def expected_bindings(plan, role):
    result = {}
    for response in plan[role]:
        observation, = response['sampled_anchors']
        positions = [position for batch in observation['positions'] for position in batch]
        keep = [value for batch in observation['keep_mask'] for value in batch]
        if len(positions) != len(keep):
            raise ValueError('Sealed anchor and keep counts differ')
        for index, (position, kept) in enumerate(zip(positions, keep)):
            key = response['record_id'], index
            if key in result:
                raise ValueError('Duplicate sealed role/response/block binding')
            result[key] = position, kept
    if len(plan[role]) != 2:
        raise ValueError('Every sealed role must have exactly two response bindings')
    return result


def inspect_blocks(path, event, step, plan):
    groups = {}
    with path.open() as stream:
        for line in stream:
            block = json.loads(line)
            candidate, role = block['candidate'], block['role']
            if (block.get('event') != event or block.get('step') != step
                    or block.get('pass') != (step-1)//64 or candidate not in CANDIDATES or role not in ROLES):
                raise ValueError('Per-block observation has a foreign event/window/role/candidate')
            key = candidate, role
            if key not in groups:
                groups[key] = dict(summary={name: 0 for name in SUM_FIELDS}, oracle=[], bindings={},
                                   anchors={}, responses={})
            group = groups[key]
            totals, oracle, fingerprint = reconstruct_block(block)
            block_key = block['record_id'], block['block_index']
            if block_key in group['bindings']:
                raise ValueError('Duplicate block observation')
            group['bindings'][block_key] = fingerprint
            group['anchors'][block_key] = block['anchor'], block['kept_anchor']
            response = group['responses'].setdefault(block['record_id'],
                dict(protected_weight=0, numerator_fp64=0, positions=0))
            response['protected_weight'] += totals['protected_weight']
            response['numerator_fp64'] += totals['candidate_hinge_numerator_fp64']
            response['positions'] += len(block['canonical_tokens'])
            for name, value in totals.items():
                group['summary'][name] += value
            if totals['valid_blocks']:
                group['oracle'].append({**oracle, 'has_errors': totals['K_protected_errors'] > 0})
    for (candidate, role), group in groups.items():
        if group['anchors'] != expected_bindings(plan, role):
            raise ValueError('Observed role/response/anchor/keep differs from sealed plan')
        baseline = groups.get(('K_ONLY', role))
        if baseline is None or group['bindings'] != baseline['bindings']:
            raise ValueError('Candidates do not share exactly the same captured blocks and K_ONLY baseline')
    return groups


def validate_native_pair(raw, group=None, *, check_response_hinge=True):
    """Check FP32 pair arithmetic; optionally check its own scored margins.

    The optional bound is an analysis consistency check, never a scientific
    replay/restoration tolerance. A D gradient replay is not reconstructed from
    a different, no-grad forward's margins.
    """
    rows = raw['responses']
    if len(rows) != 2 or len({row['record_id'] for row in rows}) != 2:
        raise ValueError('Native pair must contain two distinct response contributions')
    if group is not None and {row['record_id'] for row in rows} != set(group['responses']):
        raise ValueError('Native pair response identities differ from block evidence')
    numerator = denominator = 0.0
    max_error = max_bound = 0.0
    for row in rows:
        reconstructed = group['responses'][row['record_id']] if group is not None else None
        weight = reconstructed['protected_weight'] if reconstructed is not None else row['protected_weight']
        # Integer triangular weights in the fixed bounded block layout are
        # exactly representable, including their complete response sum.
        if weight > 2**24 or int(weight) != weight:
            raise ValueError('Protected-weight sum leaves the qualified exact-integer FP32 range')
        exact(row['protected_weight'], weight, 'response protected weight')
        contribution = row['violation_numerator']
        exact(contribution, fp32(contribution), 'response FP32 numerator representation')
        if contribution < 0:
            raise ValueError('Negative native hinge numerator')
        if reconstructed is not None and check_response_hinge:
            reference = reconstructed['numerator_fp64']
            error = abs(contribution-reference)
            bound = reduction_bound(reference, reconstructed['positions'])
            if (reference == 0 and contribution != 0) or error > bound:
                raise ValueError('Response hinge differs beyond the declared FP32 reduction bound')
            max_error, max_bound = max(max_error, error), max(max_bound, bound)
        numerator = fp32(numerator + contribution)
        denominator = fp32(denominator + row['protected_weight'])
    exact(raw['numerator'], numerator, 'native pair numerator')
    exact(raw['denominator'], denominator, 'native pair denominator')
    exact(raw['V'], fp32(numerator/denominator) if denominator else numerator, 'native pair V')
    reconstructed_hinge = group is not None and check_response_hinge
    return dict(response_max_reconstruction_error=max_error if reconstructed_hinge else None,
                response_max_rounding_bound=max_bound if reconstructed_hinge else None)


def d_forward_discrepancy(gradient, no_grad):
    """Describe D forward differences; they are not an added historical gate."""
    result = {'gradient_minus_no_grad_' + key: gradient[key] - no_grad[key]
              for key in ('V', 'numerator', 'denominator')}
    original = {row['record_id']: row for row in gradient['responses']}
    baseline = {row['record_id']: row for row in no_grad['responses']}
    result['gradient_no_grad_same_response_ids'] = original.keys() == baseline.keys()
    def flatten(value):
        if isinstance(value, list):
            return [item for child in value for item in flatten(child)]
        return [value]
    for field, label in (('all_margins', 'margin'), ('proposals', 'prediction')):
        comparable = (original.keys() == baseline.keys() and
                      all(field in original[key] and field in baseline[key] for key in original))
        pairs = []
        if comparable:
            for key in original:
                left, right = flatten(original[key][field]), flatten(baseline[key][field])
                if len(left) != len(right):
                    comparable = False
                    break
                pairs.extend(zip(left, right))
        result[f'gradient_no_grad_{label}_comparable'] = comparable
        result[f'gradient_no_grad_{label}_changed_positions'] = (
            sum(left != right for left, right in pairs) if comparable else None)
        if field == 'all_margins':
            result['gradient_no_grad_margin_max_absolute_difference'] = (
                max((abs(left-right) for left, right in pairs), default=0.) if comparable else None)
    return result


def validate_candidate(entry, candidate, evidence, groups):
    expected_alpha = {'K_ONLY': 0., 'R_HALF': .5, 'R_FULL': 1., 'D_FULL': 1.}[candidate]
    exact(entry['alpha'], expected_alpha, 'candidate dose')
    state = entry['state_identity']
    if not re.fullmatch('[0-9a-f]{64}', state):
        raise ValueError('Invalid candidate state identity')
    aliases = sorted(name for name, other in evidence['candidates'].items()
                     if name != candidate and other.get('state_identity') == state)
    exact(sorted(entry['aliases']), aliases, 'candidate aliases')
    native_checks = {role: validate_native_pair(entry['roles'][role], groups[candidate, role]) for role in ROLES}
    if candidate == 'K_ONLY':
        exact(entry['branch']['status'], 'unchanged_post_K', 'K_ONLY branch')
        return native_checks
    fitted_role = 'D' if candidate == 'D_FULL' else 'R'
    exact(entry['fitted_role'], fitted_role, 'fitted role')
    original = evidence['D_post_K' if fitted_role == 'D' else 'original_R_post_K']
    baseline = evidence['candidates']['K_ONLY']['roles'][fitted_role]
    if fitted_role == 'R':
        validate_native_pair(original, groups['K_ONLY', fitted_role])
        for key in ('V', 'numerator', 'denominator'):
            exact(original[key], baseline[key], 'original versus K_ONLY ' + key)
    else:
        # D's fitted direction and predictions retain its actual grad-enabled
        # source V. The handoff requires the exact physical R witness only.
        # Immutable response IDs/weights are source bindings, not numerical
        # forward-equality criteria. Check those without comparing logits.
        validate_native_pair(original, groups['K_ONLY', fitted_role], check_response_hinge=False)
        native_checks['D'].update(d_forward_discrepancy(original, baseline))
    branch, direction, prediction = entry['branch'], entry['direction'], entry['prediction']
    status = branch['status']
    if status not in ('applied', 'empty_protection', 'zero_violation', 'unresolved_zero_gradient'):
        raise ValueError('Unqualified correction branch')
    for key in ('proposed_norm', 'realized_norm'):
        if not math.isfinite(direction[key]) or direction[key] < 0:
            raise ValueError('Nonfinite or negative direction norm')
    exact(direction['realized_norm'] == 0, 'K_ONLY' in aliases, 'realized displacement versus BF16 alias')
    coefficient, squared = branch['coefficient_fp32'], branch['gradient_norm_squared_fp32']
    if not math.isfinite(squared) or squared < 0:
        raise ValueError('Nonfinite or negative fitted gradient norm')
    active = status == 'applied'
    if active:
        if original['V'] <= 0 or squared <= 0 or coefficient is None or not math.isfinite(coefficient) or coefficient > 0:
            raise ValueError('Active fitted correction has invalid V/norm/coefficient')
        exact(coefficient, fp32(-original['V']/squared), 'qualified FP32 coefficient')
    else:
        exact(coefficient, None, 'inactive coefficient')
        for key in ('proposed_norm', 'realized_norm', 'proposed_dot', 'realized_dot'):
            exact(direction[key], 0, 'inactive ' + key)
        if status == 'empty_protection':
            exact(original['denominator'], 0, 'empty protection')
        elif status == 'zero_violation':
            exact(original['V'], 0, 'zero violation')
        else:
            exact(squared, 0, 'unresolved zero gradient')
    value, actual = original['V'], entry['roles'][fitted_role]['V']
    proposed, realized = direction['proposed_dot'], direction['realized_dot']
    if not all(math.isfinite(number) for number in (value, actual, proposed, realized)):
        raise ValueError('Nonfinite fitted prediction input')
    formal = (1-expected_alpha)*value
    expected = dict(V_before=value, alpha=expected_alpha, branch_status=status, formal_ideal=formal,
        ideal=formal if active else None,
        ideal_null_reason=None if active else 'inactive_correction:' + status,
        g_dot_proposed_fp32_displacement=proposed, g_dot_realized_bf16_displacement=realized,
        proposed_linearized_V=value+proposed, realized_linearized_V=value+realized, actual_V=actual,
        proposed_minus_formal_ideal=value+proposed-formal, realized_minus_proposed=realized-proposed,
        actual_minus_realized_linearized=actual-value-realized)
    for key, expected_value in expected.items():
        exact(prediction[key], expected_value, 'prediction ' + key)
    return native_checks


def oracle_rows(group, identity):
    rows = group['oracle']
    output = []
    for label, field in (('FIRST_PROTECTED_ERROR', 'first'), ('ALL_PROTECTED_ERRORS', 'all')):
        gains = [row[f'oracle_{field}_prefix_gain'] for row in rows]
        result = dict(**identity, oracle=label, valid_blocks=len(rows), prefix_gain_sum=sum(gains),
            prefix_gain_mean=ratio(sum(gains), len(rows)), improved_blocks=sum(gain > 0 for gain in gains),
            zero_opportunity_blocks=sum(gain == 0 for gain in gains),
            blocks_with_protected_errors=sum(row['has_errors'] for row in rows))
        if field == 'first':
            for kind in ('wrong_native_tie', 'strict_negative_margin'):
                selected = [row for row in rows if row['oracle_first_protected_error_kind'] == kind]
                result[kind + '_blocking_errors'] = len(selected)
                result[kind + '_prefix_gain_sum'] = sum(row['oracle_first_prefix_gain'] for row in selected)
        output.append(result)
    return output


def episode_validity(root, campaign, item, seal):
    event = item['event_number']
    folder = campaign / f'event_{event:03d}'
    reasons = []
    identities = {}
    for name in ('parent_state', 'endpoint_state'):
        identity = seal.get('files', {}).get(item.get(name))
        if not identity or not re.fullmatch('[0-9a-f]{64}', identity.get('sha256', '')):
            reasons.append('missing_sealed_' + name + '_identity')
        else:
            identities[name] = identity
    endpoints = {}
    for name in ('ENDPOINT_EQUALITY', 'ORIGINAL_ENDPOINT_CHECK', 'LIVE_ENDPOINT', 'LIVE_RESUME'):
        path = folder / (name + '.json')
        endpoints[name] = read(path) if path.exists() else {}
        if endpoints[name].get('status') != 'passed':
            reasons.append(name.lower() + '_not_passed')
    equality = endpoints['ENDPOINT_EQUALITY']
    if (equality.get('exact_decoded_scientific_state') is not True
            or equality.get('scientific_mismatch_count') != 0 or equality.get('event_number') != event):
        reasons.append('original_endpoint_equality_not_established')
    for name in ('PREFLIGHT', 'POSTFLIGHT'):
        path = root / (name + '.json')
        if not path.exists() or read(path).get('status') != 'passed':
            reasons.append(name.lower() + '_not_passed')
    return dict(event=event, included=not reasons, exclusions=reasons, identities=identities,
        metadata_difference_count=equality.get('metadata_difference_count'),
        comparison='decoded scientific state, not serialized file-hash equality')


def gather_episode(folder, event, endpoint, sealed_windows):
    windows, details, predictions, oracles = [], [], [], []
    for step in WINDOWS:
        identity = dict(event=event, pass_index=(step-1)//64, local_step=step,
                        global_step=(event-1)*128+step)
        path = folder / 'mechanism' / f'window_{step:03d}.json'
        evidence = read(path) if path.exists() else {}
        reasons = list(endpoint['exclusions'])
        expected_plan = sealed_windows.get(step)
        if expected_plan is None or evidence.get('plan') != expected_plan:
            reasons.append('window_plan_differs_from_sealed_input')
        if evidence.get('status') != 'completed':
            reasons.append('window_not_completed')
        if evidence.get('restoration_passed') is not True:
            reasons.append('continuation_restoration_not_passed')
        if evidence.get('k_only_witness_passed') is not True:
            reasons.append('physical_K_ONLY_witness_not_passed')
        if evidence.get('candidate_order') != list(CANDIDATES):
            reasons.append('candidate_order_differs')
        groups = {}
        block_path = path.with_name(f'window_{step:03d}_BLOCKS.jsonl')
        if block_path.exists():
            try:
                groups = inspect_blocks(block_path, event, step, expected_plan)
            except (ValueError, KeyError, TypeError) as error:
                reasons.append('block_reconstruction_failed:' + str(error))
        else:
            reasons.append('missing_block_records')
        for candidate in CANDIDATES:
            entry = evidence.get('candidates', {}).get(candidate, {})
            excluded = list(reasons)
            if entry.get('status') != 'valid':
                excluded.append('candidate_not_valid')
            for role in ROLES:
                try:
                    group = groups[candidate, role]
                    saved = entry['roles'][role]['metrics']['summary']
                    for key in SUM_FIELDS:
                        require_equal(saved[key], group['summary'][key], key)
                except (ValueError, KeyError, TypeError) as error:
                    excluded.append('role_' + role + '_summary_invalid:' + str(error))
            native_checks = {}
            try:
                native_checks = validate_candidate(entry, candidate, evidence, groups)
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                excluded.append('candidate_scalar_evidence_invalid:' + str(error))
            included = not excluded
            direction = entry.get('direction', {})
            branch = entry.get('branch', {})
            window = dict(**identity, candidate=candidate, included=included,
                exclusion=';'.join(dict.fromkeys(excluded)), branch_status=branch.get('status'),
                aliases=';'.join(entry.get('aliases', [])), proposed_norm=direction.get('proposed_norm'),
                realized_norm=direction.get('realized_norm'),
                quantized_noop=(candidate != 'K_ONLY' and branch.get('status') == 'applied'
                               and 'K_ONLY' in entry.get('aliases', [])),
                g_R_dot_g_D=evidence.get('gradient_alignment', {}).get('g_R_dot_g_D'),
                gradient_cosine=evidence.get('gradient_alignment', {}).get('cosine'),
                pre_capture_s=evidence.get('costs', {}).get('pre_capture_s'),
                candidate_sequence_s=evidence.get('costs', {}).get('candidate_sequence_s'))
            if included:
                for role in ROLES:
                    group = groups[candidate, role]
                    totals = group['summary']
                    raw = entry['roles'][role]
                    details.append(dict(**identity, candidate=candidate, role=role, **totals,
                        prefix_change_mean=ratio(totals['prefix_change_sum'], totals['valid_blocks']),
                        candidate_hinge_fp64=ratio(totals['candidate_hinge_numerator_fp64'], totals['protected_weight']),
                        native_pair_V=raw['V'], native_pair_numerator=raw['numerator'], native_pair_denominator=raw['denominator'],
                        **native_checks[role]))
                    window[f'{role}_prefix_change'] = totals['prefix_change_sum']
                    window[f'{role}_gross_loss'] = totals['gross_prefix_losses']
                    window[f'{role}_K_gain_lost'] = totals['K_prefix_gain_lost']
                    window[f'{role}_protected_errors_repaired'] = totals['protected_errors_repaired']
                    for key, value in native_checks[role].items():
                        window[f'{role}_{key}'] = value
                    if candidate == 'K_ONLY':
                        oracles.extend(oracle_rows(group, dict(**identity, role=role)))
                if candidate != 'K_ONLY':
                    prediction = entry['prediction']
                    predictions.append(dict(**identity, candidate=candidate, fitted_role=entry['fitted_role'],
                        **{key: prediction.get(key) for key in PREDICTION_FIELDS},
                        proposed_norm=direction['proposed_norm'], realized_norm=direction['realized_norm'],
                        coefficient_fp32=branch.get('coefficient_fp32'),
                        gradient_norm_squared_fp32=branch.get('gradient_norm_squared_fp32')))
            windows.append(window)
    return windows, details, predictions, oracles


def aggregate_roles(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['event'], row['pass_index'], row['candidate'], row['role']].append(row)
    result = []
    for (event, pass_index, candidate, role), selected in sorted(groups.items()):
        sums = {name: sum(row[name] for row in selected) for name in SUM_FIELDS}
        changes = [row['prefix_change_sum'] for row in selected]
        native_numerator = sum(row['native_pair_numerator'] for row in selected)
        native_denominator = sum(row['native_pair_denominator'] for row in selected)
        result.append(dict(event=event, pass_index=pass_index, candidate=candidate, role=role,
            windows=len(selected), **sums,
            prefix_change_mean=ratio(sums['prefix_change_sum'], sums['valid_blocks']),
            candidate_hinge_fp64=ratio(sums['candidate_hinge_numerator_fp64'], sums['protected_weight']),
            native_pair_pooled_V=ratio(native_numerator, native_denominator),
            native_pair_numerator=native_numerator, native_pair_denominator=native_denominator,
            response_max_reconstruction_error=max(row['response_max_reconstruction_error'] for row in selected),
            response_max_rounding_bound=max(row['response_max_rounding_bound'] for row in selected),
            window_change_min=min(changes), window_change_median=statistics.median(changes),
            window_change_max=max(changes), improved_windows=sum(value > 0 for value in changes),
            regressed_windows=sum(value < 0 for value in changes), tied_windows=sum(value == 0 for value in changes)))
    return result


def recommendation(windows, oracles):
    """Keep machine output descriptive; the full handoff map needs interpretation."""
    valid = [row for row in windows if row['included']]
    return dict(status='inconclusive', intervention=None,
        reason=('Machine summaries are descriptive. A final recommendation requires explicit interpretation of all branches of the original handoff decision map; no new numeric decision threshold is imposed.'
                if len(valid) == 80 else 'Incomplete valid coverage prevents a supported next-change recommendation.'),
        complete_handoff_interpretation_required=True,
        quantified_half_noop_windows=sum(row['quantized_noop'] for row in valid if row['candidate'] == 'R_HALF'),
        oracle_first_gain=sum(row['prefix_gain_sum'] for row in oracles if row['oracle'] == 'FIRST_PROTECTED_ERROR'),
        oracle_all_gain=sum(row['prefix_gain_sum'] for row in oracles if row['oracle'] == 'ALL_PROTECTED_ERRORS'))


def fallacy_scan(roles, exclusions):
    directions = defaultdict(list)
    for row in roles:
        directions[row['candidate'], row['role']].append(row['prefix_change_sum'])
    mixed = [f'{candidate}/{role}' for (candidate, role), values in directions.items()
             if min(values) < 0 < max(values)]
    return [
        dict(type='Simpson paradox', assessment='Mixed phase directions: ' + ', '.join(mixed) if mixed else
             'Event/pass strata retained; no pooled-between-event claim.'),
        dict(type='Ecological fallacy', assessment='Block occurrences and window summaries are not generalized to requests, seeds or users.'),
        dict(type='Berkson paradox', assessment='Fixed training-buffer sample, not a population sample; sampling/selection limits remain.'),
        dict(type='Collider bias', assessment='No post-outcome covariate adjustment or favorable-window selection.'),
        dict(type='Base rate neglect', assessment='Fixed valid-block denominators, zero opportunities and gross losses retained.'),
        dict(type='Regression to the mean', assessment='Windows fixed before outcomes; no extreme-state selection.'),
        dict(type='Survivorship bias', assessment=f'{exclusions} excluded candidate slots remain visible with reasons; no silent survivor-only coverage claim.'),
        dict(type='Look-elsewhere effect', assessment='All fixed candidates/roles/windows reported; no significance testing.'),
        dict(type='Garden of forking paths', assessment='Fixed sampling/doses/reference offsets; no outcome-driven tuning or replacement windows.'),
        dict(type='Correlation versus causation', assessment='Same-state interventions support local contrasts only; endpoint/age/serving mechanisms remain unisolated.'),
        dict(type='Reverse causality', assessment='Pre-K membership precedes K/candidates; no observational temporal-causality claim.'),
    ]


def scalar_costs(costs):
    def numbers(value):
        if isinstance(value, dict):
            return {key: numbers(item) for key, item in value.items()
                    if isinstance(item, (dict, int, float, bool)) and key not in ('correction',)}
        return value
    episodes = {}
    for name, value in costs.get('episodes', {}).items():
        episode = numbers(value)
        correction = value.get('correction', {})
        episode['original_correction_costs'] = numbers(correction.get('costs', {}))
        episode['original_correction_work'] = numbers(correction.get('work', {}))
        episode['diagnostic_windows'] = [dict(step=row['step'], costs=numbers(row.get('costs', {})),
            work=numbers(row.get('work', {}))) for row in correction.get('mechanism', {}).get('rows', [])]
        episodes[name] = episode
    return dict(actual_campaign_wall_s=costs.get('actual_campaign_wall_s'), episodes=episodes,
        work_ledger=costs.get('work_ledger', {}),
        timer_accounting='Component timers nest inside event/campaign wall time; do not sum nested costs.',
        internal_work='Vocabulary-head and layer recomputation are separate from response-forward equivalents.')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formatted(value):
    return 'undefined' if value is None else f'{value:.6g}' if isinstance(value, float) else str(value)


def render_report(summary, roles, predictions, oracles, costs, windows):
    included = summary['included_candidate_slots']
    decision = summary['recommendation']
    lines = ['# KRC mechanism diagnostic', '',
        f"Execution source: `{summary['execution_git_commit']}`. Included candidate slots: **{included}/80** across the fixed event-2/event-4 sample.", '',
        '**Recommendation: ' + (decision['intervention'] or 'inconclusive') + '.** ' + decision['reason'], '',
        'Only the original R_FULL trajectory continued. Candidates were temporary; no serving generation or replacement training arm was run.', '',
        '## Four research questions', '',
        '**1. Does correction realize its fitted-objective prediction?**', '']
    for event in (2, 4):
        for candidate in ('R_FULL', 'R_HALF', 'D_FULL'):
            selected = [row for row in predictions if row['event'] == event and row['candidate'] == candidate]
            if not selected:
                lines.append(f'- Event {event}, {candidate}: no valid fitted-objective measurements.')
                continue
            residual = [row['actual_minus_realized_linearized'] for row in selected]
            lines.append(f"- Event {event}, {candidate}: {len(selected)} windows; actual minus realized-linear prediction median/range "
                         f"{formatted(statistics.median(residual))} / [{formatted(min(residual))}, {formatted(max(residual))}]. "
                         f"Hinge decreased in {sum(row['actual_V'] < row['V_before'] for row in selected)}, increased in "
                         f"{sum(row['actual_V'] > row['V_before'] for row in selected)}; inactive ideals remain null.")
    lines += ['', 'Proposed-versus-realized differences describe rounding of the actual displacement. The remaining actual-versus-linearized residual does not uniquely establish nonlinearity.', '',
        '**2. Does lower hinge restore native correct prefixes?** The table keeps fitted and cross-role prefix effects beside protected errors; lower V alone is not a success criterion.', '',
        '| Event | Pass (zero based) | Candidate | Role | Prefix Δ | Gross gain/loss | Mean Δ per valid block | Windows +/−/= | K gain lost | Protected wrong ties |',
        '|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for row in roles:
        if row['candidate'] == 'K_ONLY':
            continue
        lines.append(f"| {row['event']} | {row['pass_index']} | {row['candidate']} | {row['role']} | {row['prefix_change_sum']:+} | "
            f"{row['gross_prefix_gains']}/{row['gross_prefix_losses']} | {formatted(row['prefix_change_mean'])} | "
            f"{row['improved_windows']}/{row['regressed_windows']}/{row['tied_windows']} | {row['K_prefix_gain_lost']} | {row['candidate_wrong_native_tie_errors']} |")
    lines += ['', '**3. Does correction undo K learning or transfer?** K gained-prefix retention and loss, original-prefix loss, and matched first-error repair identities are retained in CANDIDATE_ROLE.csv. A and D are evaluated at the same restored post-K state. Positive and negative windows remain separately visible in WINDOW_SUMMARY.csv.', '',
        '**4. How sensitive is correction to the defining reference?** R_FULL and D_FULL use their own fitted hinge and natural step length. D is fitted data for D_FULL, so own-pair benefit does not establish a better reference policy.', '']
    for event in (2, 4):
        selected = [row for row in windows if row['event'] == event and row['included'] and row['candidate'] == 'K_ONLY']
        cosines = [row['gradient_cosine'] for row in selected if row['gradient_cosine'] is not None]
        if cosines:
            lines.append(f"- Event {event}: g_R/g_D cosine median/range {formatted(statistics.median(cosines))} / "
                         f"[{formatted(min(cosines))}, {formatted(max(cosines))}], defined in {len(cosines)}/{len(selected)} windows.")
        paired = {candidate: {row['local_step']: row for row in windows if row['event'] == event
                  and row['candidate'] == candidate and row['included']} for candidate in ('R_FULL', 'D_FULL')}
        common = set(paired['R_FULL']) & set(paired['D_FULL'])
        for role in ('R', 'D'):
            if not common:
                lines.append(f'- Event {event}, scored pair {role}: no valid paired R_FULL/D_FULL comparisons.')
                continue
            sums = {candidate: sum(paired[candidate][step][f'{role}_prefix_change'] for step in common)
                    for candidate in ('R_FULL', 'D_FULL')}
            lines.append(f"- Event {event}, scored pair {role}: R_FULL/D_FULL prefix changes {sums['R_FULL']:+}/{sums['D_FULL']:+} over {len(common)} paired windows; see pass/window distributions before interpreting transfer.")
    lines += ['', 'PREDICTIONS.csv retains proposed/realized norms; WINDOW_SUMMARY.csv retains gradient alignment, BF16 aliases and paired A/R/D outcomes.', '',
        '## Oracle opportunity', '', '| Event | Role | First-only gain | All-protected gain | First blocked by wrong tie: recovered gain |',
        '|---:|---|---:|---:|---:|']
    for event in (2, 4):
        for role in ROLES:
            selected = [row for row in oracles if row['event'] == event and row['role'] == role]
            if not selected:
                lines.append(f'| {event} | {role} | unavailable | unavailable | unavailable |')
                continue
            first = [row for row in selected if row['oracle'] == 'FIRST_PROTECTED_ERROR']
            all_rows = [row for row in selected if row['oracle'] == 'ALL_PROTECTED_ERRORS']
            lines.append(f"| {event} | {role} | {sum(row['prefix_gain_sum'] for row in first)} | "
                         f"{sum(row['prefix_gain_sum'] for row in all_rows)} | {sum(row.get('wrong_native_tie_prefix_gain_sum', 0) for row in first)} |")
    ledger = costs['work_ledger']
    lines += ['', 'These are hypothetical token replacements, not attainable parameter updates or a serving-improvement bound. ORACLES.csv retains every window, including zero opportunities.', '',
        '## Validity and work', '',
        f"Decoded endpoint gates: " + '; '.join(f"event {row['event']}: {'passed' if row['included'] else 'excluded (' + ', '.join(row['exclusions']) + ')'}" for row in summary['episodes']) + '.',
        f"Recorded campaign wall time: {formatted(costs['actual_campaign_wall_s'])} s. Core response forwards/backwards: "
        f"{formatted(ledger.get('core_response_forwards'))}/{formatted(ledger.get('core_response_backwards'))}; ordinary Adam updates: "
        f"{formatted(ledger.get('ordinary_Adam_steps'))}. Separate startup forwards/backwards/Adam: "
        f"{formatted(ledger.get('startup_response_forwards'))}/{formatted(ledger.get('startup_response_backwards'))}/{formatted(ledger.get('startup_Adam_steps'))}.",
        'Nested timing components must not be added to campaign/event time. COSTS_WORK.json preserves numeric phase costs and internal-work accounting. Missing evidence is not replaced by a budget-derived measurement.',
        'The gamma_n response-reduction bound is an analysis consistency check, not a relaxed scientific replay or restoration tolerance. WINDOW_SUMMARY.csv retains each role’s observed reconstruction discrepancy and bound; CANDIDATE_ROLE.csv retains their maxima. D gradient/no-grad scalar, margin and prediction differences are descriptive and do not add a D equality gate.', '',
        '## Limits and interpretation', '']
    lines += ['- ' + item for item in LIMITS]
    lines += ['', 'Earlier non-monotonicity source: [KRC1 report, local probes](../../reports/dflash_krc_event_20261027/REPORT.md).', '',
        '## Statistical fallacy scan', '', '11/11 reviewed as descriptive design/claim checks; this is not a statistical certification.', '']
    lines += ['- ' + row['type'] + ': ' + row['assessment'] for row in summary['fallacy_scan']]
    lines += ['', '## Material Passport', '',
        '- Mode: ARS experiment-agent validate; compact evidence reconstructed without GPU replication.',
        '- Status: ANALYZED. Scientific replay validity is conditional on the recorded endpoint and isolation receipts.',
        '- Stage 1 / project None unchanged. Any proposed next intervention requires a separately authorized test.',
        '- Public files contain scalar summaries only; native token/margin/anchor arrays remain in private run artifacts.', '']
    return '\n'.join(lines)


def build_report(root, output, *, execution_commit):
    root, output = Path(root), Path(output)
    if not re.fullmatch('[0-9a-f]{40}', execution_commit):
        raise ValueError('Require the actual 40-character execution commit')
    campaign = root / 'campaign_001'
    plan, seal = read(root / 'sealed/PLAN.json'), read(root / 'sealed/SEAL.json')
    status, costs = read(campaign / 'STATUS.json'), read(campaign / 'COSTS.json')
    for source in (plan.get('execution_source', {}), seal.get('execution_source', {}),
                   status.get('execution_source', {}), costs.get('execution_source', {})):
        if source.get('execution_git_commit') != execution_commit or source.get('dirty_files') != []:
            raise ValueError('Requested commit differs from clean execution source receipts')
    if [item['event_number'] for item in plan['episodes']] != [2, 4]:
        raise ValueError('Expected independent events 2 and 4')
    all_windows, details, predictions, oracles, endpoints = [], [], [], [], []
    for item in plan['episodes']:
        endpoint = episode_validity(root, campaign, item, seal)
        endpoints.append(endpoint)
        window_path = root / 'sealed' / Path(item['window_plan']).name
        sealed_windows = read(window_path)
        if [row['step'] for row in sealed_windows] != list(WINDOWS):
            raise ValueError('Sealed window coverage differs from fixed sampling plan')
        values = gather_episode(campaign / f"event_{item['event_number']:03d}", item['event_number'], endpoint,
                                {row['step']: row for row in sealed_windows})
        for target, rows in zip((all_windows, details, predictions, oracles), values):
            target.extend(rows)
    roles = aggregate_roles(details)
    included = sum(row['included'] for row in all_windows)
    summary = dict(execution_git_commit=execution_commit, campaign_status=status.get('status'),
        expected_candidate_slots=80, included_candidate_slots=included,
        excluded_candidate_slots=80-included, episodes=endpoints,
        recommendation=recommendation(all_windows, oracles),
        fallacy_scan=fallacy_scan(roles, 80-included), limitations=list(LIMITS),
        scalar_reductions='FP64 reporting; native token decisions and original update arithmetic unchanged',
        native_pair_validation='Exact binary32 response-contribution addition and final division; no FP64 tolerance gate on native pair arithmetic',
        response_hinge_validation='Analysis consistency only, not a scientific replay/restoration tolerance: fixed pre-result bound gamma_(2N+4)*abs(FP64 numerator)+(2N+4)*2^-126, N=stored response positions; exact zero when reconstructed numerator is zero',
        D_gradient_no_grad_comparison='Descriptive scalar/margin/prediction discrepancies; no exact D forward-equality gate; predictions use actual D gradient-source V',
        prediction_validation='Exact source V, fitted role, fixed dose, alias/norm, coefficient and recorded algebra checks',
        denominators='prefix sums / fixed valid block occurrences; hinge numerator sums / frozen protected weight sums',
        public_outputs_exclude_raw_arrays=True)
    costs_public = scalar_costs(costs)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('CANDIDATE_ROLE', roles), ('WINDOW_SUMMARY', all_windows),
                       ('PREDICTIONS', predictions), ('ORACLES', oracles)):
        write_csv(output / (name + '.csv'), rows)
    for name, value in (('SUMMARY', summary), ('COSTS_WORK', costs_public)):
        (output / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    (output / 'REPORT.md').write_text(render_report(summary, roles, predictions, oracles, costs_public, all_windows))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execution-commit', required=True)
    args = parser.parse_args()
    result = build_report(args.root, args.output, execution_commit=args.execution_commit)
    print(json.dumps({key: result[key] for key in ('included_candidate_slots', 'excluded_candidate_slots', 'recommendation')}, indent=2))
