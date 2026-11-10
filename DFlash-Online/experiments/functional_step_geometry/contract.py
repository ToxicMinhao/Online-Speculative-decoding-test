"""Fixed FSG mechanism, continuation boundaries and separately counted work."""
from experiments.full_k_gsm8k.contract import SERVING
from online_feedback.kplus import kplus_contract

SCHEMA = 'kplus_functional_step_geometry_v1'
LEGACY_REALIZATION = 'legacy_raw_norm_v0'
REALIZATION_R1 = 'norm_feasible_bf16_r1'
REALIZATION_REVISIONS = (LEGACY_REALIZATION, REALIZATION_R1)
ARMS = ('KPLUS', 'FSG', 'SCALAR')
DATASETS = ('gsm8k', 'mathqa')
BASE_COMMIT = 'cdb640543d8fe3a43c8e63af4ccaaa002e09a599'
REPORT_COMMIT = '18bd839cbd85f268a07f7da3abb9f9e14ba16bf0'
START_EVENT, FINAL_EVENT = 4, 6
EVENTS = (5, 6)
PANEL_SIZE, ENDPOINT_REPEATS = 256, 2
DIAGNOSTIC_WINDOWS = (1, 32, 65, 128)
ANALYSIS = dict(resamples=10000, seed=20261108, generator='PCG64',
    percentile_method='linear', primary=[['FSG', 'KPLUS'], ['FSG', 'SCALAR']],
    datasets_pooled=False, intervals_adjusted=False)
MAIN_BUDGET = dict(continuations=6, training_events=12, feedback_generations=512,
    optimizer_steps=1536, ordinary_presentations=3072, ordinary_forwards=3072,
    ordinary_backwards=3072, auxiliary_forwards=6144, auxiliary_backwards=2048,
    total_forwards=9216, total_backwards=5120, endpoint_generations=3072)
DIAGNOSTIC_BUDGET = dict(windows=16, response_forwards=640, backwards=64)
QUALIFICATION_BUDGET = dict(optimizer_steps=8, ordinary_presentations=16,
    additional_response_forwards=96, additional_backwards=32, serving_generations=8)


def geometry_contract(realization_revision=LEGACY_REALIZATION):
    """Return a fresh, serializable description; none of these choices is tuned."""
    if realization_revision not in REALIZATION_REVISIONS:
        raise ValueError('Unknown BF16 realization revision')
    result = dict(strength=1.0, margin_floor=0.05,
        native_objective=kplus_contract(), sketch_directions_per_response=1,
        maximum_columns=2, reference_coverage='all_original_sampled_complete_blocks',
        protection='prestate_native_matching_prefix_without_seed_or_first_error',
        response_weights='triangular_over_floored_probability_margin_squared',
        response_average=0.5, metric='euclidean_native_optimizer_active_tensors',
        scaling='largest_sketch_eigenvalue', native_precision='bfloat16',
        native_moments='ordinary_kplus_gradients', scalar_reduction_precision='float64',
        gradient_precision='native_autograd_dtype', master_parameters=False,
        proposal_count=1, backtracking=False, scalar_bisection_iterations=24,
        scalar_target='own_raw_fsg_realized_bfloat16_norm',
        risk='pooled_fixed_prestate_full_vocabulary_forward_kl',
        risk_deadband_multiplier=16, risk_deadband_dtype='float64',
        finite_rejection='exact_native_endpoint', nonfinite_action='stop',
        inactive_tensors='unchanged', expanded_diagnostic_relative_cutoff=1e-6,
        expanded_diagnostic_scaling='production_sigma')
    if realization_revision == REALIZATION_R1:
        result.update(realization_revision=realization_revision,
            realization='original_continuous_displacement_before_bfloat16_rounding',
            feasibility_bisection_iterations=24, feasibility_endpoint='lower_feasible_bracket',
            scalar_target='own_norm_feasible_fsg_realized_bfloat16_norm',
            endpoint_identity='native_bypass_or_prestate_fallback',
            candidate_count_after_realization=1, kl_guided_search=False)
    return result


def experiment_contract(binding, arm, dataset, *, origin_binding, realization_revision=LEGACY_REALIZATION):
    if arm not in ARMS or dataset not in DATASETS:
        raise ValueError('Unknown FSG dataset or arm')
    for value in (binding, origin_binding):
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Require sealed experiment and retained PR32 bindings')
    geometry = geometry_contract(realization_revision)
    result = dict(schema=SCHEMA, binding=binding, arm=arm, dataset=dataset,
        origin_binding=origin_binding, origin_method='KPLUS', origin_event=START_EVENT,
        start_adam_step=512, final_adam_step=768, events=list(EVENTS),
        initialization='exact_complete_dataset_specific_PR32_KPLUS4',
        feedback='byte_identical_shared_evolving_KPLUS_packets',
        native_sampling=True, seed_reset=False, inherited_horizon=7424,
        geometry=None if arm == 'KPLUS' else geometry)
    if realization_revision == REALIZATION_R1:
        result['realization_revision'] = realization_revision
    return result


def expected_work(arm, completed_event):
    if arm not in ARMS or type(completed_event) is not int or not START_EVENT <= completed_event <= FINAL_EVENT:
        raise ValueError('FSG continuation boundary must be event 4, 5 or 6')
    events = completed_event - START_EVENT
    responses = events * 256
    filtered = arm != 'KPLUS'
    return dict(optimizer_steps=events*128, ordinary_presentations=responses,
        ordinary_forwards=responses, ordinary_backwards=responses,
        auxiliary_forward_maximum=3*responses if filtered else 0,
        auxiliary_backward_maximum=responses if filtered else 0)
