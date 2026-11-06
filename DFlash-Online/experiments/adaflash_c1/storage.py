"""Conservative complete-campaign admission with the qualified rolling lifecycle."""
from functools import partial

from experiments.kplus_full_gsm8k import storage as native
from .contract import NEW_BUDGET

GIB = 2 ** 30
# Retained MathQA schema estimate: all three native trace representations,
# output cap 2048, and the worst one-token progress per verification.
RAW_BYTES_PER_GENERATION = 5327030


def capacity_plan(weight_bytes):
    plan = native.capacity_plan(weight_bytes)
    plan['compact_bytes'] = (
        NEW_BUDGET['total_generations'] * RAW_BYTES_PER_GENERATION + 4 * GIB
    )
    plan['raw_bytes_per_generation'] = RAW_BYTES_PER_GENERATION
    plan['raw_budget_generations'] = NEW_BUDGET['total_generations']
    plan['metadata_and_log_allowance_bytes'] = 4 * GIB
    plan['required_initial_free_bytes'] = plan['tensor_peak_with_margin_bytes'] + sum(
        plan[key] for key in ('compact_bytes', 'compiler_bytes', 'workspace_bytes', 'reserve_bytes')
    )
    return plan


admit = partial(native.admit, arm='A')
check_growth = partial(native.check_growth, arm='A')
retire_event = partial(native.retire_event, arm='A')
retire_redundant_features = native.retire_redundant_features
