"""Full-capacity admission; only this supplement's validated states earn credit."""
from functools import partial
from experiments.adaflash_c1.storage import RAW_BYTES_PER_GENERATION, GIB
from experiments.kplus_full_gsm8k import storage as native
from .contract import NEW_BUDGET


def capacity_plan(weight_bytes):
    plan = native.capacity_plan(weight_bytes)
    plan['compact_bytes'] = NEW_BUDGET['total_generations'] * RAW_BYTES_PER_GENERATION + 4 * GIB
    # Qualification retains no serving export. Its compact records fit the
    # metadata allowance; tensor workspace overlaps the larger online peak.
    plan.update(raw_bytes_per_generation=RAW_BYTES_PER_GENERATION,
                raw_budget_generations=NEW_BUDGET['total_generations'],
                metadata_and_log_allowance_bytes=4*GIB,
                qualification_optimizer_steps=6)
    plan['required_initial_free_bytes'] = plan['tensor_peak_with_margin_bytes'] + sum(
        plan[key] for key in ('compact_bytes','compiler_bytes','workspace_bytes','reserve_bytes'))
    return plan


admit = partial(native.admit, arm='AREPO')
check_growth = partial(native.check_growth, arm='AREPO')
retire_event = partial(native.retire_event, arm='AREPO')
retire_redundant_features = native.retire_redundant_features
