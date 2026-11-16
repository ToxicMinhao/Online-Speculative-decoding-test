"""One shared work, time and storage admission for both physical cards."""
from __future__ import annotations

from dataclasses import dataclass
import math

CARD_SECONDS = 14_400
ELAPSED_SECONDS = 14_400
ALLOCATION_BYTES = 110_000_000_000
FREE_RESERVE_BYTES = 400_000_000_000
HOST_HEADROOM_BYTES = 32 * 2**30
LIMITS = {
    'adam': 648,
    'response_forward': 1310,
    'response_backward': 1310,
    'feedback_generation': 384,
    'reference_generation': 128,
    'qualification_generation': 8,
    'warmup_generation': 72,
    'scored_generation': 1152,
    'teacher_preparation': 384,
    'training_initialization': 7,
}


def storage_plan(prompt_lengths, *, weight_bytes, recovery_bytes,
                 hidden_size, captured_layers, prior_allocation_bytes=0):
    """Bound all three own-feedback pools at the fixed output cap.

    Only exact redundant native capture packets may be retired after their
    prepared replacement passes tensor/identity checks. All prepared features
    remain. Publication is serialized, so one recovery/export atomic overlap
    is admitted at a time. Four recovery payloads conservatively cover an old
    E7/S7 endpoint until its checked successor commits; original K6/N7 is not
    counted as new and never retired.
    """
    if type(prior_allocation_bytes) is not int or prior_allocation_bytes < 0:
        raise ValueError('Prior allocation must be a nonnegative integer byte count')
    lengths = list(prompt_lengths)
    if len(lengths) != 128 or any(type(n) is not int or n < 1 or n+2048 > 4096 for n in lengths):
        raise ValueError('Storage requires all128 admissible sealed event8 prompts')
    if min(weight_bytes, recovery_bytes, hidden_size, captured_layers) <= 0:
        raise ValueError('Tensor dimensions and measured file sizes must be positive')
    row_bytes = hidden_size * (captured_layers+1) * 2 + 12
    # Long masks and token IDs plus BF16 auxiliary/final teacher features.
    prepared = 3 * sum(n+2048 for n in lengths) * row_bytes
    # Both collection workers may overlap one source packet and one replacement.
    temporary_features = 2 * (max(lengths)+2048+1) * row_bytes
    checkpoints = 4*recovery_bytes + 5*weight_bytes
    publication_overlap = recovery_bytes + weight_bytes
    payload = prepared + temporary_features + checkpoints + publication_overlap
    allowance = dict(serialization_margin_bytes=math.ceil(payload*.05),
                     compact_bytes=5_000_000_000, compiler_bytes=4*2**30,
                     source_and_workspace_bytes=2*2**30,
                     prior_allocation_bytes=prior_allocation_bytes)
    peak = payload + sum(allowance.values())
    return dict(status='passed' if peak <= ALLOCATION_BYTES else 'blocked',
                peak_new_bytes=peak, allocation_cap_bytes=ALLOCATION_BYTES,
                prepared_feature_bytes=prepared, temporary_feature_bytes=temporary_features,
                checkpoint_bytes=checkpoints, atomic_publication_bytes=publication_overlap,
                row_bytes=row_bytes, **allowance,
                required_initial_free_bytes=peak+FREE_RESERVE_BYTES,
                publication_concurrency=1, unique_features_retained=True)


def time_admission(*, occupied_spent_s, remaining_components, elapsed_spent_s,
                   remaining_schedule_s):
    """Apply the prespecified margin to a full, source-priced remaining plan."""
    values = [occupied_spent_s, elapsed_spent_s, remaining_schedule_s,
              *remaining_components.values()]
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('Missing, negative or nonfinite time estimate')
    remaining = sum(remaining_components.values())
    admitted_card = occupied_spent_s + 1.5*remaining + 1800
    admitted_elapsed = elapsed_spent_s + 1.5*remaining_schedule_s + 1800
    return dict(status='passed' if admitted_card <= CARD_SECONDS and
                admitted_elapsed <= ELAPSED_SECONDS else 'blocked',
                occupied_spent_s=occupied_spent_s, raw_remaining_card_s=remaining,
                conservative_total_card_s=admitted_card,
                conservative_total_elapsed_s=admitted_elapsed,
                remaining_components=dict(remaining_components),
                card_cap_s=CARD_SECONDS, elapsed_cap_s=ELAPSED_SECONDS)


def admit_task_pair(*, available_host_bytes, new_working_sets, resident_counted=False):
    """New incremental working sets only; live residents are already deducted."""
    if any(x < 0 for x in new_working_sets):
        raise ValueError('Working-set bounds must be nonnegative')
    required = sum(new_working_sets) + HOST_HEADROOM_BYTES
    return dict(status='passed' if available_host_bytes >= required else 'serialize',
                available_host_bytes=available_host_bytes, required_bytes=required,
                resident_workers_already_in_memavailable=resident_counted)


@dataclass(frozen=True)
class Task:
    name: str
    dependencies: tuple[str, ...]
    estimated_card_s: float
    kind: str


def campaign_tasks(training_s, feedback_s, preparation_s, reference_s):
    """Explicit per-arm dependencies, without a global event7 barrier."""
    tasks = [Task(a+'7', (), training_s, 'train') for a in ('E', 'S')]
    for arm in ('N', 'E', 'S'):
        prior = () if arm == 'N' else (arm+'7',)
        tasks += [Task(arm+'_feedback', prior, feedback_s, 'feedback'),
                  Task(arm+'_prepare', (arm+'_feedback',), preparation_s, 'prepare'),
                  Task(arm+'8', (arm+'_prepare',), training_s, 'train')]
    tasks += [Task('references', (), reference_s, 'reference')]
    return tasks


def next_ready_task(tasks, *, completed, active):
    """Choose the longest remaining dependency path, with a stable name tie."""
    by_name = {task.name: task for task in tasks}
    def remaining_path(name):
        successors = [t.name for t in tasks if name in t.dependencies]
        return by_name[name].estimated_card_s + max(
            (remaining_path(child) for child in successors), default=0.)
    ready = [task for task in tasks if task.name not in completed|active
             and set(task.dependencies) <= completed]
    return min(ready, key=lambda t: (-remaining_path(t.name), t.name)) if ready else None
