"""Price the whole fixed campaign using source metadata and native qualification."""
from __future__ import annotations

import math
from pathlib import Path

from .campaign import task_graph, planned_elapsed, current_memory
from .resources import HOST_HEADROOM_BYTES, time_admission
from .task_runner import clock_status, read


def qualify_resources(plan, training, serving):
    basis = read(plan['historical_cost_basis'])
    costs = dict(basis['rates'])
    scales = basis['data_scale']
    if not all(math.isfinite(value) and value >= 0 for value in costs.values()):
        raise ValueError('Historical fixed-work cost basis is incomplete')
    cases = [row for card in training.values() for row in card['qualification']['cases']]
    # The all-parameter preclip hash is qualification observation overhead. It
    # has already been charged to spent time and does not recur at each step.
    pair_s = max(row['pair_wall_excluding_preclip_observer_s'] for row in cases)
    costs['training_event'] = max(costs['training_event'], 128*pair_s)*scales['training']
    costs['training_startup'] = max(costs['training_startup'],
                                   max(row['startup_s'] for row in training.values()))
    costs['training_load'] = max(costs['training_load'],
                                max(row['loading_s'] for row in training.values()))
    costs['serving_load'] = max(costs['serving_load'],
                               max(row['loading_s'] for row in serving.values()))
    rows = [row for report in serving.values() for row in report['qualification']['rows']]
    # Independent warmup references give a current hardware price per output
    # token. This is an estimator, not a worst-case runtime guarantee.
    native_s_per_token = max(row['request_latency_s']/max(1,row['output_tokens']) for row in rows)
    metadata = basis['current_panel_metadata']
    known = metadata['selected_historical_frozen']['output_tokens']
    missing = metadata['missing_fixed_K6_reference_count']
    # Unknown selected-output lengths use the predeclared historical selected
    # panel's p95. Actual output caps remain2048, and time gates remain hard.
    expected_panel_length = (known['total']+missing*known['p95'])/128
    expected_feedback_length = metadata['event7_retained']['output_tokens']['mean']
    costs['feedback_response'] = max(costs['feedback_response']*scales['feedback'],
                                    native_s_per_token*expected_feedback_length)
    costs['teacher_response'] *= scales['feedback']
    costs['reference_response'] = max(costs['reference_response']*scales['reference'],
                                     native_s_per_token*known['p95'])
    costs['scored_request'] = max(costs['scored_request']*scales['scored'],
                                 native_s_per_token*expected_panel_length)
    costs['warmup_request'] = max(costs['warmup_request']*scales['warmup'],
                                 max(row['request_latency_s'] for row in rows))
    graph = task_graph(costs, missing)
    clock = clock_status(plan['root'])
    training_host = max(row['sampled_process_tree_peak_rss_bytes'] for row in training.values()) + 12*2**30
    serving_host = max(row['sampled_process_tree_peak_rss_bytes'] for row in serving.values()) + 4*2**30
    training_gpu = max(row['peak_gpu_reserved_bytes'] for row in training.values()) + 2*2**30
    serving_gpu = max(max(row['gpu_memory_after_loading']['used_bytes'],
                          row['gpu_memory_after_generations']['used_bytes'],
                          row['graph_and_physical_work']['peak_gpu_reserved_bytes'])
                      for row in serving.values()) + 4*2**30
    host = dict(training=training_host, collection=serving_host,
                reference=serving_host, final=serving_host)
    vram = dict(training=training_gpu, collection=serving_gpu,
                reference=serving_gpu, final=serving_gpu)
    available = current_memory()
    two_trainers = 2*training_host+HOST_HEADROOM_BYTES <= available
    mixed = training_host+serving_host+HOST_HEADROOM_BYTES <= available
    two_serving = 2*serving_host+HOST_HEADROOM_BYTES <= available
    # If no two safe working sets fit, price elapsed time serially. The actual
    # queue rechecks live RAM and VRAM before every task admission.
    schedule = (planned_elapsed(graph) if two_trainers and mixed and two_serving
                else sum(t.estimated_card_s for t in graph))
    timing = time_admission(occupied_spent_s=clock['occupied_s'],
        remaining_components={task.name: task.estimated_card_s for task in graph},
        elapsed_spent_s=clock['elapsed_s'], remaining_schedule_s=schedule)
    memory_ok = max(host.values())+HOST_HEADROOM_BYTES <= available
    return dict(status='passed' if timing['status']=='passed' and memory_ok else 'blocked',
        execution_commit=plan['execution_commit'], costs=costs, time=timing,
        task_memory_bytes=host, task_vram_bytes=vram,
        available_host_bytes=available, two_trainers_admitted=two_trainers,
        mixed_training_serving_admitted=mixed,
        two_serving_workers_admitted=two_serving,
        host_bound_policy='Measured qualification process-tree peak plus12GiB trainer/publication or4GiB native capture/preparation; global32GiB reserve',
        historical_cost_basis=plan['historical_cost_basis'],
        storage_gate=plan['cpu_gate'], storage_checked_again_by_production_supervisor=True,
        measured_worst_pair_compute_s=pair_s, measured_native_s_per_output_token=native_s_per_token,
        projected_panel_output_mean=expected_panel_length,
        unknown_output_length_policy='Historical selected-reference p95 for24 missing panel answers; previous128-response event mean for prospective feedback. Estimates, not guaranteed lengths.',
        candidate_outcomes_used_for_pricing=False, safety_multiplier=1.5,
        additional_card_seconds_contingency=1800, scientific_settings_changed=False)
