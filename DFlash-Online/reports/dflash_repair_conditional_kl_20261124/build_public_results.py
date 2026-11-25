"""Build public scalar evidence from the private, verified qualification receipts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plan = read(source / 'sealed/PLAN.json')
    launch = read(source / 'LAUNCH.json')
    audit = read(source / 'FINAL_CPU_AUDIT.json')
    assert launch['status'] == 'completed' and audit['status'] == 'passed'
    assert audit['cuda_initialized'] is False
    native = read(source / 'NATIVE_SCALAR_RECORDS.json')
    results, observations, artifacts, costs, retirement = {}, [], {}, {}, []
    for arm in ('BASE', 'PAIR'):
        report = read(source / 'qualification' / arm / 'TRAINING_QUALIFICATION.json')
        status = read(source / 'qualification_inputs' / arm / 'STATUS.json')
        assert report['status'] == status['status'] == 'passed'
        response_rows = [row['response_observation'] for row in report['records']]
        ratios = {name: sum(row['ratio_metrics'].get(name, [0, 1])[0]
                          / row['ratio_metrics'].get(name, [0, 1])[1]
                          for row in response_rows) / len(response_rows)
                  for name in ('forward_kl', 'repair_pair_auxiliary', 'objective_loss')}
        totals = {name: sum(row['sum_metrics'].get(name, 0) for row in response_rows)
                  for name in set().union(*(row['sum_metrics'] for row in response_rows))}
        repair_count = totals.get('repair_pair_repair_rows', 0)
        results[arm] = dict(
            status='passed', original_response_presentations=2,
            normalized_loss_means=ratios,
            pair_totals={k: v for k, v in totals.items() if k.startswith('repair_pair_')},
            mean_mu=totals.get('repair_pair_mu_sum', 0) / repair_count if repair_count else None,
            mean_complement=totals.get('repair_pair_complement_mass_sum', 0) / repair_count if repair_count else None,
            mean_conditional_kl=totals.get('repair_pair_conditional_kl_sum', 0) / repair_count if repair_count else None,
            active_fraction=totals.get('repair_pair_active_rows', 0) / repair_count if repair_count else None,
            ordinary_repair_rows=totals['kplus_priority1_error_count'] + totals['kplus_priority2_error_count'],
            available_position_occurrences=totals['available_position_occurrences'],
            selected_position_occurrences=totals['selected_position_occurrences'],
            sampled_blocks=int(totals['prefix_allocation_blocks']),
            gradient_norm=report['windows'][0]['gradient_norm'],
            clipped_windows=sum(row['clipped'] for row in report['windows']),
            optimizer_windows=len(report['windows']),
            gradient_contract=report['gradient_contract'],
            initialization=report['initialization'],
            exact_export_and_state_reload=report['exact_export_and_state_reload'],
            canonical_and_repeat_checks=status['canonical_and_repeat_checks'])
        for ordinal, row in enumerate(response_rows):
            observations.append(dict(arm=arm, presentation=ordinal, record_id=row['record_id'],
                loss=row['loss'], ratio_metrics=row['ratio_metrics'], sum_metrics=row['sum_metrics'],
                sampled_anchor_digest=hashlib.sha256(json.dumps(row['sampled_anchors'],
                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()))
        artifacts[arm] = dict(
            state=dict(path=report['state_path'], **report['state_identity'],
                       role='disposable one-step qualification state; no production event committed'),
            export=dict(path=report['export_path'] + '/model.safetensors', **report['export_identity']),
            feedback=[dict(
                path=plan['root'] + '/qualification_inputs/' + arm + '/loop/features/' + row['record_id'] + '.pt',
                **identity) for row, identity in zip(response_rows, report['feature_identities'])])
        startup = report['startup']
        physical = read(source / 'qualification_inputs' / arm / 'PHYSICAL_WORK.json')
        publications = [read(source / 'qualification_inputs' / arm / name)
                        for name in ('EXPORT_RELOAD_0.json', 'FROZEN_RELOAD.json', 'EXPORT_RELOAD_1.json')]
        assert all(not item['errors'] and item['weights']['all_equal'] for item in publications)
        costs[arm] = dict(
            phase_wall_s=status['wall_s'], native_loading_s=status['backend_loading_s'],
            training_qualification_wall_s=report['wall_s'],
            native_two_response_training_s=report['training_s'], startup_wall_s=startup['wall_s'],
            training_peak_allocated_bytes=report['peak_gpu_allocated_bytes'],
            training_peak_reserved_bytes=report['peak_gpu_reserved_bytes'],
            serving_peak_allocated_bytes=physical['peak_gpu_allocated_bytes'],
            serving_peak_reserved_bytes=physical['peak_gpu_reserved_bytes'],
            host_maxrss_bytes=report['host_maxrss_bytes'],
            learning_physical_work=report['physical_work'],
            startup_physical_work=startup['physical_work'],
            startup_work=startup['work'],
            native_initialization=physical,
            native_publications=[item['publication'] for item in publications],
            teacher_preparation=[item['scalars'] for item in native['teachers'] if item['arm'] == arm],
            pending_operations=read(source / 'qualification_inputs' / arm / 'WORK.json')['pending'])
        for path in (source / 'qualification_inputs' / arm / 'preparation_retirement').glob('*.json'):
            if path.name.endswith(('.plan.json', '.equivalence.json')):
                continue
            receipt = read(path)
            retirement.append(dict(arm=arm, removed=receipt['removed'],
                removed_bytes=receipt['removed_bytes'], retained=receipt['retained'],
                reason=receipt['reason'], free_before_bytes=receipt['free_before_bytes'],
                free_after_bytes=receipt['free_after_bytes'], lost_capability=receipt['lost_capability']))

    # Equal pre-update ordinary loss/allocation and anchors are independent of
    # the auxiliary term. Compare the retained original-forward observations.
    for first, second in zip(observations[:2], observations[2:]):
        assert first['sampled_anchor_digest'] == second['sampled_anchor_digest']
        assert first['ratio_metrics']['forward_kl'] == second['ratio_metrics']['forward_kl']
        for key in ('kplus_coefficient1', 'kplus_coefficient2', 'loss_denominator',
                    'prefix_allocation_delta_numerator'):
            assert first['sum_metrics'][key] == second['sum_metrics'][key]
    pair = results['PAIR']
    pair['auxiliary_relative_to_ordinary_loss'] = (
        pair['normalized_loss_means']['repair_pair_auxiliary']
        / pair['normalized_loss_means']['forward_kl'])
    assert len(native['records']) == 8 and all(row['canonical_valid'] for row in native['records'])
    for row in native['records']:
        assert row['native_accepted_draft_tokens'] == row['committed_progress'] - row['verification_calls']
    write(destination / 'MECHANISM.json', dict(scope='qualification original forwards only',
        response_loss_aggregation='mean of separately normalized response losses',
        repair_statistic_aggregation='sum across original-forward repair rows, excluding recomputation',
        arms=results, responses=observations))
    write(destination / 'COSTS.json', dict(
        execution_commit=plan['execution_commit'], launch=launch,
        final_resource=audit['resource'], arms=costs,
        startup_and_training_timers_are_nested=True,
        publication_timers_include_internal_cache_maintenance=True,
        export_and_serialization_not_separately_timed=True,
        claim='One step per arm, different cards and serial order; no reliable production overhead estimate.'))
    write(destination / 'ARTIFACTS.json', dict(
        execution_commit=plan['execution_commit'], run_root=plan['root'], arms=artifacts,
        raw_native_records=[row['raw_record'] for row in native['records']],
        compact_retrieval=read(source / 'RETRIEVAL_MANIFEST.json'),
        new_duplicate_retirements=retirement,
        preexisting_artifacts_deleted=0))
    write(destination / 'IDENTITIES.json', dict(
        base_commit=plan['base_commit'], execution_commit=plan['execution_commit'],
        execution_source_manifest_identity=read(source / 'SOURCE_MANIFEST_IDENTITY.json'),
        plan_binding=plan['binding'], source_manifest=plan['source_manifest'],
        source_identities=plan['source_identities'], model_identities=plan['model_identities'],
        sealed_files=plan['sealed_files'], canonical_provenance=plan['canonical_provenance'],
        serving=plan['serving'], devices=plan['devices'],
        runtime_pin=read(source / 'sealed/RUNTIME_PIN.json'),
        training_pin=read(source / 'sealed/TRAINING_PIN.json'),
        resource_plan=plan['storage'], forecast=plan['forecast'],
        production_blocker=plan['production_blocker']))
    write(destination / 'AUDIT.json', dict(
        native_cpu_recheck=audit, retrieved_receipts=79,
        original_forward_anchors_base_loss_coefficients_and_denominators_equal=True,
        native_accepted_progress_identity_passed=True,
        native_execution_failures=0, unknown_operations=0, qualification_attempts=1,
        preparatory_failures='CPU package/import and source-symlink checks were repaired before the first GPU action; original preparatory logs retained privately.',
        cpu_tests=read(source / 'CPU_TEST_SUMMARY.json'),
        production_status='not admitted before GPU work',
        unexecuted_production=plan['unexecuted_production_work']))
    fields = ['arm', 'phase', 'record_id', 'request_ordinal', 'prompt_tokens', 'output_tokens',
              'finish_reason', 'verification_calls', 'accepted_draft_tokens',
              'native_accepted_draft_tokens', 'committed_progress', 'canonical_valid',
              'request_latency_s', 'native_generate_s', 'cache_maintenance_s',
              'native_accepted_per_call', 'stop_clipped_matching_prefix_per_call',
              'committed_progress_per_call']
    with (destination / 'QUALIFICATION_REQUESTS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for raw in native['records']:
            row = {key: raw[key] for key in fields if key in raw}
            row.update(raw['timings'])
            row['cache_maintenance_s'] = raw['capture_timings']['cache_maintenance_s']
            row['native_accepted_per_call'] = raw['native_accepted_draft_tokens'] / raw['verification_calls']
            row['stop_clipped_matching_prefix_per_call'] = raw['accepted_draft_tokens'] / raw['verification_calls']
            row['committed_progress_per_call'] = raw['committed_progress'] / raw['verification_calls']
            writer.writerow(row)
    write(destination / 'ENDPOINTS.json', dict(
        declared_mode='qualification_only', efficacy='unassessed',
        candidate_one_event=dict(status='not admitted before GPU work', BASE=None, PAIR=None,
            call_reduction=None, latency_reduction=None, confidence_intervals=None,
            repaired_boundaries=None, damaged_prefixes=None, net_prefix_progress=None),
        two_event_default=dict(status='not admitted before GPU work', event1=None, event2=None),
        reason=plan['production_blocker']))
    print(json.dumps(dict(pair=pair, clocks=launch['clock'],
        retired_new_duplicate_bytes=sum(row['removed_bytes'] for row in retirement)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)
