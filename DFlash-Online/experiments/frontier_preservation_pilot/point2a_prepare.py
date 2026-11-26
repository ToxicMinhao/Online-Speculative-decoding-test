"""Bind Point2A to the unchanged PR61 parent, window, controls and witnesses."""
import argparse
import json
from pathlib import Path
import shutil
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .point1_prepare import (
    PRODUCTION_PLAN_IDENTITY, WINDOW_IDENTITY, RECOVERY_IDENTITY, EXPORT_IDENTITY,
    PARENT_BINDING, authenticate_parent_export, authenticate_witness, read_bound_json,
    host_memory, gpu_inventory, resource_admission,
)

SCHEMA = 'point2a_c_referenced_scoring_v1'
HISTORICAL_CARD_S = 5510.8580226639515
CASES = ('C', 'F_BETA', 'F_C')
LIMITS = dict(elapsed_s=3600, occupied_card_s=3600, allocated_bytes=2_000_000_000,
              attempted_adam_calls=6, planned_adam_calls=3, correction_retries=1,
              reserve_bytes=400_000_000_000, oracle_backwards_per_attempt=30,
              oracle_backwards_total=60)
POINT1_PRIVATE_IDENTITY = dict(bytes=14742,sha256='f4e74fa6715ee6d3cea183e214ddb6232c27688bb993dd83b4576df507f07163')
POINT1_PUBLIC_IDENTITIES = dict(bytes=208270,sha256='d71e512ee7c5aac40ef5f1daae7a9b301c4fc2cae0f744de74e081fad09e61d3')
CONVERGENCE = dict(
    identical_allocation_or_complete_endpoint='close as ineffective on this case',
    identical_parameters_different_moments='park; no immediate prediction effect, not equivalence',
    changed_coefficients_no_native_progress='do not promote; park',
    local_gain_disjoint_tie_or_conflict='inconclusive transfer; prioritize assessment/transfer decision',
    gain_against_both_with_disjoint_support='promising for proposed bounded serving screen, no automatic run',
    regression_without_disjoint_benefit='drop reference change; retain working recipe',
    correctness_or_resource_failure='preserve completed work; report blocker',
)


def historical_controls(point1_root, published_identities):
    """Use full private case receipts authenticated by the published identity table."""
    result = {}
    for current, previous in (('C','C'),('F_BETA','F')):
        relative = f'attempt_001/native/{previous}.json'
        expected = published_identities['raw_input_receipts'][relative]
        path = Path(point1_root)/relative
        raw = read_bound_json(path, expected)
        published = published_identities['cases'][previous]
        if (raw['status'] != 'completed' or raw['endpoint'] != published['endpoint']
                or raw['gradient_identity'] != published['gradient_identity']):
            raise ValueError('Historical control does not match published scientific identity')
        result[current] = dict(path=str(path), identity=expected)
    return result


def prepare(production_plan, point1_root, root, source_manifest, execution_commit,
            local_allocated_bytes, archive_manifest, remaining_write_budget_bytes=1_000_000_000):
    from experiments.kplus_full_gsm8k.storage import compiler_snapshot
    from experiments.bounded_prefix_allocation.runtime import allocated_bytes
    from .native import select_parent
    from .qualify import verify_source
    from .point2a_math import point2a_numerical_contract

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[2]
    production = read_bound_json(production_plan, PRODUCTION_PLAN_IDENTITY)
    previous_path = Path(point1_root)/'CONTRACT.json'
    previous = read_bound_json(previous_path, POINT1_PRIVATE_IDENTITY)
    public_path = source_root/'reports/dflash_point1_update_fidelity_20261125/diagnostic_001/IDENTITIES.json'
    published = read_bound_json(public_path, POINT1_PUBLIC_IDENTITIES)
    controls = historical_controls(point1_root, published)
    parent, recovery, selected = select_parent(production['parent_plan'])
    if (parent['binding'] != PARENT_BINDING or selected != previous['parent_features']
            or previous['execution_commit'] != '269f05644eeed2f0396602c47beb2a71166f8dd0'
            or identity(production['parent_plan']) != previous['parent_plan_identity']):
        raise ValueError('Original Point1 parent or qualification features changed')
    export = authenticate_parent_export(production['parent_export'])
    if identity(recovery) != RECOVERY_IDENTITY:
        raise ValueError('Protected parent recovery changed')
    for item in selected:
        if identity(item['path']) != item['identity']:
            raise ValueError('Original qualification feature bytes changed')
    window_path = Path(previous['original_window'])
    window = read_bound_json(window_path, WINDOW_IDENTITY)
    if [len(row['anchors'][0]) for row in window['responses']] != [512,269]:
        raise ValueError('Original anchor bindings changed')
    witness = dict(status='unavailable',scope='qualification_window_only')
    try:
        witness = authenticate_witness(production['root'],archive_manifest,selected,window)
    except (OSError,ValueError,KeyError) as error:
        witness['reason'] = repr(error)
    source_manifest = Path(source_manifest)
    plan = dict(schema=SCHEMA,root=str(root),execution_commit=execution_commit,
        source_root=str(source_root),source_manifest=str(source_manifest),source_identity=identity(source_manifest),
        production_plan=str(production_plan),production_plan_identity=PRODUCTION_PLAN_IDENTITY,
        point1_private_contract=str(previous_path),point1_private_contract_identity=POINT1_PRIVATE_IDENTITY,
        point1_public_identities=POINT1_PUBLIC_IDENTITIES,historical_controls=controls,
        parent_plan=production['parent_plan'],parent_plan_identity=identity(production['parent_plan']),
        parent_binding=PARENT_BINDING,parent_recovery=str(recovery),parent_recovery_identity=RECOVERY_IDENTITY,
        parent_export=str(export),parent_export_identity=EXPORT_IDENTITY,parent_features=selected,
        original_probe=previous['original_probe'],original_window=str(window_path),original_window_identity=WINDOW_IDENTITY,
        witness=witness,cases=list(CASES),primary_contrast='F_C-F_BETA',working_contrast='F_C-C',
        context_contrast='F_BETA-C',native_steps_independently_restored=True,
        parent_counters=[128,256,1],case_counters=[129,258,1],
        scientific_change='scoring reference gradient g_beta -> actual native preclip g_C',
        allocator='unchanged beta-centered, beta-budget, beta-cap0.25 allocator',
        live_precision='native BF16',diagnostic_product_reduction='FP64',shadow_precision='FP32',
        objective_changes=False,full_events=0,generation_calls=0,
        oracle_selection='lowest-index prefix pool with at least two positions per response, before scores',
        expected_oracle_pool_lengths=[13,15],planned_oracle_backwards=28,
        numerical_criteria=point2a_numerical_contract(),
        prospective_convergence=CONVERGENCE,prior_occupied_card_s=HISTORICAL_CARD_S,limits=dict(LIMITS),
        devices=production['devices'],training_python=production['training_python'],compiler_baseline=compiler_snapshot(),
        local_allocated_bytes=int(local_allocated_bytes),charged_remote_roots=[str(root),str(source_manifest.parent)],
        remaining_write_budget_bytes=int(remaining_write_budget_bytes),created_epoch=time.time(),status='preparing',
        old_hidden_gate_status='failed_retained_diagnostic')
    verify_source(plan)
    memory,gpu=host_memory(),gpu_inventory()
    card=next(row for row in gpu['cards'] if row['uuid']==plan['devices']['B'])
    charged=allocated_bytes(plan['charged_remote_roots'])+plan['local_allocated_bytes']
    admission=resource_admission(free_bytes=shutil.disk_usage(root).free,charged_bytes=charged,
        remaining_write_bytes=plan['remaining_write_budget_bytes'],host_available_bytes=memory['available_bytes'],
        gpu_free_bytes=card['total_bytes']-card['used_bytes'],
        gpu_busy=any(plan['devices']['B'] in row for row in gpu['processes'].splitlines()))
    plan.update(status='admitted' if admission['admitted'] else 'resource_blocked',resource_admission=admission,
                host_memory=memory,gpu_inventory=gpu)
    write_json(root/'CONTRACT.json',plan)
    if not admission['admitted']:
        raise RuntimeError('Point2A bounded resource admission failed')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('production-plan','point1-root','root','source-manifest','execution-commit','archive-manifest'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--local-allocated-bytes',required=True,type=int)
    parser.add_argument('--remaining-write-budget-bytes',default=1_000_000_000,type=int)
    result=prepare(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'],witness=result['witness']['status'],admission=result['resource_admission'])))
