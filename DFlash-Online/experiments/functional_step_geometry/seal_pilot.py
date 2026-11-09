"""Seal the original conditional events-five/six pilot after completed R1 checks.

This is metadata-only preparation. Actual storage, origin decoding and native
runtime admission remain launch-time checks; qualification exports never become
pilot initialization. Local retrieval inventories are copied as evidence, never
invented as files at their original remote roots.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from experiments.ok_coding.storage import identity
from experiments.role_loss_pilot.contract import stable_digest
from experiments.role_loss_pilot.training import validate_config_text
from online_feedback.storage import write_json
from .contract import (ARMS, DATASETS, MAIN_BUDGET, DIAGNOSTIC_BUDGET,
                       REALIZATION_R1, geometry_contract)
from .evaluation import evaluation_schedule, validate_evaluation_inputs
from .native_pilot import DIAGNOSTIC_FACTORY, PILOT_STARTUP_BUDGET
from .pilot_storage import capacity_plan
from .qualify_realization import LIMITS, activity
from .seal import published_source

STUDY = Path('/home/minhao2/EAGLE_SPEC/research/dflash_functional_step_geometry_20261108')
REMOTE_ROOT = Path('/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/pilot_r1_001')
REMOTE_CODE = Path('/data/sources/minhao/fsg_source_20261109_pilot_r1/DFlash-Online')
SCOPE = 'conditional_r1_original_events_5_6_pilot'
OPERATIONAL_TIMEOUT = dict(seconds=48*3600, signal='TERM', kill_after_seconds=120, automatic_retry=False)
REFERENCE_PROOF = STUDY.parent/'dflash_role_loss_pilot_20261107/production_001/sealed'
REFERENCE_MIRRORS = {
    'gsm8k': (
        STUDY.parent/'dflash_online_osd_rerun_20261023/status_check_20261024/raw',
        Path('/data/sources/minhao/runs/dflash_online_osd_rerun_20261023')),
    'mathqa': (
        STUDY.parent/'dflash_mathqa_transfer_20261104/retrieved_002/production_002',
        Path('/data/sources/minhao/runs/dflash_mathqa_transfer_20261104/production_002')),
}


def read(path):
    return json.loads(Path(path).read_text())


def qualification_admission(plan, result, postflight, exit_code, previous):
    """Derive admission from completed raw receipts, including inactive controls."""
    if (plan.get('scope') != 'bf16_realization_r1_qualification'
            or plan.get('realization_revision') != REALIZATION_R1
            or plan.get('geometry') != geometry_contract(REALIZATION_R1)
            or plan.get('future_pilot_budget') != MAIN_BUDGET
            or plan.get('future_diagnostic_budget') != DIAGNOSTIC_BUDGET):
        raise ValueError('Qualification does not bind the unchanged R1/original pilot scope')
    commit = plan['execution_source']['commit']
    if (str(exit_code).strip() != '0' or result.get('status') != 'passed'
            or result.get('revision') != REALIZATION_R1
            or result.get('execution_commit') != commit
            or result.get('main_pilot_started') is not False
            or postflight.get('status') != 'passed' or postflight.get('postflight') is not True
            or postflight.get('execution_commit') != commit
            or previous.get('status') != 'passed' or result.get('limits') != LIMITS):
        raise ValueError('Completed qualification, postflight and exit evidence is required')
    for name, maximum in LIMITS.items():
        work = result['work']
        attempted, completed = work.get(name+'_attempted', 0), work.get(name+'_completed', 0)
        if (type(attempted) is not int or attempted != completed or attempted > maximum
                or work.get(name+'_failed', 0)
                or (name in ('adam','ordinary_forward','ordinary_backward','serving_generation')
                    and completed != maximum)):
            raise ValueError('Qualification work is failed, incomplete or over budget: '+name)
    if set(result['datasets']) != set(DATASETS):
        raise ValueError('Both qualified datasets are required')
    for dataset, item in result['datasets'].items():
        if ([case.get('case') for case in item['cases']] != ['FSG','SCALAR']
                or any(case.get('status') != 'passed' or case.get('optimizer_steps') != 1
                       or case.get('presentations') != 2 for case in item['cases'])):
            raise ValueError('Every original live FSG/SCALAR case must pass')
        serving = item['serving']
        if (serving.get('status') != 'passed' or serving.get('generations') != 2
                or serving.get('exact_output_and_termination') is not True):
            raise ValueError('Every R1 serving correctness check must pass')
    observed = activity(result['datasets'])
    if result.get('activity') != observed or not observed['mechanism_gate']:
        raise ValueError('Require observed nontrivial FSG activity and native SCALAR matching')
    prior = previous['datasets']
    formula = [prior['gsm8k']['oracles']['formula'],
               next(case['oracles'] for case in prior['mathqa']['cases'] if case['case']=='FSG')]
    tolerance = dict(rtol=0.016, atol=1e-5, maximum_relative_l2=0.016)
    if any(row.get('status') != 'passed' or row.get('numerical_protocol') != 'independent_same_native_layout'
           or row.get('tolerances') != tolerance or row['native_vjp'].get('status') != 'passed'
           for row in formula):
        raise ValueError('Both unchanged independent same-layout formula proofs must pass')
    if prior['gsm8k']['oracles']['resource'].get('status') != 'passed':
        raise ValueError('The original maximum-resource proof must pass')
    return dict(status='passed', pilot_permitted=True, realization_revision=REALIZATION_R1,
        checks=dict(native_formula_both_datasets=True, native_state=True, serving=True, resource=True),
        nontrivial_accepted_fsg_cases=sum(row['nontrivial_accepted'] for row in observed['fsg']),
        scalar_matching_exercised_cases=sum(row['scalar_matching_exercised'] and
            row['final_candidate_check'] in ('actual_bf16_replay','exact_native_risk_reuse')
            for row in observed['scalar']), activity=observed,
        qualification_execution_commit=commit, storage_admitted=False,
        historical_cross_layout='original failed records retained; not relabelled passes')


def validate_feedback(preparation, datasets, read_file=read):
    """The four already collected packets belong to the exact next shared pools."""
    if preparation.get('status') != 'completed' or set(preparation['datasets']) != set(DATASETS):
        raise ValueError('Completed four-packet preparation is required')
    for dataset in DATASETS:
        selected = [row for row in read_file(datasets[dataset]['training_manifest'])['train_requests']
                    if row['event_id'] == 'event_005']
        packets = preparation['datasets'][dataset]
        if (len(selected) != 128 or len(packets) != 2
                or [row['request_id'] for row in packets] != [row['request_id'] for row in selected[:2]]
                or any(row.get('status') != 'completed' for row in packets)):
            raise ValueError('Reuse exactly the first two original event-five packets per dataset')
        for packet in packets:
            for field in ('record','feature'):
                expected = packet[field+'_identity']
                if not packet[field+'_path'] or expected['bytes'] <= 0 or len(expected['sha256']) != 64:
                    raise ValueError('Reusable packet lacks its original file identity')
    return dict(total_feedback=512, reused_feedback=4, new_feedback=508,
                new_teacher_preparations=508, qualification_states_reused=False)


def verify_retrieval(retained, expected_root):
    """Authenticate retrieved files without registering the local inventory remotely."""
    manifest = read(retained/'COMPACT_RETRIEVAL_MANIFEST.json')
    if manifest['root'] != str(expected_root):
        raise ValueError('Retrieved qualification belongs to another root')
    for relative, expected in manifest['files'].items():
        path = retained/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or identity(path) != expected:
            raise ValueError('Qualification retrieval identity changed: '+relative)
    for required in ('qualification/QUALIFICATION.json','POSTFLIGHT.json','EXIT_CODE'):
        if required not in manifest['files']:
            raise ValueError('Missing authenticated qualification evidence: '+required)
    return manifest


def prepare_canonical_inputs(package, old, binding, reference_proof):
    """Rebind metadata to unchanged native evidence before creating the seal."""
    from .canonical_bindings import load_pr32_proof, bind_canonicals
    prepared = {}
    for dataset, item in old['datasets'].items():
        panel = read(package/Path(item['panel_manifest']).relative_to(old['root']))['requests']
        inherited = read(package/Path(item['references']).relative_to(old['root']))
        authenticated, identities, proof = load_pr32_proof(reference_proof,dataset,
            model_pin=package/'sealed/MODEL_PIN.json',runtime_pin=package/'sealed/RUNTIME_PIN.json')
        mirror, origin = REFERENCE_MIRRORS[dataset]
        references, native = bind_canonicals(panel,inherited,authenticated=authenticated,
            source_identities=identities,mirror=mirror,origin=origin,binding=binding)
        schedule = evaluation_schedule(panel,dataset)
        validate_evaluation_inputs(panel,{arm:'/future/'+dataset+'/'+arm for arm in ARMS},
            references,dataset=dataset,event=6,schedule=schedule)
        prepared[dataset] = dict(canonicals=references,proof=proof,native=native,schedule=schedule)
    return prepared


def copy_canonical_evidence(prepared, output):
    """Copy authenticated private sources; historical paths remain provenance."""
    total_bytes = 0
    for dataset, item in prepared.items():
        directory = output/'sealed/reference_evidence'/dataset
        receipt = deepcopy(dict(proof=item['proof'],native=item['native']))
        for kind, rows in (('proof',receipt['proof']['files']),('native',receipt['native']['records'])):
            for ordinal, row in enumerate(rows):
                source = Path(row['local_path'])
                name = f'{ordinal:04d}_'+source.name if kind=='proof' else f'{ordinal:04d}.json'
                destination = directory/kind/name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,destination)
                if identity(destination) != row['identity']:
                    raise ValueError('Canonical evidence changed while copying: '+str(source))
                row['retained_path'] = str(REMOTE_ROOT/destination.relative_to(output))
                total_bytes += row['identity']['bytes']
        write_json(directory/'BINDING.json',receipt)
    return total_bytes


def seal(code, commit, output, *, qualification=None, prior=None, reference_proof=None):
    code, output = Path(code), Path(output)
    qualification = Path(qualification or STUDY/'bf16_realization_r1_001')
    prior = Path(prior or STUDY/'qualification_resume_001')
    package, retained = qualification/'sealed_package', qualification/'retained'
    old = read(package/'sealed/PLAN.json'); old_root = Path(old['root'])
    old_seal = read(package/'sealed/SEAL.json')
    execution = published_source(code, commit); execution['source_path'] = str(REMOTE_CODE)
    # Confirm the commit is published, beyond the clean-checkout assertion.
    refs = subprocess.check_output(['git','for-each-ref','--format=%(refname)',
                                    '--contains',commit,'refs/remotes/origin'],cwd=code,text=True)
    if not refs.strip():
        raise ValueError('The exact execution commit must be present in a published origin ref')
    retrieval = verify_retrieval(retained, old_root)
    for path in (package/'sealed').rglob('*'):
        if path.is_file() and path.name != 'SEAL.json':
            if identity(path) != old_seal['files'][str(old_root/path.relative_to(package))]:
                raise ValueError('Original sealed inputs changed: '+str(path))
    previous_path = prior/'retained/qualification/QUALIFICATION.json'
    previous_remote = str(Path(old['supplementary_reuse']['prior_root'])/'qualification/QUALIFICATION.json')
    if identity(previous_path) != old_seal['files'][previous_remote]:
        raise ValueError('Inherited formula/resource proof identity changed')
    admission = qualification_admission(old, read(retained/'qualification/QUALIFICATION.json'),
        read(retained/'POSTFLIGHT.json'), (retained/'EXIT_CODE').read_text(), read(previous_path))
    preparation = read(retained/'qualification_feedback/PREPARATION.json')
    def old_input(name):
        return read(package/Path(name).relative_to(old_root))
    feedback = validate_feedback(preparation, old['datasets'], old_input)
    geometry = geometry_contract(REALIZATION_R1)
    binding = stable_digest(dict(qualification_binding=old['binding'], geometry=geometry,
        main_budget=MAIN_BUDGET, diagnostic_budget=DIAGNOSTIC_BUDGET,
        qualification_result=identity(retained/'qualification/QUALIFICATION.json')))
    canonical_inputs = prepare_canonical_inputs(package,old,binding,Path(reference_proof or REFERENCE_PROOF))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(package/'sealed',output/'sealed',ignore=shutil.ignore_patterns('PLAN.json','SEAL.json'))
    evidence = output/'sealed/qualification_evidence'
    shutil.copytree(retained,evidence/'r1')
    shutil.copy2(previous_path,evidence/'PREVIOUS_QUALIFICATION.json')
    canonical_evidence_bytes = copy_canonical_evidence(canonical_inputs,output)
    plan = deepcopy(old)
    plan.update(root=str(REMOTE_ROOT), scope=SCOPE, execution_source=execution,
        binding=binding, original_config=str(REMOTE_ROOT/'sealed/ORIGINAL_CONFIG.yaml'),
        geometry=geometry, realization_revision=REALIZATION_R1, events=[5,6],
        arms=list(ARMS), main_budget=deepcopy(MAIN_BUDGET), diagnostic_budget=deepcopy(DIAGNOSTIC_BUDGET),
        startup_budget=deepcopy(PILOT_STARTUP_BUDGET), diagnostic_factory=DIAGNOSTIC_FACTORY,
        operational_timeout=deepcopy(OPERATIONAL_TIMEOUT),
        pilot_admission=str(REMOTE_ROOT/'sealed/PILOT_ADMISSION.json'),
        shared_feedback_preparation=str(REMOTE_ROOT/'sealed/qualification_evidence/r1/qualification_feedback/PREPARATION.json'),
        feedback_accounting=feedback, canonical_evidence_bytes=canonical_evidence_bytes, automatic_retry=False,
        main_pilot_authorized='completed_R1_activity_and_correctness; actual_storage_still_required')
    plan['main_storage'] = capacity_plan(
        max(item['origin_weights_identity']['bytes'] for item in old['datasets'].values()),
        max(item['origin_state_identity']['bytes'] for item in old['datasets'].values()))
    for dataset, item in plan['datasets'].items():
        for key in ('training_manifest','panel_manifest','references'):
            item[key] = str(REMOTE_ROOT/Path(item[key]).relative_to(old_root))
        item['evaluation_schedule'] = canonical_inputs[dataset]['schedule']
        write_json(output/'sealed'/dataset/'CANONICALS.json',canonical_inputs[dataset]['canonicals'])
        item['canonical_binding_evidence'] = str(REMOTE_ROOT/'sealed/reference_evidence'/dataset/'BINDING.json')
        for arm, path in item['configs'].items():
            relative = Path(path).relative_to(old_root); local = output/relative
            cfg = yaml.safe_load(local.read_text())
            cfg['output_dir'] = str(REMOTE_ROOT/'campaign_001'/dataset/arm)
            text = yaml.safe_dump(cfg,sort_keys=False)
            validate_config_text(text,(output/'sealed/ORIGINAL_CONFIG.yaml').read_text())
            local.write_text(text); item['configs'][arm] = str(REMOTE_ROOT/relative)
    admission.update(execution_commit=commit,binding=binding,
        qualification_result=str(REMOTE_ROOT/'sealed/qualification_evidence/r1/qualification/QUALIFICATION.json'),
        qualification_postflight=str(REMOTE_ROOT/'sealed/qualification_evidence/r1/POSTFLIGHT.json'))
    write_json(output/'sealed/PILOT_ADMISSION.json',admission)
    write_json(output/'sealed/PLAN.json',plan)
    files = dict(old_seal['files'])
    # Only retrieved file rows attest original remote presence. The generated
    # local retrieval manifest exists remotely only in our newly copied package.
    for relative, expected in retrieval['files'].items():
        files[str(old_root/relative)] = expected
    for rows in preparation['datasets'].values():
        for row in rows:
            for field in ('record','feature'):
                files[row[field+'_path']] = row[field+'_identity']
    for encoded in subprocess.check_output(['git','ls-files','-z','.'],cwd=code).split(b'\0'):
        if encoded:
            path = code/encoded.decode()
            if path.is_file(): files[str(REMOTE_CODE/path.relative_to(code))] = identity(path)
    for path in sorted(output.rglob('*')):
        if path.is_file(): files[str(REMOTE_ROOT/path.relative_to(output))] = identity(path)
    write_json(output/'sealed/SEAL.json',dict(execution_source=execution,files=files,
        scope=SCOPE,qualification_binding=old['binding'],binding=binding,
        main_budget=MAIN_BUDGET,diagnostic_budget=DIAGNOSTIC_BUDGET,
        historical_failures_preserved=True,actual_storage_admitted=False))
    return dict(status='sealed',root=str(REMOTE_ROOT),source=execution,binding=binding,
                files=len(files),feedback_accounting=feedback,storage=plan['main_storage'],
                canonical_evidence_bytes=canonical_evidence_bytes,
                qualification_admitted=True,actual_storage_admitted=False,model_work=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--commit',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--qualification'); parser.add_argument('--prior')
    parser.add_argument('--reference-proof')
    args = parser.parse_args()
    print(json.dumps(seal(args.code,args.commit,args.output,
        qualification=args.qualification,prior=args.prior,reference_proof=args.reference_proof),indent=2))
