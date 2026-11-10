"""Seal only the authorized disposable qualification, without model execution.

The prior final-boundary receipts supply the expected recovery/weight identities;
the A800 preflight must hash and decode those actual files before using them.
This entry point does not authorize or launch the events-5–6 pilot.
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
from .contract import (ANALYSIS, ARMS, BASE_COMMIT, DATASETS, DIAGNOSTIC_BUDGET,
                       MAIN_BUDGET, QUALIFICATION_BUDGET, REPORT_COMMIT, SCHEMA,
                       SERVING, geometry_contract)


WORKSPACE = Path('/home/minhao2/EAGLE_SPEC')
STUDY = WORKSPACE / 'research/dflash_functional_step_geometry_20261108'
PRIOR = WORKSPACE / 'research/dflash_role_loss_pilot_20261107'
REMOTE_CODE = Path('/data/sources/minhao/fsg_source_20261108/DFlash-Online')
REMOTE_ROOT = Path('/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/qualification_001')
REPOSITORY = 'https://github.com/ToxicMinhao/Online-Speculative-decoding-test'
PR17_COMMIT = '49b921ef95b744d373670c4f9a7b6a6917a9b6ef'
PR17_BLOB = 'c4c77be79fbf3ce6065e5a40c7d1eed783a291d4'
PR17_PATH = 'DFlash-Online/reports/dflash_krc_mechanism_diagnostic_20261028/REPORT.md'
PIN_FILES = ('ORIGINAL_CONFIG.yaml', 'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json')
PREPARATION_FILES = ('PREPARATION.json', 'INPUT_AUDIT.json', 'MATHQA_EXCLUSIONS.json')


def origin_from_status(status, *, dataset, prior_binding):
    """Use the exact authenticated event-4 boundary, not a guessed latest path."""
    boundary = status['boundary']
    scientific = boundary['scientific_check']
    live = scientific['live']
    if (status['status'] != 'completed' or status['dataset'] != dataset
            or status['arm'] != 'KPLUS' or status['binding'] != prior_binding
            or boundary['status'] != 'passed' or boundary['dataset'] != dataset
            or boundary['arm'] != 'KPLUS' or boundary['event'] != 4
            or scientific['status'] != 'passed' or scientific['global_step'] != 512
            or scientific['presentations'] != 1024 or not scientific['export_exact']
            or live['status'] != 'passed' or not live['exact_resume'] or live['mismatches']):
        raise ValueError('Require the authenticated complete dataset-specific PR32 KPLUS4 boundary')
    export = Path(status['final_checkpoint'])
    if (status['final_state'] != boundary['state']
            or Path(boundary['weights']) != export / 'model.safetensors'
            or export.parent.name != 'event_004'):
        raise ValueError('Final state/export do not identify the retained event-4 boundary')
    for key in ('state_identity', 'weights_identity'):
        item = boundary[key]
        if (set(item) != {'bytes', 'sha256'} or type(item['bytes']) is not int
                or item['bytes'] <= 0 or len(item['sha256']) != 64
                or any(character not in '0123456789abcdef' for character in item['sha256'])):
            raise ValueError('Retained boundary lacks a complete file identity')
    return dict(origin_state=boundary['state'], origin_export=str(export),
        origin_weights=boundary['weights'], origin_state_identity=deepcopy(boundary['state_identity']),
        origin_weights_identity=deepcopy(boundary['weights_identity']), origin_binding=prior_binding)


def published_source(code, commit):
    code = Path(code)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True)
    if head != commit or dirty:
        raise ValueError('Seal the exact requested clean published commit, after publication')
    return dict(commit=commit, dirty=False, source_path=str(REMOTE_CODE), repository=REPOSITORY)


def check_prepared(prepared):
    """Recheck the sealed preparation assertions before any qualification work."""
    prepared = Path(prepared)
    receipt = json.loads((prepared / 'PREPARATION.json').read_text())
    audit = json.loads((prepared / 'INPUT_AUDIT.json').read_text())
    if (receipt['schema'] != SCHEMA or receipt['status'] != 'sealed_inputs'
            or receipt['model_executions'] != 0 or receipt['training_requests'] != 512
            or receipt['events'] != [5, 6] or receipt['source_answers_used']
            or audit['status'] != 'passed' or audit['model_executions'] != 0
            or audit['selection_uses_model_results'] or audit['source_answers_used']):
        raise ValueError('Preparation must seal next inputs before model work')
    all_panels, prior_training = [], []
    for dataset in DATASETS:
        matches = [name for name in receipt['input_identities']
                   if name.endswith(f'/{dataset}/TRAINING.json')]
        if len(matches) != 1:
            raise ValueError('Preparation must identify each retained 512-request training stream')
        path = Path(matches[0])
        if identity(path) != receipt['input_identities'][str(path)]:
            raise ValueError('Retained training input bytes changed since preparation')
        previous = json.loads(path.read_text())['train_requests']
        if len(previous) != 512 or any(row['dataset'] != dataset for row in previous):
            raise ValueError('Missing or foreign PR32 training prefix')
        prior_training.extend(previous)
        panel = json.loads((prepared / dataset / 'PANEL.json').read_text())['requests']
        references = json.loads((prepared / dataset / 'CANONICALS.json').read_text())
        if len(panel) != 256 or set(references) != {row['request_id'] for row in panel}:
            raise ValueError('Require the complete unchanged 256-request panel and canonicals')
        all_panels.extend(panel)
    from experiments.role_loss_pilot.data import overlap_index, overlap_keys, overlap_reasons
    seen = overlap_index(all_panels + prior_training)
    for dataset in DATASETS:
        manifest = json.loads((prepared / dataset / 'TRAINING.json').read_text())
        rows = manifest['train_requests']
        if (manifest['dataset'] != dataset or len(rows) != 256
                or len({row['request_id'] for row in rows}) != 256
                or manifest['source_answers_used'] or manifest['selection_uses_outputs']
                or manifest['start_ordinal'] != 512 or manifest['stop_ordinal'] != 768):
            raise ValueError('Continuation inputs must be the next complete 256-request stream')
        for ordinal, row in enumerate(rows, 512):
            if (row['dataset'] != dataset or row['event_id'] != f'event_{ordinal//128+1:03d}'
                    or not row['collect_feedback'] or row['max_new_tokens'] != 2048
                    or not row['prompt_ids'] or len(row['prompt_ids']) + 2048 > 2272
                    or overlap_reasons(row, seen)):
                raise ValueError('Continuation input identity, event, overlap or context contract differs')
            for key, value in overlap_keys(row).items():
                seen[key].add(value)
    return receipt


def numerical_recipe():
    from online_feedback import functional_geometry_math as math
    from .native_oracles import NATIVE_VJP_TOLERANCES, ORACLE_BUDGET
    from .work import LIMITS
    return dict(geometry=geometry_contract(), geometry_probability_precision='float32',
        vector_chunk_size=math.VECTOR_CHUNK_SIZE,
        gram_roundoff_multiplier=math.GRAM_ROUNDOFF_MULTIPLIER,
        risk_roundoff_multiplier=math.RISK_ROUNDOFF_MULTIPLIER,
        expanded_relative_cutoff=math.EXPANDED_RELATIVE_CUTOFF,
        solver='symmetric_float64_direct_solve', damping=False,
        native_vjp_tolerances=dict(NATIVE_VJP_TOLERANCES),
        native_oracle_budget=dict(ORACLE_BUDGET), qualification_work_limits=dict(LIMITS),
        resource_fixture=dict(anchor_occurrences=512, native_gradient_vectors=4,
            construction='repeat_retained_anchor_bindings_in_order_without_sampling'))


def seal_evidence(code, sealed):
    """Resolve Appendix B's former branch link to one exact retained report blob."""
    revision = f'{PR17_COMMIT}:{PR17_PATH}'
    blob = subprocess.check_output(['git', 'rev-parse', revision], cwd=code, text=True).strip()
    if blob != PR17_BLOB:
        raise ValueError('The resolved PR17 mechanism report blob differs')
    content = subprocess.check_output(['git', 'show', revision], cwd=code)
    directory = sealed / 'evidence'
    directory.mkdir()
    path = directory / 'PR17_REPORT.md'
    path.write_bytes(content)
    evidence = dict(pr17=dict(commit=PR17_COMMIT, blob=blob, repository_path=PR17_PATH,
        local_report='evidence/PR17_REPORT.md', identity=identity(path),
        url=f'{REPOSITORY}/blob/{PR17_COMMIT}/{PR17_PATH}'),
        pr32_report_commit=REPORT_COMMIT, qualified_source_commit=BASE_COMMIT,
        pr21_report_commit='649e0b34066e217a3f918d82012d2a631c447ad0')
    write_json(sealed / 'EVIDENCE_MANIFEST.json', evidence)
    return evidence


def seal(args):
    code, prepared, prior, retained, output = map(Path,
        (args.code, args.prepared, args.prior_sealed, args.retained, args.output))
    execution = published_source(code, args.commit)
    preparation = check_prepared(prepared)
    old = json.loads((prior / 'PLAN.json').read_text())
    origins, status_files = {}, {}
    for dataset in DATASETS:
        path = retained / 'campaign_001/trajectories' / dataset / 'KPLUS/STATUS.json'
        origins[dataset] = origin_from_status(json.loads(path.read_text()),
            dataset=dataset, prior_binding=old['binding'])
        status_files[dataset] = path
    inputs = {str(path.relative_to(prepared)): identity(path)
              for path in sorted(prepared.rglob('*.json'))}
    binding = stable_digest(dict(source=execution, prepared=inputs, origins=origins,
        geometry=geometry_contract(), scope='bounded_qualification_only'))
    output.mkdir(parents=True, exist_ok=False)
    sealed = output / 'sealed'
    sealed.mkdir()
    for name in PIN_FILES:
        shutil.copy2(prior / name, sealed / name)
    for name in PREPARATION_FILES:
        shutil.copy2(prepared / name, sealed / name)
    shutil.copy2(args.handoff, sealed / 'HANDOFF.md')
    evidence = seal_evidence(code, sealed)
    numerics = numerical_recipe()
    write_json(sealed / 'NUMERICAL_RECIPE.json', numerics)
    original_config = (sealed / 'ORIGINAL_CONFIG.yaml').read_text()
    datasets = {}
    for dataset in DATASETS:
        directory = sealed / dataset
        directory.mkdir()
        for name in ('TRAINING.json', 'PANEL.json', 'CANONICALS.json'):
            shutil.copy2(prepared / dataset / name, directory / name)
        shutil.copy2(status_files[dataset], directory / 'ORIGIN_STATUS.json')
        configs = {}
        for arm in ARMS:
            raw = yaml.safe_load(original_config)
            raw['run_id'] = f'fsg-qualification-{dataset}-{arm.lower()}'
            raw['output_dir'] = str(REMOTE_ROOT / 'qualification' / dataset / 'trainer')
            text = yaml.safe_dump(raw, sort_keys=False)
            validate_config_text(text, original_config)
            name = f'config_{arm}.yaml'
            (directory / name).write_text(text)
            configs[arm] = str(REMOTE_ROOT / 'sealed' / dataset / name)
        datasets[dataset] = dict(origins[dataset], configs=configs,
            training_manifest=str(REMOTE_ROOT / 'sealed' / dataset / 'TRAINING.json'),
            panel_manifest=str(REMOTE_ROOT / 'sealed' / dataset / 'PANEL.json'),
            references=str(REMOTE_ROOT / 'sealed' / dataset / 'CANONICALS.json'))
    plan = dict(schema=SCHEMA, root=str(REMOTE_ROOT), scope='bounded_qualification_only',
        main_pilot_authorized=False, automatic_retry=False, execution_source=execution,
        source_base_commit=BASE_COMMIT, prior_report_commit=REPORT_COMMIT,
        binding=binding, arms=list(ARMS), datasets=datasets,
        original_config=str(REMOTE_ROOT / 'sealed/ORIGINAL_CONFIG.yaml'),
        target=old['target'], draft=old['draft'], training_python=old['training_python'],
        worker_python=old['worker_python'], serving_gpu=0, training_gpu=1,
        geometry=geometry_contract(), serving=SERVING, analysis=ANALYSIS,
        qualification_budget=QUALIFICATION_BUDGET,
        numerical_recipe=numerics, evidence=evidence,
        qualification_feedback=dict(generations=4, teacher_preparations=4,
            allocation='first two new event-5 requests per dataset; retained within future 512 feedback budget'),
        future_pilot_budget=MAIN_BUDGET, future_diagnostic_budget=DIAGNOSTIC_BUDGET,
        preparation=dict(model_executions=preparation['model_executions'], source_inputs=inputs))
    write_json(sealed / 'PLAN.json', plan)
    files = {}
    for encoded in subprocess.check_output(['git', 'ls-files', '-z', '.'], cwd=code).split(b'\0'):
        if encoded:
            name = encoded.decode()
            if (code / name).is_file():
                files[str(REMOTE_CODE / name)] = identity(code / name)
    for path in sorted(output.rglob('*')):
        if path.is_file():
            files[str(REMOTE_ROOT / path.relative_to(output))] = identity(path)
    write_json(sealed / 'SEAL.json', dict(execution_source=execution, files=files,
        scope='bounded_qualification_only',
        origin_identity_validation='actual remote file hashes, complete-state decode and export equality required by preflight'))
    return dict(status='sealed_before_model_work', execution_source=execution, binding=binding,
        scope=plan['scope'], files=len(files), qualification_budget=QUALIFICATION_BUDGET,
        shared_feedback_generations=4, main_pilot_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True, help='Exact clean published execution commit')
    parser.add_argument('--code', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--prepared', default=str(STUDY / 'prepared_001'))
    parser.add_argument('--prior-sealed', default=str(PRIOR / 'production_001/sealed'))
    parser.add_argument('--retained', default=str(PRIOR / 'result_review_001/retained'))
    parser.add_argument('--output', default=str(STUDY / 'qualification_001'))
    parser.add_argument('--handoff', default=str(STUDY / 'HANDOFF.md'))
    print(json.dumps(seal(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
