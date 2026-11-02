"""Create the fixed Track2 plan from verified retained inputs, without model work."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from experiments.full_k_gsm8k.contract import SERVING, stable_digest
from experiments.ok_events_5_8.storage import identity
from online_feedback.storage import write_json
from .contract import ARMS, BASE_COMMIT, BUDGET, CODING_BINDING, EVENTS, SELECTION_SEED, final_schedule
from .data import build_inputs
from .storage import capacity_plan
from .training import validate_config_text

PIN_NAMES = ('MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json')
TARGET_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
DRAFT_REVISION = '9b41424b7109f9c5413454f481b09a82b85333f4'


def read(path):
    return json.loads(Path(path).read_text())


def verify_preflight(index_path):
    """Verify the receipt bundle and every selected retained source byte binding."""
    index_path = Path(index_path).absolute()
    index = read(index_path)
    if index['status'] != 'completed_read_only_preflight':
        raise ValueError('Require completed read-only preflight evidence')
    receipts = {}
    for row in index['files']:
        path = Path(row['path'])
        if path.parent.resolve() != index_path.parent.resolve():
            raise ValueError('Preflight index refers outside its own evidence directory')
        if identity(path) != {key: row[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Preflight receipt changed: ' + str(path))
        receipts[path.name] = read(path)
    required = {'INPUT_SOURCE_CHECK.json', 'CANONICAL_SOURCE_CHECK.json',
                'INPUT_DISJOINTNESS.json', 'REMOTE_INVENTORY.json',
                'CODING_CANONICAL_ORIGINAL.json', 'STORAGE_ADMISSION_PLAN.json'}
    if not required <= receipts.keys():
        raise ValueError('Incomplete preflight evidence bundle')
    sources = receipts['INPUT_SOURCE_CHECK.json']
    canonical = receipts['CANONICAL_SOURCE_CHECK.json']
    disjoint = receipts['INPUT_DISJOINTNESS.json']
    if (sources['status'] != 'passed' or sources['model_work'] != 0
            or sources['coding_data_binding'] != CODING_BINDING
            or canonical['status'] != 'passed' or canonical['errors']
            or canonical['model_work'] != 0 or disjoint['status'] != 'passed'):
        raise ValueError('Input or canonical preflight did not pass the fixed contract')
    for row in sources['checks']:
        if (row['old_receipt_match'] is not True or row['local_remote_match'] is not True
                or identity(row['local_path']) != row['identity']):
            raise ValueError('Retained input differs from its old receipt: ' + row['source'])
    if len(canonical['records']) != 256 or len(disjoint['records']) != 1280:
        raise ValueError('Preflight does not cover the complete fixed training/evaluation panel')
    return receipts


def prepare_inputs(receipts):
    """Decode the pinned tokenizer only; selection never reads answers or losses."""
    from tokenizers import Tokenizer
    sources = {row['source']: row for row in receipts['INPUT_SOURCE_CHECK.json']['checks']}
    expected = {'ORIGINAL_MANIFEST.json', 'CANONICALS.json',
                'TRAINING_REQUESTS.json', 'EVALUATION_REQUESTS.json'}
    if set(sources) != expected:
        raise ValueError('Require the four exact retained source inventories')
    disjoint = receipts['INPUT_DISJOINTNESS.json']
    if identity(disjoint['tokenizer_path']) != disjoint['tokenizer_identity']:
        raise ValueError('Pinned tokenizer bytes changed')
    tokenizer = Tokenizer.from_file(disjoint['tokenizer_path'])
    canonical_rows = receipts['CANONICAL_SOURCE_CHECK.json']['records']
    stop_sets = {tuple(row['stop_token_ids']) for row in canonical_rows}
    if stop_sets != {(151645, 151643)}:
        raise ValueError('Retained canonical stop semantics differ')
    manifest = read(sources['ORIGINAL_MANIFEST.json']['local_path'])
    if stable_digest(manifest) != receipts['INPUT_SOURCE_CHECK.json']['math_manifest_binding']:
        raise ValueError('Original math manifest binding differs')
    inputs = build_inputs(manifest, read(sources['TRAINING_REQUESTS.json']['local_path']),
        read(sources['EVALUATION_REQUESTS.json']['local_path']),
        math_canonicals=read(sources['CANONICALS.json']['local_path']),
        coding_canonicals=receipts['CODING_CANONICAL_ORIGINAL.json'],
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False), stop_ids=[151645, 151643])
    original_checks = {row['request_id']: row for row in disjoint['records']}
    for request in inputs['training'] + inputs['evaluation']:
        old = original_checks[request['request_id']]
        if (stable_digest(request['prompt_ids']) != old['token_digest']
                or request['normalized_rendered_sha256'] != old['rendered_normalized_sha256']
                or request['normalized_user_sha256'] != old['user_normalized_sha256']):
            raise ValueError('Selected prompt differs from verified disjointness evidence')
    canonical_checks = {row['request_id']: row for row in canonical_rows}
    for request in inputs['evaluation']:
        canonical = inputs['canonicals'][request['request_id']]
        old = canonical_checks[request['request_id']]
        if (canonical['source_record_path'] != old['source_record_path']
                or canonical['finish_reason'] != old['finish_reason']
                or len(canonical['response_ids']) != old['output_tokens']):
            raise ValueError('Selected canonical differs from the native source receipt')
    return inputs


def verify_original_recipe(receipts):
    sources = {row['source']: row for row in receipts['INPUT_SOURCE_CHECK.json']['checks']}
    local = Path(sources['ORIGINAL_MANIFEST.json']['local_path']).parent
    remote = Path(sources['ORIGINAL_MANIFEST.json']['remote_path']).parent
    inventory = {row['path']: row for row in receipts['REMOTE_INVENTORY.json']['files']}
    for name in ('PLAN.json', 'SEAL.json'):
        expected = inventory[str(remote / name)]
        if identity(local / name) != {key: expected[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Original full-K plan/seal changed: ' + name)
    original_seal = read(local / 'SEAL.json')
    for name in ('ORIGINAL_CONFIG.yaml', *PIN_NAMES):
        if identity(local / name) != original_seal['files'][str(remote / name)]:
            raise ValueError('Original recipe or model/runtime pin changed: ' + name)
    plan = read(local / 'PLAN.json')
    pins = read(local / 'MODEL_PIN.json')
    if (not pins['target']['source'].endswith('/' + TARGET_REVISION)
            or not pins['draft']['source'].endswith('/' + DRAFT_REVISION)
            or plan['serving'] != SERVING
            or read(local / 'RUNTIME_PIN.json')['python'] != plan['worker_python']):
        raise ValueError('Qualified model revision or serving runtime changed')
    for kind in ('target', 'draft'):
        current = receipts['REMOTE_INVENTORY.json']['models'][kind]
        if (current['path'] != plan[kind]
                or {key: current['config_identity'][key] for key in ('bytes', 'sha256')}
                   != pins[kind]['files']['config.json']):
            raise ValueError('Resource inventory model differs from pinned original')
    config_text = (local / 'ORIGINAL_CONFIG.yaml').read_text()
    validate_config_text(config_text, 'K', config_text)
    config = yaml.safe_load(config_text)
    if (config['model']['target_model_path'] != plan['target']
            or config['model']['draft_checkpoint_path'] != plan['draft']):
        raise ValueError('Original configuration model paths differ')
    return local, plan, pins


def execution_source(code, remote_code):
    code = Path(code)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True):
        raise ValueError('Publish the exact clean implementation before sealing')
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE_COMMIT, commit], cwd=code, check=True)
    return dict(commit=commit, dirty=False, source_path=str(remote_code),
                repository='https://github.com/ToxicMinhao/Online-Speculative-decoding-test',
                scientific_base=BASE_COMMIT)


def validate_plan(plan):
    if (plan['schema'] != 'k_workload_replay_v1' or plan['arms'] != list(ARMS)
            or plan['events'] != EVENTS or plan['training_seed'] != 17
            or plan['selection_seed'] != SELECTION_SEED or plan['budget'] != BUDGET
            or plan['serving'] != SERVING or plan['automatic_retry'] is not False
            or plan['execution_source']['dirty'] is not False
            or plan['binding'] != stable_digest(plan['binding_material'])):
        raise ValueError('Sealed plan differs from the fixed scientific contract')
    if set(plan['configs']) != set(ARMS) or plan['storage']['reserve_bytes'] != 8 * 2**30:
        raise ValueError('Incomplete arm configuration or changed storage reserve')
    return dict(status='passed', model_work=0, events=8, arms=list(ARMS), budget=BUDGET)


def seal(args):
    code, index, output = map(Path, (args.code, args.preflight, args.output))
    remote, remote_code = Path(args.remote_root), Path(args.remote_code)
    if not remote.is_absolute() or not remote_code.is_absolute():
        raise ValueError('Remote run and source roots must be absolute')
    source = execution_source(code, remote_code)
    receipts = verify_preflight(index)
    inputs = prepare_inputs(receipts)
    original_dir, original_plan, model_pins = verify_original_recipe(receipts)
    inventory = receipts['REMOTE_INVENTORY.json']
    # The independent preflight conservatively budgets serialized file bytes
    # per W, including the safetensors header; retain raw tensor bytes separately.
    weight_bytes = sum(row['file_bytes'] for row in inventory['models']['draft']['weights'])
    compiler_baseline = [{key: row[key] for key in ('path', 'allocated_bytes')}
                         for row in inventory['compiler_caches']]
    storage = capacity_plan(inputs['training'], weight_bytes, compiler_baseline=compiler_baseline)
    previous_storage = receipts['STORAGE_ADMISSION_PLAN.json']
    if (storage['required_initial_free_bytes'] != previous_storage['required_initial_free_bytes']
            or storage['tensor_peak_bytes'] != previous_storage['peak']['tensor_bytes']
            or storage['row_bytes'] != previous_storage['prepared_feature_row_bytes']):
        raise ValueError('Implementation capacity differs from the independently computed preflight')
    storage['preflight_full_phase_table'] = previous_storage['phases']
    storage['weight_file_bytes'] = weight_bytes
    storage['weight_tensor_bytes'] = sum(row['tensor_bytes'] for row in inventory['models']['draft']['weights'])
    output.mkdir(parents=True, exist_ok=False)
    sealed = output / 'sealed'
    sealed.mkdir()
    preflight_copy = sealed / 'preflight'
    preflight_copy.mkdir()
    shutil.copy2(index, preflight_copy / index.name)
    for row in read(index)['files']:
        shutil.copy2(row['path'], preflight_copy / Path(row['path']).name)
    for name in ('ORIGINAL_CONFIG.yaml', *PIN_NAMES):
        shutil.copy2(original_dir / name, sealed / name)
    configs, config_contracts = {}, {}
    for arm in ARMS:
        raw = yaml.safe_load((original_dir / 'ORIGINAL_CONFIG.yaml').read_text())
        raw['run_id'] = 'k-workload-replay-v1-' + arm.lower()
        raw['output_dir'] = str(remote / 'campaign_001/arms' / arm)
        text = yaml.safe_dump(raw, sort_keys=False)
        validate_config_text(text, arm, (original_dir / 'ORIGINAL_CONFIG.yaml').read_text())
        name = f'config_{arm}.yaml'
        (sealed / name).write_text(text)
        configs[arm], config_contracts[arm] = str(remote / 'sealed' / name), raw
    canonicals = inputs.pop('canonicals')
    write_json(sealed / 'REQUESTS.json', inputs)
    write_json(sealed / 'CANONICALS.json', canonicals)
    binding_material = dict(source=source, requests=identity(sealed / 'REQUESTS.json'),
        canonicals=identity(sealed / 'CANONICALS.json'), configs=config_contracts,
        preflight_index=identity(index), source_pins={name: identity(sealed / name) for name in PIN_NAMES},
        budget=BUDGET, training_seed=17, selection_seed=SELECTION_SEED)
    source_inputs = receipts['INPUT_SOURCE_CHECK.json']['checks']
    plan = dict(schema='k_workload_replay_v1', arms=list(ARMS), events=EVENTS,
        training_seed=17, selection_seed=SELECTION_SEED, budget=BUDGET, serving=SERVING,
        automatic_retry=False, execution_source=source, binding_material=binding_material,
        binding=stable_digest(binding_material), configs=configs,
        original_config=str(remote / 'sealed/ORIGINAL_CONFIG.yaml'),
        requests=str(remote / 'sealed/REQUESTS.json'), canonicals=str(remote / 'sealed/CANONICALS.json'),
        evaluation_schedule=final_schedule(inputs['evaluation']),
        source_pins={name: str(remote / 'sealed' / name) for name in PIN_NAMES},
        request_source_identities=source_inputs, storage=storage,
        target=original_plan['target'], draft=original_plan['draft'],
        training_python=original_plan['training_python'], worker_python=original_plan['worker_python'],
        serving_gpu=0, training_gpu=1,
        preflight_index=str(remote / 'sealed/preflight' / index.name),
        provenance_limits=[receipts['CANONICAL_SOURCE_CHECK.json']['coding_provenance_limit']],
        compiler_cache_policy='reuse_qualified_global_caches_with_bounded_additional_growth')
    write_json(sealed / 'PLAN_CHECK.json', validate_plan(plan))
    write_json(sealed / 'PLAN.json', plan)
    files = {}
    for encoded in subprocess.check_output(['git', 'ls-files', '-z', '.'], cwd=code).split(b'\0'):
        if encoded:
            relative = encoded.decode()
            if (code / relative).is_file():
                files[str(remote_code / relative)] = identity(code / relative)
    for path in sorted(output.rglob('*')):
        if path.is_file():
            files[str(remote / path.relative_to(output))] = identity(path)
    for row in source_inputs:
        files[row['remote_path']] = row['identity']
    for row in receipts['CANONICAL_SOURCE_CHECK.json']['records']:
        files[row['source_record_path']] = row['identity']
        files[row['native_source_path']] = row['native_identity']
    for kind in ('target', 'draft'):
        for name, expected in model_pins[kind]['files'].items():
            files[str(Path(plan[kind]) / name)] = expected
    folder = code / 'experiments/k_workload_replay_v1'
    exact = {str(remote_code / 'experiments/k_workload_replay_v1'):
             sorted(path.name for path in folder.iterdir() if path.is_file())}
    write_json(sealed / 'SEAL.json', dict(execution_source=source, files=files, exact_file_sets=exact))
    return dict(status='sealed_before_model_work', execution_source=source, files=len(files),
                binding=plan['binding'], budget=BUDGET, storage=storage,
                plan_path=str(sealed / 'PLAN.json'), model_work=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'preflight', 'output', 'remote-root', 'remote-code'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(seal(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
