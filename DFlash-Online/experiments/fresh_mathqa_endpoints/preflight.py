"""Authenticate source, fixed endpoints and fresh-panel readiness on CPU."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import unicodedata

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.bounded_prefix_allocation.preflight import verify_source
from experiments.osd_components.prepare import verify_native_runtime
from experiments.persistent_prefix_withholding.launch import check_compilers
from .runtime import read, storage, schedule_rows


def postflight_exports(plan):
    """Rehash all three read-only input exports after every scored worker exits."""
    from .evaluation import METHODS
    began = time.perf_counter()
    report = dict(status='running', before_timing=False, after_scored_workers=True,
                  outside_scored_latency=True, exports={}, errors=[])
    for method in METHODS:
        path = Path(plan['checkpoints'][method])/'model.safetensors'
        entry = dict(path=str(path), expected_identity=plan['checkpoint_identities'][method])
        try:
            entry['actual_identity'] = identity(path)
            entry['unchanged'] = entry['actual_identity'] == entry['expected_identity']
            if not entry['unchanged']:
                report['errors'].append(method+': endpoint export changed')
        except (OSError, ValueError) as error:
            entry.update(unchanged=False,error=repr(error))
            report['errors'].append(method+': '+repr(error))
        report['exports'][method] = entry
    report.update(status='failed' if report['errors'] else 'passed', wall_s=time.perf_counter()-began)
    write_json(Path(plan['root'])/'POSTFLIGHT_EXPORT_IDENTITIES.json', report)
    if report['errors']:
        raise ValueError('Frozen endpoint postflight identity failed: '+repr(report['errors']))
    return report


def verify_panel(plan):
    """Authenticate the sealed selection and reproduce ranking without outcomes."""
    from .evaluation import evaluation_schedule, dataset_contract
    seal_path = Path(plan['panel_seal'])
    seal = read(seal_path)
    if (seal.get('status') != 'sealed_before_new_model_work' or seal.get('scored') != 1024
            or seal.get('warmup') != 2 or seal.get('new_model_work') != 0):
        raise ValueError('Fresh input seal is incomplete')
    for name, pin in seal['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or identity(seal_path.parent/name) != pin:
            raise ValueError('Sealed fresh input changed: '+name)
    for key,name in [('panel','PANEL.json'),('schedule','SCHEDULE.json'),
                     ('warmups','WARMUPS.json'),('warmup_canonicals','WARMUP_CANONICALS.json')]:
        if identity(plan[key]) != seal['files'][name]:
            raise ValueError('Execution input differs from the original panel seal: '+key)
    pool = read(plan['pool_audit'])
    if identity(plan['pool_audit']) != seal['input_audit']['identity']:
        raise ValueError('Exclusion audit differs from its selected panel seal')
    counts = pool.get('counts', {})
    coverage = pool.get('historical_coverage', {})
    if (pool.get('status') != 'passed_pool_availability'
            or counts.get('source_rows') != 2985 or counts.get('total_excluded') != 1799
            or counts.get('eligible_rows') != 1186 or counts.get('required_scored') != 1024
            or counts.get('known_used_or_reserved_rows') != 1799
            or counts.get('duplicates') != {} or counts.get('input_ineligible') != {}
            or coverage.get('missing_essential_manifests') != []
            or coverage.get('unknown_later_roots') != []
            or coverage.get('historic_receipts_reauthenticated') is not True):
        raise ValueError('Fresh-panel exclusion audit has not passed its complete coverage/count gate')
    panel = read(plan['panel'])['requests']
    selection = read(seal_path.parent/'SELECTION.json')
    exclusions = read(seal_path.parent/'EXCLUSIONS.json')['rows']
    excluded = {r['source_id'] for r in exclusions}
    revision = '4f958013c164a40745a5b771dfb005df308deb4a'
    if (selection.get('seed') != 20261119 or selection.get('dataset_revision') != revision
            or selection.get('split') != 'test' or selection.get('required') != 1024
            or selection.get('no_answer_or_response_or_performance_inputs') is not True):
        raise ValueError('Selection metadata differs from the approved protocol')
    ranked = []
    for candidate in selection['ranked_eligible']:
        source = candidate['source_identity']
        if source.get('dataset_revision') != revision or source.get('split') != 'test':
            raise ValueError('Fresh source revision or split changed')
        text = json.dumps(dict(seed=20261119, **source), sort_keys=True, separators=(',', ':'))
        key = hashlib.sha256(text.encode()).hexdigest()
        if candidate['selection_key'] != key or source['source_id'] in excluded:
            raise ValueError('Fresh input ranking/exclusion differs')
        ranked.append((key, json.dumps(source,sort_keys=True,separators=(',',':')),source['source_id']))
    if (len(ranked) != 1186 or len({r[2] for r in ranked}) != 1186 or len(excluded) != 1799
            or [r['source_id'] for r in panel] != [r[2] for r in sorted(ranked)[:1024]]):
        raise ValueError('Panel differs from the approved input-only SHA256 rank')
    for row in panel:
        normalized = ' '.join(unicodedata.normalize('NFKC',row['rendered_prompt']).split())
        expected = dict(rendered_prompt_sha256=hashlib.sha256(row['rendered_prompt'].encode()).hexdigest(),
            normalized_prompt_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
            prompt_ids_sha256=hashlib.sha256(json.dumps(row['prompt_ids'],separators=(',',':')).encode()).hexdigest())
        if (any(row[k] != v for k,v in expected.items()) or row['dataset'] != 'mathqa'
                or row['max_new_tokens'] != 2048 or row['stop_token_ids'] != [151645,151643]
                or row['collect_feedback'] is not False or not 0 < len(row['prompt_ids']) <= 2048):
            raise ValueError('Fresh prompt identity or decoding contract changed')
    for field in ('request_id','source_id','normalized_prompt_sha256','prompt_ids_sha256'):
        if len({r[field] for r in panel}) != 1024:
            raise ValueError('Duplicate fresh-panel identity: '+field)
    if evaluation_schedule(panel) != schedule_rows(plan['schedule']):
        raise ValueError('Fresh panel or fixed schedule changed')
    if plan['session'] != dataset_contract('mathqa')['session']:
        raise ValueError('Native session identity differs from qualified measurement scope')
    warmups = read(plan['warmups'])['requests']
    for field in ('source_id','normalized_prompt_sha256','prompt_ids_sha256'):
        if {r[field] for r in warmups} & {r[field] for r in panel}:
            raise ValueError('Warmup identity overlaps scored panel: '+field)
    return dict(status='passed',counts=counts,selection_reproduced=True,
                warmups_byte_identical_to_seal=True,no_output_dependent_selection=True)


def verify(plan):
    from .evaluation import METHODS, evaluation_schedule, validate_references
    began = time.perf_counter()
    report = dict(status='running', model_work=0, execution_commit=plan['execution_commit'])
    try:
        content = {key:value for key,value in plan.items() if key != 'binding'}
        actual_binding = hashlib.sha256(json.dumps(content, sort_keys=True,
                                                   separators=(',', ':')).encode()).hexdigest()
        if plan.get('binding') != actual_binding:
            raise ValueError('Sealed execution plan changed')
        expected_serving = dict(block_size=16, max_new_tokens=2048, context_limit=4096,
            concurrency=1, greedy=True, precision='BF16', attention_backend='fa3', capture=False,
            cache_policy='idle_full_flush_before_every_request_and_publication')
        if plan['serving'] != expected_serving or plan['optimizer_steps'] != 0 or plan['new_trained_checkpoints'] != 0:
            raise ValueError('The frozen-endpoint serving contract changed')
        report['source'] = verify_source(plan)
        for path, expected in plan['input_identities'].items():
            if identity(path) != expected:
                raise ValueError('Sealed input changed: '+path)
        for entry in plan['authenticated_files']:
            if identity(entry['path']) != entry['expected_identity']:
                raise ValueError('Required model/tokenizer/export changed: '+entry['path'])
        if set(plan['checkpoints']) != set(METHODS):
            raise ValueError('Only O58/AREPO58/KPLUS_C58 endpoints are allowed')
        if (set(plan['canonical_provenance']) != {'target_identity','tokenizer_identity','runtime_identity'}
                or any(not isinstance(v,dict) or not v for v in plan['canonical_provenance'].values())):
            raise ValueError('Canonical provenance must contain exactly the authenticated target/tokenizer/runtime identities')
        if plan['reference_checkpoints']['scored'] != plan['checkpoints']['O58']:
            raise ValueError('Only O58 may produce fresh canonical witnesses')
        tensors = {}
        for method, checkpoint in plan['checkpoints'].items():
            path = Path(checkpoint)/'model.safetensors'
            if identity(path) != plan['checkpoint_identities'][method]:
                raise ValueError('Endpoint weights differ: '+method)
            with path.open('rb') as stream:
                header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
            tensors[method] = {k:{'dtype':v['dtype'], 'shape':v['shape']}
                for k,v in header.items() if k != '__metadata__'}
            if len(tensors[method]) != 58 or any(v['dtype'] != 'BF16' for v in tensors[method].values()):
                raise ValueError('Endpoint BF16 tensor schema changed')
            config = read(Path(checkpoint)/'config.json')
            if config['block_size'] != 16 or config['num_hidden_layers'] != 5:
                raise ValueError('Endpoint draft geometry changed')
        if not all(v == tensors['O58'] for v in tensors.values()):
            raise ValueError('Endpoint tensor shapes differ')
        report['endpoint_tensor_shapes'] = tensors
        report['panel'] = verify_panel(plan)
        panel = read(plan['panel'])
        closeout = read(plan['pr54_closeout'])
        if closeout.get('status') != 'passed':
            raise ValueError('PR54 archive delivery/keeper closeout is incomplete')
        warmups = read(plan['warmups'])['requests']
        validate_references(warmups, read(plan['warmup_canonicals']),
                            reference_checkpoint=plan['reference_checkpoints']['warmup'])
        if len(warmups) != 2 or set(r['source_id'] for r in warmups) & set(r['source_id'] for r in panel['requests']):
            raise ValueError('Warmups must be two fixed previously seen non-panel requests')
        report['runtime'] = verify_native_runtime(plan['worker_python'], read(plan['runtime_pin']))
        report['compilers'] = check_compilers()
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                         '--format=csv,noheader,nounits'], text=True)
        if active.strip():
            raise RuntimeError('Qualification admission requires both project cards idle')
        report['storage'] = storage(plan, complete_forecast=True)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['wall_s'] = time.perf_counter()-began
        write_json(Path(plan['root'])/'CPU_PREFLIGHT.json', report)
    return report
