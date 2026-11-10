"""Read-only source/artifact admission before any GPU or model loading."""
from pathlib import Path
import shutil
import subprocess

from .contract import CHECKPOINTS, DATASETS, scientific_config
from .data import check_receipt, identity

RESERVE_BYTES = 8 * 1024**3
# Compact JSON traces + copies, logs and failure evidence; live KV snapshots stay
# in host memory and are released per request, never persisted as model tensors.
ARTIFACT_ALLOWANCE_BYTES = 2 * 1024**3


def verify(plan, output_root, code_root):
    if plan['config'] != scientific_config():
        raise ValueError('Sealed scientific configuration changed')
    requests = plan['requests']
    if len(requests) != 128 or [(r['selection_rank'],r['dataset']) for r in requests] != [
            (i,d) for i in range(64) for d in DATASETS]:
        raise ValueError('Selected request execution order changed')
    if len({(r['dataset'],r['request_id']) for r in requests}) != 128:
        raise ValueError('Selected request identities are duplicated')
    checked = []
    for dataset in DATASETS:
        for checkpoint in CHECKPOINTS:
            item = plan['checkpoints'][dataset][checkpoint]
            path = Path(item['weights'])
            if path.name != 'model.safetensors' or path.parent != Path(item['export']):
                raise ValueError('Checkpoint/export identity mismatch')
            if identity(path) != item['weights_identity']:
                raise ValueError('Checkpoint missing or changed: ' + str(path))
            checked.append(dict(path=str(path), **item['weights_identity']))
    for path, expected in plan['bound_native_files'].items():
        check_receipt(dict(path=path, **expected),'path')
        checked.append(dict(path=path, **expected))
    shared = plan['shared']
    for name, expected in shared['target_files'].items():
        path = Path(shared['target_path'])/name
        if identity(path) != expected:
            raise ValueError('Target/tokenizer artifact changed: ' + str(path))
        checked.append(dict(path=str(path), **expected))
    runtime = shared['runtime']
    version = '.'.join(runtime['python_version'].split('.')[:2])
    package = Path(runtime['python']).parent.parent/'lib'/f'python{version}'/'site-packages/sglang/srt'
    for name, expected in runtime['native_runtime_sha256'].items():
        path = package/name
        if identity(path)['sha256'] != expected:
            raise ValueError('Pinned native runtime changed: ' + str(path))
        checked.append(dict(path=str(path),sha256=expected))
    for item in requests:
        request, canonical = item['request'], item['canonical']
        if (not request['prompt_ids'] or request['max_new_tokens'] != 2048
                or len(request['prompt_ids'])+2048 > 4096
                or canonical['prompt_ids'] != request['prompt_ids']
                or not canonical['response_ids']):
            raise ValueError('Original request or canonical input changed')
    target = Path(output_root).resolve()
    parent = target
    while not parent.exists():
        parent = parent.parent
    if not target.is_relative_to('/data/sources/minhao/runs'):
        raise ValueError('Native diagnostic output must use approved A800 run storage')
    free = shutil.disk_usage(parent).free
    if free < RESERVE_BYTES + ARTIFACT_ALLOWANCE_BYTES:
        raise RuntimeError('Diagnostic storage admission requires 10 GiB free including reserve')
    code = Path(code_root)
    commit = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True)
    if dirty:
        raise ValueError('Publish the exact experimental source before native execution')
    sources = {str(p.relative_to(code)):identity(p) for p in sorted(code.rglob('*.py'))
               if '__pycache__' not in p.parts and '.git' not in p.parts}
    return dict(status='passed',execution_commit=commit,code_root=str(code),source_files=sources,
                artifact_checks=checked,free_bytes=free,reserve_bytes=RESERVE_BYTES,
                artifact_allowance_bytes=ARTIFACT_ALLOWANCE_BYTES,new_model_work=0)
