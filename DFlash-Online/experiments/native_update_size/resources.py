"""CPU-only resource and import-origin receipts for the bounded size control.

The full inventory runs once during preparation. Later admission checks only
read free RAM/disk and process/GPU metadata. This module never constructs a model,
initializes CUDA, changes a queue, or deletes an artifact.
"""
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

from online_feedback.storage import write_json

RUNTIME_FILES = (
    'entrypoints/engine.py', 'layers/logits_processor.py', 'models/dflash.py',
    'models/qwen3.py', 'server_args.py', 'speculative/dflash_info.py',
    'speculative/dflash_worker.py',
)


def command(argv, timeout=30, *, env=None):
    """Return missing-command/timeout evidence without inventing an empty result."""
    began = time.perf_counter()
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, env=env)
        return dict(argv=list(argv), status='passed' if result.returncode == 0 else 'failed',
                    returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    wall_s=time.perf_counter()-began)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=list(argv), status='unavailable', error=repr(error),
                    wall_s=time.perf_counter()-began)


def memory_available(path='/proc/meminfo'):
    values = dict(line.split(':', 1) for line in Path(path).read_text().splitlines())
    return {name: int(values[name].split()[0])*1024 for name in ('MemTotal', 'MemAvailable')}


def inventory(roots, top_n=20):
    """One bounded owned-root walk; count physical inodes once and retain aliases.

Top-directory rows overlap and must not be added. File/directory symlinks are
recorded, never traversed; any symlink root is REVIEW rather than a new walk.
"""
    seen, files, links, errors, root_rows = set(), [], [], [], []
    directory_bytes = defaultdict(int)
    total_bytes = logical_bytes = 0
    began = time.perf_counter()
    for raw in roots:
        root = Path(raw)
        row = dict(path=str(root), realpath=str(root.resolve()), allocated_bytes=0,
                   logical_bytes=0, unique_inodes=0, status='scanned')
        if root.is_symlink():
            row['status'] = 'REVIEW_symlink_root_not_followed'
            links.append(dict(path=str(root), realpath=str(root.resolve()), target=os.readlink(root)))
            root_rows.append(row)
            continue
        if not root.is_dir():
            row['status'] = 'unavailable_root'
            errors.append(dict(path=str(root), error='Not an existing directory'))
            root_rows.append(row)
            continue
        row['owner_uid'] = root.stat().st_uid
        def walk_error(error):
            errors.append(dict(path=error.filename, error=repr(error)))
        for parent, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            parent = Path(parent)
            entries = [parent] + [parent/name for name in dirs if (parent/name).is_symlink()]
            dirs[:] = [name for name in dirs if not (parent/name).is_symlink()]
            entries.extend(parent/name for name in names)
            for path in entries:
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    errors.append(dict(path=str(path), error='Disappeared during inventory'))
                    continue
                except OSError as error:
                    errors.append(dict(path=str(path), error=repr(error)))
                    continue
                if stat.S_ISLNK(info.st_mode):
                    links.append(dict(path=str(path), realpath=str(path.resolve()), target=os.readlink(path)))
                inode = (info.st_dev, info.st_ino)
                if inode in seen:
                    continue
                seen.add(inode)
                allocated = info.st_blocks*512
                logical = info.st_size if stat.S_ISREG(info.st_mode) else 0
                row['allocated_bytes'] += allocated
                row['logical_bytes'] += logical
                row['unique_inodes'] += 1
                total_bytes += allocated
                logical_bytes += logical
                relative = path.relative_to(root)
                directory_bytes[str(root)] += allocated
                # Attribute the object to its containing directories, not to
                # a file-named pseudo-directory.
                parents = relative.parts if path.is_dir() and not path.is_symlink() else relative.parts[:-1]
                for depth in range(1, min(2, len(parents))+1):
                    directory_bytes[str(root.joinpath(*parents[:depth]))] += allocated
                if stat.S_ISREG(info.st_mode):
                    files.append(dict(path=str(path), realpath=str(Path(row['realpath'])/path.relative_to(root)),
                                      owner_uid=info.st_uid, device=info.st_dev,
                                      inode=info.st_ino, links=info.st_nlink,
                                      logical_bytes=logical, allocated_bytes=allocated))
        root_rows.append(row)
    return dict(status='passed' if not errors and all(r['status']=='scanned' for r in root_rows) else 'incomplete', roots=root_rows,
                allocated_bytes=total_bytes, logical_bytes=logical_bytes,
                unique_inodes=len(seen), regular_files=len(files), symlinks=links,
                top_files=sorted(files, key=lambda x:(-x['allocated_bytes'],x['path']))[:top_n],
                top_directories=[dict(path=p, allocated_bytes=n) for p,n in
                                 sorted(directory_bytes.items(),key=lambda x:(-x[1],x[0]))[:top_n]],
                directory_rows_overlap=True, errors=errors,
                wall_s=time.perf_counter()-began)


def gpu_snapshot():
    devices = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                       '--format=csv,noheader,nounits'])
    jobs = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits'])
    for result, names in [(devices, ('index','uuid','name','memory_total_mib','memory_used_mib','utilization_percent')),
                          (jobs, ('gpu_uuid','pid','process_name','memory_used_mib'))]:
        result['rows'] = []
        if result['status'] == 'passed':
            result['rows'] = [dict(zip(names, [v.strip() for v in row]))
                              for row in csv.reader(io.StringIO(result['stdout'])) if row]
    return dict(devices=devices, compute_jobs=jobs,
                cuda_context_created=False, method='nvidia-smi read-only queries')


def job_snapshot(plan):
    processes = command(['ps','-eo','pid=,ppid=,uid=,etimes=,args='])
    roots = plan.get('resource_owned_roots', ['/data/sources/minhao','/data/conda_envs/minhao'])
    if processes['status'] == 'passed':
        processes['project_rows'] = [line for line in processes.pop('stdout').splitlines()
                                    if any(root in line for root in roots)]
    queue = command(['atq'])
    declared = []
    for name in plan.get('queued_job_records', []):
        path = Path(name)
        declared.append(dict(path=str(path), realpath=str(path.resolve()), exists=path.is_file(),
                             content=json.loads(path.read_text()) if path.is_file() else None))
    return dict(processes=processes, at_queue=queue, declared_job_records=declared,
                coverage='Running project command lines, local at queue, and explicitly supplied queue records. '
                         'No global scheduler/remote job registry is assumed available; absence is not proof of no queued work.')


# Run inside each pinned interpreter. Top-level imports only; no model classes,
# cuda availability query, device allocation or torch.compile invocation.
IMPORT_PROBE = r'''
import hashlib, importlib, importlib.metadata, json, os, pathlib, site, sys
roles=json.loads(sys.argv[1]); result={'executable':sys.executable,'prefix':sys.prefix,
 'python':sys.version.split()[0],'sys_path':sys.path,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
 'modules':{},'errors':[]}
for name in roles['modules']:
 try:
  module=importlib.import_module(name); origin=getattr(module,'__file__',None)
  result['modules'][name]={'file':origin,'realpath':str(pathlib.Path(origin).resolve()) if origin else None,
   'version':str(getattr(module,'__version__','unavailable'))}
 except Exception as error: result['errors'].append({'module':name,'error':repr(error)})
torch=sys.modules.get('torch'); result['cuda_initialized']=bool(torch and torch.cuda.is_initialized())
result['cuda_build']=getattr(getattr(torch,'version',None),'cuda',None)
result['distribution_versions']=sorted([{'name':d.metadata.get('Name',''), 'version':d.version}
 for d in importlib.metadata.distributions()],key=lambda x:(x['name'].lower(),x['version']))
def ident(path):
 data=path.read_bytes();return {'path':str(path),'realpath':str(path.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
result['runtime_files']={}
if 'sglang' in result['modules']:
 root=pathlib.Path(result['modules']['sglang']['realpath']).parent/'srt'
 for relative in roles['runtime_files']:
  path=root/relative
  if path.is_file():result['runtime_files'][relative]=ident(path)
  else:result['errors'].append({'runtime_file':str(path),'error':'missing'})
result['editable_and_pth']=[]
for directory in sorted(set(site.getsitepackages()+[site.getusersitepackages()])):
 root=pathlib.Path(directory)
 for path in sorted(list(root.glob('*.pth'))+list(root.glob('*.dist-info/direct_url.json'))):
  item=ident(path);item['content']=path.read_text(errors='replace');result['editable_and_pth'].append(item)
result['conda_packages']=[]
for path in sorted((pathlib.Path(sys.prefix)/'conda-meta').glob('*.json')):
 record=json.loads(path.read_text());result['conda_packages'].append({k:record.get(k) for k in ['name','version','build','channel','sha256']})
print('RESOURCE_IMPORT_PROBE='+json.dumps(result,allow_nan=False))
'''


def import_snapshot(python, code_path, role, expected):
    modules = ['torch','transformers','safetensors'] + (['specforge'] if role=='training' else ['sglang'])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       PYTHONPATH=str(code_path)+os.pathsep+str(Path(code_path)/'training'))
    prefix = Path(python).resolve().parent.parent
    environment['PATH'] = str(prefix/'bin')+os.pathsep+environment.get('PATH','')
    environment['CONDA_PREFIX'] = str(prefix)
    if role == 'serving':
        environment['LD_LIBRARY_PATH'] = ':'.join([str(prefix/'lib/adaflash_cuda12'),str(prefix/'lib'),
                                                  environment.get('LD_LIBRARY_PATH','')])
    result = command([str(python),'-c',IMPORT_PROBE,json.dumps(dict(modules=modules,runtime_files=RUNTIME_FILES))],
                     timeout=90, env=environment)
    result['role'] = role
    if result['status'] != 'passed':
        return result
    lines = [line for line in result['stdout'].splitlines() if line.startswith('RESOURCE_IMPORT_PROBE=')]
    if len(lines) != 1:
        result.update(status='failed', error='Import probe did not return exactly one receipt')
        return result
    data = json.loads(lines[0].split('=',1)[1]); result['receipt'] = data
    # Keep warning output, without duplicating the potentially large package list.
    result['stdout'] = '\n'.join(line for line in result['stdout'].splitlines() if not line.startswith('RESOURCE_IMPORT_PROBE='))
    errors = list(data['errors'])
    if role == 'training':
        origin = data['modules'].get('specforge', {}).get('realpath')
        expected_root = (Path(code_path)/'training/specforge').resolve()
        if origin is None or not Path(origin).is_relative_to(expected_root):
            errors.append(dict(component='specforge_import_origin', expected_root=str(expected_root), actual=origin))
    if data['cuda_initialized']:
        errors.append(dict(error='CPU import probe initialized CUDA'))
    for name, version in expected.items():
        if name in ('native_runtime_sha256','runtime_sha256'):
            for relative, sha in version.items():
                actual = data['runtime_files'].get(relative,{}).get('sha256')
                if actual != sha:errors.append(dict(runtime_file=relative, expected=sha, actual=actual))
        elif name in ('python','python_version','cuda','cuda_build'):
            actual = data['python'] if name in ('python','python_version') else data['cuda_build']
            if actual != version:errors.append(dict(component=name, expected=version, actual=actual))
        elif name in modules:
            actual = data['modules'].get(name,{}).get('version')
            if actual != version:errors.append(dict(component=name, expected=version, actual=actual))
    result.update(status='passed' if not errors else 'failed', mismatches=errors)
    return result


def dependency_path(path):
    path=Path(path); row=dict(path=str(path),realpath=str(path.resolve()),exists=path.exists(),symlink=path.is_symlink())
    if path.exists():
        info=path.stat();row.update(owner_uid=info.st_uid,device=info.st_dev,inode=info.st_ino,
                                   links=info.st_nlink,bytes=info.st_size,allocated_bytes=info.st_blocks*512)
    return row


def lightweight_snapshot(plan):
    """Read current resource availability without a full disk walk or imports."""
    space=shutil.disk_usage(plan['root'])
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                disk=dict(path=plan['root'],total_bytes=space.total,used_bytes=space.used,free_bytes=space.free),
                host=memory_available(),gpu=gpu_snapshot(),jobs=job_snapshot(plan),
                model_forwards=0,optimizer_steps=0,cuda_operations=0)


def resource_snapshot(plan, full_inventory=True):
    """Preparation receipt; callers decide admission using the sealed policy."""
    result=lightweight_snapshot(plan)
    if not full_inventory:
        return result
    roots=plan.get('resource_owned_roots',['/data/sources/minhao','/data/conda_envs/minhao'])
    result['owned_inventory']=inventory(roots)
    contract=plan['runtime_contract']
    result['imports']={role:import_snapshot(plan[key],plan['code_path'],role,contract.get(role,{}))
                       for role,key in [('training','training_python'),('serving','worker_python')]}
    # The original contract records the seven native file pins at top level.
    pins=contract.get('native_runtime_sha256',{})
    serving=result['imports']['serving']
    for relative, expected in pins.items():
        actual=serving.get('receipt',{}).get('runtime_files',{}).get(relative,{}).get('sha256')
        if actual!=expected:
            serving.setdefault('mismatches',[]).append(dict(runtime_file=relative,expected=expected,actual=actual))
            serving['status']='failed'
    paths=plan.get('resource_dependency_paths',[
        '/data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh',
        '/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc',
        '/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++'])
    result['external_path_dependencies']=[dependency_path(path) for path in paths]
    required=[result['owned_inventory']['status'],*(value['status']for value in result['imports'].values()),
              result['gpu']['devices']['status'],result['gpu']['compute_jobs']['status'],result['jobs']['processes']['status']]
    result['status']='passed' if all(value=='passed'for value in required) and all(r['exists']for r in result['external_path_dependencies']) else 'incomplete_or_mismatched'
    result['allocation_note']='Allocated project bytes and filesystem free space are separate. Root totals deduplicate physical inodes; top-directory rows overlap. Other users may change filesystem free space.'
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);parser.add_argument('--light',action='store_true')
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text())
    write_json(args.output,resource_snapshot(plan,full_inventory=not args.light))
