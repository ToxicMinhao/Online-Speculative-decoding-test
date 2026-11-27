"""Authenticate a narrow infrastructure repair without changing the sealed plan.

The original commit remains the scientific/state binding. A separate receipt
names the clean checkout that actually runs the remaining work. Only these
orchestration/compaction files may differ in production; learning, serving,
configuration and resource-policy code must remain byte-identical.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from .contract import digest
from .runtime import read, write_json

SCHEMA = 'continuation_repair_fullrun_source_amendment_v1'
PREFIX = 'experiments/continuation_repair_fullrun/'
APPROVED_PRODUCTION_CHANGES = tuple(PREFIX + name for name in (
    'compact.py', 'entry.py', 'launch.py', 'source_amendment.py'))


def identity(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return dict(bytes=Path(path).stat().st_size, sha256=checksum.hexdigest())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _documentary(path):
    path = Path(path)
    return path.parts[0] in ('tests', 'reports') or path.suffix.lower() == '.md'


def verify_tree(root, manifest_path, expected_identity, commit):
    """Verify every staged entry and reject unmanifested Python import sources."""
    _require(identity(manifest_path) == expected_identity, 'Source manifest identity changed')
    manifest = read(manifest_path)
    _require(manifest['commit'] == commit and manifest.get('clean') is True,
             'Require the actual clean execution source manifest')
    root = Path(root).resolve()
    for relative, expected in manifest['files'].items():
        path = Path(relative)
        _require(not path.is_absolute() and '..' not in path.parts, 'Invalid source member path')
        actual = root / path
        if expected.get('kind') == 'symlink':
            valid = actual.is_symlink() and os.readlink(actual) == expected['target']
        else:
            valid = not actual.is_symlink() and identity(actual) == expected
        _require(valid, 'Staged source changed: ' + relative)
    unexpected = {str(path.relative_to(root)) for path in root.rglob('*.py')
                  if path.is_file()} - set(manifest['files'])
    _require(not unexpected, 'Unmanifested Python source: ' + repr(sorted(unexpected)))
    return manifest


def compare_sources(original, repaired):
    changed = sorted(name for name in set(original['files']) | set(repaired['files'])
                     if original['files'].get(name) != repaired['files'].get(name))
    production = [name for name in changed if not _documentary(name)]
    _require(set(production) <= set(APPROVED_PRODUCTION_CHANGES),
             'Repair changes unapproved production code: ' + repr(production))
    _require(PREFIX + 'compact.py' in production, 'Repair must include the scoped compaction fix')
    _require(PREFIX + 'source_amendment.py' in repaired['files'], 'Repair lacks its source verifier')
    return dict(changed_production_paths=production,
                changed_diagnostic_or_documentation_paths=[name for name in changed if _documentary(name)],
                unchanged_production_python_files=sum(
                    name.endswith('.py') and not _documentary(name) and name not in production
                    for name in original['files']))


def _original(plan_path):
    plan = read(plan_path)
    _require(plan['binding'] == digest({k: v for k, v in plan.items() if k != 'binding'}),
             'Original scientific plan changed')
    manifest = verify_tree(plan['source_root'], plan['source_manifest'],
                           plan['source_identity'], plan['execution_commit'])
    return plan, manifest


def verify_amendment(plan_path, amendment_path, *, expected_sha256=None, check_running_source=True):
    plan, original = _original(plan_path)
    amendment_path = Path(amendment_path).resolve()
    amendment_identity = identity(amendment_path)
    if expected_sha256 is not None:
        _require(amendment_identity['sha256'] == expected_sha256, 'Source amendment identity changed')
    amendment = read(amendment_path)
    _require(amendment['schema'] == SCHEMA and amendment['binding'] ==
             digest({k: v for k, v in amendment.items() if k != 'binding'}), 'Source amendment seal changed')
    _require(Path(amendment['original_plan_path']).resolve() == Path(plan_path).resolve()
             and amendment['original_plan_identity'] == identity(plan_path)
             and amendment['plan_binding'] == plan['binding']
             and amendment['execution_commit'] == plan['execution_commit'],
             'Amendment does not bind the unchanged original plan')
    _require(amendment['approved_production_changes'] == list(APPROVED_PRODUCTION_CHANGES),
             'Infrastructure whitelist changed')
    root = Path(amendment['source_root']).resolve()
    charged_root = Path(plan['root']).resolve() / 'infrastructure_sources'
    _require(root.is_relative_to(charged_root) and root != Path(plan['source_root']).resolve(),
             'Repair source must be separate and charged inside the existing run')
    repaired = verify_tree(root, amendment['source_manifest'], amendment['source_identity'],
                           amendment['actual_execution_commit'])
    comparison = compare_sources(original, repaired)
    _require(amendment['source_comparison'] == comparison, 'Source difference receipt changed')
    evidence = amendment['resume_evidence']
    _require(identity(evidence['path']) == evidence['identity'], 'CPU resume evidence changed')
    receipt = read(evidence['path'])
    _require(receipt['status'] in ('passed', 'completed') and receipt['new_model_work'] == 0,
             'Require completed CPU-only compaction recovery evidence')
    _require(receipt['plan_binding'] == plan['binding'] and receipt['execution_commit'] == plan['execution_commit'],
             'CPU resume evidence belongs to another scientific plan/source')
    if check_running_source:
        _require(Path(__file__).resolve().parents[2] == root,
                 'This process is not imported from the authenticated repaired source')
    return dict(execution_commit=plan['execution_commit'],
        actual_execution_commit=amendment['actual_execution_commit'],
        actual_source_root=str(root), actual_source_identity=amendment['source_identity'],
        source_amendment_path=str(amendment_path), source_amendment_identity=amendment_identity,
        source_amendment_binding=amendment['binding'])


def execution_source(plan_path, amendment_path=None, *, expected_sha256=None):
    if amendment_path is not None:
        return verify_amendment(plan_path, amendment_path, expected_sha256=expected_sha256)
    _require(expected_sha256 is None, 'Amendment identity was given without its receipt')
    plan, _ = _original(plan_path)
    root = Path(plan['source_root']).resolve()
    _require(Path(__file__).resolve().parents[2] == root,
             'This process is not imported from the authenticated original source')
    return dict(execution_commit=plan['execution_commit'], actual_execution_commit=plan['execution_commit'],
        actual_source_root=str(root), actual_source_identity=plan['source_identity'],
        source_amendment_path=None, source_amendment_identity=None, source_amendment_binding=None)


def seal_amendment(plan_path, *, source_root, source_manifest, resume_evidence, output):
    """CPU-only seal after publication/staging and metadata recovery, before launch."""
    output = Path(output)
    _require(not output.exists(), 'Preserve an existing source amendment')
    plan, original = _original(plan_path)
    repaired = read(source_manifest)
    verify_tree(source_root, source_manifest, identity(source_manifest), repaired['commit'])
    value = dict(schema=SCHEMA, original_plan_path=str(Path(plan_path).resolve()),
        original_plan_identity=identity(plan_path), plan_binding=plan['binding'],
        execution_commit=plan['execution_commit'], actual_execution_commit=repaired['commit'],
        source_root=str(Path(source_root).resolve()), source_manifest=str(Path(source_manifest).resolve()),
        source_identity=identity(source_manifest), approved_production_changes=list(APPROVED_PRODUCTION_CHANGES),
        source_comparison=compare_sources(original, repaired),
        resume_evidence=dict(path=str(Path(resume_evidence).resolve()), identity=identity(resume_evidence)),
        scope='Infrastructure-only compaction recovery and execution provenance; original plan, scientific code, clocks and states unchanged')
    value['binding'] = digest(value)
    write_json(output, value)
    verify_amendment(plan_path, output, check_running_source=False)
    return value


def preserve_resume(plan, source):
    """Copy old receipts byte-for-byte before a supervisor overwrites its status."""
    root = Path(plan['root'])
    parent = root / 'infrastructure_resumes'
    parent.mkdir(exist_ok=True)
    output = parent / f'resume_{len(list(parent.glob("resume_*"))) + 1:03d}'
    output.mkdir(exist_ok=False)
    paths = [root / name for name in ('STATUS.json', 'CLOCK.json', 'WORK.json')]
    qualification = root / 'qualification/QUALIFICATION.json'
    if qualification.exists():
        paths.append(qualification)
    paths += sorted((root / 'processes').glob('*.json'))
    paths += sorted((root / 'phases').glob('*/STATUS.json'))
    paths += [p for arm in ('BASE', 'TAIL') if (p := root / 'campaign_001/arms' / arm / 'STATUS.json').exists()]
    preserved = {}
    for path in paths:
        relative = path.relative_to(root)
        target = output / 'prior' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = identity(path)
        shutil.copy2(path, target)
        _require(identity(target) == expected and identity(path) == expected, 'Resume evidence changed during copy')
        preserved[str(relative)] = expected
    receipt = dict(status='preserved', plan_binding=plan['binding'], **source,
        original_model_load_epoch=read(root / 'CLOCK.json')['first_model_load_epoch'],
        prior_files=preserved, new_model_work=0)
    write_json(output / 'RESUME.json', receipt)
    return dict(path=str(output / 'RESUME.json'), identity=identity(output / 'RESUME.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--resume-evidence', required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    seal_amendment(arguments.plan, source_root=arguments.source_root,
        source_manifest=arguments.source_manifest, resume_evidence=arguments.resume_evidence, output=arguments.output)
