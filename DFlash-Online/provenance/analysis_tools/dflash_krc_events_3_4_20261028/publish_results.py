"""Stage an explicit compact report package locally; never commit or upload it."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

EXECUTION_COMMIT = '11b7718a04416a89025a6df69c0c33e558e5fe1b'
ANALYSIS_COMMIT = 'ea4f289c6e18383b7560c8c4103a9d91ca3943bf'
BUILDER_SHA256 = '3c8e8462a15b7490bc074732a69b13216816d156977f5116d7c62b5e3d20a454'
REPOSITORY = 'https://github.com/ToxicMinhao/Online-Speculative-decoding-test'
CORE_FILES = ('REPORT.md', 'SUMMARY.json', 'TRAINING.csv', 'EXECUTION_SOURCE.json')
ALLOWLIST = CORE_FILES + tuple(
    f'analysis/{panel}/{name}.csv'
    for panel in ('development', 'additional')
    for name in ('COMPARISONS', 'EVENT_COUNT_CURVE', 'PER_REQUEST', 'ALL_REPEATS', 'REPEAT_TOTALS'))
MANIFEST = 'PUBLICATION_MANIFEST.json'
RAW_FIELDS = frozenset(('prompt_ids', 'response_ids', 'input_ids', 'token_ids',
                        'proposed_tokens', 'committed_tokens', 'prompt', 'response',
                        'question', 'prompt_text', 'response_text'))
SECRET_FIELDS = frozenset(('api_key', 'access_token', 'secret_key', 'password', 'authorization'))
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'provider_token': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    'authenticated_url': re.compile(r'https?://[^\s/@:]+:[^\s/@]+@'),
    'raw_token_array': re.compile(r'''["']?(?:prompt_ids|response_ids|input_ids|token_ids|proposed_tokens|committed_tokens)["']?\s*[:=]\s*\['''),
    'credential_assignment': re.compile(r'''(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\b["']?\s*[:=]\s*["'][^"'\s]+["']'''),
}
LINK_REPLACEMENTS = (
    ('[sealed plan](raw/sealed/PLAN.json)', 'sealed plan (retained private evidence; not published)'),
    ('[preflight](raw/PREFLIGHT.json)', 'preflight (retained private evidence; not published)'),
    ('[postflight](raw/POSTFLIGHT.json)', 'postflight (retained private evidence; not published)'),
    ('[launch script](run_campaign.sh)',
     f'[launch script]({REPOSITORY}/blob/{EXECUTION_COMMIT}/DFlash-Online/provenance/run_tools/dflash_krc_events_3_4_20261028/run_campaign.sh)'),
    ('[independent audit](FINAL_AUDIT.json)',
     'independent audit (full record retained privately; [compact status and integrity counts](SUMMARY.json))'),
    ('[complete costs](raw/campaign_001/COSTS.json)',
     'complete costs (full record retained privately; [compact costs](SUMMARY.json) and [event cost table](TRAINING.csv))'),
)
AVAILABILITY_NOTE = (
    '\n## Public evidence availability\n\n'
    'This public copy preserves the report’s result values. Only the recorded evidence links '
    'and this availability note differ from the retained original. Compact metrics and execution '
    'identities are supplied alongside the report; the existing launch and review records remain '
    'in this report directory. Raw plans, prompts, responses, token trajectories, complete audit '
    'and runtime records, replay features, model weights and optimizer states are outside this '
    'public package. Absolute paths, usernames and host metadata are operational provenance, '
    'not public download locations. Original and public file hashes, missing compact files and '
    'the exact text transformations are recorded in [publication manifest](PUBLICATION_MANIFEST.json). '
    'The execution and analysis revisions retain their distinct roles.\n'
)


def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def scan_value(value, location):
    """Conservative payload/secret checks; error messages never echo matched values."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace('-', '_')
            if normalized in SECRET_FIELDS and item not in (None, ''):
                raise ValueError(f'{location}: credential field requires manual review')
            if normalized in RAW_FIELDS and isinstance(item, (str, list, dict)):
                raise ValueError(f'{location}: raw prompt/response field requires manual review')
            scan_value(item, location)
    elif isinstance(value, list):
        for item in value:
            scan_value(item, location)
    elif isinstance(value, str):
        for name, pattern in PATTERNS.items():
            if pattern.search(value):
                raise ValueError(f'{location}: {name} requires manual review')


def scan_file(name, data):
    text = data.decode('utf-8')
    scan_value(text, name)
    if name.endswith('.json'):
        scan_value(json.loads(text), name)
    elif name.endswith('.csv'):
        reader = csv.DictReader(io.StringIO(text))
        if any(field.lower().replace('-', '_') in RAW_FIELDS | SECRET_FIELDS for field in reader.fieldnames or ()):
            raise ValueError(f'{name}: raw/credential CSV column requires manual review')
        for row in reader:
            if None in row:
                raise ValueError(f'{name}: malformed CSV row')
            scan_value(row, name)


def public_report(original):
    text = original.decode('utf-8')
    binding = f'Execution commit: `{EXECUTION_COMMIT}`. Postprocessing commit: `{ANALYSIS_COMMIT}`'
    if binding not in text:
        raise ValueError('REPORT.md: execution/analysis identity binding mismatch')
    if '## Public evidence availability' in text:
        raise ValueError('REPORT.md: source already has publication transformations')
    transforms = []
    for old, new in LINK_REPLACEMENTS:
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            transforms.append({'operation': 'literal_replace', 'old': old, 'new': new, 'count': count})
    if re.search(r'\]\((?:raw/|FINAL_AUDIT\.json|run_campaign\.sh)', text):
        raise ValueError('REPORT.md: an unrecognized private evidence link needs manual review')
    transforms.append({'operation': 'append', 'text': AVAILABILITY_NOTE})
    return (text + AVAILABILITY_NOTE).encode('utf-8'), transforms


def stage_results(source, destination):
    """Create a fresh staging directory; sources and existing report folders stay untouched."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Use a new staging directory; existing files will not be overwritten')
    if source == destination.resolve() or destination.resolve() in source.parents:
        raise ValueError('Staging destination must not replace the source directory')
    original, missing = {}, []
    for name in ALLOWLIST:
        path = source / name
        if any(candidate.is_symlink() for candidate in (path, *path.parents)):
            raise ValueError(f'{name}: symlink inputs are excluded')
        if not path.exists():
            if name in CORE_FILES:
                raise FileNotFoundError(f'Required compact file is missing: {name}')
            missing.append(name)
            continue
        data = path.read_bytes()
        scan_file(name, data)
        original[name] = data
    execution = json.loads(original['EXECUTION_SOURCE.json'])
    summary = json.loads(original['SUMMARY.json'])
    if (execution.get('execution_git_commit') != EXECUTION_COMMIT or execution.get('dirty_files') != []
            or execution.get('repository') != REPOSITORY or summary.get('source') != execution):
        raise ValueError('Execution source identity mismatch')
    if summary.get('postprocessing') != {'git_commit': ANALYSIS_COMMIT, 'script_sha256': BUILDER_SHA256}:
        raise ValueError('Analysis source identity mismatch')
    if summary.get('completed_audited_endpoint') and missing:
        raise ValueError('A completed endpoint is missing compact result files')
    public = dict(original)
    public['REPORT.md'], transforms = public_report(original['REPORT.md'])
    scan_file('REPORT.md', public['REPORT.md'])
    manifest = {
        'schema': 'krc34_compact_publication_v1', 'status': 'staged_locally_not_published',
        'execution_git_commit': EXECUTION_COMMIT, 'analysis_git_commit': ANALYSIS_COMMIT,
        'builder_sha256': BUILDER_SHA256, 'publication_helper': identity(Path(__file__).read_bytes()),
        'allowlist': list(ALLOWLIST), 'missing_files': missing,
        'original_files_unchanged': True,
        'review_limit': 'Bounded payload/credential checks passed; inspect exact final bytes and freeform errors before publication.',
        'files': {name: {'original': identity(original[name]), 'public': identity(data),
                         'transforms': transforms if name == 'REPORT.md' else []}
                  for name, data in public.items()},
    }
    # Every input is validated before creating a staging directory. No raw directory is traversed.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.compact-publication-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'package'
        staged.mkdir()
        for name, data in public.items():
            output = staged / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
        (staged / MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        staged.rename(destination)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Local run root containing generated compact reports')
    parser.add_argument('destination', type=Path, help='New local staging directory; no Git or upload operation')
    args = parser.parse_args()
    result = stage_results(args.source, args.destination)
    print(json.dumps({'status': result['status'], 'files': len(result['files']),
                      'missing_files': result['missing_files'], 'destination': str(args.destination)}))
