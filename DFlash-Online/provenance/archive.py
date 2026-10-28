"""Inspect and reconstruct one retained source contract from this archive.

Examples (run from DFlash-Online/provenance):
  python archive.py list dflash_krc_off_20261028
  python archive.py verify
  python archive.py extract dflash_krc_off_20261028 --contract ORIGINAL_SEAL_PATH \
      --destination /tmp/krc-off-source

Extraction preserves exact bytes. It never merges different historical seals.
The default source prefix selects DFlash-Online files in an absolute-path seal;
relative source paths are also accepted. Use --prefix to select external runtime
or run-driver paths instead. Dependencies not covered by that contract remain
the caller's responsibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
DEFAULT_PREFIX = '/data/sources/minhao/DFlash-Online/'


def read_json(path):
    return json.loads(path.read_text())


def object_bytes(root, digest):
    data = (root / 'objects' / digest).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        raise ValueError(f'Object digest mismatch: expected {digest}, found {actual}')
    return data


def run_manifest(root, run_id):
    index = read_json(root / 'INDEX.json')
    entry = next(row for row in index['runs'] if row['run_id'] == run_id)
    return read_json(root / entry['manifest'])


def verify(root):
    objects = read_json(root / 'OBJECTS.json')
    for digest, metadata in objects.items():
        data = object_bytes(root, digest)
        if len(data) != metadata['bytes']:
            raise ValueError(f'Object length mismatch: {digest}')
    index = read_json(root / 'INDEX.json')
    count = 0
    readable = 0
    for run in index['runs']:
        manifest = read_json(root / run['manifest'])
        for copy in manifest.get('run_tools', []) + manifest.get('config_pins', []):
            if (root / copy['published_path']).read_bytes() != object_bytes(root, copy['sha256']):
                raise ValueError(f"Readable source copy differs: {copy['published_path']}")
            readable += 1
        for contract in manifest['contracts']:
            for pin in contract['pins']:
                if pin['object'] is not None:
                    if pin['object'] != f"objects/{pin['sha256']}" or pin['sha256'] not in objects:
                        raise ValueError(f'Invalid source mapping: {pin}')
                    count += 1
    return {'verified_objects': len(objects), 'verified_source_pins': count, 'verified_readable_copies': readable,
            'missing_historical_pins': len(index['missing_exact_bytes'])}


def selected_sources(contract, prefix):
    selected = {}
    for pin in contract['pins']:
        name = pin['source_path']
        if name.startswith(prefix):
            name = name[len(prefix):]
        elif PurePosixPath(name).is_absolute():
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError(f'Unsafe retained source path: {name}')
        if pin['object'] is None:
            raise ValueError(f"Required historical source bytes are unavailable: {pin['source_path']}")
        if name in selected and selected[name] != pin['sha256']:
            raise ValueError(f'Conflicting versions within selected contract: {name}')
        selected[name] = pin['sha256']
    if not selected:
        raise ValueError('No source files matched the selected contract/prefix')
    return selected


def extract(root, run_id, contract_path, destination, prefix=DEFAULT_PREFIX):
    manifest = run_manifest(root, run_id)
    contract = next(row for row in manifest['contracts'] if row['retained_manifest'] == contract_path)
    selected = selected_sources(contract, prefix)
    data = {name: object_bytes(root, digest) for name, digest in selected.items()}
    destination.mkdir(parents=True, exist_ok=False)
    for name, contents in data.items():
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(contents)
    receipt = {'run_id': run_id, 'retained_manifest': contract_path,
               'retained_manifest_sha256': contract['retained_manifest_sha256'],
               'selected_prefix': prefix, 'files': selected}
    (destination / 'EXTRACTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list')
    listing.add_argument('run_id')
    commands.add_parser('verify')
    extraction = commands.add_parser('extract')
    extraction.add_argument('run_id')
    extraction.add_argument('--contract', required=True)
    extraction.add_argument('--destination', type=Path, required=True)
    extraction.add_argument('--prefix', default=DEFAULT_PREFIX)
    args = parser.parse_args()
    if args.command == 'verify':
        result = verify(ROOT)
    elif args.command == 'list':
        manifest = run_manifest(ROOT, args.run_id)
        result = [{'contract': row['retained_manifest'], 'source_pins': len(row['pins']),
                   'missing': sum(pin['object'] is None for pin in row['pins'])}
                  for row in manifest['contracts']]
    else:
        result = extract(ROOT, args.run_id, args.contract, args.destination, args.prefix)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
