"""Copy compact run evidence; never include model weights or feature tensors."""
from pathlib import Path
import argparse
import json
import zipfile


def collect(source, destination):
    allowed = {'.json', '.jsonl', '.csv', '.md', '.yaml', '.log', '.txt'}
    names = {'EXIT_CODE', 'STARTED_UTC', 'FINISHED_UTC'}
    inventory = []
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path == destination:
                continue
            compact_tensor = path.suffix == '.pt' and path.stat().st_size < 5 * 2**20
            if path.suffix not in allowed and path.name not in names and not compact_tensor:
                continue
            content = path.read_bytes()
            relative = path.relative_to(source).as_posix()
            archive.writestr(relative, content)
            inventory.append({'path': relative, 'bytes': len(content)})
        archive.writestr('RETRIEVAL_INVENTORY.json', json.dumps(inventory, indent=2))
    print(json.dumps({'files': len(inventory), 'archive_bytes': destination.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    collect(args.source, args.destination)
