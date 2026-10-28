"""Install only declared source changes after checking their retained base."""
import json
from pathlib import Path
import shutil
import sys
from seal_remote import identity


def install(root):
    root = Path(root)
    code = Path('/data/sources/minhao/DFlash-Online')
    manifest = json.loads((root / 'SOURCE_UPDATE.json').read_text())
    for relative, expected in manifest['base_files'].items():
        path = code / relative
        if identity(path) != expected:
            raise ValueError('Existing DFlash-Online source differs: ' + relative)
    backup = root / 'source_before'
    installed = []
    for relative, expected in manifest['changed_files'].items():
        source = root / 'source_update' / relative
        destination = code / relative
        if identity(source) != expected:
            raise ValueError('Transferred source differs: ' + relative)
        if destination.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if identity(destination) != expected:
            raise ValueError('Installed source differs: ' + relative)
        installed.append(relative)
    (root / 'INSTALL.json').write_text(json.dumps(dict(installed=installed,
        base_files_checked=len(manifest['base_files']), unchanged_preserved=True), indent=2) + '\n')
    print(json.dumps(dict(installed=installed)))


if __name__ == '__main__':
    install(sys.argv[1])
