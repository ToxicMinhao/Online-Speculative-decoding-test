"""Inventory explicit completed artifacts and pack their compact evidence.

Run on A800 with the required environment helper. This preparation never deletes.
"""
import json
from pathlib import Path
import tarfile
import shutil

PROJECT = Path('/data/sources/minhao/EAGLE_SPEC/research')
RUNS = Path('/data/sources/minhao/runs')
OUT = RUNS / 'dflash_online_osd_rerun_20261023' / 'storage'
OUT.mkdir(parents=True, exist_ok=True)

# Completed or superseded pilot exports; retain the three main matched-context
# exports and all newer OSD checkpoints and active/paused campaign recovery.
roots = [PROJECT / name for name in (
    'multiturn_online_setting_20261008', 'multiturn_matched_d199',
    'multiturn_fullhistory_d200', 'multiturn_adaptation_d202')]
kept = {
    'multiturn_matched_d199/runtime_001/connected8_001/session_27814/round_002/model_weights/model.safetensors',
    'multiturn_matched_d199/runtime_001/split4_001/session_27814/round_002/model_weights/model.safetensors',
    'multiturn_fullhistory_d200/runtime_001/fullhistory_001/session_27814/round_002/model_weights/model.safetensors',
}
exports = [p for root in roots for p in root.rglob('model.safetensors')
           if p.stat().st_size > 10**9 and str(p.relative_to(PROJECT)) not in kept]
states = [
    RUNS / 'projected_target_feasibility_20261022/recovery_001/campaign_002/sharegpt/P/trainer/training_state.pt',
    RUNS / 'dflash_online_sglang_20261023/qualification_003/trainer/training_state.pt',
]
state_exports = [p.parent / 'events' / event / 'export/model.safetensors'
                 for p, event in zip(states, ('update', 'event_1'))]
for checkpoint in state_exports:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
evidence_roots = roots + [states[0].parents[1], states[1].parents[1]]
rows, evidence = [], []
for path in exports + states:
    if not path.is_file() or path.is_symlink():
        raise ValueError('Unexpected retirement candidate: ' + str(path))
    stat = path.stat()
    rows.append({'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                 'inode': stat.st_ino, 'kind': 'optimizer_state' if path in states else 'obsolete_export'})
with tarfile.open(OUT / 'compact_evidence.tar.gz', 'w:gz') as archive:
    for root in evidence_roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in {
                    '.json', '.jsonl', '.csv', '.md', '.log', '.txt', '.yaml', '.py', '.sh'}:
                data = path.read_bytes()
                if path.suffix == '.json':
                    json.loads(data)
                archive.add(path, arcname=str(path.relative_to('/data/sources/minhao')))
                evidence.append({'path': str(path), 'bytes': len(data)})
plan = {'status': 'prepared_not_deleted', 'free_before': shutil.disk_usage(OUT).free,
        'retire': rows, 'total_bytes': sum(row['bytes'] for row in rows),
        'retained_model_exports': [str(p) for p in state_exports],
        'retained_legacy_exports': sorted(kept), 'evidence': evidence,
        'archive': str(OUT / 'compact_evidence.tar.gz'),
        'reason': 'Completed/superseded artifacts under standing aggressive retention authorization.',
        'lost_capability': 'Exact optimizer continuation for two completed runs; direct model replay of listed obsolete exports. No retired tensor is claimed archived.'}
(OUT / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({k: plan[k] for k in ('status', 'free_before', 'total_bytes', 'archive')}, indent=2))
print('candidates', len(rows), 'compact_files', len(evidence))
