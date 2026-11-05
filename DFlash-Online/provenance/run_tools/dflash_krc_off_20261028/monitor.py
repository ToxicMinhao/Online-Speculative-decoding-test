"""Read compact progress without moving artifacts or entering model code."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = root / 'campaign_001'
result = {}
for name in ('PID', 'STARTED.utc', 'FINISHED.utc', 'EXIT_CODE'):
    path = root / name
    if path.exists():
        result[name] = path.read_text().strip()
for name in ('STATUS.json', 'RESUME_SNAPSHOT.json', 'LIVE_RESUME.json', 'off/FIRST_A.json'):
    path = output / name
    if path.exists():
        value = json.loads(path.read_text())
        result[name] = {key: item for key, item in value.items()
            if key not in {'records', 'retained_records', 'traceback', 'parameter_comparisons', 'rows'}}
journal = output / 'off/PRESENTATIONS.jsonl'
if journal.exists():
    rows = journal.read_text().splitlines()
    if rows:
        last = json.loads(rows[-1])
        result['training'] = dict(presentations=len(rows), global_step=last['optimizer_step_after'])
for path in output.glob('evaluation/*/STATUS.json'):
    result[str(path.relative_to(output))] = json.loads(path.read_text())
for path in output.glob('evaluation/*/MEASUREMENTS.jsonl'):
    result[str(path.relative_to(output))] = dict(executions=sum(1 for _ in path.open()))
print(json.dumps(result, indent=2))
