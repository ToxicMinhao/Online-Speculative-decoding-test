"""Read compact status; do not transfer artifacts during timed evaluation."""
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
result={}
for name in ('STARTED.utc','FINISHED.utc','EXIT_CODE'):
    path=root/name
    if path.exists(): result[name]=path.read_text().strip()
for relative in ('campaign_001/STATUS.json','campaign_001/evaluation/STATUS.json','campaign_001/evaluation/QUALIFICATION.json'):
    path=root/relative
    if path.exists(): result[relative]=json.loads(path.read_text())
print(json.dumps(result))
