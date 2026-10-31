"""Give the single bounded qualification its own empty compiler caches/process."""
import json
from pathlib import Path
import subprocess
import sys
import time
from experiments.ok_coding.campaign import fresh_worker_environment
from online_feedback.storage import write_json


def launch(root):
    root=Path(root)
    if (root/'qualification_001').exists() or (root/'QUALIFICATION_EXEC.json').exists():
        raise FileExistsError('One suite only; preserve earlier qualification')
    plan=json.loads((root/'sealed/PLAN.json').read_text())
    environment=fresh_worker_environment(root/'qualification_caches')
    command=[plan['training_python'],'-u','-m','experiments.k_prefix_protection_v1.qualification',str(root)]
    started=time.perf_counter()
    with (root/'QUALIFICATION.log').open('x') as stream:
        process=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    receipt=dict(command=command,returncode=process.returncode,wall_s=time.perf_counter()-started,
                 execution_source=plan['execution_source'],automatic_retry=False)
    write_json(root/'QUALIFICATION_EXEC.json',receipt)
    if process.returncode:
        raise RuntimeError('Bounded qualification failed; stop without production or automatic retry')
    return receipt

if __name__=='__main__':
    launch(sys.argv[1])
