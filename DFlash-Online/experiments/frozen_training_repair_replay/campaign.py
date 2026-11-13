"""Single bounded admission -> two frozen scorers -> scalar report, never retrying."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .prepare import prepare, read, write


def execute(spec_path):
    spec=read(spec_path)
    started=time.perf_counter()
    root=Path(spec['output'])
    manifest=prepare(**{k:spec[k] for k in ('previous_root','public_dir','output','execution_commit','source_manifest','prior_pins')})
    status=dict(status='running',execution_commit=spec['execution_commit'],completed_arms=[],unattempted_arms=['NATIVE7','RECENT7'],scientific_optimizer_steps=0)
    write(root/'CAMPAIGN_STATUS.json',status)
    try:
        for arm in ('NATIVE7','RECENT7'):
            status['unattempted_arms'].remove(arm);status['active_arm']=arm
            write(root/'CAMPAIGN_STATUS.json',status)
            command=[sys.executable,'-u','-m','experiments.frozen_training_repair_replay.runner','--plan',str(root/f'PLAN_{arm}.json'),'--output',str(root/arm)]
            arm_start=time.perf_counter()
            with (root/f'{arm}.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800-(time.perf_counter()-started))
            write(root/f'{arm}_PROCESS.json',dict(command=command,exit_code=result.returncode,occupied_process_s=time.perf_counter()-arm_start))
            if result.returncode:
                raise RuntimeError(f'{arm} scoring failed; no automatic retry')
            status['completed_arms'].append(arm)
        from .report import execute as report
        report(root/'NATIVE7',root/'RECENT7',root/'report',root/'MANIFEST.json',
               reproduction_command='python -m experiments.frozen_training_repair_replay.campaign --spec /path/to/private/RUN_SPEC.json')
        status.update(status='completed',active_arm=None)
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        status['wall_s']=time.perf_counter()-started
        write(root/'CAMPAIGN_STATUS.json',status)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    args=parser.parse_args()
    print(json.dumps(execute(args.spec),indent=2))
