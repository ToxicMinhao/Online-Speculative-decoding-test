"""Wait without a GPU for priority completion; execute the pilot at most once."""
import argparse
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from .data import read
from .preflight import priority_finished


def run(root, *, maximum_wait_s=12*3600, check_interval_s=60):
    root = Path(root); plan = read(root/'sealed/PLAN.json')
    if (root/'QUEUE_STATUS.json').exists() or (root/'START_UTC').exists():
        raise ValueError('Queue/run already attempted; no automatic retry')
    receipt = dict(status='waiting_for_PR36', priority_root=plan['priority_root'],
        gpu_work_started=False, maximum_wait_s=maximum_wait_s, execution_attempts=0)
    started = time.perf_counter()
    try:
        while not (Path(plan['priority_root'])/'EXIT_CODE').exists():
            receipt['waited_s'] = time.perf_counter()-started
            write_json(root/'QUEUE_STATUS.json', receipt)
            if receipt['waited_s'] >= maximum_wait_s: raise TimeoutError('Priority campaign window did not open')
            time.sleep(check_interval_s)
        receipt['priority'] = priority_finished(plan)
        # Admission is repeated inside run_all before any reconstruction/model.
        receipt.update(status='launching', execution_attempts=1, waited_s=time.perf_counter()-started)
        write_json(root/'QUEUE_STATUS.json', receipt)
        code = subprocess.call(['bash', 'experiments/survival_head/run_all.sh', str(root),
                                plan['execution_source']['source_path']], cwd=plan['execution_source']['source_path'])
        receipt.update(status='completed' if code == 0 else 'failed', returncode=code)
        if code: raise RuntimeError('Pilot stopped; preserve evidence, no automatic retry')
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['queue_and_campaign_wall_s'] = time.perf_counter()-started
        write_json(root/'QUEUE_STATUS.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root')
    args = parser.parse_args(); run(args.root)
