"""Fixed qualification → parallel untimed canonicals → quiet one-card evaluation."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from experiments.bounded_prefix_allocation.launch import stop_owned
from .entry import digest
from .runtime import read, initialize, boundary, enforce_clock, forecast, LIMITS


def children(plan_path, plan, phase, slots):
    root = Path(plan['root'])
    running = []
    with ExitStack() as files:
        try:
            for slot in slots:
                name = f'{phase}_{slot}'
                command = [plan['training_python'],'-u','-m','experiments.fresh_mathqa_endpoints.entry',
                           '--plan',str(plan_path),'--phase',phase,'--slot',slot]
                log = files.enter_context((root/'logs'/(name+'.log')).open('x'))
                process = subprocess.Popen(command, cwd=plan['source_root'], env=child_environment(plan,slot),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                receipt = dict(phase=phase,slot=slot,pid=process.pid,command=command,started_epoch=time.time())
                write_json(root/'processes'/(name+'.json'), receipt)
                running.append((name,process,receipt))
            while any(p.poll() is None for _,p,_ in running):
                enforce_clock(plan)
                if any(p.poll() not in (None,0) for _,p,_ in running):
                    raise RuntimeError('Worker failed; no automatic retry')
                time.sleep(1)
            if any(p.returncode != 0 for _,p,_ in running):
                raise RuntimeError('Worker failed; retain completed work')
        finally:
            for name,process,receipt in running:
                stop_owned(process)
                receipt.update(exit_code=process.poll(), finished_epoch=time.time())
                write_json(root/'processes'/(name+'.json'), receipt)
    results = {slot:read(root/'phases'/f'{phase}_{slot}'/'STATUS.json') for slot in slots}
    if any(r['status'] != 'completed' for r in results.values()):
        raise ValueError('Missing completed phase receipt')
    return results


def merge_canonicals(plan):
    from .evaluation import validate_references
    root = Path(plan['root'])
    merged = {}
    for slot in ('A','B'):
        shard = read(root/'phases'/f'canonical_{slot}'/'CANONICALS.json')
        if set(merged) & set(shard):
            raise ValueError('Canonical shards overlap')
        merged.update(shard)
    requests = read(plan['panel'])['requests']
    validate_references(requests, merged, reference_checkpoint=plan['checkpoints']['O58'],
        producer_weight_identity=plan['checkpoint_identities']['O58'],
        required_provenance=True, canonical_provenance=plan['canonical_provenance'])
    if (root/'CANONICALS.json').exists() or (root/'CANONICAL_SEAL.json').exists():
        raise FileExistsError('A sealed canonical population must never be overwritten')
    work = read(root/'WORK.json')
    if (work['pending'] or work['attempted']['canonical_generation'] != 1024
            or work['completed']['canonical_generation'] != 1024
            or work['attempted']['scored_generation'] != 0):
        raise ValueError('Complete canonical work must be sealed before scoring')
    write_json(root/'CANONICALS.json', merged)
    write_json(root/'CANONICAL_SEAL.json', dict(status='passed', count=len(merged),
        digest=digest(merged), producer='O58', producer_identity=plan['checkpoint_identities']['O58'],
        before_scored_generation=True, target_only=False))


def run(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    root = Path(plan['root'])
    if (root/'LAUNCH.json').exists():
        raise FileExistsError('An existing attempt must not be restarted automatically')
    (root/'logs').mkdir(exist_ok=True)
    (root/'processes').mkdir(exist_ok=True)
    report = dict(status='preflight', execution_commit=plan['execution_commit'],
                  started_epoch=time.time(), completed=[], automatic_retry=False)
    write_json(root/'LAUNCH.json', report)
    try:
        from .preflight import verify
        check_compilers()
        report['preflight'] = verify(plan)
        initialize(root)
        report['forecast'] = forecast(plan, plan['estimate_elapsed_s'], plan['estimate_card_s'])
        report['status'] = 'qualification'; write_json(root/'LAUNCH.json', report)
        report['qualification'] = children(plan_path, plan, 'qualification', ('A','B'))
        if read(root/'WORK.json')['completed']['qualification_generation'] != 24:
            raise ValueError('Qualification generation accounting differs')
        # Both cards independently match the same retained witnesses, including
        # tokens and termination; this is the cross-card parity gate.
        write_json(root/'QUALIFICATION.json', dict(status='passed', generations=24,
            cases='two fixed old warmups × three arms × two repeats × two cards',
            identical_canonical_witnesses=True, extra_model_work=0))
        report['completed'].append('qualification')
        report['status'] = 'canonicals'; write_json(root/'LAUNCH.json', report)
        forecast(plan, plan['estimate_canonical_elapsed_s']+plan['estimate_scored_s'],
                 2*plan['estimate_canonical_elapsed_s']+plan['estimate_scored_s'])
        report['canonicals'] = children(plan_path, plan, 'canonical', ('A','B'))
        merge_canonicals(plan)
        report['completed'].append('canonicals')
        report['status'] = 'scored'; write_json(root/'LAUNCH.json', report)
        report['serving_admission'] = forecast(plan,plan['estimate_scored_s'],plan['estimate_scored_s'])
        report['scored'] = children(plan_path, plan, 'scored', ('A',))
        report['completed'].append('scored')
        work = read(root/'WORK.json')
        if work['pending'] or work['attempted'] != LIMITS or work['completed'] != LIMITS:
            raise ValueError('Complete frozen-endpoint work accounting does not reconcile')
        from .preflight import postflight_exports
        report['postflight_exports'] = postflight_exports(plan)
        from .analysis import write_report
        began = time.perf_counter()
        summary = write_report(root/'evaluation')
        report['analysis_s'] = time.perf_counter()-began
        if summary['status'] != 'completed':
            raise ValueError('Final raw audit is incomplete or invalid')
        report['postflight'] = boundary(plan)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['wall_s'] = time.time()-report['started_epoch']
        write_json(root/'LAUNCH.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
