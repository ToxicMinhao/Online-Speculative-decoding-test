"""One isolated native trainer operation; no serving or automatic retries."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_events_5_8.storage import identity
from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
from .training import WorkloadReplayTrainer


def read(path):
    return json.loads(Path(path).read_text())


def run(root, arm, operation, event=0):
    root = Path(root)
    plan = read(root / 'sealed/PLAN.json')
    base = root / 'campaign_001/arms' / arm
    base.mkdir(parents=True, exist_ok=True)
    destination = base / ('INITIAL_WORKER.json' if operation == 'initialize'
                          else f'events/event_{event:03d}/WORKER.json')
    started = time.perf_counter()
    progress = dict(status='running', operation=operation, event=event, arm=arm,
                    pid=os.getpid(), extra_forwards=0, extra_optimizer_steps=0)
    write_json(destination, progress)
    trainer = None
    try:
        resume = None
        if operation == 'train':
            resume = read(base / 'INITIAL_WORKER.json')['state_path'] if event == 1 else read(
                base / f'boundaries/event_{event-1:03d}.json')['state']
        trainer = WorkloadReplayTrainer(plan['configs'][arm], base / 'trainer', arm=arm,
            binding=plan['binding'], original_config_text=Path(plan['original_config']).read_text(),
            resume_state=resume, completed_event=event-1 if operation == 'train' else 0,
            defer_export=True)
        if operation == 'initialize':
            comparator = root / 'campaign_001/arms/K/trainer/initial_state.pt'
            checked = trainer.save_initial(comparator_path=comparator if arm != 'K' else None,
                                           shared_snapshot=arm != 'K')
            progress.update(checked)
        else:
            event_id = f'event_{event:03d}'
            pass_plan = read(root / 'campaign_001/events' / event_id / 'PASS_BINDINGS.json')[arm]
            limits = dict(response_forwards=256, response_backwards=256, optimizer_steps=128)
            ledger = WorkLedger(destination.parent / 'WORK.jsonl', limits)
            ledger.wrap(trainer.core.strategy, 'forward_loss', 'response_forwards')
            ledger.wrap(trainer.backend, 'backward', 'response_backwards')
            ledger.wrap(trainer.backend.optimizer.optimizer, 'step', 'optimizer_steps')
            profile = begin_event_profile()
            try:
                result = trainer.train_event(event_id, pass_plan=pass_plan)
            finally:
                write_json(destination.parent / 'RESOURCE_COSTS.json', finish_event_profile(profile))
            work = read_work(destination.parent / 'WORK.jsonl', limits)
            require_complete_work(work, limits)
            checked = trainer.verify_boundary(receipt_path=base / 'boundaries' / f'{event_id}_validation.json')
            # Keep this committed inode even when a later pending save replaces
            # trainer.latest. The predecessor remains durable until validation.
            snapshot = trainer.events_root / event_id / 'training_state.pt'
            os.link(trainer.latest, snapshot)
            boundary = dict(status='passed', arm=arm, event=event, state=str(snapshot),
                state_identity=identity(snapshot), scientific_check=checked, work=work,
                checkpoint_path=None, export_status='deferred')
            write_json(base / 'boundaries' / f'{event_id}.json', boundary)
            progress.update(boundary=boundary, training_costs=result['costs'], work=work)
        progress['status'] = 'passed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        cleanup_error = None
        try:
            if trainer is not None:
                trainer.close()
        except BaseException as error:
            cleanup_error = error
            progress.update(status='failed', cleanup_error=repr(error))
        progress['worker_wall_s'] = time.perf_counter() - started
        write_json(destination, progress)
        if cleanup_error is not None:
            raise cleanup_error
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arm', choices=('K', 'U', 'R'), required=True)
    parser.add_argument('--operation', choices=('initialize', 'train'), required=True)
    parser.add_argument('--event', type=int, default=0)
    args = parser.parse_args()
    if (args.operation == 'initialize' and args.event != 0
            or args.operation == 'train' and not 1 <= args.event <= 8):
        parser.error('Initialize event zero or train one chronological event 1..8')
    print(json.dumps(run(args.root, args.arm, args.operation, args.event), indent=2))


if __name__ == '__main__':
    main()
