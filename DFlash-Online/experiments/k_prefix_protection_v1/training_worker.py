"""One fresh interpreter, one own-state KP or K_HALF event, no serving work."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work, startup_metadata
from experiments.ok_events_5_8.campaign import identity, verify_pool
from experiments.ok_coding import training as original
from online_feedback.storage import save_torch, write_json
from .training import (ARMS, PrefixProtectionTrainer, live_state, validate_snapshot, verify_live)


def original_contract(plan, seed):
    return plan['original_campaign_contracts'][str(seed)]


def source_path(value):
    return value['state'] if isinstance(value, dict) else value


def validate_own(state, plan, config, arm, seed, event):
    return validate_snapshot(state, arm=arm, seed=seed, binding=plan['training_binding'],
        original_contract=original_contract(plan, seed), config_text=Path(config).read_text(),
        completed_event=event)


def train_worker(root, seed, arm, event):
    import torch
    root = Path(root)
    plan = json.loads((root/'sealed/PLAN.json').read_text())
    if seed not in (17, 29) or arm not in ARMS or event not in (1, 2, 3, 4):
        raise ValueError('Training job is outside the sealed two-seed/four-event campaign')
    seed_root = root/'campaign_001'/f'seed_{seed}'
    arm_root = seed_root/arm
    audit = seed_root/f'resume_{event:03d}_{arm}'
    audit.mkdir(parents=True, exist_ok=True)
    limits = dict(forwards=256, backwards=256, optimizer_steps=128)
    if arm == 'KP':
        limits['reference_forwards'] = 256
    ledger = WorkLedger(audit/'WORK.jsonl', limits)
    config = plan['configs'][str(seed)][arm]
    pool_binding = plan['pools'][str(seed)][str(event)]
    if identity(pool_binding['path']) != pool_binding['identity']:
        raise ValueError('Sealed original pool identity changed')
    pool = json.loads(Path(pool_binding['path']).read_text())
    verify_pool(pool)
    schedule = plan['original_schedules'][str(seed)][str(event)]
    schedule = schedule['path'] if isinstance(schedule, dict) else schedule
    ordinary = json.loads(Path(schedule).read_text())
    original.validate_event_receipt(ordinary, event_number=event, objective='osd')
    receipt = dict(status='running', seed=seed, arm=arm, event=event,
                   automatic_retry=False, additional_qualification_steps=0)
    trainer = None
    started = time.perf_counter()
    try:
        receipt['startup'] = startup_metadata()
        write_json(audit/'WORKER.json', receipt)
        parent = arm_root/'events'/f'event_{event-1:03d}'/'training_state.pt'
        if event == 1:
            initial_path = source_path(plan['original_initial_states'][str(seed)])
            state = torch.load(initial_path, map_location='cpu', weights_only=False)
            original.validate_snapshot(state, arm='K', seed=seed,
                campaign_contract=original_contract(plan, seed), config_text=state['config_text'],
                completed_event=0)
        else:
            state = torch.load(parent, map_location='cpu', weights_only=False)
            saved = validate_own(state, plan, config, arm, seed, event-1)
            original._export_equal(state, parent.parent/'export')
            write_json(audit/'SAVED.json', dict(saved, export_exact=True, state_path=str(parent)))
            used = {feature['record_id'] for old in state['events'] for feature in old['features']}
            if used & set(pool['record_ids']):
                raise ValueError('New pool overlaps the arm\'s own earlier training')
        trainer = PrefixProtectionTrainer(config, arm_root, arm=arm, seed=seed,
            binding=plan['training_binding'], original_contract=original_contract(plan, seed),
            source_state=state if event == 1 else None,
            resume_state=parent if event > 1 else None, completed_event=event-1)
        if event == 1:
            initial = live_state(trainer)
            validate_own(initial, plan, config, arm, seed, 0)
            original._export_equal(initial, trainer.cfg.model.draft_checkpoint_path)
            save_torch(arm_root/'initial_state.pt', initial)
            verify_live(trainer, initial, audit/'LIVE.json')
            write_json(audit/'MIGRATION.json', trainer.migration_receipt)
            del initial
        else:
            verify_live(trainer, state, audit/'LIVE.json')
        del state
        receipt['startup']['device_name'] = torch.cuda.get_device_name(0)
        receipt['startup']['device_uuid'] = str(torch.cuda.get_device_properties(0).uuid)
        trainer.response_observer_path = audit/'RESPONSE_OBSERVATIONS.jsonl'
        loading_s = time.perf_counter()-started
        loading_peak = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        ledger.wrap(trainer.core.strategy, 'forward_loss', 'forwards')
        ledger.wrap(trainer.backend, 'backward', 'backwards')
        ledger.wrap(trainer.backend.optimizer.optimizer, 'step', 'optimizer_steps')
        if arm == 'KP':
            ledger.wrap(trainer.bundle.model.released_reference, 'hidden', 'reference_forwards')
        event_started = time.perf_counter()
        result = trainer.train_event(f'event_{event:03d}', pool['features'], passes=2,
                                     presentation_schedule=ordinary)
        event_wall_s = time.perf_counter()-event_started
        require_complete_work(read_work(audit/'WORK.jsonl', limits), limits)
        check = original.validate_event_receipt(result, event_number=event,
            objective='osd_first_error', reference_schedule=ordinary, trainer_status=trainer.status())
        write_json(audit/'EVENT_CHECK.json', check)
        retained = trainer.events_root/f'event_{event:03d}'/'training_state.pt'
        os.link(trainer.latest, retained)
        result.update(arm=arm, latest_state_path=str(trainer.latest), full_state_path=str(retained),
                      state_retention='Immutable event hard link; latest atomically replaced')
        write_json(retained.parent/'RESULT.json', result)
        validation_started = time.perf_counter()
        saved = torch.load(retained, map_location='cpu', weights_only=False)
        checked = validate_own(saved, plan, config, arm, seed, event)
        original._export_equal(saved, result['export_path'])
        write_json(audit/'SAVED_ENDPOINT.json', dict(checked, export_exact=True))
        verify_live(trainer, saved, audit/'LIVE_BOUNDARY.json')
        del saved
        bindings = {str(retained): identity(retained)}
        bindings.update({str(p): identity(p) for p in sorted(Path(result['export_path']).iterdir()) if p.is_file()})
        write_json(audit/'STATE_IDENTITIES.json', bindings)
        costs = dict(loading_initialization_resume_validation_s=loading_s,
            frozen_reference_setup_s=trainer.reference_setup_s, event_wall_s=event_wall_s,
            **result['costs'], objective_work=result['objective_work'],
            state_validation_hash_s=time.perf_counter()-validation_started,
            loading_peak_allocated_bytes=loading_peak['allocated'], loading_peak_reserved_bytes=loading_peak['reserved'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            device_name=torch.cuda.get_device_name(0),
            compilation_scope='Cold native work, no extra warmup; checkpoint recomputation counted by objective',
            response_observations=len(trainer.response_records))
        write_json(audit/'COSTS.json', costs)
        receipt.update(status='completed', result_path=str(retained.parent/'RESULT.json'))
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        if trainer is not None:
            receipt['trainer_status'] = trainer.status()
            trainer.close()
            write_json(audit/'CLOSED.json', trainer.status())
        receipt['work'] = read_work(audit/'WORK.jsonl', limits)
        receipt['worker_wall_s'] = time.perf_counter()-started
        write_json(audit/'WORKER.json', receipt)
        gc.collect()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--event', type=int, required=True)
    args = parser.parse_args()
    train_worker(args.root, args.seed, args.arm, args.event)


if __name__ == '__main__':
    main()
