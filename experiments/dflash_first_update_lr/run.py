"""Execute exactly the fixed six-stream first-update comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.dflash_per_turn.dataset import load_streams
from experiments.dflash_per_turn.framing import load_pinned_tokenizer, encode_history
from experiments.dflash_per_turn.run import write_json
from .experiment import run_stream, validate_reference, EXPECTED_STREAMS, digest, require
from .report import write_reports

HERE = Path(__file__).resolve().parent


def source_inventory():
    paths = []
    for name in ('dflash_first_update_lr', 'dflash_per_turn', 'dflash_two_update_followup', 'dflash_two_update'):
        paths.extend((HERE.parent/name).glob('*.py'))
        paths.extend((HERE.parent/name).glob('*.json'))
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if not p.name.startswith('test_')}


def build_references(original_root, dataset, tokenizer):
    references = []
    streams = load_streams(dataset)
    require([s['stream_id'] for s in streams] == list(EXPECTED_STREAMS), 'Dataset order differs')
    for stream in streams:
        path = Path(original_root)/stream['stream_id']/'RESULT.json'
        original = json.loads(path.read_text())
        require(original['status'] == 'complete' and original['implementation_checks']['status'] == 'passed',
                'Original pilot was not qualified')
        canonical = original['canonical'][:2]
        messages = []
        for turn, row in enumerate(canonical, 1):
            prompt = stream['requests'][turn-1]['prompt']
            require(row['source_user_text'] == prompt, 'Source whitespace/text changed')
            messages.append(dict(role='user', content=prompt))
            require(encode_history(tokenizer, messages) == row['input_ids'], 'Exact saved history reconstruction failed')
            messages.append(dict(role='assistant', content=row['text'], output_ids=row['output_ids']))
        reference = dict(episode_id=stream['stream_id'], source=stream['source'], synthetic=stream['synthetic'],
            canonical=canonical, original_update=original['updates'][0],
            original_support=original['support'][0],
            original_scores=[r for r in original['scores'] if r['turn'] in (1, 2)],
            original_result_binding=dict(path=str(path), bytes=path.stat().st_size,
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        validate_reference(reference)
        references.append(reference)
    return references


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--runtime-config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--allow-gpu-execution', action='store_true')
    args = p.parse_args(argv)
    require(args.allow_gpu_execution, 'Explicit execution authorization required')
    inventory = source_inventory()
    require(digest(inventory) == args.source_sha256, 'Source inventory differs')
    require(not args.out.exists(), 'Preserve prior attempt; use fresh output path')
    references = build_references(args.original_root, args.dataset, load_pinned_tokenizer(args.tokenizer))
    config = json.loads(args.runtime_config.read_text())
    require(config['output_cap'] == 1024 and config['block_size'] == 10, 'Serving protocol changed')
    args.out.mkdir(parents=True)
    write_json(args.out/'SOURCE_MANIFEST.json', dict(files=inventory, sha256=args.source_sha256))
    write_json(args.out/'REFERENCES.json', references)
    write_json(args.out/'RUNTIME_CONFIG.json', config)
    write_json(args.out/'PLAN.json', dict(updates=12, optimizer_steps=24, timed_executions=108,
        seed=17, learning_rates=[3e-4, 1e-4], new_answers_generated_for_training=0,
        output_cap=1024, block_size=10, order='request2 then request1; D1/L1/F rotates each repeat',
        note='First D1 request2 timed replay is reproduction gate, not an extra execution.'))
    from .native import NativeBackend
    results = []; started = time.monotonic()
    campaign = dict(status='running', results=[])
    for reference in references:
        stream_out = args.out/reference['episode_id']; stream_out.mkdir()
        def emit(kind, data):
            with (stream_out/'EVENTS.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(event=kind, data=data), allow_nan=False)+'\n')
                handle.flush(); os.fsync(handle.fileno())
        backend = NativeBackend({**config, 'output_root': str(stream_out/'native')})
        result = run_stream(reference, backend, emit=emit,
            snapshot=lambda r: write_json(stream_out/'PROGRESS.json', r, replace=True))
        write_json(stream_out/'RESULT.json', result)
        results.append(result)
        campaign['results'].append(dict(episode_id=result['episode_id'], status=result['status'],
            updates=len(result['updates']), scores=len(result['scores']),
            reproduction_check=result['reproduction_check'], error=result.get('error')))
        campaign['actual_wall_s'] = time.monotonic()-started
        write_reports(results, args.out/'report')
        if result['status'] != 'complete':
            campaign.update(status='stopped', reason='Failed validity/reproduction gate; investigate, no automatic retry.')
            write_json(args.out/'CAMPAIGN_STATE.json', campaign, replace=True)
            return 1
        write_json(args.out/'CAMPAIGN_STATE.json', campaign, replace=True)
    campaign.update(status='complete', actual_wall_s=time.monotonic()-started)
    write_json(args.out/'CAMPAIGN_STATE.json', campaign, replace=True)
    print(json.dumps(campaign))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
