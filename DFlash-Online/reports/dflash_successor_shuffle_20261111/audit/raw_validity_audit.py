"""Independent raw-output, native-trace and scalar reconciliation; CPU only."""

from pathlib import Path
import argparse
import hashlib
import json
import time
import traceback


def audit(root):
    files = {}

    def read(relative):
        path = root / relative
        content = path.read_bytes()
        files[str(relative)] = {'bytes': len(content),
                                'sha256': hashlib.sha256(content).hexdigest()}
        if path.suffix == '.jsonl':
            return [json.loads(line) for line in content.splitlines()]
        return json.loads(content)

    plan = read('sealed/PLAN.json')
    panel = read('sealed/PANEL.json')['requests']
    references = read('references/CANONICAL.json')
    checkpoints = read('CHECKPOINTS.json')
    rows = read('evaluation/MEASUREMENTS.jsonl')
    assert len(rows) == 1536 and len(panel) == 256
    assert read('POSTFLIGHT.json')['status'] == 'passed'
    assert read('STATUS.json')['status'] == 'completed'
    seen, calls, sessions, count_repeats, first_blocks = set(), 0, set(), {}, {}
    for measurement in rows:
        ordinal, arm, repeat = (measurement[k] for k in ('panel_ordinal', 'method', 'repeat'))
        key = ordinal, arm, repeat
        assert key not in seen and arm in ('K', 'TRUE', 'SHUFFLED') and repeat in (0, 1)
        seen.add(key)
        request = panel[ordinal]
        canonical = references[request['request_id']]
        record = read('evaluation/' + measurement['label'] + '.json')
        native_path = Path(record['metadata']['native_trace_path']).relative_to(plan['root'])
        sidecar = read(native_path)
        blocks = record['metadata']['trace']
        assert record['metadata']['capture'] == sidecar
        assert sidecar['blocks'] == blocks
        assert record['record_id'] == measurement['label']
        assert record['purpose'] == 'evaluation'
        assert record['checkpoint_id'] == checkpoints[arm]['checkpoint_path']
        assert record['prompt_ids'] == request['prompt_ids'] == canonical['prompt_ids']
        assert record['response_ids'] == canonical['response_ids']
        assert record['finish_reason'] == canonical['finish_reason']
        assert record['max_new_tokens'] == 2048
        assert record['metadata']['feedback_capture_in_timed_request'] is False
        assert record['metadata']['block_size'] == 16
        assert record['metadata']['context_limit'] == 4096
        sessions.add(record['session_id'])
        tokens = record['response_ids']
        if record['finish_reason'] == 'eos':
            assert tokens[-1] in record['stop_token_ids']
        else:
            assert record['finish_reason'] == 'length' and len(tokens) == 2048
        cursor = 0
        for block in blocks:
            assert block['output_start'] == cursor
            assert block['anchor_token'] == tokens[cursor]
            committed = block['committed_tokens']
            assert committed == tokens[cursor + 1:cursor + 1 + len(committed)]
            assert block['committed_progress'] == len(committed)
            accepted = block['accepted_draft_tokens']
            assert block['draft_tokens'][:accepted] == committed[:accepted]
            cursor += len(committed)
        assert cursor == len(tokens) - 1
        metrics = {'verification_calls': len(blocks),
                   'accepted_draft_tokens': sum(b['accepted_draft_tokens'] for b in blocks),
                   'native_accepted_draft_tokens': sum(b['native_accepted_draft_tokens'] for b in blocks),
                   'committed_progress': cursor, 'output_tokens': len(tokens)}
        for name, value in metrics.items():
            assert measurement[name] == value, (measurement['label'], name)
            if name not in ('native_accepted_draft_tokens', 'output_tokens'):
                assert record['metadata'][name] == value
        call_key = ordinal, arm
        if call_key in count_repeats:
            assert count_repeats[call_key] == metrics
            assert first_blocks[call_key] == (blocks[0] if blocks else None)
        else:
            count_repeats[call_key] = metrics
            first_blocks[call_key] = blocks[0] if blocks else None
        assert measurement['valid_measurement'] is True and measurement['errors'] == []
        assert measurement['validation_status'] == 'valid'
        calls += len(blocks)
    assert seen == {(i, arm, repeat) for i in range(256)
                    for arm in ('K', 'TRUE', 'SHUFFLED') for repeat in (0, 1)}
    assert len(sessions) == 1
    return {'status': 'passed', 'execution_commit': plan['execution_source']['commit'],
            'completed_generations': len(rows), 'paired_requests': len(panel),
            'native_blocks_checked': calls, 'raw_token_and_termination_matches': len(rows),
            'native_sidecars_checked': len(rows), 'repeat_counts_exact': True,
            'first_block_repeats_exact': True, 'files': files,
            'scope': 'Direct raw token, trace and scalar checks; no production validator imports or model work.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier audit')
    start = time.perf_counter()
    result = {'status': 'failed'}
    try:
        result = audit(args.root)
    except BaseException:
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result['wall_s'] = time.perf_counter() - start
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
