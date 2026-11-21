"""CPU-only audit of the two immutable private evidence bundles."""
import argparse
import io
import json
import tarfile
from pathlib import Path

import torch


def audit(original, diagnostic):
    with tarfile.open(original) as old, tarfile.open(diagnostic) as new:
        def read(archive, name):
            return json.load(archive.extractfile(name))

        retained = {arm: read(old, f'qualification/B/records/B_{arm}_cross_card_0_2048_0.json') for arm in ('AR', 'F')}
        native = {arm: read(new, f'{arm}/OUTPUT.json') for arm in ('AR', 'F')}
        rows = {arm: [json.loads(line) for line in new.extractfile(f'{arm}/TARGET_LOGITS.jsonl')] for arm in ('AR', 'F')}
        for arm in ('AR', 'F'):
            assert native[arm]['output_ids'] == retained[arm]['response_ids']
            assert read(new, f'{arm}/REPRODUCTION.json')['passed']
            assert all(row['finite'] for row in rows[arm])
            assert read(new, f'{arm}/PARAMETER_STABILITY.json')['unchanged_identity_versions_and_layout']
        assert retained['AR']['prompt_ids'] == retained['F']['prompt_ids']
        ar_tokens, f_tokens = native['AR']['output_ids'], native['F']['output_ids']
        first = next(i for i, (a, b) in enumerate(zip(ar_tokens, f_tokens)) if a != b)
        prefix = retained['AR']['prompt_ids'] + ar_tokens[:first]
        position = len(prefix) - 1
        trace = native['F']['trace']['blocks']
        assert trace == retained['F']['metadata']['trace']
        assert len(rows['F']) == len(trace) + 1
        for row, block in zip(rows['F'][1:], trace):
            assert row['argmax'] == block['target_argmax']
            assert row['positions'] == block['positions']
        assert [row['argmax'][0] for row in rows['AR'][:len(ar_tokens)]] == ar_tokens
        summaries = {}
        for arm in ('AR', 'F'):
            # Earlier speculative blocks visit this absolute position with
            # rejected hypotheses. Only the block anchored at the common
            # canonical prefix is the disputed committed decision.
            candidates = [row for row in rows[arm] if row['positions'][0] == position]
            assert len(candidates) == 1
            row = candidates[0]
            assert row['input_ids'][0] == prefix[-1]
            vector = torch.load(io.BytesIO(new.extractfile(f'{arm}/LOGITS_{row["call"]:04d}.pt').read()), map_location='cpu', weights_only=True)
            assert torch.isfinite(vector).all()
            assert vector.argmax().item() == native[arm]['output_ids'][first]
            maximum = vector.max().item()
            ties = torch.nonzero(vector == maximum).flatten().tolist()
            assert vector[[ar_tokens[first], f_tokens[first]]].tolist() == row['rival_logits'][0]
            work = read(new, f'{arm}/WORK.json')
            summaries[arm] = dict(call=row['call'], position=position, native_rows=len(row['logit_positions']),
                logit_dtype=str(vector.dtype), native_choice=int(vector.argmax()),
                rival_logits=row['rival_logits'][0], maximum=maximum, maximum_tied_token_ids=ties,
                output_tokens=len(native[arm]['output_ids']), eos_token=native[arm]['output_ids'][-1],
                verification_calls=len(trace) if arm == 'F' else None,
                forward_calls=read(new, f'{arm}/FORWARD_COUNTS.json'),
                initialization_forwards=sum(item['model_forward_calls'] for item in work['initialization']),
                peak_gpu_allocated_bytes=work['peak_gpu_allocated_bytes'],
                peak_gpu_reserved_bytes=work['peak_gpu_reserved_bytes'],
                costs=read(new, f'{arm}/COST.json'))
        clock = read(new, 'GLOBAL_CLOCK.json')
        return dict(status='passed', scope='Replay fidelity and native-score diagnosis, not AR/F equality',
            execution_commit='7ba747702d5b86628cfed95ac064db4d54469586',
            original_execution_commit='93a53f11abbfbf00e36c1bcc309a33943f972099',
            first_difference=first, common_prefix_tokens=len(prefix), rival_token_ids=[ar_tokens[first], f_tokens[first]],
            arms=summaries, complete=read(new, 'COMPLETE.json'),
            diagnostic_card_s=sum(row['occupied_card_s'] for row in clock['finished'] if row['task'].startswith('native_mismatch_')),
            cumulative_card_s=clock['completed_card_s'], original_clock_epoch=clock['first_model_load_epoch'],
            global_work=read(new, 'GLOBAL_WORK.json'), all_prior_failures_preserved=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.original, args.diagnostic), indent=2) + '\n')
