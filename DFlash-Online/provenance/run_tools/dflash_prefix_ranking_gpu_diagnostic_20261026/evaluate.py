"""Sixteen fixed-input generations to assess one-step numerical differences."""

import argparse
import json
from pathlib import Path
import time

from online_feedback.recipe import load_event_config
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.storage import write_json
from experiments.osd_components.protocol import (
    publication_errors, weight_check_errors, record_errors,
)


def evaluate(config_path, manifest_path, trial_root, output):
    config, _ = load_event_config(config_path)
    requests = json.loads(Path(manifest_path).read_text())['eval_requests'][:4]
    checkpoints = {'F': str(config.model.draft_checkpoint_path)}
    checkpoints.update({arm: str(Path(trial_root) / arm / 'export')
                        for arm in ('O1', 'O2', 'U0')})
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'PLAN.json', dict(checkpoints=checkpoints, generations=16,
               panel_ordinals=[r['panel_ordinal'] for r in requests],
               repeats=1, capture=False, timing_claim=False))
    backend = None
    rows = []
    started = time.perf_counter()
    status = dict(status='running', attempted=0, completed=0, valid=0)
    try:
        backend = SGLangDFlashBackend(config.model.target_model_path,
            config.model.draft_checkpoint_path,
            worker_python='/data/conda_envs/minhao/adaflash_author/bin/python',
            output_root=root / 'native', serving_gpu='0', context_limit=4096, block_size=16)
        status['loading_s'] = backend.loading_s
        # Group by checkpoint to avoid needless loading. There are no timing claims.
        for arm, checkpoint in checkpoints.items():
            publication = backend.publish(checkpoint)
            errors = publication_errors(publication, checkpoint)
            write_json(root / f'{arm}_PUBLICATION.json', dict(receipt=publication, errors=errors))
            if errors:
                raise RuntimeError(f'{arm} publication: {errors}')
            weights = backend.check_weights(checkpoint)
            errors = weight_check_errors(weights, checkpoint)
            write_json(root / f'{arm}_WEIGHTS.json', dict(receipt=weights, errors=errors))
            if errors:
                raise RuntimeError(f'{arm} exported/native weights: {errors}')
            for request in requests:
                label = f'{arm}_{request["panel_ordinal"]:02d}'
                status['attempted'] += 1
                write_json(root / 'STATUS.json', status)
                record = backend.generate(request['prompt_ids'],
                    max_new_tokens=request['max_new_tokens'], request_id=label,
                    session_id='lambda_zero_diagnostic', request_index=request['panel_ordinal'],
                    purpose='evaluation', save_to=root / f'{label}.json')
                status['completed'] += 1
                errors = record_errors(record, request, arm, checkpoint)
                metadata = record['metadata']
                row = dict(arm=arm, request_id=request['request_id'],
                    panel_ordinal=request['panel_ordinal'], errors=errors, valid=not errors,
                    output_tokens=len(record['response_ids']),
                    verification_calls=metadata['verification_calls'],
                    accepted_draft_tokens=metadata['accepted_draft_tokens'],
                    committed_progress=metadata['committed_progress'],
                    request_latency_s=metadata['request_latency_s'])
                rows.append(row)
                status['valid'] += int(not errors)
                write_json(root / 'ROWS.json', rows)
                write_json(root / 'STATUS.json', status)
                if errors:
                    raise RuntimeError(f'{label} generation validity: {errors}')
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        if backend is not None:
            backend.close()
        status['wall_s'] = time.perf_counter() - started
        status['unattempted'] = 16 - status['attempted']
        write_json(root / 'STATUS.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'trial-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.manifest, args.trial_root, args.output)
