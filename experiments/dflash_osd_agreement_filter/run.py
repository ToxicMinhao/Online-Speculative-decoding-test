"""Metadata planning and explicitly reviewed execution of the bounded pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.dflash_per_turn.dataset import load_streams, MANIFEST_SHA256, PACKAGE_SHA256
from experiments.dflash_per_turn.run import qualify_text, write_json
from experiments.dflash_per_turn.framing import load_pinned_tokenizer
from .experiment import plan, run_stream, require, digest
from .references import prepare_references, load_references, file_binding
from .report import write_reports

HERE = Path(__file__).resolve().parent


def source_inventory():
    paths = []
    for package in ('dflash_osd_agreement_filter', 'dflash_per_turn', 'dflash_two_update_followup', 'dflash_two_update'):
        paths.extend((HERE.parent / package).glob('*.py'))
        paths.extend((HERE.parent / package).glob('*.json'))
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if not p.name.startswith('test_')}


def run_phase(streams, references, backend_factory, out, *, config, qualification):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    results = []
    started = time.monotonic()
    state = dict(status='running', qualification=qualification, results=[])
    for stream in streams:
        folder = out / stream['stream_id']; folder.mkdir()
        def emit(kind, data):
            with (folder / 'EVENTS.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(event=kind, data=data), ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush(); os.fsync(handle.fileno())
        def factory(arm, stream_id):
            require(stream_id == stream['stream_id'], 'Backend stream mismatch')
            return backend_factory({**config, 'output_root': str(folder / 'native' / arm),
                'agreement_arm': arm, 'agreement_qualification': qualification,
                'remaining_diagnostic_cases': 0, 'dataset_source': stream['source']})
        result = run_stream(stream, references[stream['stream_id']], factory, emit=emit,
            snapshot=lambda value: write_json(folder / 'PROGRESS.json', value, replace=True),
            qualification=qualification)
        results.append(result)
        write_json(folder / 'RESULT.json', result)
        write_reports([result], folder / 'report')
        state['results'].append(dict(stream_id=stream['stream_id'], status=result['status'],
            result_binding=file_binding(folder / 'RESULT.json'), checks=result['implementation_checks']))
        state['actual_wall_s'] = time.monotonic() - started
        write_json(out / 'STATE.json', state, replace=True)
        if result['status'] != 'complete' or result['implementation_checks']['status'] != 'passed':
            state.update(status='stopped', reason='Correctness/output gate failed; preserve state, no automatic retry.')
            break
    else:
        state['status'] = 'complete'
    state['actual_wall_s'] = time.monotonic() - started
    write_json(out / 'STATE.json', state, replace=True)
    write_reports(results, out / 'report')
    return state


def run_qualified_campaign(streams, references, backend_factory, out, *, config, fixtures):
    started = time.monotonic()
    require(fixtures.get('status') == 'passed', 'Deterministic correctness fixtures failed')
    write_json(Path(out) / 'DETERMINISTIC_QUALIFICATION.json', fixtures)
    qualification = run_phase(streams[:1], references, backend_factory, Path(out) / 'qualification',
                              config=config, qualification=True)
    result = dict(qualification=qualification, campaign=None, status='qualification_failed')
    if qualification['status'] == 'complete':
        # Distinct directories and freshly constructed workers/optimizers/RNG for every arm.
        result['campaign'] = run_phase(streams, references, backend_factory, Path(out) / 'campaign',
                                      config=config, qualification=False)
        result['status'] = result['campaign']['status']
    result['actual_campaign_wall_s_including_qualification'] = time.monotonic() - started
    write_json(Path(out) / 'EXECUTION.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-references', 'plan', 'qualify-text', 'run'))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--historical-results-root', type=Path)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--allow-gpu-execution', action='store_true')
    parser.add_argument('--reviewed-code-sha256')
    parser.add_argument('--reviewed-reference-sha256')
    args = parser.parse_args(argv)
    require(not args.out.exists() and not args.out.resolve().is_relative_to(HERE.parents[1]),
            'Use a fresh artifact directory outside the public checkout')
    streams = load_streams(args.dataset)
    sources = source_inventory(); code_hash = digest(sources)
    if args.action != 'run':
        require(not any((args.allow_gpu_execution, args.runtime_config, args.reviewed_code_sha256,
                         args.reviewed_reference_sha256)), 'Metadata actions cannot launch GPU workers')
    if args.action == 'prepare-references':
        require(args.historical_results_root is not None and args.references is None,
                'Bind the six existing complete historical RESULT.json artifacts')
        files = {s['stream_id']: args.historical_results_root / s['stream_id'] / 'RESULT.json' for s in streams}
        bundle = prepare_references(streams, files)
        args.out.mkdir(parents=True)
        write_json(args.out / 'REFERENCES.json', bundle)
        write_json(args.out / 'SOURCE_BINDINGS.json', [x['source_binding'] for x in bundle['streams']])
        print(json.dumps(dict(status='prepared', model_loaded=False, benchmark_contexts=12,
                              reference_binding=file_binding(args.out / 'REFERENCES.json'))))
        return 0
    require(args.references is not None and args.historical_results_root is None, 'Prepared reference bundle required')
    references = load_references(args.references, streams)
    reference_binding = file_binding(args.references)
    if args.action == 'run':
        require(args.allow_gpu_execution and args.runtime_config and args.tokenizer,
                'Review and explicit GPU execution arguments required')
        require(args.reviewed_code_sha256 == code_hash, 'Code differs from reviewed source inventory')
        require(args.reviewed_reference_sha256 == reference_binding['sha256'], 'Canonical reference binding changed')
    args.out.mkdir(parents=True)
    write_json(args.out / 'PLAN.json', plan(streams))
    write_json(args.out / 'SOURCE_MANIFEST.json', dict(code_sha256=sources, reviewed_code_sha256=code_hash,
        dataset_manifest_sha256=MANIFEST_SHA256, dataset_package_sha256=PACKAGE_SHA256,
        reference_binding=reference_binding))
    if args.action == 'plan':
        print(json.dumps(dict(status='review_only', gpu_started=False, reviewed_code_sha256=code_hash,
                              reference_sha256=reference_binding['sha256'], update_events=48,
                              optimizer_steps=96, scored_replays=60)))
        return 0
    require(args.tokenizer is not None, 'Pinned tokenizer required')
    tokenizer = load_pinned_tokenizer(args.tokenizer)
    write_json(args.out / 'TEXT_QUALIFICATION.json', qualify_text(streams, tokenizer))
    del tokenizer
    if args.action == 'qualify-text':
        print(json.dumps(dict(status='passed', gpu_started=False)))
        return 0
    config = json.loads(args.runtime_config.read_text())
    require(config.get('output_cap') == 1024 and config.get('block_size') == 10 and
            config.get('remaining_diagnostic_cases', 0) == 0, 'Qualified protocol changed')
    write_json(args.out / 'RUNTIME_CONFIG.json', config)
    from .qualification import run_deterministic_checks
    from .native import NativeBackend
    result = run_qualified_campaign(streams, references, NativeBackend, args.out, config=config,
                                    fixtures=run_deterministic_checks())
    print(json.dumps(result))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
