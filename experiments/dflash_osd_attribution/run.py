"""Explicit inventory, offline analysis and bounded native diagnostic commands.

No command follows another automatically. Probe execution requires the reviewed
source digest and a new output directory; claimed blocks are never retried.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from experiments.dflash_correction_bank.artifacts import atomic_json, binding, sha
from .bound import TOLERANCE_FORMULA
from .prepare import Resolver, inventory_campaign
from .protocol import (STREAMS, SCHEMA, MAX_PAIRS, MAX_BLOCK_EVALUATIONS,
                       digest, require, compare_anchor)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GIB = 2**30


def source_manifest():
    # Include dependencies, not tests or historical measurements. External
    # native runtime/author/model pins stay in each original NATIVE_CONFIG.
    folders = ('dflash_osd_attribution', 'dflash_correction_bank', 'dflash_per_turn',
               'dflash_two_update_followup', 'dflash_two_update')
    paths = [p for name in folders for p in (HERE.parent/name).iterdir()
             if p.suffix in ('.py', '.json') and not p.name.startswith('test_')]
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def seal_plan(plan):
    value = copy.deepcopy(plan)
    value.update(source_sha256=source_manifest(), tolerance_formula=TOLERANCE_FORMULA,
                 bound_radius_scope='combined_effective_matrix_including_fusion',
                 mathematical_precision='float64', native_precision_reported_separately=True)
    value['source_digest'] = digest(value['source_sha256'])
    value['plan_digest'] = digest(value)
    return value


def validate_plan(plan):
    value = copy.deepcopy(plan); expected = value.pop('plan_digest')
    require(digest(value) == expected, 'Plan changed after inventory')
    require(plan['schema'] == SCHEMA and plan['source_sha256'] == source_manifest() and
            plan['source_digest'] == digest(source_manifest()), 'Reviewed implementation changed')
    require(plan['tolerance_formula'] == TOLERANCE_FORMULA and
            plan['bound_radius_scope'] == 'combined_effective_matrix_including_fusion', 'Bound declaration changed')
    anchors = plan['anchors']
    require(len(anchors) <= MAX_PAIRS and plan['planned_draft_block_evaluations'] == 2*len(anchors)
            <= MAX_BLOCK_EVALUATIONS, 'Block budget changed')
    require(len({a['anchor_id'] for a in anchors}) == len(anchors), 'Duplicate anchor')
    for stream, count in STREAMS.items():
        selected = [a for a in anchors if a['stream'] == stream]
        require(len(selected) <= 32 and all(a['turn'] == count for a in selected), 'Non-final or excess anchors')
    require(all(a['stream'] in STREAMS for a in anchors), 'Undeclared stream')
    return Resolver(plan.get('mapping', []))


def fresh_output(path, plan):
    path = Path(path).resolve()
    require(not path.is_relative_to(Path(plan['campaign_root'])), 'Outputs must not modify the original campaign')
    path.mkdir(parents=True, exist_ok=False)
    return path


def pinned_decoder(plan, resolver):
    from transformers import AutoTokenizer
    errors = []
    for case in plan['cases']:
        try:
            config = resolver.load(case['native_config'])
            target = config['target']; model_path = resolver.path(target['model_path'])
            tokenizer = resolver.path(config['tokenizer_path'])
            require(tokenizer == model_path, 'Tokenizer is not the pinned target directory')
            files = target['checkpoint_files']
            relevant = [name for name in files if name.startswith(('tokenizer', 'vocab', 'merges', 'special_tokens', 'added_tokens'))]
            require('tokenizer.json' in relevant and 'tokenizer_config.json' in relevant, 'Tokenizer bindings unavailable')
            for name in relevant:
                resolver.checked(dict(path=str(Path(target['model_path'])/name), **files[name]))
            tok = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True)
            return (lambda token: tok.decode([token], skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)), dict(
                path=str(tokenizer), files={n:files[n] for n in relevant}, source=case['native_config'])
        except (OSError, KeyError, ValueError) as error:
            errors.append(str(error))
    raise ValueError('No available pinned tokenizer: '+repr(errors))


def analyze(plan, out):
    from .analysis import analyze_campaign
    from .report import write_report
    resolver = validate_plan(plan)
    output = fresh_output(out, plan)
    atomic_json(output/'PLAN.json', plan)
    results, missing = [], []
    for case in plan['cases']:
        try:
            result = resolver.load(case['result'])
            require(result['episode_id'] == case['stream'] and
                    result['planned_requests'] == STREAMS[case['stream']] and
                    result['status'] == 'complete' and result['qualification'] is False,
                    'Result is not the completed main-stream record')
            results.append(result)
        except (KeyError, OSError, ValueError) as error:
            missing.append(dict(stream=case['stream'], reason=str(error)))
    try:
        decoder, tokenizer = pinned_decoder(plan, resolver)
        tokenizer['status'] = 'available'
    except (ImportError, OSError, ValueError) as error:
        decoder = lambda _: None
        tokenizer = dict(status='unavailable', reason=str(error), substitute_tokenizer=False,
                         token_categories='uncategorized; exact token-ID recurrence still available')
    value = analyze_campaign(results, resolver.load, decoder, require_full_campaign=False)
    value.update(missing_streams=missing, tokenizer=tokenizer, plan_digest=plan['plan_digest'],
                 original_request_budget=94, planned_requests=94, newly_executed_models=0)
    write_report(output/'report', analysis=value, inventory=plan)
    return value


def storage_preflight(path, pairs):
    # No checkpoint export, optimizer, full vocabulary storage, or downloads.
    # 1GiB covers <=320 small feature artifacts/config/JSON; allow another
    # 2GiB for native compiler/cache/log growth, in addition to an 8GiB reserve.
    require(0 <= pairs <= MAX_PAIRS, 'Unbounded diagnostic storage scope')
    free = shutil.disk_usage(path).free
    value = dict(free_bytes=free, evidence_peak_bytes=GIB, runtime_growth_bytes=2*GIB,
                 reserve_bytes=8*GIB, required_bytes=11*GIB,
                 checkpoint_copies=0, training_states=0, planned_pairs=pairs)
    require(free >= value['required_bytes'], 'Insufficient diagnostic storage: '+repr(value))
    return value


def preflight_anchors(plan, resolver):
    """Verify before any model launch; record missing cases, never replace them."""
    from .prepare import inventory_checkpoint
    ready, missing = [], []
    for stream in STREAMS:
        selected = [a for a in plan['anchors'] if a['stream'] == stream]
        if not selected:
            continue
        try:
            a = selected[0]
            resolver.checked(a['native_config']); resolver.checked(a['frozen_trajectory'])
            resolver.checked(a['projections'])
            for arm in ('F', 'OSD'):
                receipt = inventory_checkpoint(a['checkpoints'][arm], Resolver())
                require(receipt['status'] == 'available', 'Exact serving checkpoint unavailable: '+arm)
            ready.extend(selected)
        except (OSError, KeyError, ValueError) as error:
            missing.append(dict(stream=stream, selected_anchor_ids=[a['anchor_id'] for a in selected], reason=str(error)))
    return ready, missing


def probe(plan, out, reviewed_source_digest, *, client_factory=None, preflight=storage_preflight):
    from .report import write_report
    resolver = validate_plan(plan)
    require(reviewed_source_digest == plan['source_digest'], 'Supply the exact reviewed source digest')
    output = fresh_output(out, plan)
    atomic_json(output/'PLAN.json', plan)
    ready, missing = preflight_anchors(plan, resolver)
    atomic_json(output/'PREFLIGHT.json', dict(storage=preflight(output, len(ready)),
                missing=missing, ready_pairs=len(ready), model_work_started=False))
    # One durable claim at the PLAN's original location also prevents silently
    # rerunning the same diagnostic into a different output directory.
    claim_root = Path(plan['execution_claim_root'])
    claim_root.mkdir(parents=True, exist_ok=False)
    atomic_json(claim_root/'RUN.json', dict(plan_digest=plan['plan_digest'], output=str(output)))
    if client_factory is None:
        from .native_probe import NativeProbeClient
        client_factory = NativeProbeClient
    started = time.monotonic(); rows, comparisons, consumed = {}, [], 0
    failure = None
    try:
        for stream in STREAMS:
            selected = [a for a in ready if a['stream'] == stream]
            if not selected:
                continue
            preflight(output, len(ready))
            config = resolver.load(selected[0]['native_config'])
            original_config = copy.deepcopy(config)
            from experiments.dflash_correction_bank import native
            config.update(attribution_probe_plan=selected, output_root=str(output),
                episode_out=str(output/stream), released=selected[0]['checkpoints']['F'],
                correction_bank_source_sha256=native.source_pins())
            # Original runtime, model, author, precision and serving pins remain
            # unchanged. Bank source pin refresh is explicit for this new entry
            # point; the original config is retained beside the execution config.
            atomic_json(output/stream/'ORIGINAL_NATIVE_CONFIG.json', original_config)
            atomic_json(output/stream/'PROBE_CONFIG.json', config)
            client = client_factory(config, output/stream/'native')
            try:
                # No timing comparison here. Grouping reduces checkpoint loads.
                for arm in ('F', 'OSD'):
                    for anchor in selected:
                        require(consumed < MAX_BLOCK_EVALUATIONS, 'Block budget exhausted')
                        key = digest([anchor['anchor_id'], arm])
                        atomic_json(claim_root/(key+'.json'), dict(anchor_id=anchor['anchor_id'], arm=arm,
                                    claimed_before_model_work=True))
                        consumed += 1
                        row = client.evaluate(anchor, anchor['checkpoints'][arm], arm, projections=anchor['projections'])
                        atomic_json(output/'blocks'/(key+'.json'), row)
                        rows[(anchor['anchor_id'], arm)] = row
                        if arm == 'OSD':
                            paired = compare_anchor(anchor, rows[(anchor['anchor_id'], 'F')], row)
                            comparisons.append(paired)
                            atomic_json(output/'pairs'/(digest(anchor['anchor_id'])+'.json'), paired)
            finally:
                client.close()
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error), automatic_retry=False)
    diagnostic = dict(comparisons=comparisons, missing=missing, claimed_block_evaluations=consumed,
        returned_block_evaluations=len(rows), max_block_evaluations=MAX_BLOCK_EVALUATIONS,
        wall_s=time.monotonic()-started, failure=failure, plan_digest=plan['plan_digest'],
        timing_scope='Diagnostic wall time only; no serving speedup claim', training_steps=0,
        regenerated_responses=0, extra_timing_repeats=0)
    atomic_json(output/'DIAGNOSTIC.json', diagnostic)
    write_report(output/'report', diagnostic=diagnostic, inventory=plan)
    if failure:
        raise RuntimeError('Diagnostic stopped without retry; see '+str(output/'DIAGNOSTIC.json'))
    return diagnostic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    sub = subs.add_parser('inventory')
    sub.add_argument('--campaign-root', required=True, type=Path)
    sub.add_argument('--out', required=True, type=Path)
    sub.add_argument('--map-root', nargs=2, action='append', default=[], metavar=('ORIGINAL', 'RETAINED'))
    sub = subs.add_parser('report')
    sub.add_argument('--plan', required=True, type=Path)
    sub.add_argument('--analysis', required=True, type=Path, help='Offline report/REPORT.json')
    sub.add_argument('--diagnostic', required=True, type=Path, help='Native DIAGNOSTIC.json')
    sub.add_argument('--out', required=True, type=Path)
    for name in ('analyze', 'probe'):
        sub = subs.add_parser(name)
        sub.add_argument('--plan', required=True, type=Path)
        sub.add_argument('--out', required=True, type=Path)
        if name == 'probe':
            sub.add_argument('--reviewed-source-digest', required=True)
    args = parser.parse_args(argv)
    if args.command == 'inventory':
        require(not args.out.exists(), 'Inventory must have a fresh output directory')
        require(not args.out.resolve().is_relative_to(args.campaign_root.resolve()), 'Inventory cannot modify original evidence')
        value = inventory_campaign(args.campaign_root, Resolver(args.map_root))
        value['execution_claim_root'] = str(args.out.resolve()/'execution_claims')
        plan = seal_plan(value)
        atomic_json(args.out/'PLAN.json', plan)
        print(json.dumps(dict(plan=str(args.out/'PLAN.json'), ready_pairs=len(plan['anchors']),
                              source_digest=plan['source_digest'], models_executed=0)))
    else:
        plan = json.loads(args.plan.read_text())
        if args.command == 'report':
            from .report import write_report
            validate_plan(plan)
            analysis = json.loads(args.analysis.read_text())['analysis']
            diagnostic = json.loads(args.diagnostic.read_text())
            require(analysis['plan_digest'] == diagnostic['plan_digest'] == plan['plan_digest'], 'Evidence plans differ')
            output = fresh_output(args.out, plan)
            write_report(output, analysis=analysis, diagnostic=diagnostic, inventory=plan)
        elif args.command == 'analyze':
            analyze(plan, args.out)
        else:
            probe(plan, args.out, args.reviewed_source_digest)


if __name__ == '__main__':
    main()
