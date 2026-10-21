"""Select parents, seal new prompts, plan, or explicitly run a read-only control.

Preparation is CPU-only. A separate reviewed invocation is required for native
execution; no default command generates responses or starts a training worker.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from experiments.dflash_baseline_learning_curve.run import write_json, _git_identity
from experiments.dflash_baseline_learning_curve.experiment import require, digest, preserve_invalid_numbers
from experiments.dflash_kl_native_diagnostic.evidence import file_binding

HERE = Path(__file__).resolve().parent


def _separate(path, protected):
    for root in protected:
        require(not path.is_relative_to(root) and not root.is_relative_to(path),
                'New artifacts must remain separate from originals and public checkout')


def plan(bundle, manifest, groups):
    from .protocol import SCHEMA, PLANNED, SERVING_CONTRACT
    from .storage import build_storage_plan
    return dict(schema=SCHEMA, status='planned', gpu_started=False,
        manifest_sha256=digest(manifest), group_order=[g['group_id'] for g in groups],
        group_bindings=[g['group_payload_sha256'] for g in groups],
        arms=['F', 'S8'], **PLANNED,
        order='Manifest group order; F child0, F child1, own S8 child0, own S8 child1',
        initialization='Qualified runtime initialization retained; no additional generation warmups',
        reference='Fresh natural F per child; exact S8 output-token and termination equality',
        serving_contract=dict(SERVING_CONTRACT),
        storage_budget=build_storage_plan(groups, dict(bundle['runtime_config'],serving_contract=dict(SERVING_CONTRACT))),
        continuation='First complete group correctness gate within budget, then remaining five unchanged',
        interpretation='Constructed related instances; summed paired calls primary; no fixed-order speedup claim',
        paused='Other arms, agreement-filter, correction-bank and matched-prefix campaigns remain paused')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('select', 'seal', 'plan', 'run', 'report'))
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--diagnostic-root', type=Path)
    parser.add_argument('--parent-selection', type=Path)
    parser.add_argument('--construction', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--sealed-plan', type=Path)
    parser.add_argument('--native-root', type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--approved-code-sha256')
    parser.add_argument('--allow-gpu-execution', action='store_true')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    require(not out.exists(), 'Use a fresh private output directory')
    _separate(out, [HERE.parents[1]])
    execution_options = (args.allow_gpu_execution, args.approved_code_sha256, args.sealed_plan, args.native_root)
    if args.action != 'run':
        require(not any(execution_options), 'Only run accepts execution options')
    if args.action == 'report':
        require(args.result and not any((args.baseline_root, args.diagnostic_root, args.parent_selection, args.construction,
                args.tokenizer, args.manifest)), 'Report reads only an existing result')
        _separate(out, [args.result.resolve().parent])
        from .report import write_reports
        write_reports(json.loads(args.result.read_text()), out)
        return 0
    require(args.baseline_root and args.result is None, 'Original baseline evidence is required')
    protected = [HERE.parents[1], args.baseline_root.resolve()]
    if args.diagnostic_root:
        protected.append(args.diagnostic_root.resolve())
    _separate(out, protected)
    from .evidence import load_evidence, seal_children, bind_children
    require(args.action not in ('plan','run') or args.diagnostic_root is not None,
            'Plan/run must bind retained PR10 parent-source native context')
    bundle = load_evidence(args.baseline_root, args.diagnostic_root)
    from .native import source_pins
    inventory = source_pins()
    code_hash = digest(inventory)
    git = _git_identity()
    provenance = dict(git=git, related_instance_source_sha256=inventory,
        approved_code_sha256=code_hash, original_evidence=bundle['bindings'],
        command_argv=sys.argv if argv is None else argv)
    if args.action == 'select':
        require(not any((args.parent_selection, args.construction, args.tokenizer, args.manifest)),
                'Selection only reads original evidence')
        from .evidence import write_prompt_only_views
        write_prompt_only_views(bundle, out)
        write_json(out/'SOURCE_MANIFEST.json', provenance)
        print(json.dumps(dict(status='parents_selected', gpu_started=False, selected_parents=12)))
        return 0
    if args.action == 'seal':
        require(args.parent_selection and args.construction and args.tokenizer and args.manifest is None,
                'Sealing requires frozen parent selection, construction notes and pinned tokenizer')
        selection = json.loads(args.parent_selection.read_text())
        require(selection == bundle['parent_selection'], 'Frozen parent selection changed before construction')
        from experiments.dflash_per_turn.framing import load_pinned_tokenizer
        tokenizer = load_pinned_tokenizer(args.tokenizer)
        manifest = seal_children(bundle, json.loads(args.construction.read_text()), tokenizer)
        out.mkdir(parents=True)
        write_json(out/'REQUEST_MANIFEST.json', manifest)
        provenance.update(parent_selection=file_binding(args.parent_selection),
                          construction=file_binding(args.construction))
        write_json(out/'SOURCE_MANIFEST.json', provenance)
        print(json.dumps(dict(status='sealed', gpu_started=False, requests=12,
                              manifest_sha256=digest(manifest))))
        return 0
    require(args.manifest and not any((args.parent_selection,args.construction)),
            'Plan/run requires the complete sealed request manifest')
    manifest = json.loads(args.manifest.read_text())
    groups = bind_children(bundle, manifest)
    allocation = plan(bundle, manifest, groups)
    allocation['plan_sha256'] = digest(allocation)
    if args.action == 'run':
        require(args.allow_gpu_execution and args.approved_code_sha256 == code_hash and not git['dirty'],
                'Native execution requires explicit authorization and clean reviewed source')
        require(args.sealed_plan and args.tokenizer, 'Sealed plan and pinned tokenizer required for execution')
        require(json.loads(args.sealed_plan.read_text()) == allocation, 'Current assignments differ from sealed plan')
        from experiments.dflash_per_turn.framing import load_pinned_tokenizer, encode_history
        tokenizer = load_pinned_tokenizer(args.tokenizer)
        for group in groups:
            for child in group['children']:
                require(encode_history(tokenizer,[dict(role='user',content=child['text'])]) == child['input_ids'],
                        'Child exact input-token IDs differ on reencoding')
        del tokenizer
    else:
        require(args.tokenizer is None, 'Plan reads the already sealed token IDs')
    out.mkdir(parents=True)
    write_json(out/'PLAN.json', allocation)
    write_json(out/'REQUEST_MANIFEST.json', manifest)
    provenance['request_manifest'] = file_binding(args.manifest)
    provenance['sealed_plan'] = file_binding(args.sealed_plan) if args.sealed_plan else None
    write_json(out/'SOURCE_MANIFEST.json', provenance)
    if args.action == 'plan':
        print(json.dumps(dict(status='planned',gpu_started=False,approved_code_sha256=code_hash,
                              plan_sha256=allocation['plan_sha256'])))
        return 0
    native_root = args.native_root.resolve() if args.native_root else out/'native'
    require(not native_root.exists(), 'Native destination must be fresh')
    _separate(native_root, protected)
    require(native_root != out and not out.is_relative_to(native_root),
            'Native destination cannot contain controller output')
    from .protocol import SERVING_CONTRACT
    from .native import NativeBackend, storage_preflight
    config = dict(bundle['runtime_config'],output_root=str(native_root),controller_root=str(out),
        related_instance_source_sha256=inventory,serving_contract=dict(SERVING_CONTRACT),
        storage_budget=allocation['storage_budget'])
    try:
        preflight = storage_preflight(config)
    except Exception as error:
        write_json(out/'STORAGE_PREFLIGHT.json',dict(getattr(error,'storage_preflight',{}),
            status='failed',error=str(error),gpu_started=False,original_evidence_preserved=True))
        raise
    write_json(out/'STORAGE_PREFLIGHT.json',preflight)
    write_json(out/'RUNTIME_CONFIG.json',config)
    from .experiment import run_campaign
    from .report import write_reports
    def emit(kind,value):
        with (out/'EVENTS.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=kind,data=preserve_invalid_numbers(value)),allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
    result = run_campaign(groups,lambda group:NativeBackend(config),emit=emit,
                          snapshot=lambda value:write_json(out/'PROGRESS.json',value))
    result['source_manifest'] = file_binding(out/'SOURCE_MANIFEST.json')
    result['request_manifest'] = file_binding(out/'REQUEST_MANIFEST.json')
    result['plan_sha256'] = allocation['plan_sha256']
    write_json(out/'RESULT.json',result)
    write_reports(result,out/'report')
    print(json.dumps(dict(status=result['status'],planned_native_generations=24)))
    return 0 if result['status']=='complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
