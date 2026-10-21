"""Select parents and seal private, prompt-only child construction.

Exact historical validation is delegated to the unchanged PR10 reader. Nothing
here loads model weights, generates an answer, trains, or modifies source files.
Semantic novelty is a recorded manual review, not an inference from token hashes.
"""
from __future__ import annotations

import copy
import json
import re
import unicodedata
from pathlib import Path

from experiments.dflash_baseline_learning_curve.experiment import digest, require
from experiments.dflash_baseline_learning_curve.data import encode_request, qualified_tokenizer_binding
from experiments.dflash_kl_native_diagnostic.evidence import load_evidence as load_retained_evidence
from experiments.dflash_per_turn import framing
from .protocol import (ARMS, EXPECTED_PARENT_SUFFIXES, PARENT_SELECTION, PLANNED,
    SAMPLE_MANIFEST_SHA256, SCHEMA, SEMANTIC_REVIEW_FIELDS, SERVING_CONTRACT)

CONSTRUCTION_CHILD_FIELDS = {
    'request_id', 'parent_request_id', 'group_id', 'domain', 'role', 'text',
    'construction_method', 'changes_made', 'why_required_answer_differs', 'semantic_review'}
SEALED_CHILD_ADDITIONS = {
    'input_ids', 'input_ids_sha256', 'prompt_sha256', 'parent_binding', 'checkpoint_bindings'}


def _payload(value, field):
    return digest({k: v for k, v in value.items() if k != field})


def _normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split()).casefold()


def select_parents(groups, manifest):
    """Use only F output lengths and manifest position, never adaptation gains."""
    require([g['group_id'] for g in groups] == list(EXPECTED_PARENT_SUFFIXES) ==
            [g['group_id'] for g in manifest['groups']], 'Original six-group order differs')
    rows = []
    for group, selected in zip(groups, manifest['groups']):
        source = group['feedback']
        require(len(source) == 4 and len(selected['feedback']) == 4,
                'Exactly four source parents per group required')
        require([r['request_id'] for r in source] ==
                [r['request_id'] for r in selected['feedback']], 'Parent manifest order differs')
        ordered = sorted(enumerate(source), key=lambda item: (len(item[1]['output_ids']), item[0]))
        chosen = ordered[1:3]
        require([r['request_id'] for _, r in chosen] ==
                [group['group_id'] + '_feedback_' + suffix
                 for suffix in EXPECTED_PARENT_SUFFIXES[group['group_id']]],
                'Length-ranked parents differ from handoff')
        for rank, (index, native) in enumerate(chosen, 2):
            prompt = selected['feedback'][index]
            require(native['input_ids'] == prompt['input_ids'], 'Parent prompt binding differs')
            rows.append(dict(group_id=group['group_id'], domain=group['domain'],
                parent_request_id=native['request_id'], selected_rank=rank,
                original_manifest_order=index, retained_F_output_tokens=len(native['output_ids']),
                parent_input_ids_sha256=digest(prompt['input_ids']),
                parent_prompt_sha256=digest(prompt['text']),
                checkpoint_bindings={arm: group['checkpoints'][arm]['checkpoint_identity_sha256']
                                     for arm in ARMS}))
    result = dict(schema=SCHEMA, sample_manifest_sha256=SAMPLE_MANIFEST_SHA256,
                  rule=copy.deepcopy(PARENT_SELECTION), parents=rows)
    result['parent_selection_sha256'] = _payload(result, 'parent_selection_sha256')
    return result


def prompt_catalog(manifest):
    """An allowlist intentionally excludes benchmark answers and runtime metrics."""
    rows = [dict(group_id=g['group_id'], domain=g['domain'], role=role,
                 request_id=row['request_id'], text=row['text'])
            for g in manifest['groups'] for role in ('feedback', 'evaluation')
            for row in g[role]]
    require(len(rows) == 48 and len({r['request_id'] for r in rows}) == 48,
            'Complete original48 prompt-only catalog required')
    result = dict(schema=SCHEMA, sample_manifest_sha256=SAMPLE_MANIFEST_SHA256, prompts=rows)
    result['original_prompts_sha256'] = _payload(result, 'original_prompts_sha256')
    return result


def load_evidence(root, diagnostic_root=None):
    """Missing/tampered retained files fail closed through the unchanged PR10 API."""
    if diagnostic_root is None:
        bundle = load_retained_evidence(root)
    else:
        # Full retained paired context is needed only for later reporting. This
        # unchanged reader delegates original validation to the same PR10 API.
        from experiments.dflash_lower_lr_control.evidence import load_evidence as load_paired_evidence
        bundle = load_paired_evidence(root, diagnostic_root)
    require(bundle['bindings']['files']['SAMPLE_MANIFEST.json']['sha256'] == SAMPLE_MANIFEST_SHA256,
            'Original manifest pin differs')
    require(bundle['manifest']['tokenizer'] == qualified_tokenizer_binding(), 'Qualified tokenizer differs')
    bundle['parent_selection'] = select_parents(bundle['groups'], bundle['manifest'])
    bundle['original_prompts'] = prompt_catalog(bundle['manifest'])
    bundle['bindings']['parent_selection_sha256'] = bundle['parent_selection']['parent_selection_sha256']
    bundle['bindings']['original_prompts_sha256'] = bundle['original_prompts']['original_prompts_sha256']
    return bundle


def write_prompt_only_views(bundle, directory):
    """Freeze parent identities before construction; never overwrite another seal."""
    directory = Path(directory).resolve()
    require(not directory.is_relative_to(Path(__file__).resolve().parents[2]),
            'Private prompt preparation must remain outside the public checkout')
    directory.mkdir(parents=True, exist_ok=True)
    selection = bundle['parent_selection']
    catalog = bundle['original_prompts']
    by_id = {row['request_id']: row for row in catalog['prompts']}
    # The author view omits F lengths and checkpoint/metric information.
    parents = dict(schema=SCHEMA,
        parent_selection_sha256=selection['parent_selection_sha256'],
        original_prompts_sha256=catalog['original_prompts_sha256'],
        parents=[dict(parent_request_id=p['parent_request_id'], group_id=p['group_id'],
                      domain=p['domain'], text=by_id[p['parent_request_id']]['text'])
                 for p in selection['parents']])
    outputs = {'PARENT_SELECTION.json': selection, 'PARENTS_PROMPT_ONLY.json': parents,
               'ORIGINAL_PROMPTS_ONLY.json': catalog}
    for name, value in outputs.items():
        path = directory / name
        if path.exists():
            require(json.loads(path.read_text()) == value, 'Refusing to replace frozen preparation: ' + name)
        else:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
    return {name: str((directory / name).resolve()) for name in outputs}


def _validate_construction(bundle, construction):
    expected = {'author', 'parent_selection_sha256', 'original_prompts_sha256',
                'no_model_outputs_observed', 'pre_seal_revisions', 'children'}
    require(isinstance(construction, dict) and set(construction) == expected,
            'Construction fields differ or contain unapproved output/history data')
    require(construction['parent_selection_sha256'] ==
            bundle['parent_selection']['parent_selection_sha256'] and
            construction['original_prompts_sha256'] == bundle['original_prompts']['original_prompts_sha256'],
            'Construction did not use frozen parent selection/original48 prompts')
    require(construction['no_model_outputs_observed'] is True,
            'Children must be finalized before observing any child model output')
    author = construction['author']
    require(isinstance(author, dict) and set(author) ==
            {'name', 'tool', 'version', 'prior_outcome_exposure', 'blind_to_prior_outcomes'},
            'Author/tool/version and prior outcome exposure disclosure required')
    require(all(isinstance(author[k], str) and author[k].strip()
                for k in ('name', 'tool', 'version', 'prior_outcome_exposure')) and
            author['blind_to_prior_outcomes'] is False,
            'Do not claim blinded construction; disclose prior outcome exposure')
    revisions = construction['pre_seal_revisions']
    require(isinstance(revisions, list) and all(isinstance(r, dict) and set(r) ==
            {'request_id', 'before_text', 'after_text', 'reason', 'before_model_outputs'} and
            all(isinstance(r[k], str) and r[k].strip() for k in
                ('request_id', 'before_text', 'after_text', 'reason')) and
            r['before_model_outputs'] is True for r in revisions),
            'Pre-seal revisions must document exact before/after texts and timing')
    children = construction['children']
    require(isinstance(children, list) and len(children) == 12, 'Exactly twelve reviewed children required')
    originals = bundle['original_prompts']['prompts']
    ids = {r['request_id'] for r in originals}
    normalized = {_normalized(r['text']) for r in originals}
    for child, parent in zip(children, bundle['parent_selection']['parents']):
        require(isinstance(child, dict) and set(child) == CONSTRUCTION_CHILD_FIELDS,
                'Child fields differ or contain output/history data')
        require(child['role'] == 'related_instance', 'Child must have a new related_instance role')
        require(all(child[k] == parent[k] for k in ('parent_request_id', 'group_id', 'domain')),
                'Child parent/group/domain/order differs')
        identifier = child['request_id']
        require(isinstance(identifier, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', identifier)
                and identifier not in ids, 'New globally unique safe child ID required')
        ids.add(identifier)
        require(all(isinstance(child[k], str) and child[k].strip() for k in
                    ('text', 'construction_method', 'changes_made', 'why_required_answer_differs')),
                'Child text and answer-relevant construction explanations required')
        require(_normalized(child['text']) not in normalized,
                'Child duplicates an original48 or another child prompt')
        normalized.add(_normalized(child['text']))
        review = child['semantic_review']
        require(isinstance(review, dict) and set(review) == set(SEMANTIC_REVIEW_FIELDS) and
                all(review[k] is True for k in SEMANTIC_REVIEW_FIELDS),
                'Every child requires complete prompt-pair semantic review')
    child_by_id = {r['request_id']: r for r in children}
    last_revision = {}
    for revision in revisions:
        request = revision['request_id']
        require(request in child_by_id, 'Revision belongs to an unknown child')
        require(request not in last_revision or last_revision[request] == revision['before_text'],
                'Pre-seal revision chain is discontinuous')
        require(revision['before_text'] != revision['after_text'], 'Empty pre-seal revision')
        last_revision[request] = revision['after_text']
    require(all(child_by_id[k]['text'] == text for k, text in last_revision.items()),
            'Final child text differs from last recorded revision')


def seal_children(bundle, construction, tokenizer=None, *, render=None):
    """Render once, revalidate all original48, and bind before any model output.

    A tokenizer uses the pinned lossless single-user renderer. A CPU test or
    independently loaded tokenizer may supply ``render(text) -> list[int]``;
    that callable must reproduce every original48 prompt exactly. Native launch
    must additionally verify actual tokenizer files and re-render every child.
    """
    _validate_construction(bundle, construction)
    require((tokenizer is None) != (render is None), 'Provide exactly one tokenizer or render callable')
    if tokenizer is not None:
        if callable(tokenizer) and not hasattr(tokenizer, 'encode'):
            render = tokenizer
        else:
            framing.validate_tokenizer(tokenizer)
            render = lambda text: encode_request(tokenizer, text)
    require(callable(render), 'A callable prompt renderer is required')
    original_ids = set()
    for group in bundle['manifest']['groups']:
        for role in ('feedback', 'evaluation'):
            for row in group[role]:
                require(render(row['text']) == row['input_ids'],
                        'Renderer fails exact original48 prompt reproduction')
                original_ids.add(digest(row['input_ids']))
    children = []
    for original, parent in zip(construction['children'], bundle['parent_selection']['parents']):
        child = copy.deepcopy(original)
        ids = render(child['text'])
        framing.check_input(ids)
        token_digest = digest(ids)
        require(token_digest not in original_ids, 'Child token sequence duplicates an original or child')
        original_ids.add(token_digest)
        child.update(input_ids=ids, input_ids_sha256=token_digest, prompt_sha256=digest(child['text']),
                     parent_binding=copy.deepcopy(parent), checkpoint_bindings=copy.deepcopy(parent['checkpoint_bindings']))
        children.append(child)
    manifest = dict(schema=SCHEMA, status='sealed_before_generation',
        sample_manifest_sha256=SAMPLE_MANIFEST_SHA256,
        parent_selection=copy.deepcopy(bundle['parent_selection']),
        original_prompts_sha256=bundle['original_prompts']['original_prompts_sha256'],
        tokenizer=copy.deepcopy(qualified_tokenizer_binding()),
        rendering=dict(original48_exact_reencoded=True, no_parent_history=True,
                       native_file_verification_and_child_reencoding_required=True),
        construction=copy.deepcopy(construction), children=children,
        serving_contract=copy.deepcopy(SERVING_CONTRACT), planned=copy.deepcopy(PLANNED),
        execution_order=[dict(group_id=gid, checkpoint_id=arm, request_id=c['request_id'])
                         for gid in EXPECTED_PARENT_SUFFIXES for arm in ARMS
                         for c in children if c['group_id'] == gid])
    manifest['sealed_manifest_sha256'] = _payload(manifest, 'sealed_manifest_sha256')
    validate_sealed_manifest(bundle, manifest)
    return manifest


def validate_sealed_manifest(bundle, manifest):
    """Check all available bindings independently of the top-level digest."""
    require(isinstance(manifest, dict) and set(manifest) == {
        'schema', 'status', 'sample_manifest_sha256', 'parent_selection',
        'original_prompts_sha256', 'tokenizer', 'rendering', 'construction',
        'children', 'serving_contract', 'planned', 'execution_order', 'sealed_manifest_sha256'},
        'Sealed manifest fields differ or contain post-output data')
    require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'sealed_before_generation',
            'A pre-output child seal is required')
    require(manifest.get('sealed_manifest_sha256') == _payload(manifest, 'sealed_manifest_sha256'),
            'Sealed child manifest changed')
    require(manifest['parent_selection'] == select_parents(bundle['groups'], bundle['manifest']) ==
            bundle['parent_selection'], 'Sealed parent selection changed')
    require(manifest['original_prompts_sha256'] == prompt_catalog(bundle['manifest'])['original_prompts_sha256'] ==
            bundle['original_prompts']['original_prompts_sha256'], 'Original48 catalog binding changed')
    require(manifest['sample_manifest_sha256'] == SAMPLE_MANIFEST_SHA256 and
            manifest['tokenizer'] == qualified_tokenizer_binding() and
            manifest['planned'] == PLANNED and manifest['serving_contract'] == SERVING_CONTRACT and
            manifest['rendering'] == dict(original48_exact_reencoded=True, no_parent_history=True,
                       native_file_verification_and_child_reencoding_required=True),
            'Sealed execution/tokenizer contract changed')
    _validate_construction(bundle, manifest['construction'])
    require(len(manifest['children']) == 12, 'All twelve sealed children required')
    seen = {digest(r['input_ids']) for g in bundle['manifest']['groups']
            for role in ('feedback', 'evaluation') for r in g[role]}
    extra = SEALED_CHILD_ADDITIONS
    for child, draft, parent in zip(manifest['children'], manifest['construction']['children'],
                                    bundle['parent_selection']['parents']):
        require(set(child) == set(draft) | extra and
                {k: v for k, v in child.items() if k not in extra} == draft,
                'Sealed child differs from reviewed construction')
        framing.check_input(child['input_ids'])
        require(child['input_ids_sha256'] == digest(child['input_ids']) and
                child['prompt_sha256'] == digest(child['text']) and
                child['parent_binding'] == parent and child['checkpoint_bindings'] == parent['checkpoint_bindings'],
                'Child token/text/parent/checkpoint binding differs')
        require(child['input_ids_sha256'] not in seen, 'Duplicate rendered child tokens')
        seen.add(child['input_ids_sha256'])
    expected_order = [dict(group_id=gid, checkpoint_id=arm, request_id=c['request_id'])
                      for gid in EXPECTED_PARENT_SUFFIXES for arm in ARMS
                      for c in manifest['children'] if c['group_id'] == gid]
    require(manifest['execution_order'] == expected_order and len(expected_order) == 24,
            'Fixed F/F/S8/S8 execution order changed')
    return dict(status='passed', errors=[])


def bind_children(bundle, sealed_manifest):
    """Preserve original groups; attach explicit new child records and their seal."""
    validate_sealed_manifest(bundle, sealed_manifest)
    groups = []
    for original in bundle['groups']:
        group = copy.deepcopy(original)
        group['children'] = [copy.deepcopy(c) for c in sealed_manifest['children']
                             if c['group_id'] == original['group_id']]
        group['original_group_payload_sha256'] = digest(original)
        group['sealed_manifest_sha256'] = sealed_manifest['sealed_manifest_sha256']
        group['group_payload_sha256'] = _payload(group, 'group_payload_sha256')
        validate_group(group)
        groups.append(group)
    return groups


def validate_group(group):
    """Controller admission for an immutable original group plus two children.

    Whole-campaign admission must first call ``validate_sealed_manifest`` and
    compare these groups with ``bind_children``; a group digest alone cannot
    authenticate the private source manifest or a manual semantic judgment.
    """
    added = {'children', 'original_group_payload_sha256', 'sealed_manifest_sha256', 'group_payload_sha256'}
    require(isinstance(group, dict) and added <= set(group), 'Missing related-instance group seal')
    require(group['group_payload_sha256'] == _payload(group, 'group_payload_sha256'), 'Group payload changed')
    original = {k: v for k, v in group.items() if k not in added}
    require(group['original_group_payload_sha256'] == digest(original), 'Original group payload changed')
    gid = group['group_id']
    require(gid in EXPECTED_PARENT_SUFFIXES and group['status'] == 'complete', 'Invalid original group')
    require(isinstance(group['sealed_manifest_sha256'], str) and
            re.fullmatch(r'[0-9a-f]{64}', group['sealed_manifest_sha256']), 'Missing campaign manifest seal')
    checkpoints = group['checkpoints']
    for arm in ARMS:
        require(checkpoints[arm]['checkpoint_identity_sha256'] == digest(checkpoints[arm]['files']),
                'Checkpoint file inventory changed')
    require(checkpoints['F']['checkpoint_id'] == 'F' and checkpoints['S8']['checkpoint_id'] == 'U2' and
            checkpoints['F']['checkpoint_identity_sha256'] != checkpoints['S8']['checkpoint_identity_sha256'],
            'Retained F and own S8 checkpoints required')
    require(len(group['feedback']) == 4 and len(group['evaluation_requests']) == 4,
            'Original four-source/four-heldout populations changed')
    selected = sorted(enumerate(group['feedback']),
                      key=lambda item: (len(item[1]['output_ids']), item[0]))[1:3]
    expected_ids = [gid + '_feedback_' + suffix for suffix in EXPECTED_PARENT_SUFFIXES[gid]]
    require([row['request_id'] for _, row in selected] == expected_ids, 'Retained parent ranks changed')
    children = group['children']
    require(isinstance(children, list) and len(children) == 2, 'Exactly two group children required')
    seen_ids = {r['request_id'] for r in group['feedback'] + group['evaluation_requests']}
    seen_tokens = {digest(r['input_ids']) for r in group['feedback'] + group['evaluation_requests']}
    for rank, (child, (index, parent)) in enumerate(zip(children, selected), 2):
        require(isinstance(child, dict) and set(child) == CONSTRUCTION_CHILD_FIELDS | SEALED_CHILD_ADDITIONS,
                'Child contains unapproved output/history fields')
        require(child['role'] == 'related_instance' and child['group_id'] == gid and
                child['domain'] == group['domain'] and child['parent_request_id'] == parent['request_id'],
                'Child role/owner/parent differs')
        require(isinstance(child['request_id'], str) and
                re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', child['request_id']) and
                child['request_id'] not in seen_ids, 'Child ID collides with original or sibling')
        seen_ids.add(child['request_id'])
        framing.check_input(child['input_ids'])
        require(child['input_ids_sha256'] == digest(child['input_ids']) and
                child['input_ids_sha256'] not in seen_tokens and child['prompt_sha256'] == digest(child['text']),
                'Child exact token/text binding differs or duplicates original/sibling')
        seen_tokens.add(child['input_ids_sha256'])
        expected_parent = dict(group_id=gid, domain=group['domain'], parent_request_id=parent['request_id'],
            selected_rank=rank, original_manifest_order=index, retained_F_output_tokens=len(parent['output_ids']),
            parent_input_ids_sha256=digest(parent['input_ids']),
            parent_prompt_sha256=child['parent_binding']['parent_prompt_sha256'],
            checkpoint_bindings={arm: checkpoints[arm]['checkpoint_identity_sha256'] for arm in ARMS})
        require(child['parent_binding'] == expected_parent and
                child['checkpoint_bindings'] == expected_parent['checkpoint_bindings'],
                'Child parent/checkpoint binding changed')
        require(all(isinstance(child[k], str) and child[k].strip() for k in
                    ('text', 'construction_method', 'changes_made', 'why_required_answer_differs')) and
                set(child['semantic_review']) == set(SEMANTIC_REVIEW_FIELDS) and
                all(child['semantic_review'][k] is True for k in SEMANTIC_REVIEW_FIELDS),
                'Child construction/semantic review missing')
    return dict(status='passed', errors=[])
