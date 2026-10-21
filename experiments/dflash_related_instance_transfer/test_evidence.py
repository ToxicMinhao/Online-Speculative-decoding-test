"""CPU-only admission regressions; synthetic output IDs never enter author views."""
import copy
import json

import pytest

from . import evidence as ev
from .protocol import EXPECTED_PARENT_SUFFIXES, SAMPLE_MANIFEST_SHA256, SEMANTIC_REVIEW_FIELDS


def make_bundle():
    groups, selected = [], []
    for i, (gid, suffixes) in enumerate(EXPECTED_PARENT_SUFFIXES.items()):
        domain = gid.rsplit('_', 1)[0]
        frozen_files = {'model.safetensors': {'bytes': 1, 'sha256': 'f' * 64}}
        adapted_files = {'model.safetensors': {'bytes': 1, 'sha256': str(i) * 64}}
        checkpoints = {
            'F': dict(checkpoint_id='F', path='/models/F', files=frozen_files,
                      checkpoint_identity_sha256=ev.digest(frozen_files)),
            'S8': dict(checkpoint_id='U2', path='/retained/' + gid + '/U2', files=adapted_files,
                       checkpoint_identity_sha256=ev.digest(adapted_files))}
        group = dict(group_id=gid, domain=domain, status='complete', feedback=[],
                     evaluation_requests=[], scores=[], checkpoints=checkpoints)
        manifest_group = dict(group_id=gid, domain=domain, feedback=[], evaluation=[])
        remaining = [j for j in range(4) if f'{j:02}' not in suffixes]
        lengths = {remaining[0]: 1, int(suffixes[0]): 2, int(suffixes[1]): 3, remaining[1]: 4}
        for role_index, role in enumerate(('feedback', 'evaluation')):
            for j in range(4):
                request = f'{gid}_{role}_{j:02}'
                row = dict(request_id=request, text='Original question ' + request,
                           input_ids=[1000 + 100*i + 10*role_index + j])
                manifest_group[role].append(row)
                if role == 'feedback':
                    group['feedback'].append(dict(request_id=request, group_id=gid, domain=domain,
                        role=role, input_ids=row['input_ids'], output_ids=[200+j] * lengths[j],
                        checkpoint_identity_sha256=checkpoints['F']['checkpoint_identity_sha256']))
                else:
                    group['evaluation_requests'].append(copy.deepcopy(row))
        groups.append(group)
        selected.append(manifest_group)
    manifest = dict(groups=selected, tokenizer=ev.qualified_tokenizer_binding())
    result = dict(groups=groups, manifest=manifest, runtime_config={},
                  bindings={'files': {'SAMPLE_MANIFEST.json': {'sha256': SAMPLE_MANIFEST_SHA256}}})
    result['parent_selection'] = ev.select_parents(groups, manifest)
    result['original_prompts'] = ev.prompt_catalog(manifest)
    return result


def make_construction(bundle):
    return dict(author=dict(name='synthetic author', tool='pytest fixture', version='1',
                            prior_outcome_exposure='Synthetic fixture; no real outcomes',
                            blind_to_prior_outcomes=False),
        parent_selection_sha256=bundle['parent_selection']['parent_selection_sha256'],
        original_prompts_sha256=bundle['original_prompts']['original_prompts_sha256'],
        no_model_outputs_observed=True, pre_seal_revisions=[],
        children=[dict(request_id=f'new_child_{i:02}', parent_request_id=p['parent_request_id'],
                       group_id=p['group_id'], domain=p['domain'], role='related_instance',
                       text=f'New task with changed answer-relevant requirement {i}',
                       construction_method='Synthetic task content replacement',
                       changes_made='The required operation and supplied input differ.',
                       why_required_answer_differs='The requested result differs under the new operation.',
                       semantic_review={key: True for key in SEMANTIC_REVIEW_FIELDS})
                  for i, p in enumerate(bundle['parent_selection']['parents'])])


def make_renderer(bundle, construction):
    values = {r['text']: r['input_ids'] for g in bundle['manifest']['groups']
              for role in ('feedback', 'evaluation') for r in g[role]}
    values.update({child['text']: [10000+i, 128009, 100] for i, child in enumerate(construction['children'])})
    return lambda text: copy.deepcopy(values[text])


def make_sealed():
    bundle = make_bundle()
    construction = make_construction(bundle)
    sealed = ev.seal_children(bundle, construction, render=make_renderer(bundle, construction))
    return bundle, construction, sealed


def reseal(manifest):
    manifest['sealed_manifest_sha256'] = ev._payload(manifest, 'sealed_manifest_sha256')


def test_fixed_parent_selection_and_prompt_only_firewall(tmp_path):
    bundle = make_bundle()
    paths = ev.write_prompt_only_views(bundle, tmp_path)
    parents = json.loads(open(paths['PARENTS_PROMPT_ONLY.json']).read())
    catalog = json.loads(open(paths['ORIGINAL_PROMPTS_ONLY.json']).read())
    assert len(parents['parents']) == 12 and len(catalog['prompts']) == 48
    assert all(set(p) == {'parent_request_id', 'group_id', 'domain', 'text'} for p in parents['parents'])
    assert all(set(p) == {'request_id', 'group_id', 'domain', 'role', 'text'} for p in catalog['prompts'])
    assert ev.write_prompt_only_views(bundle, tmp_path) == paths
    (tmp_path / 'PARENT_SELECTION.json').write_text('{}')
    with pytest.raises(ValueError, match='Refusing to replace'):
        ev.write_prompt_only_views(bundle, tmp_path)


def test_prompt_packets_cannot_be_written_into_public_checkout():
    with pytest.raises(ValueError, match='outside the public checkout'):
        ev.write_prompt_only_views(make_bundle(), ev.Path(ev.__file__).resolve().parent / 'private-data')


def test_load_delegates_exact_original_validation(monkeypatch):
    calls = []
    original = make_bundle()
    monkeypatch.setattr(ev, 'load_retained_evidence', lambda root: calls.append(root) or copy.deepcopy(original))
    result = ev.load_evidence('/private/original')
    assert calls == ['/private/original']
    assert result['groups'] == original['groups']
    assert result['parent_selection'] == original['parent_selection']
    original['bindings']['files']['SAMPLE_MANIFEST.json']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='manifest pin'):
        ev.load_evidence('/private/original')


def test_optional_paired_context_does_not_change_prompt_selection_or_child_manifest(monkeypatch):
    from experiments.dflash_lower_lr_control import evidence as paired
    original = make_bundle()
    enriched = copy.deepcopy(original)
    for group in enriched['groups']:
        group['reference_pairs'] = {'source': [{'private_retained_metric': 0.1}], 'heldout': []}
    calls = []
    monkeypatch.setattr(ev, 'load_retained_evidence', lambda root: copy.deepcopy(original))
    monkeypatch.setattr(paired, 'load_evidence',
        lambda root, diagnostic: calls.append((root, diagnostic)) or copy.deepcopy(enriched))
    plain = ev.load_evidence('/private/original')
    contextual = ev.load_evidence('/private/original', '/private/diagnostic')
    assert calls == [('/private/original', '/private/diagnostic')]
    assert plain['parent_selection'] == contextual['parent_selection']
    assert plain['original_prompts'] == contextual['original_prompts']
    construction = make_construction(plain)
    sealed_plain = ev.seal_children(plain, construction, render=make_renderer(plain, construction))
    sealed_context = ev.seal_children(contextual, construction, render=make_renderer(contextual, construction))
    assert sealed_plain == sealed_context
    assert ev.bind_children(contextual, sealed_context)[0]['reference_pairs'] == enriched['groups'][0]['reference_pairs']


def test_ties_use_original_manifest_order_and_fixed_parent_ids():
    bundle = make_bundle()
    for row in bundle['groups'][0]['feedback']:
        row['output_ids'] = [9, 8]
    selected = ev.select_parents(bundle['groups'], bundle['manifest'])
    assert [p['parent_request_id'] for p in selected['parents'][:2]] == [
        'gsm8k_00_feedback_01', 'gsm8k_00_feedback_02']
    assert [p['original_manifest_order'] for p in selected['parents'][:2]] == [1, 2]
    bundle['groups'][0]['feedback'][0]['output_ids'] *= 20
    with pytest.raises(ValueError, match='handoff'):
        ev.select_parents(bundle['groups'], bundle['manifest'])


def test_seal_and_binding_preserve_original_groups_without_aliases():
    bundle, _, sealed = make_sealed()
    before = copy.deepcopy(bundle)
    groups = ev.bind_children(bundle, sealed)
    assert len(groups) == 6 and all(len(g['children']) == 2 for g in groups)
    assert sealed['execution_order'][:4] == [dict(group_id='gsm8k_00', checkpoint_id=a, request_id=r)
        for a in ('F', 'S8') for r in ('new_child_00', 'new_child_01')]
    for original, group in zip(bundle['groups'], groups):
        assert all(group[k] == v for k, v in original.items())
        assert ev.validate_group(group)['status'] == 'passed'
    groups[0]['feedback'][0]['output_ids'].append(9)
    groups[0]['children'][0]['text'] = 'changed'
    assert bundle == before
    assert sealed['children'][0]['text'] != 'changed'


@pytest.mark.parametrize('change', [
    'missing_child', 'extra_child', 'old_id', 'duplicate_id', 'wrong_role', 'foreign_parent',
    'foreign_group', 'foreign_domain', 'reordered', 'missing_changes', 'missing_answer_difference',
    'missing_review', 'failed_review', 'claim_blinded', 'observed_output', 'source_answer',
    'history', 'selection_digest', 'catalog_digest', 'duplicate_original', 'duplicate_child',
    'normalized_duplicate', 'missing_version'])
def test_bad_construction_is_rejected_before_rendering(change):
    bundle = make_bundle(); draft = make_construction(bundle)
    c = draft['children'][0]
    if change == 'missing_child': draft['children'].pop()
    if change == 'extra_child': draft['children'].append(copy.deepcopy(c))
    if change == 'old_id': c['request_id'] = c['parent_request_id']
    if change == 'duplicate_id': c['request_id'] = draft['children'][1]['request_id']
    if change == 'wrong_role': c['role'] = 'feedback'
    if change == 'foreign_parent': c['parent_request_id'] = draft['children'][2]['parent_request_id']
    if change == 'foreign_group': c['group_id'] = 'gsm8k_01'
    if change == 'foreign_domain': c['domain'] = 'codealpaca'
    if change == 'reordered': draft['children'].reverse()
    if change == 'missing_changes': c['changes_made'] = ''
    if change == 'missing_answer_difference': c['why_required_answer_differs'] = ''
    if change == 'missing_review': c['semantic_review'].pop('answer_relevant_change')
    if change == 'failed_review': c['semantic_review']['answer_relevant_change'] = False
    if change == 'claim_blinded': draft['author']['blind_to_prior_outcomes'] = True
    if change == 'observed_output': draft['no_model_outputs_observed'] = False
    if change == 'source_answer': c['source_answer'] = 'forbidden'
    if change == 'history': c['history'] = [{'role': 'assistant', 'text': 'forbidden'}]
    if change == 'selection_digest': draft['parent_selection_sha256'] = '0'*64
    if change == 'catalog_digest': draft['original_prompts_sha256'] = '0'*64
    if change == 'duplicate_original': c['text'] = bundle['original_prompts']['prompts'][0]['text']
    if change == 'duplicate_child': c['text'] = draft['children'][1]['text']
    if change == 'normalized_duplicate': c['text'] = '  '+draft['children'][1]['text'].upper().replace(' ', '\n ')+'  '
    if change == 'missing_version': draft['author']['version'] = ''
    def forbidden_render(_):
        pytest.fail('Invalid construction reached token rendering')
    with pytest.raises(ValueError): ev.seal_children(bundle, draft, render=forbidden_render)


@pytest.mark.parametrize('change', ['missing_original', 'child_duplicate_original', 'child_duplicate_child',
    'too_long', 'negative', 'float_token', 'boolean_token', 'empty'])
def test_rendering_must_reproduce_originals_and_new_children_fit(change):
    bundle = make_bundle(); draft = make_construction(bundle)
    good = make_renderer(bundle, draft)
    first_original = bundle['manifest']['groups'][0]['feedback'][0]
    def bad(text):
        if change == 'missing_original' and text == first_original['text']: return [1]
        if text == draft['children'][0]['text']:
            return {'child_duplicate_original': first_original['input_ids'],
                    'child_duplicate_child': good(draft['children'][1]['text']),
                    'too_long': [1]*31736, 'negative': [-1], 'float_token': [1.0],
                    'boolean_token': [True], 'empty': []}.get(change, good(text))
        return good(text)
    with pytest.raises(ValueError): ev.seal_children(bundle, draft, render=bad)


@pytest.mark.parametrize('change', ['parent', 'checkpoint', 'contract', 'budget', 'execution_order',
    'token_hash', 'text_hash', 'source_role', 'missing_child', 'reviewed_text', 'catalog', 'post_output'])
def test_structural_validation_detects_tampering_even_with_recomputed_top_digest(change):
    bundle, _, sealed = make_sealed()
    if change == 'parent': sealed['children'][0]['parent_binding']['selected_rank'] = 1
    if change == 'checkpoint': sealed['children'][0]['checkpoint_bindings']['S8'] = 'f'*64
    if change == 'contract': sealed['serving_contract']['parent_history'] = True
    if change == 'budget': sealed['planned']['native_generations'] = 25
    if change == 'execution_order': sealed['execution_order'].reverse()
    if change == 'token_hash': sealed['children'][0]['input_ids'].append(9)
    if change == 'text_hash': sealed['children'][0]['prompt_sha256'] = '0'*64
    if change == 'source_role': sealed['children'][0]['role'] = 'feedback'
    if change == 'missing_child': sealed['children'].pop()
    if change == 'reviewed_text': sealed['children'][0]['text'] += ' change'
    if change == 'catalog': sealed['original_prompts_sha256'] = '0'*64
    if change == 'post_output': sealed['outputs'] = [{'output_ids': [1, 2]}]
    reseal(sealed)
    with pytest.raises(ValueError): ev.validate_sealed_manifest(bundle, sealed)


def test_recorded_preseal_revision_chain_is_exact():
    bundle = make_bundle(); draft = make_construction(bundle)
    child = draft['children'][0]
    draft['pre_seal_revisions'] = [dict(request_id=child['request_id'], before_text='Old task',
        after_text=child['text'], reason='Changed required answer', before_model_outputs=True)]
    ev.seal_children(bundle, draft, render=make_renderer(bundle, draft))
    draft['pre_seal_revisions'][0]['after_text'] = 'Unrecorded final change'
    with pytest.raises(ValueError, match='last recorded revision'):
        ev.seal_children(bundle, draft, render=make_renderer(bundle, draft))


@pytest.mark.parametrize('change', ['original', 'child', 'seal', 'parent_checkpoint', 'source_role'])
def test_controller_group_admission_rejects_mutation(change):
    bundle, _, sealed = make_sealed()
    group = ev.bind_children(bundle, sealed)[0]
    if change == 'original': group['feedback'][0]['output_ids'].append(100)
    if change == 'child': group['children'][0]['input_ids'].append(100)
    if change == 'seal': group['sealed_manifest_sha256'] = ''
    if change == 'parent_checkpoint': group['children'][0]['parent_binding']['checkpoint_bindings']['S8'] = '0'*64
    if change == 'source_role': group['children'][0]['role'] = 'feedback'
    group['group_payload_sha256'] = ev._payload(group, 'group_payload_sha256')
    with pytest.raises(ValueError): ev.validate_group(group)
