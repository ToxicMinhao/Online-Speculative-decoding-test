"""Input-only selection for the independent GSM8K and MathQA role-loss pilot."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import zipfile

from experiments.mathqa_transfer.data import (
    SOURCE_BLOB, SOURCE_REVISION, TOKENIZER_HASHES, assemble, normalize, token_digest,
)
from online_feedback.storage import write_json

PREFIX = 'vfd-roleloss-pilot-v1'
REASONING_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
TRAIN_COUNT, PANEL_COUNT = 512, 256


def selection_key(dataset, split, source_id):
    return hashlib.sha256(f'{PREFIX}|{dataset}|{split}|{source_id}'.encode('utf-8')).hexdigest()


def inherited_source_id(row):
    return row.get('source_id', row.get('dataset_id'))


def prompt_fields(row, dataset, tokenizer):
    """Recover input text for overlap checks without rewriting serving tokens."""
    rendered = tokenizer.decode(row['prompt_ids'], skip_special_tokens=False)
    match = re.fullmatch(r'<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n.*',
                         rendered, flags=re.S)
    if match is None:
        raise ValueError('Expected the retained single-user Qwen wrapper')
    user = match.group(1)
    stem = row.get('question', user.removesuffix(REASONING_SUFFIX))
    return dict(source_id=inherited_source_id(row), dataset=dataset,
                question=stem, normalized_stem=normalize(stem), rendered_prompt=rendered,
                rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                prompt_ids_sha256=token_digest(row['prompt_ids']))


def select_panel(rows, dataset, tokenizer, count=PANEL_COUNT):
    ids = [inherited_source_id(row) for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('Panel must retain distinct original source identities')
    ranked = sorted(rows, key=lambda r: (selection_key(dataset, 'test', inherited_source_id(r)),
                                         inherited_source_id(r)))
    if len(ranked) < count:
        raise ValueError('Insufficient inherited evaluation inputs')
    selected = []
    for ordinal, original in enumerate(ranked[:count]):
        row = deepcopy(original)
        row.update(prompt_fields(row, dataset, tokenizer), panel_ordinal=ordinal,
                   inherited_panel_ordinal=original.get('panel_ordinal', original.get('ordinal')),
                   selection_key=selection_key(dataset, 'test', inherited_source_id(original)))
        if row['collect_feedback'] or row['max_new_tokens'] != 2048 or len(row['prompt_ids']) + 2048 > 4096:
            raise ValueError('Inherited panel serving contract changed')
        selected.append(row)
    return selected


def overlap_keys(row):
    return {'source_id': (row['dataset'], row['source_id']),
            'normalized_stem': row['normalized_stem'],
            'rendered_prompt_sha256': row['rendered_prompt_sha256'],
            'prompt_ids_sha256': row['prompt_ids_sha256']}


def overlap_index(rows):
    index = {key: set() for key in ('source_id', 'normalized_stem',
                                   'rendered_prompt_sha256', 'prompt_ids_sha256')}
    for row in rows:
        for key, value in overlap_keys(row).items():
            index[key].add(value)
    return index


def overlap_reasons(row, index):
    return [key for key, value in overlap_keys(row).items() if value in index[key]]


def read_mathqa_train(path):
    """Authenticate the ZIP and expose only question/options, never answer fields."""
    data = Path(path).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError('MathQA ZIP differs from the pinned repository blob')
    with zipfile.ZipFile(path) as archive:
        raw = archive.read('train.json')
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 29837:
        raise ValueError('Expected all 29,837 MathQA training rows')
    inputs = [{key: row.get(key) for key in ('Problem', 'options')} for row in rows]
    return inputs, dict(repository='math-QA/math-QA', revision=SOURCE_REVISION, git_blob=blob,
                        member='train.json', member_bytes=len(raw), rows=len(rows),
                        member_sha256=hashlib.sha256(raw).hexdigest(),
                        archive_sha256=hashlib.sha256(data).hexdigest(), source_answers_used=False)


def select_mathqa_training(inputs, panels, tokenizer, stop_token_ids, count=TRAIN_COUNT):
    """Deduplicate in source order, exclude panels, then select by input ID hash."""
    evaluation = overlap_index(panels)
    seen_stems, seen_prompts = set(), set()
    eligible, excluded = [], []
    for source_row, source in enumerate(inputs):
        source_id = f'mathqa:train:{source_row}'
        message = assemble(source)
        if message is None:
            excluded.append(dict(source_id=source_id, reason='malformed_or_empty_input'))
            continue
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': message}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        row = dict(source_id=source_id, source_row=source_row, dataset='mathqa',
            question=source['Problem'].strip(), normalized_stem=normalize(source['Problem']),
            rendered_prompt=rendered, prompt_ids=ids, prompt_ids_sha256=token_digest(ids),
            rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            selection_key=selection_key('mathqa', 'train', source_id))
        duplicate = row['normalized_stem'] in seen_stems or row['rendered_prompt_sha256'] in seen_prompts
        seen_stems.add(row['normalized_stem'])
        seen_prompts.add(row['rendered_prompt_sha256'])
        reasons = overlap_reasons(row, evaluation)
        if duplicate:
            reason = 'duplicate_earlier_source_row'
        elif reasons:
            reason = 'evaluation_overlap:' + ','.join(reasons)
        elif not ids or len(ids) + 2048 > 2272:
            reason = 'training_context_budget'
        else:
            eligible.append(row)
            continue
        excluded.append(dict(source_id=source_id, source_row=source_row, reason=reason,
                             prompt_tokens=len(ids)))
    eligible.sort(key=lambda r: (r['selection_key'], r['source_id']))
    if len(eligible) < count:
        raise ValueError(f'MathQA eligible shortfall: {len(eligible)} < {count}')
    selected = []
    for ordinal, row in enumerate(eligible[:count]):
        selected.append(dict(row, request_id=f'train_{ordinal:05d}', dataset_id=row['source_id'],
            dataset_index=ordinal, request_index=ordinal + 1, max_new_tokens=2048,
            stop_token_ids=stop_token_ids, collect_feedback=True,
            event_id=f'event_{ordinal // 128 + 1:03d}'))
    return selected, dict(eligible_count=len(eligible), exclusions=excluded,
        unselected_eligible_ids=[row['source_id'] for row in eligible[count:]],
        deduplication='ascending_source_row; first_valid_input_wins',
        prompt_limit=224, training_context_limit=2272, output_cap=2048)


def prepare(args):
    from transformers import AutoTokenizer
    for name, expected in TOKENIZER_HASHES.items():
        if hashlib.sha256((Path(args.tokenizer) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Tokenizer differs from the qualified native tokenizer: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    gsm_root, math_root = Path(args.gsm_sealed), Path(args.math_root)
    gsm_plan = json.loads((gsm_root / 'PLAN.json').read_text())
    original = json.loads((gsm_root / 'ORIGINAL_MANIFEST.json').read_text())
    math_plan = json.loads((math_root / 'sealed/PLAN.json').read_text())
    panels = {name: select_panel(plan['evaluation_requests'], name, tokenizer)
              for name, plan in [('gsm8k', gsm_plan), ('mathqa', math_plan)]}
    all_panel = panels['gsm8k'] + panels['mathqa']
    evaluation = overlap_index(all_panel)
    gsm_training = []
    for original_row in original['train_requests'][:TRAIN_COUNT]:
        row = deepcopy(original_row)
        row.update(prompt_fields(row, 'gsm8k', tokenizer),
                   stop_token_ids=gsm_plan['evaluation_requests'][0]['stop_token_ids'])
        reasons = overlap_reasons(row, evaluation)
        if reasons:
            raise ValueError(f'Fixed GSM8K training input overlaps a selected panel: {row["source_id"]}: {reasons}')
        gsm_training.append(row)
    if len(gsm_training) != TRAIN_COUNT:
        raise ValueError('Fixed GSM8K first-512 stream is incomplete')
    math_inputs, math_source = read_mathqa_train(args.math_archive)
    math_training, exclusions = select_mathqa_training(math_inputs, all_panel, tokenizer,
                                                      math_plan['evaluation_requests'][0]['stop_token_ids'])
    canonicals = {
        'gsm8k': json.loads((gsm_root / 'CANONICALS.json').read_text()),
        'mathqa': json.loads((math_root / 'campaign_001/evaluation/final/CANONICAL.json').read_text()),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for dataset, train in [('gsm8k', gsm_training), ('mathqa', math_training)]:
        directory = output / dataset
        directory.mkdir()
        panel = panels[dataset]
        refs = {row['request_id']: canonicals[dataset][row['request_id']] for row in panel}
        for row in panel:
            reference = refs[row['request_id']]
            if not reference['response_ids'] or reference['finish_reason'] not in ('stop', 'length', 'eos'):
                raise ValueError('Retained canonical response is incomplete')
        write_json(directory / 'TRAINING.json', dict(schema='role_loss_training_inputs_v1',
            dataset=dataset, train_requests=train, policy=original['policy'],
            source=original['provenance'] if dataset == 'gsm8k' else math_source,
            source_answers_used=False, training_request_count=len(train)))
        write_json(directory / 'PANEL.json', dict(dataset=dataset, requests=panel,
            inherited_count=1024, selection_key=f'{PREFIX}|{dataset}|test|<original_source_id>',
            previously_used_development_panel=True, input_only_selection=True))
        write_json(directory / 'CANONICALS.json', refs)
        write_json(directory / 'DIAGNOSTIC_INPUTS.json', dict(dataset=dataset,
            requests=panel[:16], max_anchors_per_request=8,
            anchor_hash_template=f'{PREFIX}|{dataset}|diagnostic-anchor|{{source_id}}|{{absolute_position}}'))
    write_json(output / 'MATHQA_EXCLUSIONS.json', exclusions)
    write_json(output / 'PREPARATION.json', dict(status='prepared', input_only=True,
        no_model_execution=True, source_answers_used=False, datasets=list(panels),
        feedback_requests=1024, evaluation_requests=512, inherited_references=512,
        missing_references=0, source_tokenizer=str(Path(args.tokenizer)),
        mathqa_source=math_source))
    return dict(status='prepared', output=str(output), training_requests=1024,
                evaluation_requests=512, missing_references=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gsm-sealed', 'math-root', 'math-archive', 'tokenizer', 'output'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
