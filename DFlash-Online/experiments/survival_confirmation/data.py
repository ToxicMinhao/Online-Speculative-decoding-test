"""Input-only official-row selection for frozen SURVIVAL confirmation.

Historical reference lookup is deliberately separate and must follow this seal.
No answer column, model output, latency, or reference-availability field ranks data.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

REVISION = '740312add88f781978c0658806c59bc2815b9866'
TOKENIZER_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
SOURCE_HASHES = {
    'test': 'ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59',
    'train': 'ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d',
}
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
PREFIX = 'survival-confirm-v1|gsm8k|test|'
STOP_IDS = [151645, 151643]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def normalize_question(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Require a nonempty source question')
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def source_lookup(questions):
    result = defaultdict(list)
    for split, values in questions.items():
        for row, question in enumerate(values):
            result[(split, normalize_question(question))].append(row)
    return result


def map_study_rows(rows, *, split, questions, label):
    """Map text to official rows; opaque request ordinals are not source rows."""
    lookup = source_lookup(questions)
    output = []
    for row in rows:
        text = normalize_question(row['question'])
        matches = lookup.get((split, text), [])
        if len(matches) != 1:
            raise ValueError('Study input has missing/ambiguous official identity: '+label)
        official = matches[0]
        explicit = row.get('source_row')
        if explicit is not None and explicit != official:
            raise ValueError('Explicit official source row disagrees with question')
        source_id = f'gsm8k:main:{split}:{official}'
        if str(row.get('source_id', '')).startswith('gsm8k:main:') and row['source_id'] != source_id:
            raise ValueError('Explicit official source ID disagrees with question')
        output.append(dict(source_split=split, source_row=official, source_id=source_id,
            request_id=row['request_id'], normalized_question=text, reason=label,
            prompt_ids_sha256=digest(row['prompt_ids']) if row.get('prompt_ids') else None,
            rendered_prompt_sha256=row.get('rendered_prompt_sha256')))
    if len({r['source_id'] for r in output}) != len(output):
        raise ValueError('Duplicate study source identity: '+label)
    return output


def select_rows(questions, exclusions, *, count, reserved_test_rows=()):
    """Choose using official input row hashes only; no exposure/ref arguments."""
    if type(count) is not int or not 136 <= count <= 768 or count % 8:
        raise ValueError('Seal a multiple of eight in [136,768]')
    reserved = set(reserved_test_rows)
    if any(type(i) is not int or not 0 <= i < len(questions['test']) for i in reserved):
        raise ValueError('Reserved rows must be valid official test indices')
    by_text = defaultdict(set); by_identity = defaultdict(set)
    for row in exclusions:
        by_text[row['normalized_question']].add(row['reason'])
        by_identity[(row['source_split'], row['source_row'])].add(row['reason'])
    eligible, excluded, seen = [], [], set()
    for index, question in enumerate(questions['test']):
        text = normalize_question(question)
        reasons = set(by_text.get(text, ())) | set(by_identity.get(('test', index), ()))
        if index in reserved:
            reasons.add('separately_reserved_final_test')
        if text in seen:
            reasons.add('duplicate_normalized_test_question')
        seen.add(text)
        if reasons:
            excluded.append(dict(source_row=index, reasons=sorted(reasons)))
        else:
            eligible.append(index)
    if len(eligible) < count:
        raise ValueError('Selected N exceeds eligible inventory; decide N before generation')
    key = lambda i: (hashlib.sha256((PREFIX+str(i)).encode('utf-8')).hexdigest(), i)
    ordered = sorted(eligible, key=key)
    return ordered[:count], dict(source_count=len(questions['test']), eligible_count=len(eligible),
        excluded_count=len(excluded), exclusions=excluded, eligible_rows_in_hash_order=ordered,
        reserved_test_rows=sorted(reserved), selection_prefix=PREFIX,
        selection_rule='ascending SHA256(UTF8(prefix + decimal official zero-based row)); row tie-break',
        normalization='NFKC and collapsed whitespace; match across splits; deduplicate before ranking',
        outcomes_used=False, reference_availability_used=False, prior_exposure_used=False)


def render_question(question, tokenizer):
    text = tokenizer.apply_chat_template([dict(role='user', content=question+SUFFIX)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids, text


def build_panel(questions, selected, audit, tokenizer, exposure, *, exclusion_rows):
    """Render a fixed selection and then annotate exposure; never replace a row."""
    exclusion_tokens = {r['prompt_ids_sha256'] for r in exclusion_rows if r['prompt_ids_sha256']}
    exclusion_prompts = {r['rendered_prompt_sha256'] for r in exclusion_rows if r['rendered_prompt_sha256']}
    requests = []
    for ordinal, index in enumerate(selected):
        question = questions['test'][index]
        ids, rendered = render_question(question, tokenizer)
        phash = digest(ids); rhash = hashlib.sha256(rendered.encode()).hexdigest()
        if phash in exclusion_tokens or rhash in exclusion_prompts:
            raise ValueError('Selected input overlaps study by rendered/token hash; do not replace silently')
        if not ids or len(ids)+2048+16 > 4096:
            raise ValueError('Selected prompt violates context budget; do not replace silently')
        history = list(exposure.get(index, []))
        source_id = f'gsm8k:main:test:{index}'
        requests.append(dict(request_id=f'survival-confirm-gsm8k-{index:04d}', dataset='gsm8k',
            dataset_id=source_id, source_id=source_id, source_split='test', source_revision=REVISION,
            source_row=index, original_index=index, request_index=ordinal, panel_ordinal=ordinal,
            question_sha256=hashlib.sha256(question.encode()).hexdigest(),
            normalized_question_sha256=hashlib.sha256(normalize_question(question).encode()).hexdigest(),
            rendered_prompt_sha256=rhash, prompt_ids_sha256=phash, prompt_ids=ids,
            stop_token_ids=list(STOP_IDS), max_new_tokens=2048, collect_feedback=False,
            selection_sha256=hashlib.sha256((PREFIX+str(index)).encode()).hexdigest(),
            prior_project_execution='documented' if history else 'unknown',
            prior_project_execution_evidence=history))
    panel = dict(schema='survival_confirmation_panel_v1', dataset='gsm8k', requests=requests,
        count=len(requests), source_revision=REVISION, held_out_scope='SURVIVAL study, not globally unused project questions',
        previously_used_development_panel=False, input_only_selection=True,
        selection=audit, projected_half_width_percentage_points=1.225*(128/len(requests))**.5,
        prior_project_execution_counts=dict(documented=sum(bool(exposure.get(i)) for i in selected),
                                            unknown=sum(not exposure.get(i) for i in selected)),
        model_work=0, selection_precedes_reference_lookup=True)
    panel['binding'] = digest(panel)
    return panel


def verify_panel(panel):
    """Check the sealed bytes and input-only identity/order contract."""
    body = {k:v for k,v in panel.items() if k != 'binding'}
    if digest(body) != panel.get('binding'):
        raise ValueError('Confirmation panel binding changed')
    rows = panel['requests']; count = panel['count']; audit = panel['selection']
    if type(count) is not int or not 136 <= count <= 768 or count % 8 or len(rows) != count:
        raise ValueError('Invalid sealed confirmation count')
    if (panel['source_revision'] != REVISION or not panel['input_only_selection']
            or panel['previously_used_development_panel'] or not panel['selection_precedes_reference_lookup']
            or any(audit[k] for k in ('outcomes_used','reference_availability_used','prior_exposure_used'))):
        raise ValueError('Changed input-only confirmation contract')
    selected = [r['source_row'] for r in rows]
    if selected != audit['eligible_rows_in_hash_order'][:count] or len(set(selected)) != count:
        raise ValueError('Changed selected official identities/order')
    for ordinal, row in enumerate(rows):
        index = row['source_row']
        if (row['source_id'] != f'gsm8k:main:test:{index}' or row['source_split'] != 'test'
                or row['source_revision'] != REVISION or row['panel_ordinal'] != ordinal
                or row['selection_sha256'] != hashlib.sha256((PREFIX+str(index)).encode()).hexdigest()
                or row['prompt_ids_sha256'] != digest(row['prompt_ids'])
                or row['stop_token_ids'] != STOP_IDS or row['max_new_tokens'] != 2048
                or row['collect_feedback']):
            raise ValueError('Changed per-request confirmation contract')
    return dict(status='passed', count=count, binding=panel['binding'])


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from experiments.mathqa_transfer.data import TOKENIZER_HASHES

    paths = {name: Path(getattr(args, name)) for name in ('source_manifest','development','lineage','continuation',
        'pilot_panel','history_inventory','reservation_review','tokenizer','output')}
    source = read(paths['source_manifest'])['sources']; questions = {}; inputs = {}
    for split, size in [('test',1319),('train',7473)]:
        entry = source['gsm8k_'+split]; path = Path(entry['local_path']); bound = identity(path)
        if entry['revision'] != REVISION or bound['sha256'] != SOURCE_HASHES[split]:
            raise ValueError('GSM8K source is not the pinned revision')
        questions[split] = pq.read_table(path,columns=['question']).column('question').to_pylist()
        if len(questions[split]) != size:
            raise ValueError('Pinned source row count changed')
        inputs['source_'+split] = bound
    dev = read(paths['development'])['requests']; lineage = read(paths['lineage'])['train_requests']
    event5 = [r for r in read(paths['continuation'])['train_requests'] if r['event_id']=='event_005']
    pilot = read(paths['pilot_panel'])['requests']
    if (len(dev),len(lineage),len(event5),len(pilot)) != (256,512,128,128):
        raise ValueError('Require all256 development,512 lineage,128 event5 and128 pilot inputs')
    if not {r['request_id'] for r in pilot} <= {r['request_id'] for r in dev}:
        raise ValueError('Pilot is not contained in the excluded source development panel')
    exclusions = []
    for rows,split,label in [(dev,'test','full_source_development_panel'),
                              (lineage,'train','KPLUS4_lineage_feedback'),
                              (event5,'train','SURVIVAL_event5_feedback')]:
        exclusions.extend(map_study_rows(rows,split=split,questions=questions,label=label))
    reservation = read(paths['reservation_review'])
    if reservation.get('status') != 'resolved' or not reservation.get('evidence'):
        raise ValueError('Require explicit supported final-reservation disposition')
    selected,audit = select_rows(questions,exclusions,count=args.count,
                                 reserved_test_rows=reservation['reserved_test_rows'])
    inputs.update({k:identity(v) for k,v in paths.items() if k not in ('tokenizer','output')})
    for name, expected in TOKENIZER_HASHES.items():
        bound=identity(paths['tokenizer']/name)
        if bound['sha256'] != expected:
            raise ValueError('Tokenizer bytes differ from the native serving pin')
        inputs['tokenizer/'+name]=bound
    tokenizer=AutoTokenizer.from_pretrained(paths['tokenizer'],local_files_only=True)
    # Reconstruct all pilot inputs using the exact serving template before emission.
    pilot_check=map_study_rows(pilot,split='test',questions=questions,label='pilot_reconstruction')
    for retained,mapped in zip(pilot,pilot_check):
        ids,rendered=render_question(questions['test'][mapped['source_row']],tokenizer)
        if ids != retained['prompt_ids'] or hashlib.sha256(rendered.encode()).hexdigest()!=retained['rendered_prompt_sha256']:
            raise ValueError('Pilot prompt/token reproduction failed')
    history=read(paths['history_inventory']); exposure=defaultdict(list)
    for entry in history['primary_completed_execution_sets']:
        for row in entry['source_rows']:
            exposure[row].append(dict(source='verified_actual_use_inventory',evidence=entry['directory']))
    for key in ('oct22_additional_completed_source_rows','async_final_additional_completed_source_rows'):
        for row in history[key]:
            exposure[row].append(dict(source='verified_actual_use_inventory',evidence=str(paths['history_inventory']),field=key))
    if args.additional_exposure:
        extra=read(args.additional_exposure)
        if extra.get('status')!='verified_completed_source_identity_sets':
            raise ValueError('Additional exposure must bind completed use and source identities')
        inputs['additional_exposure']=identity(args.additional_exposure)
        for item in extra['sets']:
            if not item.get('evidence'):
                raise ValueError('Additional prior-use set lacks evidence')
            for row in item['source_rows']:
                if type(row) is not int or not 0<=row<1319:
                    raise ValueError('Invalid exposure source row')
                exposure[row].append(dict(source=item['name'],evidence=item['evidence']))
    panel=build_panel(questions,selected,audit,tokenizer,exposure,exclusion_rows=exclusions)
    audit.update(inputs=inputs,pilot_prompt_reconstructions=len(pilot),
        full_lineage_source_rows=sorted(r['source_row'] for r in exclusions if r['reason']=='KPLUS4_lineage_feedback'),
        event5_source_rows=sorted(r['source_row'] for r in exclusions if r['reason']=='SURVIVAL_event5_feedback'),
        reservation_disposition=reservation,read_columns=['question'],source_answers_used=False,
        qualification_scope='SURVIVAL qualification/startup and six serving checks use first2 event5 feedback; PR32 diagnostics use16 source-development requests; lineage qualification4/37/72/121 lies within512 lineage; no additional intervention-specific input found in inspected sources')
    # Rebind after adding audit provenance; never overwrite an existing selection.
    panel.pop('binding');panel['binding']=digest(panel)
    verify_panel(panel)
    out=paths['output'];out.mkdir(parents=True,exist_ok=False)
    for name,value in [('PANEL.json',panel),('EXCLUSION_AUDIT.json',audit),('EXPOSURE.json',
        dict(requests=[{k:r[k] for k in ('request_id','source_row','source_id','prior_project_execution','prior_project_execution_evidence')} for r in panel['requests']],
             counts=panel['prior_project_execution_counts'],selection_uses_exposure=False))]:
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return dict(status='prepared',count=len(selected),eligible=audit['eligible_count'],binding=panel['binding'],
        prior_project_execution_counts=panel['prior_project_execution_counts'],model_work=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest','development','lineage','continuation','pilot-panel','history-inventory',
                 'reservation-review','tokenizer','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--count',type=int,required=True)
    parser.add_argument('--additional-exposure')
    print(json.dumps(prepare(parser.parse_args()),sort_keys=True))
