"""Bind KR to retained responses, per-pass anchors and the two existing panels."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

SOURCE = Path('/data/sources/minhao/DFlash-Online')
RUNS = Path('/data/sources/minhao/runs')


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def panel(name, requests, prior, source_method):
    canonical_path = prior / 'campaign_001/evaluation/CANONICAL.json'
    measurements_path = prior / 'campaign_001/evaluation/MEASUREMENTS.jsonl'
    canonical = json.loads(canonical_path.read_text())
    measurements = [json.loads(line) for line in measurements_path.read_text().splitlines()]
    counts = {}
    for request in requests:
        rid = request['request_id']
        rows = [row for row in measurements if row['request_id'] == rid and row['method'] == 'KRC']
        if (len(rows) != 3 or {row['repeat'] for row in rows} != {0, 1, 2}
                or not all(row['valid_measurement'] for row in rows)
                or len({row['verification_calls'] for row in rows}) != 1):
            raise ValueError('Incomplete retained KRC reference: ' + rid)
        counts[rid] = rows[0]['verification_calls']
    result = dict(name=name, requests=[dict(row, response_ids=canonical[row['request_id']]['response_ids'],
        finish_reason=canonical[row['request_id']]['finish_reason'], canonical_source_method=source_method)
        for row in requests], retained_krc_counts=counts,
        provenance=dict(canonical=str(canonical_path), measurements=str(measurements_path),
                        existing_panel_unchanged=True))
    return result, [canonical_path, measurements_path]


def prepare(root):
    root = Path(root)
    prior = RUNS / 'dflash_krc_event_20261027'
    confirmation = RUNS / 'dflash_krc_confirmation_20261028'
    paths = set()
    for previous in (prior, confirmation):
        if (previous / 'EXIT_CODE').read_text().strip() != '0':
            raise ValueError('Retained campaign incomplete: ' + str(previous))
        seal = json.loads((previous / 'sealed/SEAL.json').read_text())
        for name, expected in seal['files'].items():
            if identity(name) != expected:
                raise ValueError('Retained source/input changed: ' + name)
            paths.add(Path(name))
        paths.update([previous/'sealed/SEAL.json', previous/'POSTFLIGHT.json', previous/'EXIT_CODE'])
    additions = json.loads((root/'ADDITIONS.json').read_text())
    for name, expected in additions.items():
        if identity(root/'code'/name) != expected or (SOURCE/name).exists():
            raise ValueError('Require an exact additive source install: ' + name)
    for name in additions:
        target = SOURCE/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/'code'/name, target)
        paths.add(target)
    sys.path.insert(0, str(SOURCE))
    from online_feedback.reference_schedule import validate_reference_schedule
    from experiments.reference_replay_event.protocol import build_replay_schedule, validate_replay_schedule
    from experiments.reference_replay_event.evaluation_protocol import validate_plan
    old = json.loads((prior/'sealed/PLAN.json').read_text())
    ordinary = json.loads(Path(old['ordinary_receipt']).read_text())
    manifest = json.loads(Path(old['manifest']).read_text())
    reference = json.loads(Path(old['reference_schedule']).read_text())
    panel2_plan = json.loads((confirmation/'sealed/PLAN.json').read_text())
    evaluation_ids = [r['request_id'] for r in manifest['eval_requests'] + panel2_plan['requests']]
    reference_check = validate_reference_schedule(reference, ordinary, evaluation_ids=evaluation_ids)
    sealed = root/'sealed'
    sealed.mkdir(exist_ok=False)
    for name in ('config.yaml','MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json'):
        shutil.copy2(prior/'sealed'/name, sealed/name)
    shutil.copy2(old['reference_schedule'], sealed/'REFERENCE_SCHEDULE.json')
    (sealed/'REFERENCE_SCHEDULE_CHECK.json').write_text(json.dumps(reference_check,indent=2)+'\n')
    replay = build_replay_schedule(ordinary, evaluation_ids=evaluation_ids)
    replay_check = validate_replay_schedule(replay, ordinary, evaluation_ids=evaluation_ids)
    (sealed/'REPLAY_SCHEDULE.json').write_text(json.dumps(replay,indent=2)+'\n')
    (sealed/'REPLAY_SCHEDULE_CHECK.json').write_text(json.dumps(replay_check,indent=2)+'\n')
    panels = []
    for name, requests, folder, method in [('development',manifest['eval_requests'],prior,'F'),
            ('additional',panel2_plan['requests'],confirmation,'K1')]:
        result, sources = panel(name, requests, folder, method)
        panels.append(result)
        paths.update(sources)
    if [sum(p['retained_krc_counts'].values()) for p in panels] != [3005,3125]:
        raise ValueError('Retained panel KRC call totals changed')
    krc_receipt = prior/'campaign_001/KRC/events/event_001/RESULT.json'
    krc = json.loads(krc_receipt.read_text())
    plan = {key:old[key] for key in ('ordinary_receipt','k_receipt','pinned_initial','target','draft','worker_python','manifest')}
    plan.update(config=str(sealed/'config.yaml'), reference_schedule=str(sealed/'REFERENCE_SCHEDULE.json'),
        retained_krc_checkpoint=krc['checkpoint_path'], panels=panels,
        replay_schedule=str(sealed/'REPLAY_SCHEDULE.json'),
        scope='one KR ordinary-reference replay event; two unchanged panels against retained KRC',
        main=dict(arm='KR',distinct_responses=128,outer_windows=128,A_presentations=256,R_presentations=256,
                  presentations=512,optimizer_steps=256,response_forwards=512,response_backwards=512),
        startup=dict(response_evaluations=2,backwards=2,optimizer_steps=0,restoration='exact'),
        evaluation=dict(arms=['KR','KRC'],panels=2,requests_per_panel=64,repeats=3,generations=768,
                        initial_gate_requests=8,gate_generations_included=48),
        serving=dict(greedy=True,dtype='bfloat16',block_size=16,output_cap=2048,context_limit=4096,
                     concurrency=1,capture=False,extra_generation_warmups=0),
        no_automatic_retry=True,final_checkpoint_count=1,qualification_optimizer_steps=0,
        first_A_check='exact fresh model/optimizer/RNG plus first two ordinary K training records; no extra forward or protected-prefix capture; not historical post-step tensor equivalence',
        historical_KRC_costs=str(prior/'campaign_001/COSTS.json'))
    validate_plan(plan)
    (sealed/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    paths.add(krc_receipt)
    paths.add(Path(plan['historical_KRC_costs']))
    paths.update(sealed.iterdir())
    paths.update(p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.json','.md'})
    exact = {}
    for folder in map(Path,[plan['target'],plan['draft'],plan['retained_krc_checkpoint']]):
        files = [p for p in folder.iterdir() if p.is_file()]
        exact[str(folder)] = sorted(p.name for p in files)
        paths.update(files)
    result = dict(schema='kr_replay_seal_v1',sealed_before_training=True,
        sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        files={str(p):identity(p) for p in sorted(paths)},exact_file_sets=exact,
        parent_seals=[str(prior/'sealed/SEAL.json'),str(confirmation/'sealed/SEAL.json')])
    (sealed/'SEAL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='sealed',files=len(paths),new_files=list(additions),
                         panel_requests=[len(p['requests']) for p in panels],retained_KRC_calls=[3005,3125])))


if __name__ == '__main__':
    prepare(sys.argv[1])
