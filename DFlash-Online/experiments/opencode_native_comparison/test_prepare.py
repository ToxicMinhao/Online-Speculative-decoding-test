"""Sealed-plan consumer contracts, with filesystem and model identities mocked."""
import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from online_feedback.storage import write_json
from . import contract


def test_prepared_plan_supplies_real_serving_and_publication_consumers(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'experiments.opencode_native_comparison.configuration',
        SimpleNamespace(build_config=lambda text,**kwargs:json.dumps(kwargs,default=str)))
    spec=importlib.util.spec_from_file_location('experiments.opencode_native_comparison._prepare_fixture',
        Path(__file__).with_name('prepare.py'))
    prepare=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    root=tmp_path/'study'
    sealed=root/'sealed'
    sealed.mkdir(parents=True)
    commit='b'*40
    write_json(root/'SOURCE.json',dict(commit=commit,dirty=False))
    manifest={}
    offset=0
    for role,count in [('train',7424),('evaluation',1024),('warmup',2),('qualification',16)]:
        rows=[]
        for index in range(count):
            rows.append(dict(request_id=f'{role}_{index}',request_index=index,
                prompt_ids=[offset+index+1]*(1+(index%3)),max_new_tokens=2048,
                stop_token_ids=[151645,151643],**({'panel_ordinal':index} if role=='evaluation' else {})))
        manifest[role+'_requests']=rows
        offset+=count
    write_json(sealed/'INPUTS.json',manifest)
    token_files=('tokenizer.json','tokenizer_config.json','merges.txt','vocab.json')
    pin=dict(bytes=1,sha256='a'*64)
    write_json(sealed/'MODEL_PIN.json',dict(target=dict(files={name:pin for name in token_files}),
        draft=dict(files={'config.json':pin})))
    (sealed/'ORIGINAL_CONFIG.yaml').write_text('fixture')
    (sealed/'RUNTIME_PIN.json').write_text('{}')
    storage=root/'CAPACITY.json'
    write_json(storage,dict(status='admitted',projected_peak_bytes=70_000_000_000))
    monkeypatch.setattr(prepare,'identity',lambda path:pin)
    monkeypatch.setattr(prepare,'build_config',lambda text,**kwargs:json.dumps(kwargs,default=str))
    monkeypatch.setattr(prepare,'compiler_snapshot',lambda:{'directories':[],'allocated_bytes':0})
    plan=prepare.prepare(root,tmp_path/'source',commit,storage)
    assert plan['binding']==contract.digest({k:v for k,v in plan.items() if k!='binding'})
    assert plan['checkpoints']['F']==plan['draft']
    assert plan['session']=='opencode_native_comparison_20261120'
    assert plan['zstd'].endswith('/bin/zstd')
    assert len(plan['qualification_pair_ids'])==4
    assert plan['exact_qualification_work']==dict(generation=64,adam=17,response_forward=40,
        response_backward=40,teacher_preparation=16,training_initialization=3)
    schedule=json.loads((sealed/'SCHEDULE.json').read_text())
    assert schedule['balance']['activations']==120
    assert schedule['balance']['scored_generations']==15360
    assert schedule['balance']['warmup_generations']==240
    assert str(sealed/'PLAN.json') not in plan['input_identities']
    assert json.loads((root/'KEEP_MANIFEST.json').read_text())['historical_retirement_authorized'] is False
