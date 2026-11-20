"""Independent implementation of rank, identity, exclusion and schedule checks.
This is a same-agent second reduction, not independent authorship or GPU work.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,itertools,unicodedata
D=Path(__file__).absolute().parent;S=D/'sealed';checks=0

def check(ok,msg):
 global checks
 if not ok:raise AssertionError(msg)
 checks+=1

def read(p):return json.loads(p.read_text())
def digest(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def normalized(s):return ' '.join(unicodedata.normalize('NFKC',s).split())
seal=read(S/'SEAL.json')
for name,pin in seal['files'].items():
 b=(S/name).read_bytes();check(len(b)==pin['bytes'] and digest(b)==pin['sha256'],'Seal file '+name)
for name,pin in seal['scripts'].items():
 b=Path(name).read_bytes();check(len(b)==pin['bytes'] and digest(b)==pin['sha256'],'Sealed source '+name)
inputs=read(D/'ELIGIBLE_INPUTS_PRIVATE.json')['requests'];panel=read(S/'PANEL.json')['requests'];exc=read(S/'EXCLUSIONS.json')['rows']
revision='4f958013c164a40745a5b771dfb005df308deb4a'
ranked=sorted(inputs,key=lambda r:(digest(canonical({'seed':20261119,'dataset_revision':revision,'split':'test','source_id':r['source_id']})),canonical({'dataset_revision':revision,'split':'test','source_id':r['source_id']})))
check([x['source_id'] for x in panel]==[x['source_id'] for x in ranked[:1024]],'Literal independent rank recreation')
check(len(panel)==1024 and len(inputs)==1186 and len(exc)==1799,'Pool counts')
check(set(x['source_id'] for x in panel).isdisjoint(x['source_id'] for x in exc),'Source exclusion')
for r in panel:
 for text,key in [(r['rendered_prompt'],'rendered_prompt_sha256'),(normalized(r['rendered_prompt']),'normalized_prompt_sha256')]:check(digest(text.encode())==r[key],key)
 check(digest(json.dumps(r['prompt_ids'],separators=(',',':')).encode())==r['prompt_ids_sha256'],'Token identity')
 check(r['dataset_revision']==revision and r['split']=='test' and r['dataset']=='mathqa','Bound dataset identity')
 check(r['source_id']==f"mathqa:test:{r['source_row']}" and r['original_index']==r['source_row'],'Source index')
 check(r['collect_feedback'] is False and r['max_new_tokens']==2048 and r['stop_token_ids']==[151645,151643],'Decoding policy')
 check(r['prompt_tokens']==len(r['prompt_ids']) and 0<len(r['prompt_ids'])<=2048,'Input eligibility')
check([r['panel_ordinal'] for r in panel]==list(range(1024)),'Ordinal order')
for f in ('request_id','source_id','normalized_prompt_sha256','prompt_ids_sha256'):check(len({r[f] for r in panel})==1024,'Unique '+f)
check(not any(set(r)&{'answer','output','response_ids','response','metrics'} for r in panel),'No output or answer selection fields')
warm=read(S/'WARMUPS.json')['requests'];can=read(S/'WARMUP_CANONICALS.json');auth=read(S/'WARMUP_AUTHENTICATION.json')
check((S/'WARMUPS.json').read_bytes()==Path(auth['source_warmups']).read_bytes(),'Exact warmup copy')
check((S/'WARMUP_CANONICALS.json').read_bytes()==Path(auth['source_canonicals']).read_bytes(),'Exact canonical copy')
for w in warm:
 check(w['source_id'] not in {r['source_id'] for r in panel},'Warmup source distinct')
 check(tuple(w['prompt_ids']) not in {tuple(r['prompt_ids']) for r in panel},'Warmup tokens distinct')
 check(normalized(w['rendered_prompt']) not in {normalized(r['rendered_prompt']) for r in panel},'Warmup normalized prompt distinct')
 check(can[w['request_id']]['prompt_ids']==w['prompt_ids'],'Warmup reference input')
schedule=read(S/'SCHEDULE.json')['activations'];orders=sorted(itertools.permutations(('AREPO58','KPLUS_C58','O58')))
check(len(schedule)==144,'144 activations')
for i,slot in enumerate(schedule):
 repeat,group,position=i//48,(i//3)%16,i%3
 check(slot==dict(activation=i,repeat=repeat,block_ordinal=group,position=position,method=orders[(16*repeat+group)%6][position],request_ordinals=list(range(64*group,64*(group+1)))),'Literal slot '+str(i))
expected=Counter({(r,arm,o):1 for r in range(3) for arm in ('AREPO58','KPLUS_C58','O58') for o in range(1024)})
actual=Counter((s['repeat'],s['method'],o) for s in schedule for o in s['request_ordinals'])
check(actual==expected,'Exact request arm repeat coverage')
order=read(S/'ORDER.json');check(order['lexicographic_request_id_order']==sorted(r['request_id'] for r in panel),'Bootstrap order')
check(digest(canonical(order['lexicographic_request_id_order']))==order['bootstrap_order_sha256'],'Bootstrap input digest')
result={'status':'passed','checks':checks,'scored':len(panel),'warmups':len(warm),'coverage':len(actual),'authorship':'same agent second reduction; no independent execution or external authorship','model_work':0,'remote_operations':0}
(D/'SEAL_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
