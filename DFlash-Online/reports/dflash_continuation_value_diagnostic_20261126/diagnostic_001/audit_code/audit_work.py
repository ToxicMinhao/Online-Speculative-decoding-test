"""CPU retained-evidence audit; no imports from the experimental model code."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--study',type=Path,required=True)
args=parser.parse_args()
study=args.study.resolve(); raw=study/'diagnostic_001/raw'; research=study.parent
out=study/'final_audit'; out.mkdir(exist_ok=True)
started=time.perf_counter(); checks=Counter(); errors=[]; evidence=[]

def read(path): return json.loads(Path(path).read_text())
def check(condition,label,detail=None):
    checks[label]+=1
    if not condition: errors.append(dict(check=label,detail=detail))
def ident(path):
    p=Path(path); digest=hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return dict(bytes=p.stat().st_size,sha256=digest.hexdigest())
def verify(path,expected,label):
    check(ident(path)==expected,label,str(path)); evidence.append(dict(path=str(path),identity=ident(path)))

contract=read(raw/'CONTRACT.json'); status=read(raw/'STATUS.json'); clock=read(raw/'CLOCK.json')
post=read(raw/'POSTFLIGHT.json'); source=read(study/'SOURCE_MANIFEST.json'); manifest=read(raw/'RETRIEVAL_MANIFEST.json')
commit='b6f2a2c8d487bcc2d32a5e716b870c10eb5067b3'
for obj in (contract,status,manifest):check(obj['execution_commit']==commit,'execution_commit')
check(source['commit']==post['source_commit']==commit,'source_commit')
check(source['clean'] is True,'source_clean_receipt')
check(status['status']=='completed' and post['process_exit_code']==0,'completion')
check(status['clock']==clock,'clock_status_agree')
check(contract['base_commit']=='281516426411ac567129dae7af15ea443a3d0c94','base_commit')
check(contract['endpoints']['parent']['counters']==[128,256,1],'parent_lineage')
check(contract['endpoints']['child']['counters']==[256,512,2],'child_lineage')
verify(study/'SOURCE_MANIFEST.json',contract['source_identity'],'source_manifest_identity')
check(post['source_manifest_identity']==contract['source_identity'],'postflight_source_manifest')
source_root=study/'implementation_checkout/DFlash-Online'
for name, expected in source['files'].items():
    path=source_root/name
    if expected.get('kind')=='symlink': check(path.is_symlink() and os.readlink(path)==expected['target'],'source_symlink',name)
    else: check(ident(path)==expected,'source_file_identity',name)
check(len(source['files'])==post['source_entries_verified']==5410,'source_entry_count')
tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'DFlash-Online'],cwd=source_root.parent,text=True).splitlines()
check({p.removeprefix('DFlash-Online/') for p in tracked}==set(source['files']),'source_manifest_complete_commit_tree')
for row in manifest['files']:
    check(ident(raw/row['path'])==dict(bytes=row['bytes'],sha256=row['sha256']),'retrieval_file_identity',row['path'])
check(sum(r['bytes']for r in manifest['files'])==manifest['total_bytes']==112133005,'retrieval_bytes')
check(len(manifest['files'])==300,'retrieval_files')

# Independently bind historical training event and its first-pass anchor schedule.
archive_path=research/'dflash_frontier_preservation_20261125/production_001/COMPACT_MANIFEST.json'
verify(archive_path,dict(bytes=211418,sha256='dd65e3f371b4c5b6a4f9385447fb3f02ceca68642d79a9940076f0ede3576cc5'),'handoff_archive_pin')
archive=read(archive_path)
check(archive['execution_commit']=='225802657d630c0985abd68d6e7453f3e47c5b74','historical_execution_commit')
archive_entries={r['path']:dict(bytes=r['bytes'],sha256=r['sha256'])for r in archive['files']}
for role,size,digest in [('parent',2097259104,'1556409004f7c73c0007c1f01764d4fd1dd189827994596939597be43b2d21b2'),('child',2097259104,'b675b559c0fe2f7db4251a585a53740c7ea66868137cdf3be432c5cb6e634cc6')]:
    check(contract['endpoints'][role]['identity']==dict(bytes=size,sha256=digest),'handoff_export_pin')
for size,digest in [(6296473051,'728770a7abf0ba3ef96805ef760997976ad7bedbbee1d8debe165e785a4d98a2'),(6301108315,'3f3e5075926e3f08c06958c7ab77eb0eb9bf9af2cd5b8ea2cb77aeab70750ddd')]:
    check(dict(bytes=size,sha256=digest)in contract['identities'].values(),'handoff_recovery_pin')
old=research/'dflash_frontier_preservation_20261125/production_001/raw'
remote_old='/data/sources/minhao/runs/dflash_frontier_preservation_20261125/production_001'
for relative in ('sealed/PLAN.json','arms/CONTROL/STATUS.json','arms/CONTROL/trainer/events/event_002/RESULT.json','shared/FEATURE_IDENTITIES.json','witness/BINDINGS.json','witness/parent.json','witness/CONTROL.json'):
    verify(old/relative,contract['identities'][remote_old+'/'+relative],'historical_retained_identity')
    check(contract['identities'][remote_old+'/'+relative]==archive_entries[relative],'historical_archive_binding')
old_event=read(old/'arms/CONTROL/trainer/events/event_002/RESULT.json');old_status=read(old/'arms/CONTROL/STATUS.json')
check(old_event==old_status['training'],'historical_event_status_equal')
check(len(old_event['records'])==256,'historical_presentations')
first=[r for r in old_event['records'] if r['pass']==0];check(len(first)==128,'first_pass_count')
first={r['record_id']:r for r in first}; check(len(first)==128,'first_pass_unique')
feature_rows=read(old/'shared/FEATURE_IDENTITIES.json');check(len(feature_rows)==len(contract['features'])==128,'cohort_size')
for ordinal,(original,row) in enumerate(zip(feature_rows,contract['features'])):
    check(row['ordinal']==ordinal and row['source_index']==128+ordinal,'source_order')
    check(row['record_id']==original['record_id'] and row['feature_identity']==original['identity'] and row['feature_path']==original['path'],'feature_binding')
    check(contract['identities'][row['feature_path']]==row['feature_identity'],'feature_identity_contract')
    candidates=set()
    for sample in first[row['record_id']]['sampled_anchors']:
        candidates.update(int(a)for a,keep in zip(sample['positions'][0],sample['keep_mask'][0])if keep)
    prefix='continuation_value_diagnostic_v1|20261126|'
    chosen=sorted(sorted(candidates,key=lambda a:(hashlib.sha256(f'{prefix}{row["record_id"]}|{a}'.encode()).digest(),a))[:64])
    check(row['anchors']==chosen and row['available_first_pass_anchors']==len(candidates),'anchor_hash_selection')
    check(row['physical_shard']==ordinal%2,'fixed_physical_shard')
check(contract['legacy_binding']==read(raw/'legacy/BINDINGS.json')==read(old/'witness/BINDINGS.json'),'legacy_binding')
for endpoint,historical in [('parent','parent'),('child','CONTROL')]:
    check(read(raw/f'legacy/{endpoint}.json')==read(old/f'witness/{historical}.json'),'legacy_witness_exact')
    check(status['checkpoint_identities_after'][endpoint]==contract['endpoints'][endpoint]['identity'],'checkpoint_postflight_unchanged')

zero_keys=('backwards','optimizer_steps','teacher_forwards','feedback_generations','serving_generations')
all_receipts={}; per_phase={}; endpoints=[]; compiler=Counter(); row_wall=Counter(); loading_wall=Counter()
for phase in ('qualification','primary'):
    totals=Counter(); files_expected=set()
    for shard in range(2):
        worker=read(raw/phase/f'gpu{shard}/WORK.json')
        check(worker['status']=='passed' and worker['phase']==phase and worker['shard']==shard,'worker_identity_status')
        check(worker['physical_gpu']==contract['devices'][shard] and worker['execution_commit']==commit,'worker_gpu_commit')
        check(worker['elapsed_s']==worker['end_epoch']-worker['start_epoch'],'worker_elapsed')
        for endpoint in ('parent','child'):
            entry=worker['endpoints'][endpoint]; final=entry['final_state']; loading=final['loading']; mapping=loading['mapping']
            check(entry['status']==final['status']=='passed' or (entry['status']=='completed' and final['status']=='passed'),'endpoint_completed')
            check(final['endpoint']==endpoint and final['parameters_unchanged'] and final['modes_unchanged'] and not final['optimizer_created'],'endpoint_state')
            check(mapping['status']=='passed' and mapping['tensors']==58 and mapping['exact_copy'],'export_mapping')
            check(mapping['export_identity']==contract['endpoints'][endpoint]['identity'],'loaded_export_identity')
            for item in mapping['mapping']:check(item['dtype']=='torch.bfloat16' and bool(item['shape']),'mapping_precision_shape')
            check(loading['rng_before']==loading['rng_after'] and loading['frozen_components_unchanged'],'loading_state_rng')
            check(loading['construction_seed']==17 and loading['startup_forwards']==loading['startup_backwards']==0 and not loading['optimizer_created'],'loading_no_training')
            for key in zero_keys:check(loading[key]==final['work'][key]==0,'endpoint_forbidden_work')
            loading_wall[phase]+=entry['load_s']
            expected=[]
            if phase=='qualification':
                for n in range(2):expected.append((f'legacy_{n}.json',contract['legacy_binding']['sets']['new_buffer'][n],True))
                for n in range(2):expected.append((f'primary_{n}.json',contract['features'][0],False))
            else:
                expected=[(f'{r["ordinal"]:03d}.json',r,False)for r in contract['features']if r['physical_shard']==shard]
            local=Counter()
            for name,binding,legacy in expected:
                path=raw/phase/f'gpu{shard}'/endpoint/name; receipt=read(path); relative=str(path.relative_to(raw));files_expected.add(relative);all_receipts[relative]=receipt
                check(receipt['status']=='passed' and receipt['endpoint']==endpoint,'readout_status_endpoint')
                check(receipt['record_id']==binding['record_id'] and receipt['anchors']==binding['anchors'] and receipt['feature_identity']==binding['feature_identity'],'readout_binding')
                check(receipt['rng_before']==receipt['rng_after']==loading['rng_after'] and receipt['rng_preserved'],'readout_rng')
                check(all(receipt[k]is True for k in ('parameters_unchanged','gradients_unchanged','runtime_restored')),'readout_state')
                check(receipt['evaluation_mode']=='eval/no_grad' and receipt['target_features_dtype']=='torch.bfloat16','native_mode_precision')
                check(receipt['geometry']==dict(block_size=16,projection_chunk_blocks=8,anchor_count=len(binding['anchors'])),'native_geometry')
                blocks=len(binding['anchors']);heads=math.ceil(blocks/8)
                check(len(receipt['blocks'])==blocks,'blocks_retained')
                check(receipt['projection_shapes']==[[1,min(8,blocks-i),16,4096]for i in range(0,blocks,8)],'head_physical_shapes')
                expected_work=dict(response_reads_attempted=1,response_reads_completed=1,draft_forwards_attempted=int(blocks>0),draft_forwards_completed=int(blocks>0),head_calls_attempted=heads,head_calls_completed=heads,blocks_completed=blocks,**{k:0 for k in zero_keys})
                check(receipt['work']==expected_work,'work_counts_exact')
                for key,value in receipt['work'].items():local[key]+=value
                row_wall[phase]+=receipt['wall_s']
                for key,timer in receipt['compiler_timers'].items():compiler[key]+=timer['total_s']
                if legacy:
                    n=int(name.split('_')[1].split('.')[0]); retained=read(raw/f'legacy/{endpoint}.json')['sets']['new_buffer'][n]
                    check(receipt['anchors']==retained['anchors'],'legacy_anchor_equality')
                    for current,old_block in zip(receipt['blocks'],retained['blocks']):
                        check(current['roles']==old_block['current'] and current['proposals']==old_block['proposals'],'legacy_roles_proposals_exact')
                        for position in old_block['positions']:
                            current_pos=next(p for p in current['positions']if p['position']==position['position'])
                            for key in ('canonical','rival','prediction','correct','margin','margin_tie','wrong_rival_tie'):check(current_pos[key]==position[key],'legacy_margin_fields_exact')
            check(dict(local)==final['work'],'endpoint_counts_reconcile')
            totals.update(local)
            if phase=='qualification':
                q=read(raw/phase/f'gpu{shard}'/endpoint/'QUALIFICATION.json');check(q==entry['qualification'],'qualification_duplicate_identity')
                check(q['status']=='passed' and len(q['comparisons'])==3 and all(x['status']=='passed'and not x['discrepancies']for x in q['comparisons']),'qualification_receipts')
                left=all_receipts[f'{phase}/gpu{shard}/{endpoint}/primary_0.json'];right=all_receipts[f'{phase}/gpu{shard}/{endpoint}/primary_1.json']
                for key in ('record_id','anchors','feature_identity','geometry','blocks'):check(left[key]==right[key],'repeat_compact_exact')
            endpoints.append(dict(phase=phase,shard=shard,endpoint=endpoint,load_s=entry['load_s'],adapter_loading_s_nested=loading['wall_s'],device_initialization_s_nested=loading['device_initialization_s_nested'],readout_s=sum(all_receipts[p]['wall_s']for p in files_expected if p.startswith(f'{phase}/gpu{shard}/{endpoint}/')),process_peak_allocated_bytes=final['process_peak_allocated_bytes'],process_peak_reserved_bytes=final['process_peak_reserved_bytes']))
    actual={str(p.relative_to(raw))for p in (raw/phase).glob('gpu*/*/*.json')if p.name!='QUALIFICATION.json'}
    check(actual==files_expected,'planned_response_files_exact')
    per_phase[phase]=dict(totals)

for endpoint in ('parent','child'):
    a=all_receipts[f'qualification/gpu0/{endpoint}/primary_1.json'];b=all_receipts[f'qualification/gpu1/{endpoint}/primary_1.json']
    for key in ('record_id','anchors','feature_identity','geometry','blocks'):check(a[key]==b[key],'cross_card_compact_exact')
cross=read(raw/'CROSS_CARD.json');check(cross['status']=='passed' and cross['first_primary_exact'] and cross['physical_gpus']==contract['devices'],'cross_card_receipt')
for endpoint in ('parent','child'):
    a=all_receipts[f'qualification/gpu0/{endpoint}/primary_1.json']; b=all_receipts[f'primary/gpu0/{endpoint}/000.json']
    for key in ('record_id','anchors','feature_identity','geometry','blocks'):check(a[key]==b[key],'reloaded_primary_matches_qualified')
for row in contract['features']:
    pair=[all_receipts[f'primary/gpu{row["physical_shard"]}/{e}/{row["ordinal"]:03d}.json']for e in ('parent','child')]
    for a,b in zip(pair[0]['blocks'],pair[1]['blocks']):
        check(a['labels']==b['labels'] and a['valid']==b['valid'],'paired_labels_validity')
check(per_phase['qualification']['draft_forwards_completed']==contract['budget']['qualification_response_forwards']==16,'qualification_forward_budget')
check(per_phase['primary']['draft_forwards_completed']==contract['budget']['primary_response_forwards']==256,'primary_forward_budget')
check(per_phase['primary']['blocks_completed']==2*contract['budget']['primary_paired_blocks']==16384,'primary_block_budget')
check(sum(status['forbidden_work'].values())==0,'supervisor_forbidden_work')
check(len(clock['processes'])==4 and all(p['exit_code']==0 for p in clock['processes']),'processes_four_no_retry')
occupied=sum(p['end_epoch']-p['start_epoch']for p in clock['processes'])
check(occupied==clock['occupied_card_s']<=3600 and clock['elapsed_s']<=5400,'clock_caps')
check(clock['first_gpu_action_epoch']<min(p['start_epoch']for p in clock['processes']),'original_clock_precedes_all_work')
check(max(p['end_epoch']for p in clock['processes']if p['phase']=='qualification')<=min(p['start_epoch']for p in clock['processes']if p['phase']=='primary'),'qualification_before_primary')
check(contract['created_epoch']<clock['first_gpu_action_epoch'],'contract_before_work')
check(not post['gpus']['processes'] and all(g['used_bytes']==0 for g in post['gpus']['cards']),'postflight_gpus_idle')
check(contract['resource_admission']['free_bytes']-contract['resource_admission']['remote_peak_write_bytes']>=400_000_000_000,'preflight_peak_reserve')
check(post['storage']['free_bytes']>=400_000_000_000,'postflight_reserve')
check(post['storage']['charged_bytes']+contract['local_charged_bytes']<=2_000_000_000,'all_location_storage_forecast')
seen=set();local_allocated=0
for base,dirs,files in os.walk(study):
    for name in files:
        stat=(Path(base)/name).lstat();key=(stat.st_dev,stat.st_ino)
        if key not in seen:local_allocated+=stat.st_blocks*512;seen.add(key)
check(local_allocated<contract['local_charged_bytes'],'actual_local_study_below_envelope')
summary=dict(status='passed'if not errors else'failed',execution_commit=commit,checks=sum(checks.values()),check_categories=dict(checks),errors=errors,source_entries_verified=len(source['files']),retrieval_files_verified=len(manifest['files']),retrieval_bytes=manifest['total_bytes'],cohort=dict(responses=128,paired_blocks=8192,source_index_range=[128,255],scope='Retained event-2 training buffer; not held out'),physical_gpu_uuids=contract['devices'],work=per_phase,invalid=0,unknown=0,unattempted=0,retries=0,startup_response_forwards=0,startup_backwards=0,optimizer_created=0,timing=dict(occupied_card_s=occupied,elapsed_s=clock['elapsed_s'],phase_wall_s={p['phase']:p['wall_s']for p in status['phases']},load_s_by_phase=dict(loading_wall),readout_s_by_phase=dict(row_wall),passive_compiler_timer_s_nested=dict(compiler),note='Loading/readout/compiler timers overlap. Occupied-card time charges complete child-process lifetimes. Compiler categories can be nested within each other.'),endpoint_costs=endpoints,resources=dict(postflight_free_bytes=post['storage']['free_bytes'],remote_study_and_cache_allocated_bytes=post['storage']['charged_bytes'],compiler_growth_bytes=post['storage']['compiler_growth_bytes'],local_envelope_bytes=contract['local_charged_bytes'],current_local_study_allocated_bytes=local_allocated,remote_plus_local_envelope_bytes=post['storage']['charged_bytes']+contract['local_charged_bytes'],peak_gpu_allocated_bytes=max(e['process_peak_allocated_bytes']for e in endpoints),peak_gpu_reserved_bytes=max(e['process_peak_reserved_bytes']for e in endpoints)),limitations=['Reviewer authored native.py and its tests; review of preparation/supervisor, arithmetic reconciliation and historical bindings is separate from that authorship.','Instrumented source/evidence review, not independent GPU execution or hardware instruction tracing.','Full vocabulary logits and model tensors are not retained locally; immutable model state relies on exact authenticated load, runtime version/gradient checks and final authenticated export files.','Parent feature-manifest whole-file hash is newly sealed; used parent-training feature identities are historically bound by the original witness.','Current local allocation excludes additional report/publication output not yet written and shared Git object growth; root must finalize all-location charge before publication.','First CPU audit attempt stopped on a source-manifest symlink entry incorrectly treated as a regular file. The checker was fixed to read its kind/target schema; failed log retained. No experimental source or model work changed.'],cpu_audit_wall_s=time.perf_counter()-started)
(out/'AUDIT_WORK.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'AUDIT_WORK_PRIVATE.json').write_text(json.dumps(dict(summary=summary,authenticated_local_evidence=evidence),indent=2)+'\n')
print(json.dumps({k:summary[k]for k in ('status','checks','errors','work','timing','resources')},indent=2))
