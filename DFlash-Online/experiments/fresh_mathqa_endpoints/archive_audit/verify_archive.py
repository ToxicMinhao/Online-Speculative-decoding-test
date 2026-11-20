#!/usr/bin/env python3
"""Lossless private PR54 closeout audit without extracting the 10 GB archive.

This reads bytes only: no model imports, GPU work, remote operations or deletion.
A passed CLOSEOUT.json requires archive identity, every regular member hash,
original scalar/retirement evidence and the fresh remote keeper observations.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

COMMIT = 'b4459820b3822c3d760bb3fd880e4d139fa2bbc2'
CAMPAIGN_SUFFIX = ('dflash_full_c_gsm8k_20261118', 'execution_001')
BASE = 'campaign_001/arms/KPLUS_C'
C58 = dict(bytes=2097259104, sha256='705c6b8b81895837ffbe1fb926aa266078d6af2c9904a77642b797a81dc1f937')
STATE58 = dict(bytes=6532631103, sha256='5dd521062f8501eb3e399ca5e79085132602efe2627dc32ddc6744094c4782ff')
WORK = dict(adam=7430,response_forward=14862,response_backward=14862,training_initialization=4,
    feedback_generation=7473,teacher_preparation=7424,correctness_generation=4,canonical_generation=0,
    qualification_generation=24,warmup_generation=576,scored_generation=18432)


def identity(path):
    path=Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def bytes_identity(raw):
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def content_identity(row):
    return {key:row[key] for key in ('bytes','sha256')}


def load_json(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique)


def read(path):
    return load_json(Path(path).read_bytes())


def safe_name(name):
    path=PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or str(path)!=name
            or '\\' in name or '\x00' in name):
        raise ValueError('Unsafe/noncanonical archive member: '+repr(name))
    return name


class StrictZstdReader(io.RawIOBase):
    """Bounded-input decoder that requires each zstd frame to reach EOF.

    stream_reader alone can silently accept a truncated final frame. Small
    compressed input chunks also bound transient expansion before readinto.
    Concatenated complete frames are accepted; incomplete frames/junk fail.
    """
    def __init__(self, source):
        import zstandard
        super().__init__();self.source=source;self.factory=zstandard.ZstdDecompressor()
        self.decoder=None;self.pending=b'';self.ended=False;self.frames=0
    def readable(self):return True
    def readinto(self, target):
        view=memoryview(target);written=0
        while written<len(view):
            if self.pending:
                n=min(len(self.pending),len(view)-written)
                view[written:written+n]=self.pending[:n];self.pending=self.pending[n:];written+=n
                continue
            if self.ended:break
            raw=self.source.read(1024)
            if not raw:
                if self.decoder is not None and not self.decoder.eof:
                    raise EOFError('Truncated zstd frame without frame EOF')
                self.ended=True;break
            while raw:
                if self.decoder is None:self.decoder=self.factory.decompressobj()
                self.pending+=self.decoder.decompress(raw)
                if self.decoder.eof:
                    raw=self.decoder.unused_data;self.frames+=1;self.decoder=None
                else:raw=b''
        return written


class TailReader:
    """Observe every decompressed byte, including tar padding, without saving it."""
    def __init__(self,source):self.source=source;self.total=0;self.tail=b''
    def read(self,size=-1):
        value=self.source.read(size);self.total+=len(value)
        if value:self.tail=(self.tail+value[-1024:])[-1024:]
        return value


def streamed_members(archive, compact, *, on_member=None):
    """Require exactly the named regular members; never extract or follow links."""
    expected={safe_name(name):row for name,row in compact.items()}
    seen=set();logical=0;began=time.monotonic();last=began
    with ExitStack() as stack:
        raw=stack.enter_context(Path(archive).open('rb'))
        if str(archive).endswith(('.zst','.zstd')):
            decoder=StrictZstdReader(raw)
            stream=stack.enter_context(io.BufferedReader(decoder,buffer_size=1024*1024))
        elif str(archive).endswith(('.gz','.tgz')):
            stream=stack.enter_context(gzip.GzipFile(fileobj=raw,mode='rb'))
        else:stream=raw
        # ignore_zeros scans past end markers too, detecting appended tar members.
        observed=TailReader(stream)
        tar=stack.enter_context(tarfile.open(fileobj=observed,mode='r|',ignore_zeros=True))
        for member in tar:
            name=safe_name(member.name)
            if name in seen:raise ValueError('Duplicate archive member: '+name)
            if name not in expected:raise ValueError('Unexpected archive member: '+name)
            if not member.isfile():raise ValueError('Non-regular archive member: '+name)
            if name.endswith(('.pt','.safetensors')):raise ValueError('Tensor payload in compact archive: '+name)
            want=expected[name]
            if member.size!=want['bytes']:raise ValueError('Member size differs: '+name)
            selected=on_member is not None and on_member.wants(name)
            if selected and member.size>64*1024*1024:
                raise ValueError('Selected scalar receipt unexpectedly exceeds 64 MiB: '+name)
            pieces=[];sha=hashlib.sha256();length=0
            content=tar.extractfile(member)
            if content is None:raise ValueError('Unreadable archive member: '+name)
            while chunk:=content.read(1024*1024):
                sha.update(chunk);length+=len(chunk)
                if selected:pieces.append(chunk)
            if dict(bytes=length,sha256=sha.hexdigest())!=content_identity(want):
                raise ValueError('Archive member identity differs: '+name)
            seen.add(name);logical+=length
            if selected:on_member.consume(name,load_json(b''.join(pieces)))
            now=time.monotonic()
            if now-last>=20:
                print(json.dumps(dict(phase='streaming_members',verified=len(seen),total=len(expected),logical_bytes=logical)),flush=True)
                last=now
        # Require decompressor EOF/checksum even after the final tar padding.
        while observed.read(1024*1024):pass
        if observed.total%512 or observed.total<1024 or observed.tail!=b'\0'*1024:
            raise ValueError('Tar stream lacks aligned final two zero blocks')
        if isinstance(stream,io.BufferedReader) and isinstance(stream.raw,StrictZstdReader):
            if not stream.raw.ended or stream.raw.frames<1:raise ValueError('No complete zstd frame')
    missing=sorted(set(expected)-seen)
    if missing:raise ValueError('Missing archive members: '+repr(missing[:20])+f' ({len(missing)} total)')
    return dict(status='passed',regular_files=len(seen),logical_bytes=logical,
                missing=0,extra=0,duplicates=0,nonregular=0,extracted_files=0,decompressed_tar_bytes=observed.total,final_tar_zero_blocks=True,wall_s=time.monotonic()-began)


class Reconciliation:
    def __init__(self,manifest,keeper_stats,input_auth=None,external_auth=None):
        self.manifest=manifest;self.compact=manifest['compact'];self.stats=keeper_stats;self.auth=input_auth;self.external_auth=external_auth
        self.records={};self.errors=[];self.checks=0;self.preparations={};self.events={};self.prepared_successors={};self.event_summaries=[]
        self.retired=Counter();self.retired_bytes=Counter();self.removed_paths=set();self.source_checks=0
        self.fixed={'WORK.json','CLOCK.json','FINAL_AUDIT.json','QUALIFICATION_RETIREMENT.json',
            'admin/SOURCE004.json','admin/PRODUCTION_PLAN.json','production_resume_001/LAUNCH.json',
            BASE+'/STATUS.json',BASE+'/FINAL_STATE_CHECK.json',BASE+'/trainer/INITIAL_CHECK.json',
            BASE+'/trainer/INITIAL_COMPARISON.json'}
    def check(self,condition,message):
        self.checks+=1
        if not condition:self.errors.append(message)
    def wants(self,name):
        return (name in self.fixed or bool(re.fullmatch(re.escape(BASE)+r'/boundaries/event_\d{3}\.json',name))
            or bool(re.fullmatch(re.escape(BASE)+r'/events/event_\d{3}/(RETIREMENT|FEATURE_RETIREMENT_IDENTITIES|PREDECESSOR_TENSOR_FINGERPRINTS|PUBLICATION_CHECK)\.json',name))
            or (bool(re.fullmatch(re.escape(BASE)+r'/events/event_\d{3}/preparation_retirement/[^/]+\.json',name))
                and not name.endswith(('.plan.json','.equivalence.json'))))
    def relative(self,path):
        prefix=self.manifest['source_root'].rstrip('/')+'/'
        if not path.startswith(prefix):raise ValueError('Receipt references outside source root: '+path)
        return safe_name(path[len(prefix):])
    def consume(self,name,row):
        if '/preparation_retirement/' in name:
            event=int(name.split('/events/event_')[1][:3]);self.preparations.setdefault(event,[]).append(name)
            self.prepared_successors.setdefault(event,[]).append(row.get('retained',{}).get('prepared_packet'))
            self.retirement(name,row,preparation=True)
        else:
            if name.endswith('/FEATURE_RETIREMENT_IDENTITIES.json'):
                row={'status':row.get('status'),'packets':[{'path':x['path'],'identity':x['identity']} for x in row.get('packets',[])]}
            elif name.endswith('/PREDECESSOR_TENSOR_FINGERPRINTS.json'):
                row={k:row.get(k) for k in ('status','identity','export_identity','exact_export_tensors','path','export_path')}
            self.records[name]=row
            if name.endswith('/RETIREMENT.json'):
                event=int(name.split('/events/event_')[1][:3]);self.events[event]=name
                self.retirement(name,row,preparation=False)
    def retirement(self,name,row,*,preparation):
        self.check(row.get('status')=='completed',name+': retirement not completed')
        self.check(row.get('removed')==row.get('files'),name+': planned/removed paths differ')
        removed=row.get('removed',[])
        if preparation:
            self.check(dict(Counter(x['kind'] for x in removed))=={'native_capture':1,'terminal_feature':1},name+': preparation retirement kinds')
        self.check(row.get('removed_bytes')==sum(x['bytes'] for x in removed),name+': retired byte sum differs')
        self.check('not archived' in row.get('lost_capability',''),name+': lost tensor capability not disclosed')
        for entry in removed:
            path=self.relative(entry['path'])
            self.check(path not in self.removed_paths,name+': duplicate retired path '+path)
            self.removed_paths.add(path)
            self.check(path not in self.compact and path not in self.manifest['tensor_keepers'],name+': retired path remains in fresh inventory '+path)
            kind=('preparation:' if preparation else 'event:')+entry['kind']
            self.retired[kind]+=1;self.retired_bytes[kind]+=entry['bytes']
        for absolute,want in row.get('compact_evidence',{}).items():
            path=self.relative(absolute);actual=self.compact.get(path)
            self.check(actual is not None and content_identity(actual)==want,name+': compact evidence identity '+path)
            self.source_checks+=1
    def finish(self):
        for name in self.fixed:self.check(name in self.records,'Missing scalar reconciliation input '+name)
        if self.errors:return self.result()
        d=self.records;keepers=self.manifest['tensor_keepers'];root=self.manifest['source_root']
        self.check(PurePosixPath(root).is_absolute() and tuple(PurePosixPath(root).parts[-2:])==CAMPAIGN_SUFFIX,'Unexpected PR54 execution root')
        initial=BASE+'/trainer/initial_state.pt';alias=BASE+'/trainer/training_state.pt'
        state=BASE+'/trainer/events/event_058/training_state.pt';weights=BASE+'/trainer/events/event_058/export/model.safetensors'
        self.check(set(keepers)=={initial,alias,state,weights},'Retained tensor path set differs')
        for name,want in [(alias,STATE58),(state,STATE58),(weights,C58)]:
            self.check(name in keepers and content_identity(keepers[name])==want,'Historical keeper identity differs '+name)
        initial_check=d[BASE+'/trainer/INITIAL_CHECK.json']
        for key,value in dict(status='passed',completed_event=0,global_step=0,presentations=0,
                              adam_state_entries=0,export_exact=True).items():
            self.check(initial_check.get(key)==value,'Initial state evidence '+key)
        self.check(initial_check.get('state_path')==root+'/'+initial,'Initial state receipt path')
        self.check(initial_check.get('original_K_initialization',{}).get('exact_released_weights_empty_adam_rng') is True,'Original K0 equality evidence')
        self.check(d[BASE+'/trainer/INITIAL_COMPARISON.json'].get('status')=='passed','Initial exact comparison status')
        status=d[BASE+'/STATUS.json'];final=d[BASE+'/FINAL_STATE_CHECK.json']
        self.check(status.get('status')=='completed' and status.get('events_completed')==58 and status.get('tail_completed')==49,'C58 completion counts')
        self.check(final.get('status')=='passed' and final.get('completed_event')==58 and final.get('global_step')==7424 and final.get('export_exact') is True,'Final state/export equality evidence')
        self.check(final.get('live',{}).get('exact_resume') is True,'Final complete-state live equality evidence')
        self.check(d['WORK.json'].get('attempted')==WORK and d['WORK.json'].get('completed')==WORK and not d['WORK.json'].get('pending'),'Full work counts including original failures/qualification')
        self.check(d['FINAL_AUDIT.json'].get('status')=='passed' and d['FINAL_AUDIT.json'].get('execution_commit')==COMMIT,'Final endpoint audit execution')
        self.check(all(d['FINAL_AUDIT.json'].get('datasets',{}).get(k,{}).get('status')=='passed' for k in ('gsm8k','mathqa')),'Both final dataset audits')
        self.check(d['admin/PRODUCTION_PLAN.json'].get('execution_commit')==COMMIT,'Production plan source identity')
        self.check(d['admin/SOURCE004.json'].get('commit')==COMMIT and d['admin/SOURCE004.json'].get('dirty') is False,'Clean published production source manifest')
        self.check(d['production_resume_001/LAUNCH.json'].get('status')=='completed','Production continuation completed')
        for name in ('LAUNCH.json','CPU_PREFLIGHT.json','QUALIFICATION.json','QUALIFICATION_RETIREMENT.json',
                     'initial_retry_001/LAUNCH.json','initial_retry_001/prior_failure/PRESERVATION.json',
                     'diagnostics/constructor_002/DETAILS.json','admin/ELAPSED_EXTENSION_AUTHORIZATION.json'):
            self.check(name in self.compact,'Preserved original failure/qualification/provenance member '+name)
        self.check(set(self.events)==set(range(1,59)),'All 58 event retirement receipts')
        self.check(set(self.preparations)==set(range(1,59)) and all(len(v)==128 for v in self.preparations.values()),'All 7424 preparation retirement receipts')
        for event in range(1,59):
            eroot=f'{BASE}/events/event_{event:03d}';bname=f'{BASE}/boundaries/event_{event:03d}.json'
            required=[bname,eroot+'/RETIREMENT.json',eroot+'/FEATURE_RETIREMENT_IDENTITIES.json',eroot+'/PUBLICATION_CHECK.json']
            if event>1:required.append(eroot+'/PREDECESSOR_TENSOR_FINGERPRINTS.json')
            for name in required:self.check(name in d,'Missing event receipt '+name)
            if any(name not in d for name in required):continue
            boundary=d[bname];retired=d[eroot+'/RETIREMENT.json'];features=d[eroot+'/FEATURE_RETIREMENT_IDENTITIES.json']
            self.check(boundary.get('status')=='passed' and boundary.get('event')==event,'Boundary event '+str(event))
            self.check(boundary.get('scientific_check',{}).get('global_step')==event*128 and boundary['scientific_check'].get('export_exact') is True,'Boundary step/export '+str(event))
            expected={('prepared_feature'):128,**({'previous_state':1,'previous_export':1} if event>1 else {})}
            self.check(dict(Counter(x['kind'] for x in retired['removed']))==expected,'Retired kinds event '+str(event))
            self.check(retired.get('retained',{}).get('current_state')==boundary['state'] and retired['retained'].get('current_weights')==boundary['weights'] and retired['retained'].get('initial_state')==root+'/'+initial,'Retained successor event '+str(event))
            packets=features.get('packets',[]);prepared={x['path']:x for x in retired['removed'] if x['kind']=='prepared_feature'}
            self.check(features.get('status')=='passed' and len(packets)==128 and {x['path'] for x in packets}==set(prepared),'Feature identity set event '+str(event))
            self.check(len(self.prepared_successors.get(event,[]))==128 and set(self.prepared_successors[event])==set(prepared),'Preparation to event retirement chain '+str(event))
            self.check(all(x['identity']['bytes']==prepared[x['path']]['bytes'] for x in packets if x['path'] in prepared),'Feature retired byte identity event '+str(event))
            publication=d[eroot+'/PUBLICATION_CHECK.json']
            self.check(publication.get('status')=='passed' and publication.get('all_equal') is True,'Publication evidence event '+str(event))
            if event>1:
                previous=d.get(f'{BASE}/boundaries/event_{event-1:03d}.json',{});finger=d[eroot+'/PREDECESSOR_TENSOR_FINGERPRINTS.json']
                self.check(finger.get('status')=='passed' and finger.get('identity')==previous.get('state_identity') and finger.get('export_identity')==previous.get('weights_identity') and finger.get('exact_export_tensors') is True,'Predecessor state/export fingerprint event '+str(event))
                self.check(finger.get('path')==previous.get('state') and finger.get('export_path')==str(PurePosixPath(previous['weights']).parent),'Predecessor fingerprint path event '+str(event))
                predecessor={entry['kind']:entry for entry in retired['removed'] if entry['kind'] in ('previous_state','previous_export')}
                for kind,field in [('previous_state','state'),('previous_export','weights')]:
                    self.check(predecessor.get(kind,{}).get('path')==previous.get(field)
                        and predecessor.get(kind,{}).get('bytes')==previous.get(field+'_identity',{}).get('bytes'),
                        'Exact retired predecessor '+kind+' event '+str(event))
            self.event_summaries.append(dict(event=event,boundary_member=bname,boundary_identity=content_identity(self.compact[bname]),
                state_identity=boundary['state_identity'],weights_identity=boundary['weights_identity'],
                retirement_member=eroot+'/RETIREMENT.json',retirement_identity=content_identity(self.compact[eroot+'/RETIREMENT.json']),
                retired_files=len(retired['removed']),cumulative_retired_logical_bytes=retired['removed_bytes'],prepared_features=len(packets),
                original_complete_state_and_export_equality=True,successor_retained_then_predecessor_retired_in_next_event=event<58))
            if event==58:
                self.check(boundary['state_identity']==STATE58 and boundary['weights_identity']==C58,'Final boundary hashes')
                self.check(status.get('boundary')==boundary,'Final status/boundary identity')
        rows={r['path']:r for r in self.stats.get('rows',[])};unique={};keeper_rows=[]
        for name,idn in keepers.items():
            absolute=root+'/'+name;row=rows.get(absolute)
            self.check(row is not None,'Fresh keeper stat missing '+name)
            if row is None:continue
            self.check(row['size']==idn['bytes'] and row['allocated_bytes']==idn['allocated_bytes'],'Fresh keeper stat size/allocation '+name)
            self.check(row['resolved']==absolute and row['uid']==0 and row['links']>=1,'Fresh keeper ownership/link path '+name)
            physical=(row['device'],row['inode']);unique.setdefault(physical,row)
            keeper_rows.append(dict(path=absolute,identity=content_identity(idn),stat=row,disposition='KEEP',
                archived_payload=False,verification='Fresh remote file hash reconciled to archived scalar receipts; no tensor reload'))
        self.check(rows.get(root+'/'+alias,{}).get('inode')==rows.get(root+'/'+state,{}).get('inode') and rows.get(root+'/'+alias,{}).get('device')==rows.get(root+'/'+state,{}).get('device'),'Current recovery alias physical identity')
        self.check(len(unique)==3,'Exactly three physical retained tensor inodes')
        self.keeper_result=dict(paths=keeper_rows,physical_inodes=len(unique),unique_allocated_bytes=sum(x['allocated_bytes'] for x in unique.values()),
                               byte_hash_source_epoch=self.manifest.get('started_epoch'),stat_epoch=self.stats.get('epoch'),
                               initial_hash_scope='New current inventory identity; original semantic equality comes from archived exact INITIAL_CHECK/INITIAL_COMPARISON')
        if self.auth is not None:
            self.check(self.auth.get('status')=='passed','External current input authentication status')
            external={r['path']:r for r in self.auth.get('files',[])}
            runs_root=PurePosixPath(root).parent.parent
            exports={
                str(runs_root/'dflash_online_osd_rerun_20261023/campaign_001/trainer/events/event_058/export'):dict(bytes=2097259104,sha256='796ec9d486046d43a2b91bd0ddf32b33eb3183b7d924e9dd9ab6bc4c6b0d0856'),
                str(runs_root/'dflash_adaflash_arepo_20261107/production_001/campaign_001/arms/AREPO/trainer/events/event_058/export'):dict(bytes=2097259104,sha256='fab2e033b9c2cc8c30d36e4fe676fc65eeb8087b2acc9eabad9279f0d92b19e7'),
                root+'/'+BASE+'/trainer/events/event_058/export':C58}
            config=dict(bytes=1211,sha256='5749d96966f9ce7e6a9662c9f482b8aa0d461845fb6b9ce09ac2c08311fab2aa')
            for export,model in exports.items():
                for filename,want in [('model.safetensors',model),('config.json',config)]:
                    observed=external.get(export+'/'+filename,{})
                    self.check(observed.get('exists') is True and observed.get('matches') is True
                        and observed.get('actual')==want and observed.get('expected')==want,
                        'O58/AREPO58/C58 exact external export/config authentication '+export+'/'+filename)
        self.external_result=dict(status='not_supplied',scope='Original K0/KPLUS58/runtime external keeper authentication')
        if self.external_auth is not None:
            external=self.external_auth
            roles={'KPLUS58_selected_endpoint','initial_state_witness','qualified_runtime_overlay'}
            dependencies=[r for r in d['admin/PRODUCTION_PLAN.json']['protected_dependencies']['rows'] if r['role'] in roles]
            expected={r['path']:r['expected_identity'] for r in dependencies}
            observed={r['path']:r for r in external.get('rows',[])}
            start_checks=self.checks;start_errors=len(self.errors)
            self.check(external.get('status')=='passed' and not external.get('errors') and external.get('new_model_work')==0,'Supplemental external keeper receipt status/work')
            self.check(len(expected)==10 and len(observed)==10 and len(external.get('rows',[]))==10 and set(observed)==set(expected),'Exact KPLUS58/K0/seven-overlay keeper paths')
            for path,want in expected.items():
                row=observed.get(path,{})
                self.check(row.get('actual_identity')==want and row.get('expected_identity')==want
                    and row.get('identity_equal') is True and row.get('resolved_path')==path
                    and row.get('observed_epoch',0)>=external.get('started_epoch',float('inf')),
                    'Supplemental fresh external keeper identity '+path)
            self.external_result=dict(status='passed' if len(self.errors)==start_errors else 'failed',
                checked_files=len(expected),checks=self.checks-start_checks,
                started_epoch=external.get('started_epoch'),finished_epoch=external.get('finished_epoch'),
                scope='KPLUS58 export/config, original K0 witness and seven native runtime overlays',
                observation_fields='actual_identity, identity_equal and observed_epoch are the fresh observations; copied current_availability wording remains historical metadata.')
        self.check(sum(1 for n in self.compact if n.startswith(BASE+'/tail/') and n.endswith('.json'))>=49,'Retained final tail record coverage')
        return self.result()
    def result(self):
        return dict(status='passed' if not self.errors else 'failed',checks=self.checks,errors=self.errors,
            retired_paths=len(self.removed_paths),retired_file_counts_by_kind=dict(self.retired),
            cumulative_retired_logical_bytes_by_kind=dict(self.retired_bytes),
            compact_evidence_reference_checks=self.source_checks,event_retirement_receipts=len(self.events),
            preparation_retirement_receipts=sum(map(len,self.preparations.values())),
            keepers=getattr(self,'keeper_result',None),events=self.event_summaries,
            supplemental_external_keepers=getattr(self,'external_result',None),
            scalar_receipt_members={name:content_identity(self.compact[name]) for name in sorted(self.fixed) if name in self.compact},
            meaning='Archived compact evidence and current remote keeper observations reconciled; tensor deletion was historical, not performed here.',
            lost_capabilities='Retired prepared/capture tensors and predecessor states/exports are not archived and cannot provide direct historical feature replay or optimizer continuation.',
            limitations=['No tensor reload, native model replication or statistical reanalysis.',
                         'Retirement byte sums are cumulative historical file sizes, not simultaneous reclaimed space.',
                         'Current remote observations apply at the recorded manifest/stat epochs.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--remote-delivery',required=True);parser.add_argument('--transport-receipt')
    parser.add_argument('--archive-sha256');parser.add_argument('--archive-bytes',type=int)
    parser.add_argument('--keeper-stats',required=True);parser.add_argument('--input-authentication')
    parser.add_argument('--external-keeper-authentication',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();started=datetime.now(timezone.utc).isoformat();before=shutil.disk_usage(out).free
    result=dict(status='in_progress',scope='PR54 whole compact archive delivery and keeper reconciliation',
                started_utc=started,model_work=0,gpu_initialized=False,deleted_paths=[],extracted_files=0)
    try:
        manifest_raw=Path(args.manifest).read_bytes();manifest=load_json(manifest_raw);delivery=read(args.remote_delivery)
        if delivery.get('status')!='passed' or bytes_identity(manifest_raw)!=delivery['manifest']:
            raise ValueError('Original remote delivery does not authenticate this exact member manifest')
        if (manifest['source_root']!=delivery['source_root'] or len(manifest['compact'])!=delivery['compact_files']
                or sum(x['bytes'] for x in manifest['compact'].values())!=delivery['compact_logical_bytes']
                or len(manifest['tensor_keepers'])!=delivery['tensor_keepers']):
            raise ValueError('Manifest counts/root differ from original delivery')
        if manifest.get('deleted_paths') or delivery.get('deleted_paths'):
            raise ValueError('Packaging receipt reports unexpected deletion')
        if manifest.get('symlinks_not_followed'):
            raise ValueError('Symlink reconciliation is required before this PR54 gate')
        if args.transport_receipt:
            transport=read(args.transport_receipt)
            if transport.get('status') not in ('passed','completed','packaged','compressed_same_tar_stream'):
                raise ValueError('Transport archive has no successful receipt')
            expected=transport.get('archive',transport)
            if transport.get('file') and transport['file']!=Path(args.archive).name:
                raise ValueError('Transport receipt identifies a different archive filename')
            if 'manifest' in transport and transport['manifest']!=bytes_identity(manifest_raw):
                raise ValueError('Transport receipt names a different member manifest')
            # A flat recompression receipt authenticates compressed bytes only.
            # The separately authenticated original manifest and complete member
            # audit below bind those bytes to the original execution inventory.
        elif args.archive_sha256 is not None and args.archive_bytes is not None:
            expected=dict(bytes=args.archive_bytes,sha256=args.archive_sha256)
        elif str(args.archive).endswith(('.gz','.tgz')):expected=delivery['archive']
        else:raise ValueError('Recompressed zstd archive requires its own exact remote size/SHA256 or transport receipt')
        archive=Path(args.archive);stat=archive.stat();actual=identity(archive)
        if actual!=content_identity(expected):raise ValueError('Compressed archive file does not match remote transport identity')
        reconcile=Reconciliation(manifest,read(args.keeper_stats),read(args.input_authentication) if args.input_authentication else None,read(args.external_keeper_authentication))
        streamed=streamed_members(archive,manifest['compact'],on_member=reconcile)
        after=archive.stat()
        if (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Archive changed while verifying')
        reconciled=reconcile.finish()
        (out/'RECONCILIATION.json').write_text(json.dumps(reconciled,indent=2)+'\n')
        result.update(status='passed' if reconciled['status']=='passed' else 'failed',archive=str(archive.resolve()),archive_identity=actual,
            manifest=str(Path(args.manifest).resolve()),manifest_identity=bytes_identity(manifest_raw),
            source_root=manifest['source_root'],stream_verification=streamed,
            keeper_and_retirement_reconciliation=dict(status=reconciled['status'],checks=reconciled['checks'],errors=reconciled['errors']),
            receipt_inputs={name:dict(path=str(Path(value).resolve()),identity=identity(value)) for name,value in vars(args).items()
                if name in ('remote_delivery','transport_receipt','keeper_stats','input_authentication','external_keeper_authentication') and value},
            tensor_payloads='Retained remotely, not included in compact archive',
            archive_delivery_completed=True,keeper_reconciliation_completed=reconciled['status']=='passed')
    except Exception as error:
        result.update(status='failed',error=type(error).__name__+': '+str(error),archive_delivery_completed=False,keeper_reconciliation_completed=False)
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_s=time.monotonic()-began,
                  local_free_before_bytes=before,local_free_after_bytes=shutil.disk_usage(out).free,
                  verifier_identity=identity(__file__))
    (out/'CLOSEOUT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
