"""Independent, in-memory archive corruption fixtures; no extraction or model work."""
from copy import deepcopy
import gzip
import hashlib
import io
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch

import zstandard
from experiments.fresh_mathqa_endpoints.archive_audit import verify_archive as V

TEST_ROOT = "/fixture/runs/dflash_full_c_gsm8k_20261118/execution_001"


def make_tar(entries):
    result=io.BytesIO()
    with tarfile.open(fileobj=result,mode='w') as archive:
        for name,payload,kind in entries:
            member=tarfile.TarInfo(name)
            if kind=='file':
                member.size=len(payload);archive.addfile(member,io.BytesIO(payload))
            else:
                member.type={'hardlink':tarfile.LNKTYPE,'symlink':tarfile.SYMTYPE,'directory':tarfile.DIRTYPE}[kind]
                member.linkname='a.json' if kind!='directory' else ''
                archive.addfile(member)
    return result.getvalue()


def manifest(entries):
    return {name:{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()} for name,body in entries}


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.body=b'{"status":"passed"}'
        self.tar=make_tar([('a.json',self.body,'file')])
        self.members=manifest([('a.json',self.body)])

    def run_archive(self,raw,*,suffix='.tar',members=None,collector=None):
        with patch.object(Path,'open',return_value=io.BytesIO(raw)):
            return V.streamed_members('in_memory'+suffix,self.members if members is None else members,on_member=collector)

    def test_complete_archive_all_supported_compressions(self):
        for suffix,raw in [('.tar',self.tar),('.tar.gz',gzip.compress(self.tar)),
                           ('.tar.zst',zstandard.ZstdCompressor().compress(self.tar))]:
            with self.subTest(suffix=suffix):
                result=self.run_archive(raw,suffix=suffix)
                self.assertEqual(result['status'],'passed')
                self.assertEqual(result['regular_files'],1)
                self.assertEqual(result['logical_bytes'],len(self.body))
                self.assertEqual(result['extracted_files'],0)
                self.assertTrue(result['final_tar_zero_blocks'])

    def test_missing_duplicate_and_unexpected_regular_members(self):
        wanted=self.members|manifest([('missing.json',b'{}')])
        with self.assertRaisesRegex(ValueError,'Missing archive members'):self.run_archive(self.tar,members=wanted)
        for entries,word in [([('a.json',self.body,'file')]*2,'Duplicate'),
                             ([('a.json',self.body,'file'),('extra.json',b'{}','file')],'Unexpected')]:
            with self.subTest(word=word),self.assertRaisesRegex(ValueError,word):self.run_archive(make_tar(entries))

    def test_appended_tar_members_after_zero_blocks_are_detected(self):
        appended=self.tar+make_tar([('extra.json',b'{}','file')])
        with self.assertRaisesRegex(ValueError,'Unexpected'):self.run_archive(appended)

    def test_changed_payload_or_declared_size_rejected(self):
        bad=deepcopy(self.members);bad['a.json']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'identity differs'):self.run_archive(self.tar,members=bad)
        bad=deepcopy(self.members);bad['a.json']['bytes']+=1
        with self.assertRaisesRegex(ValueError,'size differs'):self.run_archive(self.tar,members=bad)

    def test_tensor_payload_is_never_a_compact_member(self):
        for name in ['model.safetensors','training_state.pt']:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Tensor payload'):
                self.run_archive(make_tar([(name,b'opaque','file')]),members=manifest([(name,b'opaque')]))

    def test_hardlinks_symlinks_and_directories_are_not_followed(self):
        for kind in ['hardlink','symlink','directory']:
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'Non-regular'):
                self.run_archive(make_tar([('a.json',b'',kind)]),members=manifest([('a.json',b'')]))

    def test_unsafe_and_noncanonical_member_names_rejected(self):
        for name in ['../a.json','/a.json','a/../b.json','a//b.json','./a.json','a\\b.json']:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Unsafe/noncanonical'):
                V.safe_name(name)

    def test_complete_member_without_final_tar_blocks_is_rejected(self):
        # This retains the full member header and payload; only finalization is missing.
        with self.assertRaisesRegex(ValueError,'final two zero blocks'):self.run_archive(self.tar[:1024])

    def test_unaligned_tar_tail_is_rejected(self):
        with self.assertRaises((ValueError,tarfile.ReadError)):
            self.run_archive(self.tar+b'\0')

    def test_truncated_zstd_final_frame_is_rejected(self):
        encoded=zstandard.ZstdCompressor().compress(self.tar)
        with self.assertRaises((EOFError,zstandard.ZstdError)):
            self.run_archive(encoded[:-1],suffix='.tar.zst')

    def test_complete_concatenated_zstd_frames(self):
        compressor=zstandard.ZstdCompressor()
        encoded=compressor.compress(self.tar[:4096])+compressor.compress(self.tar[4096:])
        self.assertEqual(self.run_archive(encoded,suffix='.tar.zst')['status'],'passed')

    def test_gzip_checksum_and_zstd_junk_tail_are_checked(self):
        encoded=bytearray(gzip.compress(self.tar));encoded[-8]^=1
        with self.assertRaises(gzip.BadGzipFile):self.run_archive(bytes(encoded),suffix='.tar.gz')
        encoded=zstandard.ZstdCompressor().compress(self.tar)+b'junk'
        with self.assertRaises(zstandard.ZstdError):self.run_archive(encoded,suffix='.tar.zst')

    def test_selected_json_consumed_only_after_its_hash_passes(self):
        class Collector:
            def __init__(self):self.rows=[]
            def wants(self,name):return True
            def consume(self,name,row):self.rows.append((name,row))
        collector=Collector();self.run_archive(self.tar,collector=collector)
        self.assertEqual(collector.rows,[('a.json',{'status':'passed'})])
        bad=deepcopy(self.members);bad['a.json']['sha256']='f'*64;collector=Collector()
        with self.assertRaisesRegex(ValueError,'identity differs'):self.run_archive(self.tar,members=bad,collector=collector)
        self.assertEqual(collector.rows,[])

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):V.load_json(b'{"a":1,"a":2}')


class ReconciliationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'source_root':TEST_ROOT,'compact':{},'tensor_keepers':{}}
        self.reconcile=V.Reconciliation(self.manifest,{'rows':[]})

    def test_retirement_and_boundary_selection_excludes_companions(self):
        prefix=V.BASE+'/events/event_001/preparation_retirement/train_00000'
        self.assertTrue(self.reconcile.wants(prefix+'.json'))
        self.assertFalse(self.reconcile.wants(prefix+'.plan.json'))
        self.assertFalse(self.reconcile.wants(prefix+'.equivalence.json'))
        self.assertTrue(self.reconcile.wants(V.BASE+'/boundaries/event_001.json'))
        self.assertFalse(self.reconcile.wants(V.BASE+'/boundaries/event_001_validation.json'))

    def test_retirement_cannot_claim_payload_is_still_retained(self):
        relative=V.BASE+'/events/event_001/loop/features/train_00000.pt'
        self.manifest['tensor_keepers'][relative]={'bytes':12,'sha256':'f'*64}
        removed=[{'path':TEST_ROOT+'/'+relative,'bytes':12,'kind':'prepared_feature'}]
        row={'status':'completed','files':removed,'removed':removed,'removed_bytes':12,
             'lost_capability':'Deleted tensors are not archived; direct tensor replay is lost.',
             'compact_evidence':{}}
        self.reconcile.retirement('test',row,preparation=False)
        self.assertTrue(any('remains in fresh inventory' in error for error in self.reconcile.errors))

    def test_retirement_preserved_evidence_identity_must_match(self):
        self.manifest['compact']['retained.json']={'bytes':2,'sha256':'a'*64}
        row={'status':'completed','files':[],'removed':[],'removed_bytes':0,
             'lost_capability':'Retired, not archived.',
             'compact_evidence':{TEST_ROOT+'/retained.json':{'bytes':2,'sha256':'b'*64}}}
        self.reconcile.retirement('test',row,preparation=True)
        self.assertTrue(any('compact evidence identity' in error for error in self.reconcile.errors))


def complete_reconciliation_fixture():
    """Only scalar audit structures; all actual archive bytes are tested separately."""
    initial=V.BASE+'/trainer/initial_state.pt'
    alias=V.BASE+'/trainer/training_state.pt'
    final_state=V.BASE+'/trainer/events/event_058/training_state.pt'
    final_weights=V.BASE+'/trainer/events/event_058/export/model.safetensors'
    keepers={initial:dict(bytes=2,sha256='1'*64,allocated_bytes=4096),
             alias:dict(V.STATE58,allocated_bytes=6532632576),
             final_state:dict(V.STATE58,allocated_bytes=6532632576),
             final_weights:dict(V.C58,allocated_bytes=2097262592)}
    compact={};manifest_value={'source_root':TEST_ROOT,'compact':compact,'tensor_keepers':keepers}
    rows=[]
    for path,inode,links in [(initial,1,1),(alias,2,2),(final_state,2,2),(final_weights,3,1)]:
        idn=keepers[path];absolute=TEST_ROOT+'/'+path
        rows.append(dict(path=absolute,resolved=absolute,size=idn['bytes'],allocated_bytes=idn['allocated_bytes'],
                         inode=inode,device=2066,links=links,uid=0,gid=0))
    result=V.Reconciliation(manifest_value,dict(status='observed',rows=rows,epoch=1))
    d=result.records
    d.update({'WORK.json':dict(attempted=V.WORK,completed=V.WORK,pending={}), 'CLOCK.json':{},
              'FINAL_AUDIT.json':dict(status='passed',execution_commit=V.COMMIT,
                   datasets={name:{'status':'passed'} for name in ('gsm8k','mathqa')}),
              'QUALIFICATION_RETIREMENT.json':{},'admin/SOURCE004.json':dict(commit=V.COMMIT,dirty=False),
              'admin/PRODUCTION_PLAN.json':{'execution_commit':V.COMMIT},
              'production_resume_001/LAUNCH.json':{'status':'completed'},
              V.BASE+'/STATUS.json':dict(status='completed',events_completed=58,tail_completed=49),
              V.BASE+'/FINAL_STATE_CHECK.json':dict(status='passed',completed_event=58,global_step=7424,
                                                  export_exact=True,live={'exact_resume':True}),
              V.BASE+'/trainer/INITIAL_CHECK.json':dict(status='passed',completed_event=0,global_step=0,
                    presentations=0,adam_state_entries=0,export_exact=True,state_path=TEST_ROOT+'/'+initial,
                    original_K_initialization={'exact_released_weights_empty_adam_rng':True}),
              V.BASE+'/trainer/INITIAL_COMPARISON.json':{'status':'passed'}})
    prior=None
    for event in range(1,59):
        eroot=f'{V.BASE}/events/event_{event:03d}';troot=f'{V.BASE}/trainer/events/event_{event:03d}'
        bname=f'{V.BASE}/boundaries/event_{event:03d}.json'
        state=TEST_ROOT+'/'+troot+'/training_state.pt';weights=TEST_ROOT+'/'+troot+'/export/model.safetensors'
        state_id=V.STATE58 if event==58 else dict(bytes=10,sha256=f'{event:064x}')
        weights_id=V.C58 if event==58 else dict(bytes=10,sha256=f'{event+100:064x}')
        boundary=dict(status='passed',event=event,arm='KPLUS_C',state=state,weights=weights,
                      state_identity=state_id,weights_identity=weights_id,
                      scientific_check=dict(global_step=event*128,export_exact=True))
        paths=[TEST_ROOT+'/'+eroot+f'/loop/features/train_{n:05d}.pt' for n in range(128)]
        removed=[dict(path=path,kind='prepared_feature',bytes=10) for path in paths]
        if prior:
            removed.extend([dict(path=prior['state'],kind='previous_state',bytes=prior['state_identity']['bytes']),
                            dict(path=prior['weights'],kind='previous_export',bytes=prior['weights_identity']['bytes'])])
            d[eroot+'/PREDECESSOR_TENSOR_FINGERPRINTS.json']=dict(status='passed',identity=prior['state_identity'],
                export_identity=prior['weights_identity'],path=prior['state'],
                export_path=prior['weights'].rsplit('/',1)[0],exact_export_tensors=True)
        d[bname]=boundary
        d[eroot+'/RETIREMENT.json']=dict(removed=removed,removed_bytes=sum(row['bytes'] for row in removed),
                                       retained=dict(current_state=state,current_weights=weights,initial_state=TEST_ROOT+'/'+initial))
        d[eroot+'/FEATURE_RETIREMENT_IDENTITIES.json']=dict(status='passed',packets=[
            dict(path=path,identity=dict(bytes=10,sha256='a'*64)) for path in paths])
        d[eroot+'/PUBLICATION_CHECK.json']=dict(status='passed',all_equal=True,event_id=f'event_{event:03d}',checkpoint_path=weights.rsplit('/',1)[0])
        result.events[event]=eroot+'/RETIREMENT.json'
        result.preparations[event]=[eroot+f'/preparation_retirement/train_{n:05d}.json' for n in range(128)]
        result.prepared_successors[event]=paths
        prior=boundary
    d[V.BASE+'/STATUS.json']['boundary']=prior
    for name in d:compact[name]=dict(bytes=2,sha256='b'*64)
    for name in ('LAUNCH.json','CPU_PREFLIGHT.json','QUALIFICATION.json','QUALIFICATION_RETIREMENT.json',
                 'initial_retry_001/LAUNCH.json','initial_retry_001/prior_failure/PRESERVATION.json',
                 'diagnostics/constructor_002/DETAILS.json','admin/ELAPSED_EXTENSION_AUTHORIZATION.json'):
        compact.setdefault(name,dict(bytes=2,sha256='b'*64))
    for n in range(49):compact[V.BASE+f'/tail/train_{n:05d}.json']=dict(bytes=2,sha256='b'*64)
    return result


class KeeperAndChainTests(unittest.TestCase):
    def test_complete_scalar_chain_and_three_physical_keepers(self):
        result=complete_reconciliation_fixture().finish()
        self.assertEqual(result['status'],'passed',result['errors'])
        self.assertEqual(result['keepers']['physical_inodes'],3)
        self.assertEqual(result['keepers']['unique_allocated_bytes'],4096+6532632576+2097262592)

    def test_wrong_recovery_alias_inode_blocks_closeout(self):
        fixture=complete_reconciliation_fixture();fixture.stats['rows'][1]['inode']=99
        result=fixture.finish();self.assertEqual(result['status'],'failed')
        self.assertTrue(any('alias physical identity' in error for error in result['errors']))

    def test_preparation_successor_must_be_the_event_retired_feature(self):
        fixture=complete_reconciliation_fixture();fixture.prepared_successors[3][0]='unexpected.pt'
        result=fixture.finish();self.assertEqual(result['status'],'failed')
        self.assertTrue(any('Preparation to event retirement chain' in error for error in result['errors']))

    def test_wrong_predecessor_retired_path_blocks_closeout(self):
        fixture=complete_reconciliation_fixture()
        removed=fixture.records[V.BASE+'/events/event_002/RETIREMENT.json']['removed']
        next(row for row in removed if row['kind']=='previous_state')['path']=TEST_ROOT+'/wrong/training_state.pt'
        result=fixture.finish();self.assertEqual(result['status'],'failed')



class ExternalKeeperTests(unittest.TestCase):
    def fixture(self):
        fixture=complete_reconciliation_fixture()
        roles=['KPLUS58_selected_endpoint']*2+['initial_state_witness']+['qualified_runtime_overlay']*7
        deps=[dict(role=role,path=TEST_ROOT+f'/external/{i}',expected_identity=dict(bytes=i+1,sha256=f'{i:064x}')) for i,role in enumerate(roles)]
        fixture.records['admin/PRODUCTION_PLAN.json']['protected_dependencies']={'rows':deps}
        fixture.external_auth=dict(status='passed',errors=[],new_model_work=0,started_epoch=10,finished_epoch=11,
            rows=[dict(path=row['path'],resolved_path=row['path'],expected_identity=row['expected_identity'],
                actual_identity=row['expected_identity'],identity_equal=True,observed_epoch=11,
                current_availability='NOT_CHECKED_LOCAL_ONLY') for row in deps])
        return fixture
    def test_fresh_external_observations_supersede_inherited_availability_note(self):
        result=self.fixture().finish()
        self.assertEqual(result['status'],'passed',result['errors'])
        self.assertEqual(result['supplemental_external_keepers']['checked_files'],10)
    def test_changed_external_identity_blocks_closeout(self):
        fixture=self.fixture();fixture.external_auth['rows'][0]['actual_identity']={'bytes':1,'sha256':'f'*64}
        result=fixture.finish();self.assertEqual(result['status'],'failed')
        self.assertEqual(result['supplemental_external_keepers']['status'],'failed')

if __name__=='__main__':unittest.main()
