"""CPU fixtures for relocated inputs, original-state admission and peak storage."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from experiments.mathqa_fixed_lr_transfer import prepare as P, preflight as F
from experiments.mathqa_fixed_lr_transfer import data as D, state as S
from experiments.mathqa_fixed_lr_transfer.runtime import LIMITS
from test_mathqa_fixed_lr_transfer_data import row, adaptation
from test_mathqa_fixed_lr_transfer_training import origin, arm_config


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def package(directory, remote_root):
    directory.mkdir()
    sealed = directory / 'sealed'
    sealed.mkdir()
    requests = D.seal_requests([row(i) for i in range(260)])
    train = adaptation()
    schedule = dict(activations=P.evaluation_schedule(requests['scored']),
                    scored_executions=1536, warmup_executions=48)
    values = dict(data_panel=dict(requests=requests['scored']), warmups=dict(requests=requests['warmup']),
        adaptation=train, evaluation_schedule=schedule,
        source_provenance=dict(status='passed',ancestry_count=768,selected_ancestry_content_overlap=0))
    for event, offset in ((5, 0), (6, 128)):
        values[f'event{event}_inputs'] = dict(event=event, start_ordinal=512+offset,
            stop_ordinal=640+offset, requests=train['train_requests'][offset:offset+128])
    for key, value in values.items():
        write(sealed/P.INPUTS[key], value)
    values['data_seal'] = dict(status='passed', first_model_work_before_seal=False,
        input_only_eligibility='Nonempty original prompt within the fixed context cap',
        files={P.INPUTS[key]:P.identity(sealed/P.INPUTS[key]) for key in values})
    write(sealed/'DATA_SEAL.json', values['data_seal'])
    plan = dict(root=str(remote_root), input_identities={},
        references=str(remote_root)+'/references/CANONICALS.json',
        warmup_references=str(remote_root)+'/references/WARMUP_CANONICALS.json')
    for key, name in P.INPUTS.items():
        plan[key] = str(remote_root)+'/sealed/'+name
        plan['input_identities'][plan[key]] = P.identity(sealed/name)
    write(sealed/'PLAN.json', plan)
    return plan


class StorageTests(unittest.TestCase):
    def test_complete_peak_and_verified_duplicate_policy(self):
        args = dict(weight_bytes=2097259104, recovery_bytes=6297245019,
                    hidden_size=4096, captured_layers=5)
        peak = P.storage_plan([96]*128, [96]*128, **args)
        self.assertEqual(peak['status'], 'passed')
        self.assertEqual(peak['reserve_bytes'], 400_000_000_000)
        self.assertEqual(peak['allocation_cap_bytes'], 140_000_000_000)
        self.assertEqual(peak['checkpoint_bytes'], 4*peak['atomic_publication_bytes'])
        self.assertEqual(peak['qualification_payload_bytes'], 2*peak['atomic_publication_bytes'])
        self.assertEqual(peak['feature_pools'], 3)
        self.assertEqual(peak['common_event5_copies'], 1)
        self.assertEqual(peak['required_initial_free_bytes'], peak['peak_new_bytes']+400_000_000_000)
        self.assertEqual(P.storage_plan([96]*128, [96]*128,
            retire_verified_duplicates=False, **args)['status'], 'blocked')
        with self.assertRaises(ValueError): P.storage_plan([96]*127, [96]*128, **args)

    def test_binding_preserves_science_but_allows_measured_resource_updates(self):
        plan = dict(root='/run', configs={'N':'native.yaml', 'S':'scalar.yaml'},
                    compiler_baseline_allocated_bytes=0, estimated_seconds={'N5':600})
        digest = P.plan_binding(plan)
        plan.update(binding=digest, compiler_baseline_allocated_bytes=9000,
                    estimated_seconds={'N5':720})
        self.assertEqual(P.plan_binding(plan), digest)
        plan['configs']['S'] = 'different.yaml'
        self.assertNotEqual(P.plan_binding(plan), digest)


class PackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_archive_relocation_uses_new_consumer_without_existing_references(self):
        source = self.root/'package'
        plan = package(source, Path('/remote/execution_001'))
        self.assertEqual(P.reference_readiness(plan, package_root=source)['new_canonical_generations'], 258)
        archive = self.root/'package.tar.gz'
        P.build_archive(plan, source, archive)
        result = P.unpack_archive(archive, self.root/'unpacked')
        self.assertEqual(result['readiness']['status'], 'passed')
        self.assertEqual(result['readiness']['model_work'], 0)
        self.assertFalse((self.root/'unpacked/references/CANONICALS.json').exists())

    def test_input_mutation_and_preexisting_reference_stop_before_worker(self):
        source = self.root/'package';plan = package(source, Path('/remote/execution_001'))
        write(source/'references/CANONICALS.json', {})
        with self.assertRaisesRegex(ValueError, 'already been attempted'):
            P.reference_readiness(plan, package_root=source)
        (source/'references/CANONICALS.json').unlink()
        with (source/'sealed/PANEL.json').open('a') as stream: stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'changed packaged input'):
            P.reference_readiness(plan, package_root=source)

    def test_event_slice_and_balanced_order_cannot_be_resealed_accidentally(self):
        source = self.root/'package';plan = package(source, Path('/remote/execution_001'))
        path = source/'sealed/EVENT6_INPUTS.json'
        value = json.loads(path.read_text());value['requests'].reverse();write(path,value)
        plan['input_identities'][plan['event6_inputs']] = P.identity(path)
        sealpath = source/'sealed/DATA_SEAL.json';seal = json.loads(sealpath.read_text())
        seal['files']['EVENT6_INPUTS.json'] = P.identity(path);write(sealpath,seal)
        plan['input_identities'][plan['data_seal']] = P.identity(sealpath)
        with self.assertRaisesRegex(ValueError, 'Event slice'):
            P.load_inputs(plan, package_root=source)

    def test_allocation_paths_cannot_adopt_historical_or_linked_roots(self):
        run = self.root/P.TASK_NAME/'execution_001';run.mkdir(parents=True)
        source = self.root/'DFlash-Online-mathqa-fixed-lr-transfer-20261116-001';source.mkdir()
        plan = dict(root=str(run),source_root=str(source),allocation_roots=[str(run),str(source)])
        self.assertEqual(F.validate_allocation_roots(plan)['status'],'passed')
        plan['allocation_roots'].append('/old/checkpoints')
        with self.assertRaises(ValueError): F.validate_allocation_roots(plan)


class OriginTests(unittest.TestCase):
    def test_complete_cpu_origin_and_two_lr_forks_preserve_saved_payload(self):
        source, args = origin()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);state_path = root/'origin.pt'
            torch.save(source,state_path);before = P.identity(state_path)
            (root/'original.yaml').write_text(args['original_config_text'])
            configs = {}
            for arm in ('N','S'):
                path = root/(arm+'.yaml');path.write_text(arm_config(source,arm));configs[arm]=str(path)
            records = [dict(request_id=r['record_id']) for event in source['events'] for r in event['features']]
            write(root/'ancestry.json',dict(train_requests=records))
            identity = dict(bytes=10,sha256='b'*64)
            binding = source['algorithm_contract']['role_loss_pilot_v1']['binding']
            plan = dict(origin_state=str(state_path),origin_export=str(root/'export'),
                origin_config=str(root/'original.yaml'),origin_status=str(root/'status.json'),
                expected_ancestry=str(root/'ancestry.json'),configs=configs,binding='a'*64,
                origin_binding=binding,origin_state_identity=before,origin_weights_identity=identity)
            write(root/'status.json',dict(status='completed',dataset='mathqa',arm='KPLUS',binding=binding,
                boundary=dict(event=4,state_identity=before,weights_identity=identity)))
            with patch('experiments.ok_coding.training._export_equal') as export:
                result = F.inspect_states(plan,expected_cuda_rng_states=0)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['ancestry_records'],512)
            self.assertEqual(result['forks']['S']['actual_learning_rate'],2.88e-4)
            self.assertEqual(P.identity(state_path),before)
            export.assert_called_once()
            records[0]['request_id']='wrong'
            write(root/'ancestry.json',dict(train_requests=records))
            with patch('experiments.ok_coding.training._export_equal'):
                with self.assertRaisesRegex(ValueError,'ancestry'):
                    F.inspect_states(plan,expected_cuda_rng_states=0)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__': unittest.main()
