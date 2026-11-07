"""Fixed-context selection, useful-prefix transitions and author-forward isolation."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

import torch

from experiments.role_loss_pilot import diagnostics as D
from online_feedback.objective import TargetObjectiveModel
from online_feedback.rng import capture_rng
from test_objective_integration import tiny_author,response_batch
from test_vfd_scoring import assert_rng_equal


def block(predictions,valid=None):
    labels=list(range(len(predictions)))
    return dict(anchor_id=0,labels=labels,predictions=predictions,
        valid=valid or [False]+[True]*(len(labels)-1),weights=[0.]+[1.]*(len(labels)-1))


def snapshot(blocks):
    return dict(dataset='gsm8k',request_id='r',source_id='source',policy_version=D.POLICY_VERSION,
        conditioning_id='prepared',anchor_ids=list(range(len(blocks))),input_ids_sha256='ids',
        loss_mask_sha256='mask',context_length=80,block_size=5,eos_token_ids=[151645],
        blocks=[dict(b,anchor_id=i) for i,b in enumerate(blocks)])


class PrefixTests(unittest.TestCase):
    def test_repair_requires_intact_earlier_prefix_and_persistence(self):
        initial=snapshot([block([0,1,9,3,4]),block([0,1,2,3,9])])
        e2=snapshot([block([0,1,2,3,4]),block([0,9,2,3,4])])
        e4=snapshot([block([0,1,2,9,4]),block([0,1,2,3,4])])
        result=D.compare_snapshots(initial,e2,e4)
        totals=result['totals']
        self.assertEqual(totals['event2_repair_with_prefix_intact'],1)
        self.assertEqual(totals['event2_broken_prefix'],1)
        self.assertEqual(totals['repair_persists'],1)
        self.assertEqual(totals['event2_gained_prefix_tokens'],3)
        self.assertEqual(totals['persisted_gain_tokens'],1)
        self.assertEqual(totals['lost_gain_tokens'],2)
        self.assertEqual(totals['event2_gain_fully_persists'],0)

    def test_fully_matched_block_no_original_error_and_gap_cannot_reopen(self):
        initial=snapshot([block([0,1,2,3,4])])
        self.assertEqual(D.compare_snapshots(initial,initial,initial)['totals']['original_first_error'],0)
        malformed=block([0,1,2,3],valid=[False,True,False,True])
        with self.assertRaisesRegex(ValueError,'reopens'):D.prefix_length(malformed)

    def test_changed_conditioning_or_validity_rejected(self):
        initial=snapshot([block([0,1,9,3,4])]);after=deepcopy(initial)
        after['conditioning_id']='changed'
        with self.assertRaisesRegex(ValueError,'conditioning'):D.compare_snapshots(initial,after,initial)
        after=deepcopy(initial);after['blocks'][0]['valid'][-1]=False
        with self.assertRaisesRegex(ValueError,'validity'):D.compare_snapshots(initial,after,initial)

    def test_exact_hash_selects_eight_native_eligible_anchors(self):
        mask=torch.tensor([[0]*4+[1]*40])
        expected=sorted(sorted(range(4,43),key=lambda p:(hashlib.sha256(
            f'vfd-roleloss-pilot-v1|gsm8k|diagnostic-anchor|source|{p}'.encode()).hexdigest(),p))[:8])
        before=capture_rng()
        self.assertEqual(D.select_anchors(mask,'gsm8k','source'),expected)
        assert_rng_equal(self,before,capture_rng())


class AuthorIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_no_backward_one_backbone_no_teacher_head_and_exact_restore(self):
        model=TargetObjectiveModel(tiny_author(),'osd_first_error',loss_precision='bfloat16')
        raw=response_batch(model);model.train();model.draft_model.layers[0].eval()
        modes=[(m,m.training) for m in model.modules()]
        parameters=[(p,p.detach().clone()) for p in model.parameters()]
        parameter=next(model.draft_model.parameters());parameter.grad=torch.ones_like(parameter)
        gradient=parameter.grad.clone();before=capture_rng();sampler=model._sample_anchor_positions
        anchors=D.select_anchors(raw['loss_mask'],'gsm8k','source')
        with patch.object(model,'_forward_draft_blocks',wraps=model._forward_draft_blocks) as forward:
            result=D.snapshot_response(model,raw,dataset='gsm8k',source_id='source',request_id='r',
                anchor_ids=anchors,conditioning_id='sha')
        self.assertEqual(forward.call_count,1)
        self.assertEqual(result['costs']['draft_forwards'],1)
        self.assertEqual(result['costs']['projection_calls'],4)
        self.assertEqual(model._sample_anchor_positions,sampler)
        self.assertTrue(all(m.training==training for m,training in modes))
        self.assertTrue(all(torch.equal(p,saved) for p,saved in parameters))
        self.assertTrue(torch.equal(parameter.grad,gradient))
        assert_rng_equal(self,before,capture_rng())
        self.assertFalse(torch.cuda.is_initialized())

    def test_failure_restores_mode_and_rng(self):
        model=TargetObjectiveModel(tiny_author(),'osd_first_error',loss_precision='bfloat16')
        raw=response_batch(model);model.train();before=capture_rng()
        with patch.object(model.lm_head,'forward',side_effect=RuntimeError('head failure')):
            with self.assertRaisesRegex(RuntimeError,'head failure'):
                D.snapshot_response(model,raw,dataset='gsm8k',source_id='source',request_id='r',
                    anchor_ids=D.select_anchors(raw['loss_mask'],'gsm8k','source'),conditioning_id='sha')
        self.assertTrue(model.training);assert_rng_equal(self,before,capture_rng())


class PreparationTests(unittest.TestCase):
    def test_shared_pool_retires_only_proven_duplicate_tensors(self):
        from pathlib import Path
        import tempfile
        from online_feedback.sglang_backend import SGLangDFlashBackend
        from online_feedback.storage import save_torch,write_json
        class Backend:
            target_layer_ids=[0,1]
            prepare_feedback=SGLangDFlashBackend.prepare_feedback
            def _call(self,operation,**args):
                path=Path(args['outputfile'])
                if operation=='roleloss_reconstruct':
                    raw=deepcopy(args['record']);prompt=raw['prompt_ids'];response=raw['response_ids']
                    packet=dict(input_ids=(prompt+response)[:-1],target_argmax_ids=response,
                        hidden_states=torch.zeros(len(prompt+response)-1,4,dtype=torch.bfloat16),
                        target_last_hidden_states=torch.zeros(len(prompt+response)-1,2,dtype=torch.bfloat16))
                    capture=dict(feature_path=str(path.with_suffix('.capture.pt')),native_capture_validated=True,
                        collect_features=True,covered_token_ids=(prompt+response)[:-1],
                        label_binding_scope='native_prefill_and_committed_verification',target_argmax_ids=response,
                        prompt_ids=prompt,response_ids=response)
                    save_torch(path.with_suffix('.capture.pt'),packet)
                    write_json(path.with_suffix('.capture.json'),capture)
                    raw.update(purpose='serving',metadata={'capture':capture})
                    receipt=dict(status='passed',new_generations=0,prefill_forwards=1,verification_forwards=2,record=raw)
                    write_json(path,receipt)
                    return receipt
                if operation=='teacher':
                    ids=args['input_ids']
                    receipt=dict(input_ids=ids,positions=[len(ids)-1],final_feature_logits_used_for_labels=False,
                        tensor_path=str(path.with_suffix('.pt')))
                    save_torch(path.with_suffix('.pt'),dict(receipt,hidden_states=torch.zeros(1,4,dtype=torch.bfloat16),
                        target_last_hidden_states=torch.zeros(1,2,dtype=torch.bfloat16)))
                    write_json(path,receipt)
                    return receipt
                raise AssertionError(operation)
        requests=[dict(request_id=f'r{i}',source_id=f's{i}',prompt_ids=[1,2]) for i in range(16)]
        refs={r['request_id']:dict(response_ids=[3,4,5]) for r in requests}
        def source(request,reference):
            return dict(record_id=request['request_id'],prompt_ids=request['prompt_ids'],response_ids=reference['response_ids'])
        with tempfile.TemporaryDirectory() as directory, patch.object(D,'_read_reference',side_effect=source):
            root=Path(directory)/'pool'
            result=D.prepare_pool(Backend(),requests,refs,root,dataset='gsm8k',
                diagnostic_manifest=D.diagnostic_manifest(requests,refs,'gsm8k'))
            self.assertEqual(result['completed_preparations'],16)
            self.assertEqual(result['target_prefill_forwards'],16)
            self.assertEqual(result['target_verification_replay_forwards'],32)
            self.assertEqual(result['terminal_target_forwards'],16)
            self.assertEqual(result['new_generations'],0)
            for feature in result['features']:
                self.assertTrue(Path(feature['path']).is_file())
                self.assertEqual(feature['redundant_retirement']['status'],'completed')
            self.assertFalse(list(root.glob('*.capture.pt')))
            self.assertFalse(list(root.rglob('*.terminal.pt')))
            self.assertEqual(len(list(root.rglob('*.equivalence.json'))),16)


if __name__=='__main__':unittest.main()
