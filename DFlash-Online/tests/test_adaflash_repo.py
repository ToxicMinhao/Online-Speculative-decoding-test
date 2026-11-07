"""CPU source-expression parity and BF16 failure characterization for A_repo."""
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback.adaflash_repo import (
    ReleasedAdaFlashOPD, LiteralSourceAdaFlashOPD, adaflash_repo_contract,
    install_adaflash_repo, literal_source_loss, literal_source_terms,
    repo_position_terms, NUMERICAL_TOLERANCES,
)
from specforge.core.chunking import checkpointed_chunk_reduce


def independent_source(logits, teacher, weights):
    """Standalone literal equations; no port or qualification helper called."""
    z = logits.reshape(-1, logits.shape[-1])
    t = teacher.detach().reshape_as(z)
    w = weights.flatten()
    denominator = w.sum() + 1e-6
    ce = F.cross_entropy(z, t.argmax(-1), reduction='none')
    hard = (ce * w).sum() / denominator
    q = F.softmax(z / 1.0, dim=-1)
    log_p = F.log_softmax(t.float() / 1.0, dim=-1).to(z.dtype)
    rkl = F.kl_div(log_p.detach(), q, reduction='none').sum(-1) * 1.0
    rkl /= denominator
    rkl = torch.clamp(rkl, max=.01)
    soft = (rkl * w).sum()
    a = .8
    return a * hard + (1-a) * soft


def evaluate(extension, logits, teacher, weights, chunk_size=3, checkpoint=True):
    extension.prepare_response(weights)
    terms = checkpointed_chunk_reduce(
        extension.chunk_terms, logits, teacher, teacher.detach().argmax(-1), weights,
        chunk_size=chunk_size, dim=1, checkpoint=checkpoint,
    )
    return extension.reduce_response(terms, terms[1]+1e-6), terms


class ReleasedAdaFlashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def fixture(self, blocks=9):
        gen=torch.Generator().manual_seed(111)
        logits=torch.randn(1,blocks,4,11,generator=gen).to(torch.bfloat16)
        teacher=torch.randn(1,blocks,4,11,generator=gen).to(torch.bfloat16)
        teacher[:,::2]=logits[:,::2]
        weights=torch.tensor([0.,1.,.8,.6]).expand(1,blocks,4).clone()
        weights[:,-1,-1]=0
        return logits,teacher,weights

    def test_literal_operations_values_gradients_and_dtypes(self):
        logits,teacher,weights=self.fixture()
        lhs=logits.clone().requires_grad_();rhs=logits.clone().requires_grad_()
        D=weights.flatten().sum()+1e-6
        actual=repo_position_terms(lhs,teacher,D)
        expected=literal_source_terms(rhs,teacher,D)
        for name in actual:
            torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
        for name in ('ce','probabilities','teacher_log_probabilities','entries','normalized_rkl','clipped_rkl'):
            self.assertEqual(actual[name].dtype,torch.bfloat16,name)
        self.assertEqual(D.dtype,torch.float32)
        left=literal_source_loss(lhs,teacher,weights)
        right=independent_source(rhs,teacher,weights)
        left.backward();right.backward()
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        torch.testing.assert_close(lhs.grad,rhs.grad,rtol=0,atol=0)
        self.assertEqual(left.dtype,torch.float32)

    def test_teacher_log_softmax_uses_fp32_before_native_cast(self):
        logits,teacher,weights=self.fixture()
        rows=repo_position_terms(logits,teacher,weights.sum()+1e-6)
        source=F.log_softmax(teacher.float().reshape(-1,11),-1).to(torch.bfloat16)
        native_only=F.log_softmax(teacher.reshape(-1,11),-1)
        torch.testing.assert_close(rows['teacher_log_probabilities'],source,rtol=0,atol=0)
        self.assertTrue(bool((source!=native_only).any()))

    def test_full_response_denominator_across_chunks_and_backward(self):
        logits,teacher,weights=self.fixture()
        expected_leaf=logits.clone().requires_grad_()
        expected=independent_source(expected_leaf,teacher,weights);expected.backward()
        for size in (1,3,8,64):
            leaf=logits.clone().requires_grad_();extension=ReleasedAdaFlashOPD()
            actual,terms=evaluate(extension,leaf,teacher,weights,size)
            actual.backward()
            torch.testing.assert_close(actual,expected,**NUMERICAL_TOLERANCES['scalar_loss'])
            torch.testing.assert_close(leaf.grad,expected_leaf.grad,rtol=0,atol=0)
            torch.testing.assert_close(extension.denominator,weights.flatten().sum()+1e-6,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(leaf.grad).all())

    def test_port_and_literal_chunk_interfaces_are_exact(self):
        logits,teacher,weights=self.fixture()
        results=[]
        for cls in (ReleasedAdaFlashOPD,LiteralSourceAdaFlashOPD):
            leaf=logits.clone().requires_grad_();extension=cls()
            loss,terms=evaluate(extension,leaf,teacher,weights)
            loss.backward();results.append((loss.detach(),leaf.grad,terms))
        for a,b in zip(results[0][:2],results[1][:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(results[0][2],results[1][2]):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_mixture_occurs_after_component_response_reductions(self):
        logits,teacher,weights=self.fixture()
        extension=ReleasedAdaFlashOPD();loss,terms=evaluate(extension,logits,teacher,weights)
        direct=.8*(terms[0]/extension.denominator)+(1-.8)*terms[4]
        torch.testing.assert_close(loss,direct,rtol=0,atol=0)
        wrong=(.8*terms[0]+(1-.8)*terms[4])/extension.denominator
        self.assertGreater(abs(float(loss-wrong)),.001)

    def test_clipping_cap_and_raw_gradient_stop_are_preserved(self):
        z=torch.tensor([[[[4.,-4.,0.],[.1,.2,.3]]]],dtype=torch.bfloat16,requires_grad=True)
        t=torch.tensor([[[[-4.,4.,0.],[.1,.2,.3]]]],dtype=torch.bfloat16)
        rows=repo_position_terms(z,t,torch.tensor(2.))
        capped=rows['normalized_rkl']>.01
        self.assertEqual(capped.tolist(),[True,False])
        gradient,=torch.autograd.grad(rows['clipped_rkl'].sum(),z)
        self.assertTrue(torch.equal(gradient.reshape(-1,3)[0],torch.zeros(3,dtype=torch.bfloat16)))
        self.assertEqual(float(rows['clipped_rkl'][0].detach()),float(torch.tensor(.01,dtype=torch.bfloat16)))
        large=repo_position_terms(z.detach(),t,torch.tensor(10000.))
        self.assertFalse(bool((large['normalized_rkl']>.01).any()))

    def test_teacher_is_detached_and_masked_positions_have_zero_finite_gradient(self):
        logits,teacher,weights=self.fixture();leaf=logits.requires_grad_();teacher.requires_grad_()
        loss,_=evaluate(ReleasedAdaFlashOPD(),leaf,teacher,weights);loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(leaf.grad[weights==0],torch.zeros_like(leaf.grad[weights==0])))

    def test_zero_supervision_has_literal_zero_loss_for_finite_input(self):
        logits,teacher,weights=self.fixture();weights.zero_();leaf=logits.requires_grad_()
        extension=ReleasedAdaFlashOPD();loss,terms=evaluate(extension,leaf,teacher,weights)
        loss.backward();self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(torch.equal(leaf.grad,torch.zeros_like(leaf)))
        sums,_=extension.response_metrics(terms)
        self.assertEqual(float(sums['adaflash_repo_available_positions']),0.)
        self.assertEqual(float(sums['adaflash_repo_clipped_positions']),0.)

    def test_small_positive_probabilities_remain_finite(self):
        z=torch.tensor([[[[0.,-20.,-80.]]]],dtype=torch.bfloat16,requires_grad=True)
        t=torch.tensor([[[[.1,.2,.3]]]],dtype=torch.bfloat16);w=torch.ones(1,1,1)
        loss=literal_source_loss(z,t,w);loss.backward()
        self.assertTrue(bool((F.softmax(z.detach(),-1)>0).all()))
        self.assertTrue(bool(torch.isfinite(loss) & torch.isfinite(z.grad).all()))

    def test_literal_zero_probability_failure_is_not_stabilized(self):
        z=torch.tensor([[[[0.,-92.,-95.]]]],dtype=torch.bfloat16,requires_grad=True)
        t=torch.tensor([[[[.1,.2,.3]]]],dtype=torch.bfloat16);w=torch.ones(1,1,1)
        extension=ReleasedAdaFlashOPD();loss,terms=evaluate(extension,z,t,w)
        loss.backward();sums,_=extension.response_metrics(terms)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(sums['adaflash_repo_probability_zeros']),1.)
        self.assertTrue(bool(torch.isnan(z.grad).all()))
        reference=z.detach().clone().requires_grad_();independent_source(reference,t,w).backward()
        self.assertTrue(torch.equal(torch.isnan(z.grad),torch.isnan(reference.grad)))

    def test_masking_does_not_hide_literal_zero_probability_backward_failure(self):
        z=torch.tensor([[[[0.,-100.,-200.],[0.,.1,.2]]]],dtype=torch.bfloat16,requires_grad=True)
        t=torch.zeros_like(z);w=torch.tensor([[[0.,1.]]])
        loss,_=evaluate(ReleasedAdaFlashOPD(),z,t,w);loss.backward()
        self.assertTrue(torch.isnan(z.grad[0,0,0]).all())

    def test_consecutive_checkpoint_backwards_rebind_denominator_safely(self):
        extension=ReleasedAdaFlashOPD();logits,teacher,weights=self.fixture()
        for scale in (1.,100.):
            w=weights*scale;leaf=logits.clone().requires_grad_();loss,_=evaluate(extension,leaf,teacher,w)
            loss.backward()
            expected_leaf=logits.clone().requires_grad_();expected=independent_source(expected_leaf,teacher,w);expected.backward()
            torch.testing.assert_close(loss,expected,**NUMERICAL_TOLERANCES['scalar_loss'])
            torch.testing.assert_close(leaf.grad,expected_leaf.grad,rtol=0,atol=0)

    def test_one_bf16_adam_update_matches_literal_source(self):
        logits,teacher,weights=self.fixture();results=[]
        for cls in (ReleasedAdaFlashOPD,LiteralSourceAdaFlashOPD):
            parameter=torch.nn.Parameter(logits.clone())
            optimizer=torch.optim.AdamW([parameter],lr=3e-4,weight_decay=.01)
            loss,_=evaluate(cls(),parameter,teacher,weights);loss.backward()
            gradient=parameter.grad.clone()
            torch.nn.utils.clip_grad_norm_([parameter],1.,error_if_nonfinite=True)
            # Avoid PyTorch's optional CUDA-graph query for this CPU-only step.
            with mock.patch('torch.cuda.is_current_stream_capturing', return_value=False):
                optimizer.step()
            results.append((parameter.detach(),gradient,optimizer.state[parameter]))
        for a,b in zip(results[0][:2],results[1][:2]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for name in results[0][2]:torch.testing.assert_close(results[0][2][name],results[1][2][name],rtol=0,atol=0)

    def test_installer_guards_and_contract(self):
        model=SimpleNamespace(objective='osd',loss_dtype=torch.bfloat16)
        contract=install_adaflash_repo(model)
        self.assertIsInstance(model.point_loss_extension,ReleasedAdaFlashOPD)
        self.assertFalse(contract['adaptive_length_head'])
        self.assertEqual(contract['temperature'],1.)
        self.assertEqual(contract['source_commit'],'478bf0ff7a3a5232b13d0f14e752d5fe1da0b718')
        self.assertEqual(adaflash_repo_contract()['tolerances'],NUMERICAL_TOLERANCES)
        with self.assertRaisesRegex(ValueError,'combine'):install_adaflash_repo(model)
        with self.assertRaisesRegex(ValueError,'all-position'):install_adaflash_repo(SimpleNamespace(objective='osd_first_error',loss_dtype=torch.bfloat16))


if __name__=='__main__':unittest.main()
