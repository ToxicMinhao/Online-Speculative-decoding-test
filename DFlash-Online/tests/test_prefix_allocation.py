"""Independent coefficient and gradient checks; no model execution on a GPU."""
import copy
import math
import random
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback.first_error import first_error_mask, valid_prefix_mask
from online_feedback.kplus import install_kplus, MASS_ATOL, MASS_RTOL
from online_feedback.objective import TargetObjectiveModel
from online_feedback.prefix_allocation import (
    PrefixAllocation, allocation_summaries, bounded_prefix_coefficients, coefficients_from_scores,
    install_prefix_allocation, prefix_allocation_contract,
)
from specforge.core.chunking import checkpointed_chunk_reduce
from test_kplus_objective import mixed_logits
from test_objective_integration import tiny_author

TOL = dict(rtol=MASS_RTOL, atol=MASS_ATOL)


def scalar_coefficients(probabilities, base, valid, prefix, cap=.25):
    """Python scalar definition, independent from tensor reductions/masking."""
    product=1.; scores=[0.] * len(base)
    for j in range(len(base)):
        if valid[j]:
            product *= .5 + .5 * probabilities[j]
            for i in range(j+1):
                scores[i] += product
    selected=[j for j in range(len(base)) if prefix[j]]
    answer=list(base)
    if len(selected)<=1 or cap==0:
        return answer
    mass=sum(base[j] for j in selected)
    norm=sum(scores[j] for j in selected)
    relative={j:mass*scores[j]/norm/base[j]-1 for j in selected}
    maximum=max(abs(v) for v in relative.values())
    factor=0. if maximum==0 else min(1.,cap/maximum)
    for j in selected:answer[j]=base[j]*(1+factor*relative[j])
    return answer


def coefficient_fixture(q, length):
    q=torch.tensor([0.,*q],dtype=torch.float32).reshape(1,1,-1)
    width=q.shape[-1]
    base=torch.exp(-(torch.arange(width).float()-1).clamp_min(0)/7).reshape(1,1,-1)
    base[...,0]=0
    valid=torch.ones_like(q,dtype=torch.bool);valid[...,0]=False
    prefix=valid.clone();prefix[...,length+1:]=False
    return q,base,valid,prefix


class PrefixAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def tearDown(self):self.assertFalse(torch.cuda.is_initialized())

    def model(self, cap=None):
        model=TargetObjectiveModel(tiny_author(chunk_blocks=8),'osd_first_error',loss_precision='bfloat16')
        model.block_size=16;model.lm_head=torch.nn.Identity();install_kplus(model)
        if cap is not None:install_prefix_allocation(model,relative_cap=cap)
        model.begin_event()
        return model

    def test_worked_example_and_full_block_tail(self):
        q,b,v,p=coefficient_fixture([.9,.8,.6,.7,.2],4)
        a=bounded_prefix_coefficients(q,b,v,p)
        torch.testing.assert_close(a[0,0,1:5],torch.tensor([1.182480069,.917595685,.681139203,.488579293]),**TOL)
        torch.testing.assert_close(a[...,5],b[...,5],rtol=0,atol=0)
        changed=q.clone();changed[...,5]=.99
        after=bounded_prefix_coefficients(changed,b,v,p)
        self.assertFalse(torch.equal(a,after))
        # The score-only post-error slot never gains a direct coefficient.
        self.assertTrue(torch.equal(after[~p],b[~p]))

    def test_random_mass_bounds_and_scalar_oracle(self):
        rng=random.Random(20261116)
        for _ in range(400):
            length=rng.randrange(1,25);prefix=rng.randrange(length+1)
            q,b,v,p=coefficient_fixture([rng.random() for _ in range(length)],prefix)
            a=bounded_prefix_coefficients(q,b,v,p)
            expected=scalar_coefficients(q.flatten().tolist(),b.flatten().tolist(),v.flatten().tolist(),p.flatten().tolist())
            torch.testing.assert_close(a.flatten(),torch.tensor(expected),**TOL)
            torch.testing.assert_close((a*p).sum(),(b*p).sum(),**TOL)
            self.assertTrue(bool((a[p]>=.75*b[p]-MASS_ATOL).all()))
            self.assertTrue(bool((a[p]<=1.25*b[p]+MASS_ATOL).all()))
            self.assertTrue(torch.equal(a[~p],b[~p]))
            self.assertFalse(a.requires_grad)

    def test_empty_one_cap_zero_and_equal_scores_preserve_exact_coefficients(self):
        for length in [0,1,5]:
            q,b,v,p=coefficient_fixture([.4]*5,length)
            if length<=1:self.assertTrue(torch.equal(bounded_prefix_coefficients(q,b,v,p),b))
            self.assertTrue(torch.equal(bounded_prefix_coefficients(q,b,v,p,relative_cap=0),b))
        base=torch.tensor([[[0.,1.,1.,1.]]]);prefix=base>0
        self.assertTrue(torch.equal(coefficients_from_scores(torch.ones_like(base),base,prefix),base))
        # Scores proportional to existing weights also give M=0 exactly.
        self.assertTrue(torch.equal(coefficients_from_scores(base,base,prefix),base))

    def test_gap_eos_seed_and_first_error_support(self):
        labels=torch.tensor([[[0,1,2,3,4,5]]]*4)
        proposals=labels.clone();weights=torch.ones_like(labels,dtype=torch.float32);weights[...,0]=0
        weights[0,0,3]=0;labels[1,0,2]=9;proposals[1,0,2]=9
        labels[2,0,0]=9;proposals[3,0,1]=8
        valid=valid_prefix_mask(labels,weights,(9,));selection=first_error_mask(proposals,labels,weights,(9,))
        prefix=selection&valid&(proposals==labels)
        q=torch.full_like(weights,.6)
        first=bounded_prefix_coefficients(q,weights,valid,prefix)
        changed=q.clone();changed[~valid]=.01
        second=bounded_prefix_coefficients(changed,weights,valid,prefix)
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(prefix.sum(-1).flatten().tolist(),[2,2,0,0])
        self.assertTrue(torch.equal(first[~prefix],weights[~prefix]))

    def test_summary_handles_full_prefix_axis_and_empty_block_batch(self):
        base = torch.ones(2, 4)
        prefix = torch.tensor([[True] * 4, [False] * 4])
        valid = torch.ones_like(prefix)
        counts, by_slot, by_length = allocation_summaries(base, base, valid, prefix)
        self.assertEqual(tuple(by_slot.shape), (5, 4, 5))
        self.assertEqual(by_length.tolist(), [1., 0., 0., 0., 1.])
        self.assertEqual(float(by_slot[4, :, 0].sum()), 4.)
        self.assertEqual(float(counts[0]), 2.)

        empty = torch.ones(0, 4)
        counts, by_slot, by_length = allocation_summaries(
            empty, empty, empty.bool(), empty.bool())
        self.assertEqual(tuple(by_slot.shape), (5, 4, 5))
        self.assertEqual(by_length.tolist(), [0.] * 5)
        self.assertEqual(float(counts.sum()), 0.)

    def test_summary_repeated_prefix_lengths_match_independent_slot_oracle(self):
        base = torch.arange(1, 33, dtype=torch.float32).reshape(2, 4, 4) / 8
        lengths = torch.tensor([[3, 2, 3, 4], [2, 3, 0, 4]])
        prefix = torch.arange(4) < lengths.unsqueeze(-1)
        difference = torch.tensor([.125, -.125, .0625, -.0625]).expand_as(base) * prefix
        candidate = base + difference
        _, actual, histogram = allocation_summaries(base, candidate, torch.ones_like(prefix), prefix)
        expected = torch.zeros(5, 4, 5)
        for block_base, block_candidate, block_prefix in zip(
                base.reshape(-1, 4), candidate.reshape(-1, 4), prefix.reshape(-1, 4)):
            length = int(block_prefix.sum())
            for slot in range(length):
                old, new = float(block_base[slot]), float(block_candidate[slot])
                expected[length, slot] += torch.tensor([1., old, new, new-old, abs(new-old)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(histogram.tolist(), [1., 0., 2., 3., 2.])
        repeated = allocation_summaries(base, candidate, torch.ones_like(prefix), prefix)[1]
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)

    def test_zero_value_delta_keeps_its_nonzero_derivative_and_signed_loss(self):
        q,b,v,p=coefficient_fixture([.9,.8,.6,.7,.2],4)
        labels=torch.zeros_like(b,dtype=torch.long);proposals=labels.clone();proposals[...,-1]=1
        # Exact scalar log-probabilities suffice for this row-gradient fixture.
        logp=q.clamp_min(.01).log().unsqueeze(-1).requires_grad_()
        extension=PrefixAllocation();row=torch.zeros_like(b,requires_grad=True)
        terms=extension.chunk_terms(logp,row,proposals,labels,b,v)
        native=torch.tensor(3.,requires_grad=True)
        answer,_=extension.reduce_response(native,terms)
        self.assertEqual(float(terms[0].detach()),0.)
        self.assertIsNot(answer,native)
        answer.backward()
        self.assertGreater(float(row.grad.abs().sum()),0.)
        self.assertIsNone(logp.grad)
        self.assertEqual(float(row.grad[...,-1]),0.)
        a=bounded_prefix_coefficients(q,b,v,p)
        for sign in [-1,1]:
            values=sign*(a-b)
            local=extension.chunk_terms(logp,values,proposals,labels,b,v)[0]
            self.assertEqual(int(torch.sign(local)),sign)

    def test_cap_zero_returns_native_numerator_object(self):
        logits,teacher,labels,weights=mixed_logits()
        model=self.model(0.)
        terms=model._chunk_loss(logits,None,labels,weights,teacher)
        self.assertEqual(len(terms),15)
        native,_=model.first_error_redistribution.reduce_response(terms[:11])
        candidate,_=model.prefix_allocation.reduce_response(native,terms[11:])
        self.assertIs(candidate,native)
        self.assertEqual(model.prefix_allocation.last_summary['changed_positions'],0)

    def test_native_eleven_terms_and_direct_error_gradients_remain_exact(self):
        logits,teacher,labels,weights=mixed_logits()
        answers=[]
        for cap in [None,0.,.25]:
            model=self.model(cap);leaf=logits.clone().requires_grad_()
            terms=checkpointed_chunk_reduce(model._chunk_loss,leaf,None,labels,weights,teacher,chunk_size=8,dim=1,checkpoint=True)
            numerator,metrics=model.first_error_redistribution.reduce_response(terms[:11])
            if cap is not None:numerator,_=model.prefix_allocation.reduce_response(numerator,terms[11:])
            loss=numerator/(terms[1]+1e-6);loss.backward()
            answers.append((terms,leaf.grad,loss))
        for t in answers[1:]:
            for a,b in zip(answers[0][0][:11],t[0][:11]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(answers[0][1],answers[1][1],rtol=0,atol=0)
        torch.testing.assert_close(answers[0][2],answers[1][2],rtol=0,atol=0)
        selected=first_error_mask(logits.argmax(-1),labels,weights)
        errors=selected&(logits.argmax(-1)!=labels)
        torch.testing.assert_close(answers[0][1][errors],answers[2][1][errors],rtol=0,atol=0)
        self.assertGreater(float((answers[0][1]-answers[2][1]).float().abs().sum()),0)
        self.assertTrue(bool((answers[2][1][~selected]==0).all()))

    def test_full_forward_preserves_normalization_and_counts_once(self):
        logits,teacher,labels,weights=mixed_logits();logits.requires_grad_()
        blocks=labels.shape[1];anchors=torch.arange(blocks).reshape(1,-1)*16
        keep=torch.ones_like(anchors,dtype=torch.bool);input_ids=labels.reshape(1,-1)
        model=self.model(.25)
        with mock.patch.object(model,'_forward_draft_blocks',return_value=(anchors,keep,logits.reshape(1,-1,31))) as forward, \
             mock.patch.object(model,'_block_labels_and_weights',return_value=(labels,weights,labels)), \
             mock.patch.object(model,'_aligned_target_hidden',return_value=teacher), \
             mock.patch.object(model.first_error_redistribution,'reduce_response',wraps=model.first_error_redistribution.reduce_response) as reduce, \
             mock.patch.object(model.lm_head,'forward',wraps=model.lm_head.forward) as head:
            rng=torch.get_rng_state().clone()
            loss,_,metrics=model(input_ids,torch.zeros(1),torch.ones_like(input_ids),teacher)
            summary=copy.deepcopy(model.prefix_allocation.last_summary)
            self.assertEqual(head.call_count,6);self.assertEqual(forward.call_count,1)
            self.assertEqual(len(reduce.call_args.args[0]),11)
            selection=first_error_mask(logits.detach().argmax(-1),labels,weights)
            torch.testing.assert_close(metrics['loss_terms'][1],(weights*selection).sum()+1e-6,rtol=0,atol=0)
            expected=metrics['sum_metrics']['prefix_allocation_native_numerator']+metrics['sum_metrics']['prefix_allocation_delta_numerator']
            torch.testing.assert_close(loss,expected/metrics['loss_terms'][1],rtol=0,atol=0)
            loss.backward()
            self.assertEqual(summary,model.prefix_allocation.last_summary)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(summary['blocks'],17)
            self.assertEqual(sum(summary['prefix_length_counts']),17)
            self.assertEqual(sum(row['occurrences'] for row in summary['by_prefix_length_and_slot']),summary['prefix_positions'])

    def test_probability_gather_exponentiates_in_fp32_after_bf16_rounding(self):
        logits,teacher,labels,weights=mixed_logits()
        model=self.model(.25)
        with mock.patch('online_feedback.prefix_allocation.bounded_prefix_coefficients',
                        wraps=bounded_prefix_coefficients) as coefficients:
            model._chunk_loss(logits,None,labels,weights,teacher)
        actual=coefficients.call_args.args[0]
        gathered=F.log_softmax(logits,-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
        expected=gathered.float().exp()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(actual.dtype,torch.float32)
        self.assertFalse(torch.equal(actual,gathered.exp().float()))

    def test_two_response_mean_preserves_each_original_denominator(self):
        outputs=[]
        model=self.model(.25)
        logits,teacher,labels,weights=mixed_logits()
        for local_weights in [weights,weights*.31]:
            leaf=logits.clone().requires_grad_()
            terms=checkpointed_chunk_reduce(model._chunk_loss,leaf,None,labels,local_weights,teacher,
                                           chunk_size=8,dim=1,checkpoint=True)
            native,_=model.first_error_redistribution.reduce_response(terms[:11])
            candidate,_=model.prefix_allocation.reduce_response(native,terms[11:])
            denominator=terms[1]+1e-6
            selected=first_error_mask(leaf.detach().argmax(-1),labels,local_weights)
            torch.testing.assert_close(denominator,(local_weights*selected).sum()+1e-6,rtol=0,atol=0)
            self.assertEqual(model.prefix_allocation.last_summary['blocks'],17)
            outputs.append((candidate,denominator,leaf))
        mean=sum(numerator/denominator for numerator,denominator,_ in outputs)/2
        mean.backward()
        separately=[]
        for numerator,denominator,_ in outputs:separately.append(float(numerator.detach()/denominator))
        self.assertAlmostEqual(float(mean.detach()),sum(separately)/2,places=6)
        self.assertTrue(all(leaf.grad is not None for _,_,leaf in outputs))

    def test_nonfinite_score_diagnostic_and_conflicting_hooks_fail(self):
        model=self.model(.25)
        with self.assertRaises(ValueError):install_prefix_allocation(model)
        with self.assertRaises(ValueError):prefix_allocation_contract(.3)
        logits,teacher,labels,weights=mixed_logits()
        logits[0,0,2,0]=float('nan')
        terms=model._chunk_loss(logits,None,labels,weights,teacher)
        native,_=model.first_error_redistribution.reduce_response(terms[:11])
        with self.assertRaisesRegex(RuntimeError,'invariant'):
            model.prefix_allocation.reduce_response(native,terms[11:])


if __name__=='__main__':unittest.main()
