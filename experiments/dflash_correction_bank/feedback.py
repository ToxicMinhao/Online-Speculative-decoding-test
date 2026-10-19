"""Canonical author-path anchors and exact fixed-feature teacher compression.

Backbones/teacher remain frozen. Float64 is used for the mathematical head
objective, moments and factors; serving keeps its pinned native dtype. GPU
qualification must report this numerical boundary rather than treating a KL
bound as a native greedy-speed guarantee.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from experiments.dflash_per_turn import training as osd_training

author = osd_training.author


def sampled_features(draft, embed_tokens, teacher_features, response, *, generator,
                     mask_token_id=128002, attention_backend='flex_attention', synchronize=lambda: None):
    """One fixed author-sampled objective, reused for every owner and both steps.

    Full causal target replay, own-block bidirectional draft mask, target label
    position minus one, and decay exp(-(offset-1)/7) match the qualified path.
    The author mask excludes the anchor slot and all earlier-response loss.
    """
    import torch
    audit = author.audit_response(response)
    for module in (draft, embed_tokens):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError('Bank feature extraction requires frozen eval modules')
    device = next(draft.parameters()).device
    rng_before = generator.get_state().cpu().numpy().copy()
    synchronize(); start = time.perf_counter()
    ids = torch.tensor([response['input_ids'] + response['output_ids']], device=device, dtype=torch.long)
    with torch.no_grad():
        hidden, teacher = teacher_features(ids)
    if hidden.shape[:2] != ids.shape or teacher.shape[:2] != ids.shape:
        raise ValueError('Full teacher history missing')
    synchronize(); teacher_s = time.perf_counter() - start; start = time.perf_counter()
    pad = author.BLOCK_SIZE - 1
    ids = torch.cat((ids, ids.new_full((1, pad), mask_token_id)), dim=1)
    hidden = torch.cat((hidden, hidden.new_zeros((1, pad, hidden.shape[-1]))), dim=1)
    teacher = torch.cat((teacher, teacher.new_zeros((1, pad, teacher.shape[-1]))), dim=1)
    loss_mask = torch.zeros_like(ids, dtype=torch.float32)
    loss_mask[:, audit['response_start']:audit['response_end_exclusive']] = 1
    anchors, keep = author.sample_anchor_positions(ids.shape[1], loss_mask, generator=generator)
    count = anchors.shape[1]
    offsets = torch.arange(author.BLOCK_SIZE, device=device).view(1, 1, -1)
    labels = anchors.unsqueeze(-1) + offsets
    safe = labels.clamp(max=ids.shape[1] - 1)
    noise = torch.full((1, count, author.BLOCK_SIZE), mask_token_id, device=device, dtype=torch.long)
    noise[:, :, 0] = torch.where(keep, ids.gather(1, anchors), mask_token_id)
    positions = torch.cat((torch.arange(ids.shape[1], device=device).unsqueeze(0), labels.reshape(1, -1)), dim=1)
    mask = author.create_attention_mask(anchors, keep, ids.shape[1], backend=attention_backend)
    with torch.no_grad():
        output = draft(position_ids=positions, noise_embedding=embed_tokens(noise.reshape(1, -1)),
                       target_hidden=hidden, attention_mask=mask)
    if isinstance(output, tuple):
        if len(output) != 2 or output[1] is not None:
            raise ValueError('Adaptive head is outside the bank protocol')
        output = output[0]
    if output.shape[:2] != (1, count * author.BLOCK_SIZE):
        raise ValueError('Unexpected frozen draft feature shape')
    binary = keep.unsqueeze(-1).expand(-1, -1, author.BLOCK_SIZE).float() * (labels < ids.shape[1]).float()
    binary *= (offsets > 0).float()
    binary *= loss_mask.unsqueeze(1).expand(-1, count, -1).gather(2, safe)
    weights = binary * torch.exp(-(offsets - 1).clamp(min=0).float() / author.RECIPE['loss_decay_gamma'])
    selected = (weights.reshape(-1) > 0).nonzero(as_tuple=True)[0]
    h = output.reshape(-1, output.shape[-1])[selected].detach().cpu()
    teacher_indices = (safe.reshape(-1)[selected] - 1).clamp(min=0)
    teacher_h = teacher[0, teacher_indices].detach().cpu()
    raw_weights = weights.reshape(-1)[selected].double().cpu().numpy()
    # A zero-supervision sample is recorded as a no-op, never resampled.
    omega = raw_weights / raw_weights.sum() if raw_weights.size else raw_weights
    supervised = labels.reshape(-1)[selected].cpu().tolist()
    if any(not audit['response_start'] <= pos < audit['response_end_exclusive'] for pos in supervised):
        raise ValueError('Prefix position received new response loss')
    metadata = {**audit, 'anchor_positions': anchors[0].cpu().tolist(), 'block_keep_mask': keep[0].cpu().tolist(),
        'label_positions': labels[0].cpu().tolist(), 'teacher_positions': (safe[0]-1).clamp(min=0).cpu().tolist(),
        'binary_loss_mask': binary[0].cpu().tolist(), 'position_weights': weights[0].cpu().tolist(),
        'eligible_positions': len(supervised), 'unique_supervised_position_ids': sorted(set(supervised)),
        'unique_supervised_positions': len(set(supervised)), 'zero_supervision': not supervised,
        'eligible_blocks': int((binary.sum(-1)>0).sum()), 'backbone_forward_calls': 1,
        'normalization': 'positive author position weights divided by exact sum; PDF equation3',
        'weighted_position_sum': float(raw_weights.sum()), 'fixed_objective_across_candidates_and_steps': True}
    metadata['sampling_and_weights_sha256'] = author.digest(metadata)
    synchronize()
    return dict(h=h, teacher_h=teacher_h, omega=omega, metadata=metadata,
                rng_before=rng_before, rng_after=generator.get_state().cpu().numpy().copy(),
                costs={'teacher_preparation_s': teacher_s, 'frozen_feature_s': time.perf_counter()-start})


class FrozenResponseObjective:
    """Exact affine-head loss differences and gradients using a teacher moment.

    Feature rows remain on CPU between bounded vocabulary chunks. No vocabulary
    distributions or full Fisher matrices are retained. P and E are frozen.
    Each method call records its entire synchronized cost, including all owners.
    """
    def __init__(self, h, teacher_h, omega, E, U, V_in, *, projection_chunk=32, synchronize=lambda: None):
        import torch
        self.torch = torch; self.sync = synchronize; self.chunk = int(projection_chunk)
        self.device = E.device
        self.E_native = E.detach()
        self.E = E.detach().to(dtype=torch.float64)
        self.U = torch.as_tensor(np.array(U, copy=True), dtype=torch.float64, device=self.device)
        self.V_in = torch.as_tensor(np.array(V_in, copy=True), dtype=torch.float64, device=self.device)
        self.P = self.E @ self.U
        self.h = h.detach().cpu().double().contiguous()
        self.teacher_h = teacher_h.detach().cpu().double().contiguous()
        self.omega = np.asarray(omega, dtype=np.float64).copy()
        if len(self.h) != len(self.omega) or len(self.teacher_h) != len(self.h):
            raise ValueError('Anchor/teacher/weight length mismatch')
        if len(self.omega) and (not np.all(self.omega > 0) or not np.isclose(self.omega.sum(),1.,atol=1e-12)):
            raise ValueError('Response weights must be normalized')
        self.rank = self.U.shape[1]
        projected = self.h @ self.V_in.cpu()
        self.phi = (projected / projected.norm(dim=-1, keepdim=True).clamp_min(1.)).numpy()
        self.S = (self.phi.T * self.omega) @ self.phi
        self.M = np.zeros((self.rank, self.V_in.shape[1]), dtype=np.float64)
        self.G = np.zeros_like(self.M)
        self.T = np.empty((len(self.h), self.rank, self.rank), dtype=np.float64)
        self.calls = []; self.sync(); started = time.perf_counter()
        self.teacher_precision = {'logits_dtype': str(E.dtype), 'probabilities_dtype': 'torch.float64',
            'native_probability_max_difference': 0., 'native_probability_mass_max_error': 0.}
        for sl, h0, th, phi, w in self._chunks():
            z0 = h0 @ self.E.T
            native_logits = self._teacher_logits(th)
            teacher = torch.softmax(native_logits.double(), dim=-1)
            native_probability = torch.softmax(native_logits, dim=-1).double()
            self.teacher_precision['native_probability_max_difference'] = max(
                self.teacher_precision['native_probability_max_difference'],
                float((native_probability-teacher).abs().max()))
            self.teacher_precision['native_probability_mass_max_error'] = max(
                self.teacher_precision['native_probability_mass_max_error'],
                float((native_probability.sum(-1)-1).abs().max()))
            q0 = torch.softmax(z0, dim=-1)
            qp, pp = q0 @ self.P, teacher @ self.P
            self.M += ((pp*w[:,None]).T @ phi).cpu().numpy()
            self.G += (((qp-pp)*w[:,None]).T @ phi).cpu().numpy()
            # Exact finite-anchor factors; a row loop bounds peak memory and
            # avoids [anchors,vocabulary,rank] materialization.
            for local, i in enumerate(range(sl.start, sl.stop)):
                covariance = self.P.T @ (self.P * q0[local,:,None]) - torch.outer(qp[local], qp[local])
                self.T[i] = ((covariance+covariance.T)*.5).cpu().numpy()
        self.sync(); self.compression_s = time.perf_counter()-started

    def _teacher_logits(self, features):
        # Preserve the qualified target's native head computation. Normalize its
        # logits in float64 for a probability vector of unit mass, as assumed by
        # the mathematical compression and curvature bounds. The untouched OSD
        # control continues to use its author-path native probability arithmetic.
        return self.torch.nn.functional.linear(features.to(self.E_native.dtype), self.E_native)

    def _chunks(self):
        torch = self.torch
        for start in range(0, len(self.h), self.chunk):
            sl = slice(start, min(start+self.chunk,len(self.h)))
            yield sl, self.h[sl].to(self.device), self.teacher_h[sl].to(self.device), \
                torch.as_tensor(self.phi[sl], device=self.device), torch.as_tensor(self.omega[sl], device=self.device)

    def loss_gradient(self, C):
        torch = self.torch; self.sync(); started = time.perf_counter()
        matrix = torch.as_tensor(np.array(C,copy=True), dtype=torch.float64, device=self.device)
        total = 0.; gradient = -self.M.copy()
        for _, h, _, phi, w in self._chunks():
            z0 = h @ self.E.T
            z = z0 + (phi @ matrix.T) @ self.P.T
            total += float(((torch.logsumexp(z,-1)-torch.logsumexp(z0,-1))*w).sum())
            gradient += ((torch.softmax(z,-1) @ self.P * w[:,None]).T @ phi).cpu().numpy()
        total -= float(np.sum(C*self.M))
        self.sync(); self.calls.append({'kind':'compressed_loss_gradient','seconds':time.perf_counter()-started})
        return total, gradient

    def direct_loss_gradient(self, C):
        """Independent teacher-distribution recomputation for qualification only."""
        torch=self.torch; self.sync(); started=time.perf_counter()
        matrix=torch.tensor(np.array(C,copy=True),device=self.device,dtype=torch.float64,requires_grad=True)
        total=torch.sum(matrix)*0.
        for _,h,teacher_h,phi,w in self._chunks():
            z0=h@self.E.T
            z=z0+(phi@matrix.T)@self.P.T
            p=torch.softmax(self._teacher_logits(teacher_h).double(),-1)
            total=total+((p*(torch.log_softmax(z0,-1)-torch.log_softmax(z,-1))).sum(-1)*w).sum()
        gradient=torch.autograd.grad(total,matrix)[0].detach().cpu().numpy()
        self.sync();self.calls.append({'kind':'direct_qualification_loss_gradient','seconds':time.perf_counter()-started})
        return float(total.detach()),gradient

    def portable(self):
        return {'h':self.h.numpy(), 'teacher_h':self.teacher_h.numpy(), 'omega':self.omega,
                'phi':self.phi, 'M':self.M, 'G':self.G, 'S':self.S, 'T':self.T,
                'teacher_precision':self.teacher_precision,
                'numeric_contract':'native frozen target logits normalized in float64; float64 affine draft KL at canonical frozen sampled features'}
