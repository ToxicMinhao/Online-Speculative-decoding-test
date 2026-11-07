"""Released AdaFlash loss under the common online response-training protocol.

The numerical path follows rkl_model.py at 478bf0f, including BF16 softmax,
KL target differentiation and in-place normalization before position clipping.
Nonfinite source behavior is deliberately preserved for qualification to detect.
"""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


SOURCE_COMMIT = '478bf0ff7a3a5232b13d0f14e752d5fe1da0b718'
SOURCE_PATH = 'pipeline/training/rkl_model.py'
SOURCE_LINES = (168, 207)
SOURCE_SHA256 = '21e88533318991abb7165e9d1235f64f81c3ec269fa5f3a27c3fc7a43529fe24'
CE_FRACTION = 0.8
TEMPERATURE = 1.0
POSITION_CAP = 0.01
NUMERICAL_TOLERANCES = {
    'scalar_loss': {'rtol': 2e-6, 'atol': 2e-6},
    'bfloat16_logits_gradient': {'rtol': 0.0, 'atol': 0.0},
    'bfloat16_parameter_update': {'rtol': 0.0, 'atol': 0.0},
    'cap_decisions': 'exact',
    'nonfinite_policy': 'stop_before_optimizer; never stabilize or accept NaN parity',
}
DIAGNOSTIC_NAMES = (
    'adaflash_repo_ce_numerator',
    'adaflash_repo_rkl_loss',
    'adaflash_repo_normalized_rkl_before_cap',
    'adaflash_repo_clipped_positions',
    'adaflash_repo_available_positions',
    'adaflash_repo_probability_zeros',
    'adaflash_repo_probability_entries',
    'adaflash_repo_nonfinite_entries',
    'adaflash_repo_teacher_argmax_canonical_mismatches',
)


def adaflash_repo_contract():
    return dict(
        name='released_adaflash_loss_v1', source_commit=SOURCE_COMMIT,
        source_path=SOURCE_PATH, source_lines=list(SOURCE_LINES), source_sha256=SOURCE_SHA256,
        temperature=TEMPERATURE, ce_fraction=CE_FRACTION,
        reverse_kl_fraction=1.0-CE_FRACTION, normalized_position_cap=POSITION_CAP,
        ce='F.cross_entropy(draft_logits,teacher_argmax,reduction=none)',
        draft_probability='F.softmax(native_bfloat16_logits/T)',
        teacher_probability='F.log_softmax(teacher_logits.float()/T).to(draft_dtype)',
        rkl='F.kl_div(detached_log_p_teacher,p_draft,reduction=none)',
        reduction='vocabulary_sum; multiply_T_squared; inplace_divide_full_D; '
                  'position_clamp; weight_sum; late_0.8_CE_plus_0.2_RKL',
        response_denominator='full_flat_response_weights.sum()+1e-6',
        projection_chunking='shared_memory_chunking; global_response_denominator',
        reference_model=None, adaptive_length_head=False,
        nonfinite_policy=NUMERICAL_TOLERANCES['nonfinite_policy'],
        tolerances=NUMERICAL_TOLERANCES,
    )


def source_identity(path):
    """Bind a retained literal source file without importing its runtime."""
    path = Path(path)
    return dict(commit=SOURCE_COMMIT, path=SOURCE_PATH, lines=list(SOURCE_LINES),
                bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def repo_position_terms(draft_logits, teacher_logits, denominator):
    """Return literal numerical intermediates, preserving their source dtypes."""
    logits = draft_logits.reshape(-1, draft_logits.shape[-1])
    target_logits = teacher_logits.detach().reshape_as(logits)
    targets = target_logits.argmax(-1)
    ce = F.cross_entropy(logits, targets, reduction='none')
    probabilities = F.softmax(logits / TEMPERATURE, dim=-1)
    teacher_log_probabilities = F.log_softmax(
        target_logits.float() / TEMPERATURE, dim=-1,
    ).to(logits.dtype)
    entries = F.kl_div(teacher_log_probabilities.detach(), probabilities, reduction='none')
    normalized_rkl = entries.sum(dim=-1) * (TEMPERATURE * TEMPERATURE)
    normalized_rkl /= denominator
    clipped_rkl = torch.clamp(normalized_rkl, max=POSITION_CAP)
    return dict(ce=ce, probabilities=probabilities,
                teacher_log_probabilities=teacher_log_probabilities,
                entries=entries, normalized_rkl=normalized_rkl,
                clipped_rkl=clipped_rkl, teacher_targets=targets)


def literal_source_terms(flat_logits, t_logits, valid_token_count):
    """Independent source-expression oracle for bounded parity qualification.

This function intentionally repeats the pinned operations instead of calling
repo_position_terms. Inputs are projected logits; teacher projection and target
conditioning are supplied by the unchanged shared training forward.
"""
    flat_logits = flat_logits.reshape(-1, flat_logits.size(-1))
    t_logits = t_logits.detach().reshape_as(flat_logits)
    flat_teacher_targets = torch.argmax(t_logits, dim=-1)
    loss_per_token = F.cross_entropy(
        flat_logits, flat_teacher_targets, reduction='none'
    )
    T = 1.0
    p_s = F.softmax(flat_logits / T, dim=-1)
    log_p_t = F.log_softmax(t_logits.float() / T, dim=-1).to(flat_logits.dtype)
    kd_entries = F.kl_div(log_p_t.detach(), p_s, reduction='none')
    kd_raw = kd_entries.sum(dim=-1) * (T * T)
    kd_raw /= valid_token_count
    kd_row = torch.clamp(kd_raw, max=0.01)
    return dict(ce=loss_per_token, probabilities=p_s, teacher_log_probabilities=log_p_t,
                entries=kd_entries, normalized_rkl=kd_raw, clipped_rkl=kd_row,
                teacher_targets=flat_teacher_targets)


def literal_source_loss(draft_logits, teacher_logits, weights):
    """Full-response oracle: the two components are reduced before mixing."""
    flat_weights = weights.reshape(-1)
    valid_token_count = flat_weights.sum() + 1e-6
    rows = literal_source_terms(draft_logits, teacher_logits, valid_token_count)
    hard_loss = (rows['ce'] * flat_weights).sum() / valid_token_count
    rkl_loss = (rows['clipped_rkl'] * flat_weights).sum()
    a = 0.8
    return a * hard_loss + (1.0 - a) * rkl_loss


def _weighted_terms(rows, draft_logits, labels, weights):
    """Additive response components; neither component is mixed per chunk."""
    flat_weights = weights.reshape(-1)
    ce_sum = (rows['ce'] * flat_weights).sum()
    rkl_sum = (rows['clipped_rkl'] * flat_weights).sum()
    with torch.no_grad():
        active = flat_weights > 1e-8
        predictions = draft_logits.reshape(-1, draft_logits.shape[-1]).argmax(-1)
        correct = ((predictions == rows['teacher_targets']) & active).sum().float()
        positions = active.sum().float()
        diagnostics = torch.stack((
            ce_sum.detach(), rkl_sum.detach(),
            (rows['normalized_rkl'] * flat_weights).sum(),
            ((rows['normalized_rkl'] > POSITION_CAP) & active).sum().float(), positions,
            (rows['probabilities'] == 0).sum().float(),
            positions.new_tensor(rows['probabilities'].numel()),
            (~torch.isfinite(rows['entries'])).sum().float(),
            ((rows['teacher_targets'] != labels.reshape(-1)) & active).sum().float(),
        ))
    return ce_sum, weights.sum(), correct, positions, rkl_sum, diagnostics


class ReleasedAdaFlashOPD:
    """Point-loss extension with a full-response, final mixture reduction.

Finish backward before preparing another response: checkpoint recomputation
must retain that response's denominator. No logits or graphs are retained here.
"""
    def __init__(self):
        self.denominator = None

    def prepare_response(self, weights):
        self.denominator = (weights.detach().reshape(-1).sum() + 1e-6).clone()
        return self.denominator

    def chunk_terms(self, draft_logits, teacher_logits, labels, weights):
        rows = repo_position_terms(draft_logits, teacher_logits, self.denominator)
        return _weighted_terms(rows, draft_logits, labels, weights)

    def reduce_response(self, terms, outer_denominator=None):
        hard_loss = terms[0] / self.denominator
        rkl_loss = terms[4]
        return CE_FRACTION * hard_loss + (1.0 - CE_FRACTION) * rkl_loss

    def response_metrics(self, terms, outer_denominator=None):
        diagnostics = dict(zip(DIAGNOSTIC_NAMES, terms[5]))
        diagnostics.update(
            adaflash_response_count=self.denominator.new_ones(()),
            adaflash_repo_response_count=self.denominator.new_ones(()),
            adaflash_repo_full_response_denominator=self.denominator,
            adaflash_repo_loss=self.reduce_response(terms).detach(),
        )
        one = self.denominator.new_ones(())
        ratios = {
            'adaflash_repo_teacher_argmax_ce': (terms[0].detach(), self.denominator),
            'adaflash_repo_reverse_kl': (terms[4].detach(), one),
            'adaflash_repo_position_clipping_rate': (terms[5][3], terms[5][4]),
            'adaflash_repo_probability_zero_rate': (terms[5][5], terms[5][6]),
        }
        return diagnostics, ratios


class LiteralSourceAdaFlashOPD(ReleasedAdaFlashOPD):
    """Same memory interface, but literal source operations supply every row."""
    def chunk_terms(self, draft_logits, teacher_logits, labels, weights):
        rows = literal_source_terms(draft_logits, teacher_logits, self.denominator)
        return _weighted_terms(rows, draft_logits, labels, weights)

    def reduce_response(self, terms, outer_denominator=None):
        a = float(0.8)
        hard_loss = terms[0] / self.denominator
        rkl_loss = terms[4]
        return a * hard_loss + (1.0 - a) * rkl_loss


def install_adaflash_repo(model):
    """Use all ordinary response positions, without ranking or correction."""
    if model.objective != 'osd' or model.loss_dtype != torch.bfloat16:
        raise ValueError('Released AdaFlash loss requires all-position BF16 OSD')
    if any(getattr(model, name, None) is not None for name in (
            'prefix_guard', 'first_error_redistribution', 'point_loss_extension')):
        raise ValueError('Released AdaFlash loss cannot combine objective extensions')
    model.point_loss_extension = ReleasedAdaFlashOPD()
    return adaflash_repo_contract()
