"""Create the readable report from measured probabilities."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
data = json.loads((root / 'results.json').read_text())
vector = lambda v: '(' + ', '.join(f'{x:.6f}' for x in v) + ')'
lines = ['# Token-level projected-target walkthrough', '',
'This deterministic CPU diagnostic uses the deployed target constructor and actual `TargetObjectiveModel._chunk_loss`. An identity head exposes four freely adjustable FP32 logits. It is not DFlash model training; no checkpoint, backbone, production optimizer or source was modified. No A800 operation occurred.', '',
'ARS experiment supervision and an independent mathematical review were used. Constructor/loss source was byte-compared with the exact bundle previously installed on A800.', '',
'## Fixed targets and changing predictions', '',
'For the same sampled position and context, the pre-event reference q0 determines r once. Learning changes the current logits and probabilities q, not r. Different sampled positions or contexts may have different targets; there is no global r attached to a token ID. Production recomputes targets from the same reference to avoid retaining vocabulary tensors.', '',
'## Projection examples', '',
'All vectors use token order (A, B, C, D), with c=1.1.', '',
'| Case | Correct token | Reference q0 | Fixed r |', '| --- | --- | --- | --- |']
for case in data['cases'][:5]:
    lines.append(f"| {case['name']} | {case['label']} | {vector(case['reference_q0'])} | {vector(case['target_r'])} |")
lines += ['', 'Ties use the lowest token ID on original logits. A tie already selecting the correct token is preserved without imposing a positive margin; another tied target is repaired.', '',
'## Main learning trajectory', '',
'The illustrative update is `z <- z - 1.0 * gradient_z softCE(r, z)`, repeated 100 times on free logits. It does not use the real BF16-backbone/FP32-master AdamW training recipe; its step size and convergence speed must not be transferred to DFlash. The production loss is used with a single position of weight one.', '',
'| Step | q(A) | q(B) | q(C), correct | q(D) | Selected token | Soft CE |', '| --- | --- | --- | --- | --- | --- | --- |']
case = data['cases'][0]
for row in case['rows']:
    if row['step'] in [0, 1, 2, 5, 10, 12, 20, 50, 100]:
        values = ' | '.join(f"{row[f'q_{t}']:.6f}" for t in 'ABCD')
        lines.append(f"| {row['step']} | {values} | {row['argmax']} | {row['soft_ce']:.6f} |")
lines += ['', f"Target r stays {vector(case['target_r'])} at every step. C first becomes the highest-probability token at step 12 in this synthetic trace. Soft CE approaches the nonzero entropy of r, not zero. Raw hard-CE and soft-CE losses are not comparable learning-quality scores.", '',
'![Token learning](token_learning.png)', '',
'## Already-correct predictions and restoring gradients', '',
'An already-correct reference `(0.15, 0.60, 0.15, 0.10)` is preserved. Starting there gives zero logit gradient to numerical precision. A separate synthetic probe manually disturbs current probabilities to `(0.35, 0.35, 0.20, 0.10)` while keeping the original reference and r. The initial gradient is approximately `(0.20, -0.25, 0.05, 0.00)`, and descent returns toward r. This checks the restoring loss term; it does not demonstrate protection under shared-backbone training.', '',
'In the main case D has r(D)=q0(D)=0.20 and initially zero logit gradient, but q(D) temporarily rises because other logits change the softmax denominator. An unchanged target probability does not guarantee an unchanged current probability at every step.', '',
'## Validation', '',
f"All six cases passed. Maximum difference between production autograd and q-r: **{data['max_gradient_identity_error']:.3g}**. Targets remain detached and exactly unchanged across all 101 observations per case. Probability/margin atol: 2e-6. Gradient atol: 2e-7; rtol: 2e-5. The handoff worked example is checked explicitly. Final maximum distance to r is below 2e-6 in every case.", '',
'| Case | First correctly selected step | Final maximum probability error |', '| --- | --- | --- |']
for case in data['cases']:
    first = next((row['step'] for row in case['rows'] if row['argmax'] == case['label']), None)
    lines.append(f"| {case['name']} | {first} | {case['rows'][-1]['max_distance_to_target']:.3g} |")
lines += ['', 'Files: [script](trace.py), [all 606 observations](trajectory.csv), [structured results](results.json), [log](run.log), [plot source](plot.py), [SVG](token_learning.svg).', '',
'Reproduce with:', '', '```bash',
"PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/minhao2/DFlash-Research/training /home/minhao2/.venvs/dflash-specforge/bin/python /home/minhao2/EAGLE_SPEC/research/projected_target_token_trace_20261022/trace.py",
'```', '',
'The model environment lacks matplotlib; existing system matplotlib rendered the saved JSON separately, without installation. One report-generation command had a quoting syntax error and was corrected; the numerical trace passed on its first execution.', '',
'These examples verify local construction and loss behavior. They do not establish source fitting by full DFlash, future-request transfer, verification-call reduction or serving speedup. No scientific configuration changed; native qualification and the campaign remain pending review.']
(root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
