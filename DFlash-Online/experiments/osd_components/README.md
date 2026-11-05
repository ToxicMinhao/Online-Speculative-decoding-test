# OSD checkpoint-component comparison

This experiment asks which **retained changes** in the successful jointly trained
Qwen3 DFlash checkpoint remain useful. It does not train a model or introduce a
new loss. Production code lives in A800 `DFlash-Online`; the Fenrir research
worktree is an implementation/evidence snapshot. The clean Fenrir
`/home/minhao2/DFlash-Research` remains unchanged.

## Reading the code

1. `prepare.py` checks the old execution seal and native runtime, selects 64
   requests with seed 20261024, and checks their saved references. It seals the
   panel before constructing hybrids.
2. `checkpoints.py` copies exact BF16 tensors into two hybrids. C is precisely
   `fc.weight` and `hidden_norm.weight`; B is the other 56 exported tensors
   (five decoder layers and `norm.weight`). There is no interpolation or update.
3. `run.py` uses `online_feedback.sglang_backend.SGLangDFlashBackend`. Each
   checkpoint is published through the existing cache-safe path. All 58 source
   tensors / 43 fused native tensors are checked on the first use of each arm.
   Every publication checks helper freshness and empty affected caches.
4. `protocol.py` contains the fixed schedule and measurement checks. Each request
   has three repeats of F, O, C-only and B-only, rotating by selected-panel ordinal.
   The first eight requests are the 96-generation correctness qualification,
   included in the 768-generation total. Performance is not a continuation gate.
5. `analysis.py` keeps a common valid four-arm, all-repeat cohort, using repeat 0
   for verification counts and all repeats for latency. It reports gross gains,
   gross harms, conditional effects, interaction and paired request bootstrap
   sensitivity. No repeat is treated as an independent request.

## Fixed serving settings

Use the successful run's Qwen3-8B target and Qwen3 DFlash initializer/export,
exact saved prompt IDs, greedy BF16 SGLang/FA3, block size 16, concurrency 1,
2,048 output tokens or EOS, canonical context 4,096 with the runtime's existing
16-token internal allowance. Each request uses the existing cold-cache policy.
No feedback capture, training, additional warmup generations or concurrent GPU
training is permitted. F/O must reproduce retained output IDs and verification
counts. Hybrids generate naturally; saved outputs are equality references.

Complete request latency includes prefill, drafting, verification and the existing
per-request cache maintenance. Cache time is a nested diagnostic, not an extra
addition. Checkpoint publication, full weight checks, preparation and model loading
are outside request latency and separately reported. The total campaign wall time
includes benchmarking overhead. This is performance with an available checkpoint,
not adaptation-inclusive online serving.

## Preparation (CPU only)

Source `/data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh`, set the usual
DFlash-Online/training/inference PYTHONPATH, and use the existing coordinator Python:

```bash
CUDA_VISIBLE_DEVICES='' /data/conda_envs/minhao/dflash_specforge_20261022/bin/python \
  -m experiments.osd_components.prepare \
  --prior-root /data/sources/minhao/runs/dflash_online_osd_rerun_20261023 \
  --runtime-pin /data/sources/minhao/runs/osd_checkpoint_components_20261024/inputs/RUNTIME_PIN.json \
  --model-pin /data/sources/minhao/runs/osd_checkpoint_components_20261024/inputs/MODEL_PIN.json \
  --worker-python /data/conda_envs/minhao/adaflash_author/bin/python \
  --output /data/sources/minhao/runs/osd_checkpoint_components_20261024/prepared
```

Source/target/released-model hashes are compared with historical pins. The final
OSD export has a retained event-58 publication receipt, but no historical export
file digest was found. Its current digest is sealed, and historical output/count
reproduction remains mandatory. Do not call this a historical digest match.
Actual files and construction receipts are checked again once before GPU startup;
no hashing is added inside timed serving.

Preparation budgets two hybrids plus atomic overlap, 2 GB workspace and an 8 GiB
reserve, with at least 18 GB free. It refuses to replace previous artifacts.

## Explicit GPU invocation (not performed by preparation)

```bash
bash experiments/osd_components/run_campaign.sh \
  /data/sources/minhao/runs/osd_checkpoint_components_20261024/prepared \
  /data/sources/minhao/runs/osd_checkpoint_components_20261024/campaign_001
```

The wrapper checks that GPU serving is idle and applies a two-hour timeout with
120 seconds for orderly cleanup. A failed operation is not retried. Structural
failures stop; later isolated output mismatches remain recorded and exclude the
whole affected request from comparative gains. Missing/unexecuted work remains
explicit. No follow-up training or mechanism is launched automatically.

## Outputs and interpretation

Keep preparation receipts, exact input/output IDs and native traces, every timing
repeat, activation/equality costs, status and CSV/JSON reports. Pointwise bootstrap
intervals describe sensitivity to request composition, not seed robustness.
A strong C-only result motivates a later, separately reviewed experiment training
only C with matched exposure. A strong B-only result motivates the complementary
question. An interaction indicates dependence of these retained endpoint changes;
failed hybrids do not prove that full-backbone training is necessary.
