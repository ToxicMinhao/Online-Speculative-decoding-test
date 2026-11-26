# Execution commands and provenance

These are the commands used by this completed-or-in-progress record, not authorization to start another run. The original run directory must not be overwritten. Private sealed inputs and their identities remain outside Git; source/runtime paths below identify the executed environment. Source staging authenticated every tracked entry before execution.

| Phase | Source commit | Outcome |
|---|---|---|
| Original qualification, TAIL training, first readout process | `b3637cfe4ac3fb4096368cc0c93896b420bdb9a8` | Qualification/training complete; readout stopped after 74 forwards |
| First repaired readout process | `b3a467200af0b255fe2a491924d03c2c1f0c286b` | Checker rejected before model loading; zero forwards |
| CPU-only actual-record preflight | `d819c243235e5a650ce5292f2fa26ebfa563a758` | Passed; no model work |
| Remaining readouts and serving | `b059d0ea7595f24a9db114b46a98cae60f0602b0` | 63 remaining readouts and original 792 serving budget |

The original sealed plan retains its four-hour elapsed limit. The explicit user authorization changes only the continuation execution view to twelve hours from the original start. The four occupied GPU-card-hour cap and storage/scientific settings remain unchanged. Failed processes and costs are preserved.

## Preparation and CPU checks

```bash
#!/bin/bash
set -euo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/conda_envs/minhao/dflash_specforge_20261022
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CUDA_VISIBLE_DEVICES=''
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online:/data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online/training:/data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online/inference:/data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online/tests
cd /data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online
python -m unittest discover -s tests -p 'test_continuation_repair*.py' -v
python -m experiments.continuation_repair_pilot.prepare \
 --production-plan /data/sources/minhao/runs/dflash_frontier_preservation_20261125/production_001/sealed/PLAN.json \
 --archive /data/sources/minhao/point2a_c_reference_source_20261125_001/RETAINED_COMPACT_MANIFEST.json \
 --diagnostic-root /data/sources/minhao/runs/dflash_continuation_value_diagnostic_20261126/diagnostic_001 \
 --root /data/sources/minhao/runs/dflash_continuation_repair_pilot_20261126/pilot_001 \
 --source-manifest /data/sources/minhao/continuation_repair_source_20261126_001/SOURCE.json \
 --commit b3637cfe4ac3fb4096368cc0c93896b420bdb9a8 \
 --local-bytes 1500000000 \
 --training-gpu GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1 \
 --serving-gpu GPU-9986735a-3e68-d1bd-5998-9634ef24341e
```

## Original supervised run

```bash
#!/bin/bash
set -euo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/conda_envs/minhao/dflash_specforge_20261022
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CUDA_VISIBLE_DEVICES=''
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online:/data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online/training:/data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online/inference:/data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online/tests
cd /data/sources/minhao/continuation_repair_source_20261126_001/DFlash-Online
PILOT_RUN=/data/sources/minhao/runs/dflash_continuation_repair_pilot_20261126/pilot_001
trap 'pilot_exit=$?; printf "%s\n" "$pilot_exit" > "$PILOT_RUN/PROCESS_EXIT_CODE"' EXIT
printf '%s\n' "$$" > "$PILOT_RUN/PROCESS_PID"
python -u -m experiments.continuation_repair_pilot.launch --plan "$PILOT_RUN/sealed/PLAN.json"
```

## First repair, failed before model loading

```bash
#!/bin/bash
set -euo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/conda_envs/minhao/dflash_specforge_20261022
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CUDA_VISIBLE_DEVICES=''
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data/sources/minhao/continuation_repair_source_20261126_002/DFlash-Online:/data/sources/minhao/continuation_repair_source_20261126_002/DFlash-Online/training:/data/sources/minhao/continuation_repair_source_20261126_002/DFlash-Online/inference:/data/sources/minhao/continuation_repair_source_20261126_002/DFlash-Online/tests
cd /data/sources/minhao/continuation_repair_source_20261126_002/DFlash-Online
PILOT_RUN=/data/sources/minhao/runs/dflash_continuation_repair_pilot_20261126/pilot_001
trap 'pilot_exit=$?; printf "%s\n" "$pilot_exit" > "$PILOT_RUN/RESUME_PROCESS_EXIT_CODE"' EXIT
printf '%s\n' "$$" > "$PILOT_RUN/RESUME_PROCESS_PID"
python -u -m experiments.continuation_repair_pilot.launch --plan "$PILOT_RUN/sealed/PLAN.json" --resume-unexecuted --repair-source-root /data/sources/minhao/continuation_repair_source_20261126_002/DFlash-Online --repair-source-manifest /data/sources/minhao/continuation_repair_source_20261126_002/SOURCE_REPAIR.json --repair-commit b3a467200af0b255fe2a491924d03c2c1f0c286b
```

## User-authorized remaining-work continuation

```bash
#!/bin/bash
set -euo pipefail
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/conda_envs/minhao/dflash_specforge_20261022
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
export CUDA_VISIBLE_DEVICES=''
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data/sources/minhao/continuation_repair_source_20261126_004/DFlash-Online:/data/sources/minhao/continuation_repair_source_20261126_004/DFlash-Online/training:/data/sources/minhao/continuation_repair_source_20261126_004/DFlash-Online/inference:/data/sources/minhao/continuation_repair_source_20261126_004/DFlash-Online/tests
cd /data/sources/minhao/continuation_repair_source_20261126_004/DFlash-Online
PILOT_RUN=/data/sources/minhao/runs/dflash_continuation_repair_pilot_20261126/pilot_001
trap 'pilot_exit=$?; printf "%s\n" "$pilot_exit" > "$PILOT_RUN/RESUME2_PROCESS_EXIT_CODE"' EXIT
printf '%s\n' "$$" > "$PILOT_RUN/RESUME2_PROCESS_PID"
python -u -m experiments.continuation_repair_pilot.launch --plan "$PILOT_RUN/sealed/PLAN.json" --resume-unexecuted --repair-source-root /data/sources/minhao/continuation_repair_source_20261126_004/DFlash-Online --repair-source-manifest /data/sources/minhao/continuation_repair_source_20261126_004/SOURCE_ELAPSED_REPAIR.json --repair-commit b059d0ea7595f24a9db114b46a98cae60f0602b0 --elapsed-authorization "$PILOT_RUN/sealed/ELAPSED_EXTENSION_AUTHORIZATION.json"
```

## Child phases and evidence

The supervisor runs the following module commands with the sealed plan and explicit physical GPU. It records process start/end, exit status, source, storage and original-clock costs. They were not additionally invoked as duplicate jobs.

```text
python -u -m experiments.continuation_repair_pilot.native --plan ORIGINAL_PLAN --phase qualification
python -u -m experiments.continuation_repair_pilot.native --plan ORIGINAL_PLAN --phase training
python -u -m experiments.continuation_repair_pilot.readout --plan ORIGINAL_PLAN
python -u -m experiments.continuation_repair_pilot.readout --plan REPAIRED_PLAN --resume-unexecuted
python -u -m experiments.continuation_repair_pilot.serving --plan REPAIRED_PLAN
```

Postflight and retrieval execute only after all GPU processes stop. Postflight checks original input and endpoint file identities, all four staged source identities and current storage. Compact evidence is streamed directly to Fenrir, rehashed after extraction and preserved without model tensors. Derived CPU analysis is written to a separate directory; the raw seals and native records remain unchanged.

## CPU analysis and independent audits

The final CPU reducer ran from clean commit `00ac1944e8d8d535c24c690917e2a2890af64659`. It changes only reporting provenance/process counts, not scientific reductions or any GPU operation. `ANALYSIS_PLAN.json` copies the original final execution view and adds only `analysis_execution_commit`; local raw and derived locations are separate CLI arguments. `ANALYSIS_SOURCE.json` binds that view and an exact copy of the used `analysis.py`. The private raw files remain unchanged; six missing control receipts were supplemented after primary retrieval and authenticated against original failure receipts.

The following environment variables denote the exact evidence locations. Commands run on Fenrir with CUDA disabled and do not load a model. Audit outputs are new paths, preserving earlier unsuccessful evidence-completeness checks.

```bash
STUDY=/home/minhao2/EAGLE_SPEC/research/dflash_continuation_repair_pilot_20261126
PILOT_RAW="$STUDY/pilot_001/raw"
PILOT_ANALYSIS="$STUDY/pilot_001/analysis"
PILOT_PUBLIC="$STUDY/implementation_checkout/DFlash-Online/reports/dflash_continuation_repair_pilot_20261126/pilot_001"
cd "$STUDY/implementation_checkout/DFlash-Online"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:training:inference:tests /home/minhao2/.venvs/dflash-specforge/bin/python -m experiments.continuation_repair_pilot.analysis --plan "$PILOT_ANALYSIS/ANALYSIS_PLAN.json" --raw-root "$PILOT_RAW" --analysis-output-root "$PILOT_ANALYSIS" --public "$PILOT_PUBLIC"
python "$STUDY/audit_native_work.py" --raw "$PILOT_RAW" --retained-event "$PILOT_RAW/audit_inputs/retained_event.json" --feature-manifest "$PILOT_RAW/audit_inputs/feature_manifest.json" --base-status "$PILOT_RAW/audit_inputs/base_status.json" --diagnostic-contract "$PILOT_RAW/audit_inputs/diagnostic_contract.json" --output "$STUDY/final_audit/native_002.json"
/home/minhao2/.venvs/dflash-specforge/bin/python "$STUDY/audit_serving_independent.py" --raw-root "$PILOT_RAW" --analysis-root "$PILOT_ANALYSIS" --public "$PILOT_PUBLIC" --output "$STUDY/final_audit/serving_002.json"
python "$STUDY/audit_mechanism_independent.py" --raw-root "$PILOT_RAW" --retained-root /home/minhao2/EAGLE_SPEC/research/dflash_continuation_value_diagnostic_20261126/diagnostic_001/raw --output "$STUDY/final_audit/mechanism_001"
python "$STUDY/audit_costs_independent.py" --raw-root "$PILOT_RAW" --cleanup "$STUDY/STORAGE_RETIREMENT.json" --base-status "$PILOT_RAW/audit_inputs/base_status.json" --postflight "$STUDY/POSTFLIGHT.json" --output "$STUDY/final_audit/costs_001"
```

The copied scripts in `audit_code/` are byte-identical to the executed study-local audit scripts. Their private inputs are intentionally not part of Git. The report's source identities remain tied to the original native phases, not the later publication commit.
