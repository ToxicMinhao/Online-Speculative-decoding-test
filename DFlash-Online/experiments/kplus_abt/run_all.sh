#!/usr/bin/env bash
# Execute qualification first, then the unchanged pilot only on explicit admission.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
run_root="$1"
run_source="$2"
cd "$run_source"
export PYTHONPATH="$PWD:$PWD/training:$PWD/inference:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
run_python="$CONDA_PREFIX/bin/python"
test ! -e "$run_root/START_UTC"
date -u +%FT%TZ > "$run_root/START_UTC"
echo "$$" > "$run_root/SUPERVISOR_PID"
finish() {
    result=$?
    "$run_python" -m experiments.kplus_abt.report "$run_root" --output "$run_root/final_report" > "$run_root/report.log" 2>&1 || { if test "$result" -eq 0; then result=1; fi; }
    date -u +%FT%TZ > "$run_root/END_UTC"
    echo "$result" > "$run_root/EXIT_CODE"
    exit "$result"
}
trap finish EXIT
CUDA_VISIBLE_DEVICES='' "$run_python" -m unittest discover -s tests/kplus_abt -v > "$run_root/CPU_CAMPAIGN_TESTS.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$run_python" -m unittest -v test_adversarial_block test_adversarial_block_runtime test_adversarial_block_diagnostics test_adversarial_block_resource test_abt_continuation > "$run_root/CPU_RUNTIME_TESTS.log" 2>&1
timeout --signal=TERM --kill-after=120s 4h "$run_python" -u -m experiments.kplus_abt.run_qualification "$run_root"
bash experiments/kplus_abt/run_campaign.sh "$run_root" "$run_source"
