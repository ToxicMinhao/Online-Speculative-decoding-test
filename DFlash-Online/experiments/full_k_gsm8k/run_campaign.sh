#!/usr/bin/env bash
# Explicit invocation after publication, sealing, verified offload and admission.
set -eu
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
export PYTHONPATH=/data/sources/minhao/DFlash-Online:/data/sources/minhao/DFlash-Online/training:/data/sources/minhao/DFlash-Online/inference:/data/sources/minhao/DFlash-Online/tests
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
export PATH=/data/conda_envs/minhao/dflash_specforge_20261022/bin:$PATH
export CONDA_PREFIX=/data/conda_envs/minhao/dflash_specforge_20261022
export CC=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/conda_envs/minhao/dflash_tree_qualification_v2/bin/x86_64-conda-linux-gnu-g++
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

run_root="$1"
run_python=/data/conda_envs/minhao/dflash_specforge_20261022/bin/python
test -f "$run_root/sealed/SEAL.json"
test ! -e "$run_root/START_UTC"
date -u +%FT%TZ > "$run_root/START_UTC"
echo "$$" > "$run_root/SUPERVISOR_PID"
trap 'result=$?; date -u +%FT%TZ > "$run_root/END_UTC"; echo "$result" > "$run_root/EXIT_CODE"' EXIT

CUDA_VISIBLE_DEVICES='' "$run_python" -m unittest discover -s tests -p 'test_full_k*.py' -v \
  > "$run_root/CPU_TESTS.log" 2>&1
"$run_python" -u -m experiments.full_k_gsm8k.preflight "$run_root"

# Forty-eight hours is a resource safety ceiling, not a performance gate.
set +e
timeout --signal=TERM --kill-after=120s 48h \
  "$run_python" -u -m experiments.full_k_gsm8k.campaign "$run_root"
campaign_exit=$?
echo "$campaign_exit" > "$run_root/CAMPAIGN_EXIT_CODE"

"$run_python" -u -c \
  'import sys; from experiments.full_k_gsm8k.audit import audit_training; audit_training(sys.argv[1], raise_on_error=False)' \
  "$run_root"
audit_exit=$?
"$run_python" -u -m experiments.full_k_gsm8k.report "$run_root"
report_exit=$?
"$run_python" -u -m experiments.full_k_gsm8k.preflight "$run_root" --postflight
postflight_exit=$?
printf 'audit=%s\nreport=%s\npostflight=%s\n' "$audit_exit" "$report_exit" "$postflight_exit" > "$run_root/REPORT_EXIT_CODES"
for result in "$campaign_exit" "$audit_exit" "$report_exit" "$postflight_exit"; do
  if [ "$result" -ne 0 ]; then
    exit "$result"
  fi
done
exit 0
