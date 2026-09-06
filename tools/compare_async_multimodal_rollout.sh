#!/bin/bash
# =============================================================================
# E2E parity + speed check for the "async load multimodal data" change.
#
# Runs the SAME multimodal rollout twice in --debug-rollout-only mode (no
# training, SGLang only):
#     1) BASELINE  = the parent commit (synchronous preprocessing)
#     2) CANDIDATE = the current working tree (async preprocessing)
# with deterministic inference, greedy sampling, no shuffle and a fixed seed, so
# the two runs are expected to be bit-for-bit identical. It then:
#     * reports slime's own `perf/rollout_time` for each run (the speed signal),
#     * diffs the saved rollout dumps with tools/diff_rollout_dumps.py to prove
#       the request tokens, the HF-processor tensors and the generated results
#       are unchanged.
#
# This is the TOP of the verification ladder and needs a real GPU box with the
# model + dataset already available (see examples/geo3k_vlm/README.md). The
# cheaper, hardware-free layers are:
#     tests/test_async_multimodal_preprocessing.py      (pure-CPU equivalence)
#     tests/test_async_multimodal_real_processor.py     (real HF processor)
#     tests/test_async_multimodal_request_parity.py     (SGLang payload parity)
#
# Usage:
#     bash tools/compare_async_multimodal_rollout.sh
# Override anything via env, e.g.:
#     NUM_GPUS=8 ROLLOUT_STEPS=3 MODEL_NAME=Qwen3.5-35B-A3B \
#         bash tools/compare_async_multimodal_rollout.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- configuration
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
BASELINE_REF="${BASELINE_REF:-4c193f1}"          # parent of "support async load multimodal data"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
DATASET_NAME="${DATASET_NAME:-chenhegu/geo3k_imgurl}"
DATASET_LOCAL_NAME="$(basename "$DATASET_NAME")"
NUM_GPUS="${NUM_GPUS:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-3}"              # step 0 = warm-up, steps 1.. are compared/timed
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
OUT_DIR="${OUT_DIR:-/root/async_mm_verify}"
WORKTREE="${WORKTREE:-/tmp/slime-baseline-async-mm}"
MODEL_DIR="${MODEL_DIR:-/root/models/${MODEL_NAME}}"
DATASET_DIR="${DATASET_DIR:-/root/datasets/${DATASET_LOCAL_NAME}}"
# path (relative to each slime tree) of the model arch config that defines MODEL_ARGS
MODEL_CONFIG="${MODEL_CONFIG:-scripts/models/qwen3.5-35B-A3B-vl.sh}"
MULTIMODAL_KEYS="${MULTIMODAL_KEYS:-'{\"image\": \"images\"}'}"

mkdir -p "$OUT_DIR"
echo "repo_root=$REPO_ROOT  baseline_ref=$BASELINE_REF  num_gpus=$NUM_GPUS  steps=$ROLLOUT_STEPS  out=$OUT_DIR"

cleanup_procs() {
   pkill -9 sglang || true
   sleep 2
   ray stop --force || true
   pkill -9 ray || true
   pkill -9 slime || true
   sleep 2
}

# ---------------------------------------------------------------- one variant
# run_variant <label> <slime_dir> <dump_dir> <log_file>
run_variant() {
   local label="$1" slime_dir="$2" dump_dir="$3" log_file="$4"
   echo "======================================================================"
   echo ">>> running variant: ${label}   (slime_dir=${slime_dir})"
   echo "======================================================================"
   mkdir -p "$dump_dir"
   cleanup_procs

   export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

   # Model architecture args (sourced from the model config, same for both runs).
   source "${slime_dir}/${MODEL_CONFIG}"

   local CKPT_ARGS=(--hf-checkpoint "${MODEL_DIR}" --load "${MODEL_DIR}")

   # Deterministic, no-training rollout: identical inputs + reproducible outputs.
   local ROLLOUT_ARGS=(
      --prompt-data "${DATASET_DIR}/train.parquet"
      --input-key problem
      --label-key answer
      --apply-chat-template
      --rm-type deepscaler
      --num-rollout "${ROLLOUT_STEPS}"
      --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
      --n-samples-per-prompt 1
      --rollout-max-response-len 1024
      --rollout-temperature 0.0
      --global-batch-size "${ROLLOUT_BATCH_SIZE}"
      --rollout-seed 42
      # NOTE: no --rollout-shuffle  -> deterministic prompt order across runs
   )

   local SGLANG_ARGS=(
      --rollout-num-gpus-per-engine "${NUM_GPUS}"        # single engine -> deterministic dp routing
      --sglang-mem-fraction-static 0.7
      --sglang-enable-deterministic-inference
      --sglang-attention-backend flashinfer
      # speculative decoding intentionally disabled for a clean parity baseline
   )

   local DEBUG_ARGS=(
      --debug-rollout-only
      --rollout-num-gpus "${NUM_GPUS}"
      --save-debug-rollout-data "${dump_dir}/rollout_{rollout_id}.pt"
   )

   local RUNTIME_ENV_JSON="{
     \"env_vars\": {
       \"PYTHONPATH\": \"/root/Megatron-LM/\",
       \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
       \"NCCL_ALGO\": \"Ring\",
       \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"0\",
       \"CUBLAS_WORKSPACE_CONFIG\": \":4096:8\"
     }
   }"

   ( cd "${slime_dir}" && ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node "${NUM_GPUS}" \
      --multimodal-keys "${MULTIMODAL_KEYS}" \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${DEBUG_ARGS[@]}" \
   ) 2>&1 | tee "${log_file}"

   cleanup_procs
}

# ---------------------------------------------------------------- baseline tree
if [ ! -d "$WORKTREE" ]; then
   echo ">>> creating baseline worktree at ${WORKTREE} @ ${BASELINE_REF}"
   git -C "$REPO_ROOT" worktree add -f "$WORKTREE" "$BASELINE_REF"
fi

BASE_DUMP="$OUT_DIR/baseline"
CAND_DUMP="$OUT_DIR/candidate"
BASE_LOG="$OUT_DIR/baseline.log"
CAND_LOG="$OUT_DIR/candidate.log"

run_variant "baseline"  "$WORKTREE"   "$BASE_DUMP" "$BASE_LOG"
run_variant "candidate" "$REPO_ROOT"  "$CAND_DUMP" "$CAND_LOG"

# ---------------------------------------------------------------- speed report
echo "======================================================================"
echo ">>> speed: perf/rollout_time per step (step 0 is warm-up)"
echo "======================================================================"
python3 - "$BASE_LOG" "$CAND_LOG" <<'PY'
import re, sys

def rollout_times(path):
    times = []
    pat = re.compile(r"'perf/rollout_time':\s*([0-9.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                times.append(float(m.group(1)))
    return times

base = rollout_times(sys.argv[1])
cand = rollout_times(sys.argv[2])
print(f"baseline  rollout_time per step: {['%.2f' % t for t in base]}")
print(f"candidate rollout_time per step: {['%.2f' % t for t in cand]}")

def mean_measured(ts):
    measured = ts[1:] if len(ts) > 1 else ts   # drop warm-up step 0
    return sum(measured) / len(measured) if measured else float('nan')

if base and cand:
    b, c = mean_measured(base), mean_measured(cand)
    print(f"\nmean rollout_time (excl. warm-up): baseline={b:.2f}s  candidate={c:.2f}s")
    if c > 0:
        print(f"speedup: {b / c:.2f}x   (saved {b - c:.2f}s/step, {100 * (b - c) / b:.1f}%)")
else:
    print("\nWARNING: could not parse perf/rollout_time from one of the logs.")
PY

# ---------------------------------------------------------------- parity report
echo "======================================================================"
echo ">>> parity: diff saved rollout dumps"
echo "======================================================================"
overall=0
for step in $(seq 0 $((ROLLOUT_STEPS - 1))); do
   base_pt="$BASE_DUMP/rollout_${step}.pt"
   cand_pt="$CAND_DUMP/rollout_${step}.pt"
   if [ -f "$base_pt" ] && [ -f "$cand_pt" ]; then
      echo "--- rollout ${step} ---"
      if ! python3 "$REPO_ROOT/tools/diff_rollout_dumps.py" "$base_pt" "$cand_pt"; then
         overall=1
      fi
   fi
done

echo "======================================================================"
if [ "$overall" -eq 0 ]; then
   echo "OVERALL: PASS — async preprocessing preserved rollout results; see speed report above."
else
   echo "OVERALL: FAIL — rollout results diverged; inspect the diffs above."
fi
exit "$overall"
