#!/bin/bash
# shellcheck shell=bash disable=SC2155
# =============================================================================
# Multi-node A/B check for the "async load multimodal data" change.
#
# Runs the SAME multimodal rollout twice on ONE Ray cluster, in
# --debug-rollout-only mode (SGLang only, no training):
#     1) BASELINE  = a chosen commit  (synchronous  preprocessing)
#     2) CANDIDATE = the current tree (asynchronous preprocessing)
# The two runs use identical arguments; the ONLY difference is the slime code
# version, selected by pointing PYTHONPATH at a different tree (see below). It
# then:
#     * prints slime's own perf/rollout_time per step (the speed signal), and
#     * diffs the saved rollout dumps with tools/diff_rollout_dumps.py to prove
#       the request tokens and the HF-processor tensors are unchanged.
#
# This is the multi-node counterpart of tools/compare_async_multimodal_rollout.sh
# (single node). It is meant for data with MANY images per sample, where the
# rollout-side image download + HF processing dominates and the async change
# shows the largest speedup.
#
# Why parity holds regardless of inference determinism
# ----------------------------------------------------
#   The async change only alters HOW the request tokens / processor tensors are
#   prepared (sync -> async), not WHAT they contain. diff_rollout_dumps.py
#   compares two groups strictly:
#     * request   : the prompt-token prefix sent to SGLang  -> must match
#     * processor : multimodal_train_inputs (image tensors) -> must match
#   Both are independent of sampling. With a fixed --rollout-seed and no
#   --rollout-shuffle the prompt order is identical across runs. The generated
#   response may differ if inference is non-deterministic, so the response group
#   is compared with --allow-response-diff by default (set STRICT_RESPONSE=1 to
#   require it too, e.g. text-only + DETERMINISTIC=1).
#
# Requirements
# ------------
#   * A multi-node box (WORLD_SIZE nodes x NUM_GPUS_PER_NODE GPUs).
#   * MODEL_DIR, PROMPT_DATA and both slime trees on shared storage visible to
#     every node (so switching the code version reaches all workers).
#   * Passwordless SSH from head to workers (or SSH_PASSWORD for sshpass).
#   * A clean git tree so the baseline worktree can be created.
#
# Usage
# -----
#   MASTER_ADDR=<head-ip> WORKER_IPS="<w1> <w2> ..." \
#   MODEL_DIR=/path/to/hf_checkpoint \
#   PROMPT_DATA=/path/to/train.jsonl \
#   MODEL_CONFIG=scripts/models/<arch>.sh \
#       bash tests/compare_async_multimodal_multinode.sh
#
#   # Quick smoke run:
#   NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=1 ... \
#       bash tests/compare_async_multimodal_multinode.sh
# =============================================================================

set -eo pipefail

# ---------------------------------------------------------------- configuration
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# Code versions. CANDIDATE = current tree; BASELINE = a git worktree at a ref.
CANDIDATE_SLIME_ROOT="${CANDIDATE_SLIME_ROOT:-${REPO_ROOT}}"
BASELINE_REF="${BASELINE_REF:-HEAD~1}"          # parent of the async-load commit
BASELINE_WORKTREE="${BASELINE_WORKTREE:-/tmp/slime-baseline-async-mm}"

# Model / data / output (fill these in for your environment).
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a local HF checkpoint}"
PROMPT_DATA="${PROMPT_DATA:?set PROMPT_DATA to a prompt dataset (jsonl/parquet)}"
OUT_DIR="${OUT_DIR:-/tmp/async_mm_verify}"
# Model-architecture args live in a per-repo config that defines MODEL_ARGS.
MODEL_CONFIG="${MODEL_CONFIG:?set MODEL_CONFIG, e.g. scripts/models/qwen3-vl.sh}"

# Dataset field names + multimodal mapping.
INPUT_KEY="${INPUT_KEY:-prompt}"
LABEL_KEY="${LABEL_KEY:-label}"
MULTIMODAL_KEYS="${MULTIMODAL_KEYS:-'{\"image\":\"images\"}'}"

# Optional custom reward. Default uses the built-in local "random" RM so the
# run has no external dependency (this test only measures preprocessing).
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-}"
REWARD_KEY="${REWARD_KEY:-}"

# Topology.
WORLD_SIZE="${WORLD_SIZE:-4}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"

# Rollout timing / parity knobs (greedy + single sample + no shuffle).
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"                  # step 0 = warm-up
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
# slime forbids --rollout-temperature 0; use a tiny positive value ~ greedy.
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1e-6}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-47500}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"

# Cluster bring-up.
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
SSH_PORT="${SSH_PORT:-22}"
SSH_PASSWORD="${SSH_PASSWORD:-}"                 # empty -> passwordless key
SSH_OPTS="-p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
JOIN_TIMEOUT_ATTEMPTS="${JOIN_TIMEOUT_ATTEMPTS:-120}"   # 120 x 5s = 10 min

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
WORKER_IPS="${WORKER_IPS:-}"                     # space-separated; empty = single node
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
RAY_BIN="$(command -v ray || echo ray)"

mkdir -p "${OUT_DIR}"
echo "candidate=${CANDIDATE_SLIME_ROOT}  baseline=${BASELINE_REF} (${BASELINE_WORKTREE})"
echo "cluster=head:${MASTER_ADDR} workers:[${WORKER_IPS}] (${WORLD_SIZE}x${NUM_GPUS_PER_NODE} GPU)  out=${OUT_DIR}"
echo "rollout: num=${NUM_ROLLOUT} bs=${ROLLOUT_BATCH_SIZE} n=${N_SAMPLES_PER_PROMPT} temp=${ROLLOUT_TEMPERATURE}"

# ---------------------------------------------------------------- baseline tree
if [ ! -d "${BASELINE_WORKTREE}" ]; then
    echo "[baseline] creating worktree at ${BASELINE_WORKTREE} @ ${BASELINE_REF}"
    git -C "${CANDIDATE_SLIME_ROOT}" worktree add -f "${BASELINE_WORKTREE}" "${BASELINE_REF}"
fi
[ -f "${BASELINE_WORKTREE}/train.py" ] || { echo "ERROR: baseline worktree is broken" >&2; exit 1; }

# ---------------------------------------------------------------- helpers
resolve_ip() {
    local addr="$1" ip
    [[ "${addr}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && { echo "${addr}"; return 0; }
    ip="$(getent hosts "${addr}" 2>/dev/null | awk '{print $1}' | head -n1)"
    [[ -n "${ip}" ]] && { echo "${ip}"; return 0; }
    return 1
}

ssh_worker() {  # ssh_worker <ip> <remote-cmd>
    if command -v sshpass >/dev/null 2>&1 && [[ -n "${SSH_PASSWORD}" ]]; then
        sshpass -p "${SSH_PASSWORD}" ssh ${SSH_OPTS} "root@${1}" "${2}"
    else
        ssh ${SSH_OPTS} "root@${1}" "${2}"
    fi
}

cleanup_local() {
    pkill -9 sglang 2>/dev/null || true; sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 -f raylet 2>/dev/null || true; pkill -9 -f gcs_server 2>/dev/null || true
    pkill -9 -f 'ray::' 2>/dev/null || true; pkill -9 ray 2>/dev/null || true; sleep 2
    rm -rf /tmp/ray/* /tmp/ray_tmp_* 2>/dev/null || true
}

MASTER_IP="$(resolve_ip "${MASTER_ADDR}")" || { echo "ERROR: cannot resolve ${MASTER_ADDR}" >&2; exit 1; }
export no_proxy="127.0.0.1,${MASTER_IP}"
DASHBOARD="http://${MASTER_IP}:${RAY_DASHBOARD_PORT}"

# ---------------------------------------------------------------- cluster (once)
bring_up_cluster() {
    echo "[cluster] cleanup + start head on ${MASTER_IP}:${RAY_PORT}"
    cleanup_local
    ray start --head --node-ip-address "${MASTER_IP}" --port "${RAY_PORT}" \
        --num-gpus "${NUM_GPUS_PER_NODE}" --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"
    sleep 5

    for wip in ${WORKER_IPS}; do
        [[ -z "${wip}" || "${wip}" == "${MASTER_IP}" ]] && continue
        echo "[ray] starting worker ${wip}"
        ssh_worker "${wip}" "pkill -9 sglang 2>/dev/null; ${RAY_BIN} stop --force 2>/dev/null; pkill -9 python 2>/dev/null; sleep 3; rm -rf /tmp/ray/* 2>/dev/null; ${RAY_BIN} start --address=${MASTER_IP}:${RAY_PORT} --num-gpus=${NUM_GPUS_PER_NODE} --node-ip-address=${wip} --disable-usage-stats" &
    done
    wait || true

    echo "[ray] waiting for ${WORLD_SIZE} node(s) x ${NUM_GPUS_PER_NODE} GPU..."
    local a gpu ngpu nodes
    for ((a=1; a<=JOIN_TIMEOUT_ATTEMPTS; a++)); do
        gpu="$(ray status 2>/dev/null | grep -oP '(?<=/)\d+\.\d+(?=\s*GPU)' | head -n1)"
        ngpu="$(echo "${gpu:-0}" | awk '{print int($1)}')"
        nodes=$(( ngpu / NUM_GPUS_PER_NODE ))
        [ "${nodes}" -ge "${WORLD_SIZE}" ] && { echo "[ray] all ${nodes} node(s) joined"; break; }
        [ "${a}" -eq "${JOIN_TIMEOUT_ATTEMPTS}" ] && { echo "ERROR: only ${nodes}/${WORLD_SIZE} joined" >&2; exit 1; }
        sleep 5
    done
    ray status
}

# ---------------------------------------------------------------- one variant
# run_variant <label> <slime_root> <dump_dir> <log_file>
run_variant() {
    local label="$1" slime_root="$2" dump_dir="$3" log_file="$4"
    echo ">>> variant ${label}  (slime_root=${slime_root})"
    mkdir -p "${dump_dir}"

    # Model-architecture args (same for both runs), sourced from the model config.
    # shellcheck disable=SC1090
    source "${slime_root}/${MODEL_CONFIG}"

    local CKPT_ARGS=(--hf-checkpoint "${MODEL_DIR}" --load "${MODEL_DIR}")

    local ROLLOUT_ARGS=(
        --prompt-data "${PROMPT_DATA}"
        --input-key "${INPUT_KEY}"
        --label-key "${LABEL_KEY}"
        --apply-chat-template
        --multimodal-keys "${MULTIMODAL_KEYS}"
        --num-rollout "${NUM_ROLLOUT}"
        --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
        --global-batch-size "$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))"
        --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
        --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
        --rollout-temperature "${ROLLOUT_TEMPERATURE}"
        --rollout-seed 42
        # no --rollout-shuffle -> identical prompt order across runs
    )
    # Reward: custom path if given, otherwise the built-in local "random" RM.
    if [ -n "${CUSTOM_RM_PATH}" ]; then
        ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
        [ -n "${REWARD_KEY}" ] && ROLLOUT_ARGS+=(--reward-key "${REWARD_KEY}")
    else
        ROLLOUT_ARGS+=(--rm-type random)
    fi

    local SGLANG_ARGS=(
        --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
        --sglang-attention-backend flashinfer
    )
    # Deterministic inference forces batch-invariant GEMM, which currently breaks
    # some multimodal vision encoders. Enable only for text-only parity checks.
    if [ "${DETERMINISTIC:-0}" != "0" ]; then
        SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
    fi

    local DEBUG_ARGS=(
        --debug-rollout-only
        --save-debug-rollout-data "${dump_dir}/rollout_{rollout_id}.pt"
        --rollout-health-check-interval 300
        --rollout-health-check-timeout 300
    )

    local RUNTIME_ENV_JSON="{
      \"env_vars\": {
        \"PYTHONPATH\": \"${slime_root}:${MEGATRON_PATH}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
      }
    }"

    ( cd "${slime_root}" && ray job submit --address="${DASHBOARD}" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 train.py \
        --actor-num-nodes "${WORLD_SIZE}" \
        --actor-num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${DEBUG_ARGS[@]}" \
    ) 2>&1 | tee "${log_file}"
}

# ---------------------------------------------------------------- run both
BASE_DUMP="${OUT_DIR}/baseline"; CAND_DUMP="${OUT_DIR}/candidate"
BASE_LOG="${OUT_DIR}/baseline.log"; CAND_LOG="${OUT_DIR}/candidate.log"

bring_up_cluster
run_variant baseline  "${BASELINE_WORKTREE}"    "${BASE_DUMP}" "${BASE_LOG}"
run_variant candidate "${CANDIDATE_SLIME_ROOT}" "${CAND_DUMP}" "${CAND_LOG}"
cleanup_local

# ---------------------------------------------------------------- speed report
echo ">>> speed: perf/rollout_time per step (step 0 = warm-up)"
python3 - "${BASE_LOG}" "${CAND_LOG}" <<'PY'
import re, sys
def times(path):
    out, pat = [], re.compile(r"'perf/rollout_time':\s*([0-9.]+)")
    try:
        for line in open(path):
            m = pat.search(line)
            if m: out.append(float(m.group(1)))
    except FileNotFoundError:
        pass
    return out
def mean(ts):
    m = ts[1:] if len(ts) > 1 else ts   # drop warm-up
    return sum(m)/len(m) if m else float('nan')
b, c = times(sys.argv[1]), times(sys.argv[2])
print(f"baseline  per step: {['%.2f' % t for t in b]}")
print(f"candidate per step: {['%.2f' % t for t in c]}")
if b and c:
    bm, cm = mean(b), mean(c)
    print(f"mean (excl. warm-up): baseline={bm:.2f}s candidate={cm:.2f}s")
    if cm > 0:
        print(f"speedup: {bm/cm:.2f}x  (saved {bm-cm:.2f}s/step, {100*(bm-cm)/bm:.1f}%)")
else:
    print("WARNING: could not parse perf/rollout_time from one of the logs.")
PY

# ---------------------------------------------------------------- parity report
echo ">>> parity: diff saved rollout dumps"
DIFF_TOOL="${CANDIDATE_SLIME_ROOT}/tools/diff_rollout_dumps.py"
DIFF_EXTRA=(); [ "${STRICT_RESPONSE:-0}" = "0" ] && DIFF_EXTRA+=(--allow-response-diff)

overall=0
for step in $(seq 0 $((NUM_ROLLOUT - 1))); do
    base_pt="${BASE_DUMP}/rollout_${step}.pt"; cand_pt="${CAND_DUMP}/rollout_${step}.pt"
    if [ -f "${base_pt}" ] && [ -f "${cand_pt}" ]; then
        echo "--- rollout ${step} ---"
        python3 "${DIFF_TOOL}" "${base_pt}" "${cand_pt}" "${DIFF_EXTRA[@]}" || overall=1
    fi
done

if [ "${overall}" -eq 0 ]; then
    echo "OVERALL: PASS - async preprocessing preserved rollout inputs; see speed report above."
else
    echo "OVERALL: FAIL - rollout inputs diverged; inspect the diffs above."
fi
exit "${overall}"
