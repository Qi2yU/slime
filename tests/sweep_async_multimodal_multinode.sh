#!/bin/bash
# shellcheck shell=bash disable=SC2155
# =============================================================================
# Speed SWEEP for the "async load multimodal data" change (multi-node).
#
# Fixes the training data and sweeps (rollout_batch_size, n_samples_per_prompt)
# from small to large, recording baseline (synchronous) vs candidate (async)
# perf/rollout_time at each point. Produces a CSV and a Markdown table suitable
# for a PR / benchmark report.
#
# Same mechanics as tests/compare_async_multimodal_multinode.sh:
#   * one Ray cluster, brought up once and reused for every point;
#   * per point, baseline and candidate run --debug-rollout-only; the ONLY
#     difference between them is the slime code version (PYTHONPATH tree);
#   * step 0 of each point is a warm-up, the rest are averaged.
#
# The async change only speeds up rollout preprocessing without changing its
# result; correctness is proven by the parity check in the compare script, this
# one only measures the speedup as load grows.
#
# Requirements: see tests/compare_async_multimodal_multinode.sh.
#
# Usage
# -----
#   MASTER_ADDR=<head-ip> WORKER_IPS="<w1> <w2> ..." \
#   MODEL_DIR=/path/to/hf_checkpoint \
#   PROMPT_DATA=/path/to/train.jsonl \
#   MODEL_CONFIG=scripts/models/<arch>.sh \
#       bash tests/sweep_async_multimodal_multinode.sh
#
#   # Custom sweep points ("bs:n" pairs) and rollouts per point:
#   SWEEP="8:1 16:2 32:4" NUM_ROLLOUT=4 ... \
#       bash tests/sweep_async_multimodal_multinode.sh
# =============================================================================

set -eo pipefail

# ---------------------------------------------------------------- configuration
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

CANDIDATE_SLIME_ROOT="${CANDIDATE_SLIME_ROOT:-${REPO_ROOT}}"
BASELINE_REF="${BASELINE_REF:-HEAD~1}"
BASELINE_WORKTREE="${BASELINE_WORKTREE:-/tmp/slime-baseline-async-mm}"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a local HF checkpoint}"
PROMPT_DATA="${PROMPT_DATA:?set PROMPT_DATA to a prompt dataset (jsonl/parquet)}"
OUT_DIR="${OUT_DIR:-/tmp/async_mm_sweep}"
MODEL_CONFIG="${MODEL_CONFIG:?set MODEL_CONFIG, e.g. scripts/models/qwen3-vl.sh}"

INPUT_KEY="${INPUT_KEY:-prompt}"
LABEL_KEY="${LABEL_KEY:-label}"
MULTIMODAL_KEYS="${MULTIMODAL_KEYS:-'{\"image\":\"images\"}'}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-}"
REWARD_KEY="${REWARD_KEY:-}"

# Sweep points "bs:n", small -> large. gen_reqs = bs*n drives the image load.
SWEEP="${SWEEP:-4:1 8:1 8:2 16:2 16:4 32:4}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"                  # per point; step 0 = warm-up

WORLD_SIZE="${WORLD_SIZE:-4}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1e-6}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-47500}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
SSH_PORT="${SSH_PORT:-22}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SSH_OPTS="-p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
JOIN_TIMEOUT_ATTEMPTS="${JOIN_TIMEOUT_ATTEMPTS:-120}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
WORKER_IPS="${WORKER_IPS:-}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
RAY_BIN="$(command -v ray || echo ray)"

mkdir -p "${OUT_DIR}"
echo "candidate=${CANDIDATE_SLIME_ROOT}  baseline=${BASELINE_REF} (${BASELINE_WORKTREE})"
echo "cluster=head:${MASTER_ADDR} workers:[${WORKER_IPS}] (${WORLD_SIZE}x${NUM_GPUS_PER_NODE} GPU)"
echo "sweep=[${SWEEP}]  num_rollout/point=${NUM_ROLLOUT}  out=${OUT_DIR}"

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
ssh_worker() {
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
    echo "[ray] waiting for ${WORLD_SIZE} node(s)..."
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

# ---------------------------------------------------------------- one run
# run_one <slime_root> <bs> <n> <dump_dir> <log_file>
run_one() {
    local slime_root="$1" bs="$2" n="$3" dump_dir="$4" log_file="$5"
    mkdir -p "${dump_dir}"

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
        --rollout-batch-size "${bs}"
        --n-samples-per-prompt "${n}"
        --global-batch-size "$(( bs * n ))"
        --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
        --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
        --rollout-temperature "${ROLLOUT_TEMPERATURE}"
        --rollout-seed 42
    )
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
    [ "${DETERMINISTIC:-0}" != "0" ] && SGLANG_ARGS+=(--sglang-enable-deterministic-inference)

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
    ) > "${log_file}" 2>&1
}

# ---------------------------------------------------------------- sweep loop
bring_up_cluster

RESULTS_CSV="${OUT_DIR}/results.csv"
echo "bs,n,gen_reqs,baseline_times,candidate_times,baseline_mean,candidate_mean,speedup,saved_pct" > "${RESULTS_CSV}"

for point in ${SWEEP}; do
    bs="${point%%:*}"; n="${point##*:}"; genreqs=$(( bs * n ))
    echo ">>> SWEEP POINT bs=${bs} n=${n} (gen_reqs=${genreqs})"

    base_dump="${OUT_DIR}/bs${bs}_n${n}/baseline"; cand_dump="${OUT_DIR}/bs${bs}_n${n}/candidate"
    base_log="${OUT_DIR}/bs${bs}_n${n}_baseline.log"; cand_log="${OUT_DIR}/bs${bs}_n${n}_candidate.log"

    run_one "${BASELINE_WORKTREE}"    "${bs}" "${n}" "${base_dump}" "${base_log}"
    run_one "${CANDIDATE_SLIME_ROOT}" "${bs}" "${n}" "${cand_dump}" "${cand_log}"

    python3 - "${bs}" "${n}" "${genreqs}" "${base_log}" "${cand_log}" "${RESULTS_CSV}" <<'PY'
import re, sys
bs, n, genreqs, base_log, cand_log, csv = sys.argv[1:7]
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
    m = ts[1:] if len(ts) > 1 else ts
    return sum(m)/len(m) if m else float('nan')
b, c = times(base_log), times(cand_log)
bm, cm = mean(b), mean(c)
speedup = (bm/cm) if (c and cm > 0) else float('nan')
saved = (100*(bm-cm)/bm) if (b and bm > 0) else float('nan')
with open(csv, "a") as f:
    f.write(f"{bs},{n},{genreqs},{'|'.join(f'{t:.2f}' for t in b)},"
            f"{'|'.join(f'{t:.2f}' for t in c)},{bm:.2f},{cm:.2f},{speedup:.2f},{saved:.1f}\n")
print(f"    baseline  times: {b}")
print(f"    candidate times: {c}")
print(f"    mean(excl warmup): baseline={bm:.2f}s candidate={cm:.2f}s speedup={speedup:.2f}x saved={saved:.1f}%")
PY
done

cleanup_local

# ---------------------------------------------------------------- summary (markdown)
SUMMARY_MD="${OUT_DIR}/summary.md"
python3 - "${RESULTS_CSV}" "${SUMMARY_MD}" "${NUM_ROLLOUT}" "${WORLD_SIZE}x${NUM_GPUS_PER_NODE}" <<'PY'
import csv, sys
results_csv, summary_md, num_rollout, topo = sys.argv[1:5]
rows = list(csv.DictReader(open(results_csv)))
L = []
L.append("### async load multimodal data - rollout time (debug-rollout-only)")
L.append("")
L.append(f"- cluster: {topo} GPU; each point: {num_rollout} rollouts, step 0 = warm-up, mean over the rest")
L.append("- baseline = synchronous preprocessing; candidate = async load")
L.append("")
L.append("| batch_size | n_samples | gen_reqs | baseline (s) | candidate (s) | speedup | time saved |")
L.append("|-----------:|----------:|---------:|-------------:|--------------:|--------:|-----------:|")
for r in rows:
    L.append(f"| {r['bs']} | {r['n']} | {r['gen_reqs']} | {r['baseline_mean']} | "
             f"{r['candidate_mean']} | {r['speedup']}x | {r['saved_pct']}% |")
L.append("")
L.append("Per-step raw rollout_time (warm-up first):")
L.append("")
L.append("| batch_size | n_samples | baseline steps | candidate steps |")
L.append("|-----------:|----------:|:---------------|:----------------|")
for r in rows:
    L.append(f"| {r['bs']} | {r['n']} | {r['baseline_times']} | {r['candidate_times']} |")
text = "\n".join(L)
open(summary_md, "w").write(text + "\n")
print("\n" + text)
PY

echo "DONE. CSV: ${RESULTS_CSV}   Markdown: ${SUMMARY_MD}"
