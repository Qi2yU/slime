# Verifying the async multimodal preprocessing change

The change (`support async load multimodal data`) offloads two blocking pieces
of rollout preprocessing to a thread pool:

1. the HF processor call in `_prepare_prompt_ids` → `_prepare_prompt_ids_async`
   (`slime/rollout/sglang_rollout.py`);
2. PNG/base64 image encoding, `encode_image_for_rollout_engine` →
   `async_encode_image_for_rollout_engine`, gathered concurrently for multi-image
   samples (`slime/utils/processing_utils.py`).

Goal of the verification: **prove the rollout result and the bytes sent to
SGLang are unchanged — the change only makes rollout faster.**

The synchronous helpers (`_prepare_prompt_ids`, `encode_image_for_rollout_engine`)
still live in the tree, so they serve as the ground-truth reference at every
layer.

## The ladder (cheap → expensive)

| Layer | What it proves | Needs | How to run |
|------|----------------|-------|------------|
| **L1** pure-CPU unit | async == sync for every branch; `gather` keeps image order under reversed completion; concurrent samples don't cross-contaminate; work runs off the event loop | nothing (fakes) | `pytest tests/test_async_multimodal_preprocessing.py -m unit` |
| **L2** real processor | with a real HF VL processor, `prompt_ids` + `multimodal_train_inputs` tensors are `torch.equal`, including under 64-way concurrency (HF thread-safety shakeout) | `transformers` + a local VL checkpoint | `SLIME_TEST_VL_CKPT=/root/models/Qwen2.5-VL-3B-Instruct pytest tests/test_async_multimodal_real_processor.py` |
| **L3** request parity | the payload `generate()` sends to SGLang (`input_ids` / `image_data` order+content / `text` / `sampling_params`) is identical to the sync reference | slime importable (no GPU/model) | `pytest tests/test_async_multimodal_request_parity.py -m unit` |
| **L4** E2E speed + parity | in `--debug-rollout-only`, baseline vs candidate produce byte-for-byte identical dumps, and candidate is faster | GPU box + model + dataset | `bash tools/compare_async_multimodal_rollout.sh` |

Run L1→L3 first; they are deterministic and hardware-free and pin down the
correctness. L4 is the end-to-end confirmation plus the speed number.

## L4 details

`compare_async_multimodal_rollout.sh`:

* checks out the baseline commit (`BASELINE_REF`, default `4c193f1`) into a git
  worktree and runs the current tree as the candidate;
* runs each in `--debug-rollout-only` with `--sglang-enable-deterministic-inference`,
  greedy sampling (`--rollout-temperature 0.0`), no shuffle and a fixed
  `--rollout-seed`, a **single SGLang engine** (deterministic DP routing) and no
  speculative decoding — so both runs see identical inputs and produce
  reproducible outputs;
* saves `--save-debug-rollout-data` dumps for each and reports slime's own
  `perf/rollout_time` per step (step 0 is warm-up);
* diffs the dumps with `diff_rollout_dumps.py`.

`diff_rollout_dumps.py` aligns samples by `index` and checks three groups:

* `request`   — the prompt-token prefix (always fatal on mismatch);
* `processor` — `multimodal_train_inputs` tensors (always fatal);
* `response`  — generated `tokens` / `response` / `response_length` / `reward`
  (fatal unless `--allow-response-diff`, for when bitwise SGLang determinism is
  not available in your environment).

```
python tools/diff_rollout_dumps.py baseline/rollout_1.pt candidate/rollout_1.pt
```

A green L1–L3 plus an L4 that shows identical dumps and a lower `rollout_time`
is the full proof: same requests, same HF-processor outputs, same results — only
faster.
