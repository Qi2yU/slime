#!/usr/bin/env python3
"""Compare two ``--save-debug-rollout-data`` dumps for semantic parity.

Used to prove that the async multimodal preprocessing changes the *speed* of the
rollout but not its *result*. Point it at a baseline dump and a new dump produced
from the same config (same seed, no shuffle, deterministic inference) and it
checks, per sample (aligned by ``index``):

  request   : the prompt-token prefix sent to SGLang            (must always match)
  processor : ``multimodal_train_inputs`` from the HF processor  (must always match)
  response  : generated text / tokens / response_length / reward (match iff
              SGLang inference was deterministic; gate with --allow-response-diff)

Exit code is non-zero if any *fatal* group mismatches.

Usage::

    python tools/diff_rollout_dumps.py BASE.pt NEW.pt
    python tools/diff_rollout_dumps.py BASE.pt NEW.pt --allow-response-diff
"""

from __future__ import annotations

import argparse
import sys

import torch

try:
    import numpy as np
except ImportError:  # numpy is a hard dep of slime, but keep the tool standalone
    np = None


def _load_samples(path: str) -> dict[int, dict]:
    payload = torch.load(path, weights_only=False)
    samples = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    by_index: dict[int, dict] = {}
    for i, sample in enumerate(samples):
        index = sample.get("index")
        if index is None:
            index = i  # fall back to positional alignment
        by_index[index] = sample
    return by_index


def _deep_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.is_floating_point():
            return torch.equal(a, b)  # exact: preprocessing is deterministic
        return torch.equal(a, b)
    if np is not None and (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def _prompt_tokens(sample: dict) -> list:
    tokens = sample.get("tokens") or []
    response_length = sample.get("response_length") or 0
    if response_length and response_length <= len(tokens):
        return list(tokens[: len(tokens) - response_length])
    return list(tokens)


def _short(value, limit=80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _describe_mismatch(field, a, b) -> str:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return f"{field}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        if a.dtype != b.dtype:
            return f"{field}: dtype {a.dtype} != {b.dtype}"
        diff = (a != b).sum().item()
        return f"{field}: {diff} differing elements (max abs diff {(a - b).abs().max().item():.3e})"
    if isinstance(a, list) and isinstance(b, list) and len(a) != len(b):
        return f"{field}: len {len(a)} != {len(b)}"
    return f"{field}: {_short(a)} != {_short(b)}"


# group -> list of (field_name, extractor)
GROUPS = {
    "request": [("prompt_tokens", _prompt_tokens)],
    "processor": [("multimodal_train_inputs", lambda s: s.get("multimodal_train_inputs"))],
    "response": [
        ("tokens", lambda s: s.get("tokens")),
        ("response", lambda s: s.get("response")),
        ("response_length", lambda s: s.get("response_length")),
        ("reward", lambda s: s.get("reward")),
    ],
}
FATAL_GROUPS_ALWAYS = {"request", "processor"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", help="baseline dump .pt (original synchronous code)")
    parser.add_argument("candidate", help="candidate dump .pt (async code)")
    parser.add_argument(
        "--allow-response-diff",
        action="store_true",
        help="do not fail on generation differences (use when SGLang was not run deterministically)",
    )
    parser.add_argument("--max-report", type=int, default=10, help="max mismatching samples to print per field")
    args = parser.parse_args()

    base = _load_samples(args.baseline)
    cand = _load_samples(args.candidate)

    print(f"baseline: {args.baseline}  ({len(base)} samples)")
    print(f"candidate: {args.candidate}  ({len(cand)} samples)")

    base_idx, cand_idx = set(base), set(cand)
    if base_idx != cand_idx:
        only_base = sorted(base_idx - cand_idx)[:20]
        only_cand = sorted(cand_idx - base_idx)[:20]
        print("FATAL: sample index sets differ.")
        if only_base:
            print(f"  only in baseline: {only_base}")
        if only_cand:
            print(f"  only in candidate: {only_cand}")
        return 2
    shared = sorted(base_idx)

    fatal_mismatch = False
    soft_mismatch = False
    for group, fields in GROUPS.items():
        group_fatal = group in FATAL_GROUPS_ALWAYS or not args.allow_response_diff
        for field, extract in fields:
            mismatches = []
            for index in shared:
                a, b = extract(base[index]), extract(cand[index])
                if not _deep_equal(a, b):
                    mismatches.append((index, a, b))
            status = "OK" if not mismatches else ("MISMATCH" if group_fatal else "DIFF")
            tag = "" if group_fatal else " (non-fatal)"
            print(f"[{group:9}] {field:24} {status}{tag}  ({len(shared) - len(mismatches)}/{len(shared)} match)")
            for index, a, b in mismatches[: args.max_report]:
                print(f"    idx={index}: {_describe_mismatch(field, a, b)}")
            if mismatches:
                if group_fatal:
                    fatal_mismatch = True
                else:
                    soft_mismatch = True

    print()
    if fatal_mismatch:
        print("RESULT: FAIL — the async path changed rollout semantics.")
        return 1
    if soft_mismatch:
        print("RESULT: PASS (request + processor identical); response differs — expected only if inference was non-deterministic.")
    else:
        print("RESULT: PASS — byte-for-byte identical rollout results. The change only affects speed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
