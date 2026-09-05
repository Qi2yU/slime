"""CPU tests for asynchronous multimodal rollout preprocessing."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from slime.rollout import sglang_rollout
from slime.utils import processing_utils
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.unit
def test_async_image_encoding_matches_sync_and_runs_off_event_loop(monkeypatch):
    image = Image.new("RGB", (2, 2), color="red")
    expected = processing_utils.encode_image_for_rollout_engine(image)
    event_loop_thread = threading.get_ident()
    worker_threads = []
    original_encode = processing_utils.encode_image_for_rollout_engine

    def tracked_encode(value):
        worker_threads.append(threading.get_ident())
        return original_encode(value)

    monkeypatch.setattr(processing_utils, "encode_image_for_rollout_engine", tracked_encode)

    actual = asyncio.run(processing_utils.async_encode_image_for_rollout_engine(image))

    assert actual == expected
    assert worker_threads and worker_threads[0] != event_loop_thread


@pytest.mark.unit
def test_async_prompt_preparation_runs_processor_off_event_loop():
    event_loop_thread = threading.get_ident()
    processor_threads = []

    def processor(*, text, **kwargs):
        processor_threads.append(threading.get_ident())
        assert text == "prompt"
        assert kwargs["images"] == ["image"]
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "pixel_values": "pixels",
        }

    sample = Sample(prompt="prompt", multimodal_inputs={"images": ["image"]})
    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: pytest.fail("tokenizer should not be called"))

    prompt_ids = asyncio.run(sglang_rollout._prepare_prompt_ids_async(sample, tokenizer, processor))

    assert prompt_ids == [1, 2, 3]
    assert sample.multimodal_train_inputs == {"pixel_values": "pixels"}
    assert processor_threads and processor_threads[0] != event_loop_thread


@pytest.mark.unit
def test_async_prompt_preparation_preserves_text_only_fast_path():
    encode_calls = []
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: encode_calls.append((text, add_special_tokens)) or [4, 5]
    )
    processor = lambda **_kwargs: pytest.fail("processor should not be called")
    sample = Sample(prompt="text only")

    prompt_ids = asyncio.run(sglang_rollout._prepare_prompt_ids_async(sample, tokenizer, processor))

    assert prompt_ids == [4, 5]
    assert encode_calls == [("text only", False)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
