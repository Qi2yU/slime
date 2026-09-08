"""CPU regression tests for overlapping processor and image-encoding work."""

import asyncio
import copy
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Match the CPU-only rollout tests: no server or model download is needed.
try:
    import transformers  # noqa: F401
except ImportError:
    sys.modules["transformers"] = SimpleNamespace(
        AutoProcessor=object,
        AutoTokenizer=object,
        PreTrainedTokenizerBase=object,
        ProcessorMixin=object,
    )

from slime.rollout import sglang_rollout
from slime.utils import processing_utils
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.fixture
def generation(monkeypatch):
    args = SimpleNamespace(
        ci_test=True,
        sglang_router_ip="localhost",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
    )
    state = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [11, 12]),
        processor=None,
    )
    requests = []

    async def post(url, payload, headers=None):
        requests.append(copy.deepcopy(payload))
        return {
            "text": "answer",
            "meta_info": {"output_token_logprobs": [[-0.1, 13, None]], "finish_reason": {"type": "stop"}},
        }

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "post", post)
    monkeypatch.setattr(
        sglang_rollout,
        "trace_span",
        lambda *args, **kwargs: nullcontext(SimpleNamespace(update=lambda *args: None)),
    )
    return args, state, requests


@pytest.mark.parametrize("lazy_images", [False, True])
def test_processor_overlaps_encoding_and_preserves_image_order(monkeypatch, generation, lazy_images):
    args, state, requests = generation
    images = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
    expected_images = [processing_utils.encode_image_for_rollout_engine(image) for image in images]
    if lazy_images:
        buffers = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            buffers.append(buffer)
        images = [Image.open(buffer) for buffer in buffers]

    # All three blocking calls must start before any can complete. The old
    # processor-before-encoding path breaks this barrier instead of passing.
    started = threading.Barrier(3, timeout=5)
    second_encoded = threading.Event()
    completion_order = []
    expected_pixels = torch.tensor([[255, 0, 0], [0, 0, 255]])
    encode = processing_utils.encode_image_for_rollout_engine

    def processor(*, text, images, **kwargs):
        assert not requests
        started.wait()
        pixels = torch.tensor([image.convert("RGB").getpixel((0, 0)) for image in images])
        return {"input_ids": [[11, 12]], "attention_mask": [[1, 1]], "pixel_values": pixels}

    def encode_out_of_order(image):
        assert not requests
        index = next(i for i, item in enumerate(images) if item is image)
        started.wait()
        if index == 0:
            assert second_encoded.wait(5)
        result = encode(image)
        completion_order.append(index)
        if index == 1:
            second_encoded.set()
        return result

    state.processor = processor
    sample = Sample(prompt="describe the images", multimodal_inputs={"images": images})
    with ThreadPoolExecutor(max_workers=3) as pool:
        monkeypatch.setattr(sglang_rollout, "_MULTIMODAL_EXECUTOR", pool)
        monkeypatch.setattr(processing_utils, "_MULTIMODAL_EXECUTOR", pool)
        monkeypatch.setattr(processing_utils, "encode_image_for_rollout_engine", encode_out_of_order)
        result = asyncio.run(sglang_rollout.generate(args, sample, {"max_new_tokens": 8}))

    assert result is sample
    assert completion_order == [1, 0]
    assert requests == [
        {
            "sampling_params": {"max_new_tokens": 8},
            "return_logprob": True,
            "image_data": expected_images,
            "text": sample.prompt,
        }
    ]
    assert sample.tokens == [11, 12, 13]
    assert sample.response == "answer"
    assert sample.rollout_log_probs == [-0.1]
    torch.testing.assert_close(sample.multimodal_train_inputs["pixel_values"], expected_pixels)


@pytest.mark.parametrize("reuse_tokens", [False, True])
def test_text_request_prepares_input_ids(generation, reuse_tokens):
    args, _, requests = generation
    sample = Sample(prompt="question", tokens=[21, 22] if reuse_tokens else [])
    expected = list(sample.tokens) if reuse_tokens else [11, 12]
    asyncio.run(sglang_rollout.generate(args, sample, {"max_new_tokens": 8}))
    assert requests[0]["input_ids"] == expected
    assert "image_data" not in requests[0]
    assert sample.tokens == expected + [13]


def test_resumed_image_request_reuses_processor_inputs(generation):
    args, state, requests = generation

    def unexpected_processor(**kwargs):
        pytest.fail("A resumed request with cached training inputs must not rerun the processor")

    state.processor = unexpected_processor
    training_inputs = {"pixel_values": torch.tensor([1, 2, 3])}
    image = Image.new("RGB", (1, 1), "red")
    sample = Sample(
        prompt="question",
        tokens=[21, 22, 99],
        response="previous",
        response_length=1,
        rollout_log_probs=[-0.2],
        multimodal_inputs={"images": [image]},
        multimodal_train_inputs=training_inputs,
        status=Sample.Status.ABORTED,
    )
    asyncio.run(sglang_rollout.generate(args, sample, {"max_new_tokens": 8}))
    assert sample.multimodal_train_inputs is training_inputs
    assert sample.tokens == [21, 22, 99, 13]
    assert sample.rollout_log_probs == [-0.2, -0.1]
    assert requests[0]["sampling_params"]["max_new_tokens"] == 7
    assert requests[0]["image_data"] == [processing_utils.encode_image_for_rollout_engine(image)]


def test_zero_budget_preserves_training_inputs_without_encoding(monkeypatch, generation):
    args, state, requests = generation
    pixels = torch.tensor([1, 2, 3])
    state.processor = lambda **kwargs: {"input_ids": [[11, 12]], "pixel_values": pixels}

    async def unexpected_encode(image):
        pytest.fail("A zero-budget request must not submit image encoding")

    monkeypatch.setattr(sglang_rollout, "async_encode_image_for_rollout_engine", unexpected_encode)
    sample = Sample(prompt="question", multimodal_inputs={"images": [object()]}, response_length=8)
    result = asyncio.run(sglang_rollout.generate(args, sample, {"max_new_tokens": 8}))
    assert result.status == Sample.Status.TRUNCATED
    assert result.multimodal_train_inputs["pixel_values"] is pixels
    assert result.tokens == []
    assert requests == []


@pytest.mark.parametrize("failure", ["processor", "image", "cancel"])
def test_preparation_failure_or_cancellation_drains_tasks(monkeypatch, generation, failure):
    args, _, requests = generation

    async def run():
        started = set()
        finished = set()
        all_started = asyncio.Event()

        async def prepare(name):
            started.add(name)
            if len(started) == 3:
                all_started.set()
            try:
                await all_started.wait()
                if name == failure:
                    raise ValueError("preparation failed")
                await asyncio.Event().wait()
            finally:
                finished.add(name)

        async def processor(*args):
            return await prepare("processor")

        monkeypatch.setattr(sglang_rollout, "_prepare_prompt_ids_async", processor)

        async def encode(image):
            return await prepare(image.info["name"])

        monkeypatch.setattr(sglang_rollout, "async_encode_image_for_rollout_engine", encode)
        images = [Image.new("RGB", (1, 1)) for _ in range(2)]
        for image, name in zip(images, ("image", "other-image"), strict=True):
            image.info["name"] = name
        sample = Sample(prompt="question", multimodal_inputs={"images": images})
        task = asyncio.create_task(sglang_rollout.generate(args, sample, {"max_new_tokens": 8}))
        if failure == "cancel":
            await asyncio.wait_for(all_started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ValueError, match="preparation failed"):
                await asyncio.wait_for(task, timeout=5)
        assert finished == {"processor", "image", "other-image"}
        assert requests == []

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
