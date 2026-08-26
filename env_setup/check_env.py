#!/usr/bin/env python
"""Agentic-TVG environment verification. Every FAIL must be fixed before moving on."""

import importlib.metadata as md
import sys
import traceback

OK, FAIL = "\033[1;32m  OK \033[0m", "\033[1;31m FAIL\033[0m"
failed = []


def check(name, fn):
    try:
        detail = fn()
        print(f"[{OK}] {name:<28} {detail}")
    except Exception as e:
        print(f"[{FAIL}] {name:<28} {type(e).__name__}: {e}")
        traceback.print_exc(limit=1, file=sys.stderr)
        failed.append(name)


def torch_check():
    import torch

    assert torch.cuda.is_available(), "CUDA is not available"
    cap = torch.cuda.get_device_capability(0)
    assert cap == (8, 0), f"expected A100 sm80, got sm{cap[0]}{cap[1]}"
    assert torch.__version__.startswith("2.11.0"), torch.__version__
    return f"{torch.__version__} | cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)} sm{cap[0]}{cap[1]}"


def fa_check():
    import torch
    from flash_attn import flash_attn_func

    q = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device="cuda")
    out = flash_attn_func(q, q, q, causal=True)
    assert out.shape == q.shape
    return f"{md.version('flash-attn')} | bf16 forward ok"


def vllm_check():
    v = md.version("vllm")
    assert v == "0.24.0", f"expected 0.24.0, got {v}"
    import vllm  # noqa: F401

    return v


def transformers_check():
    from packaging.version import parse

    v = md.version("transformers")
    assert parse("5.5.3") <= parse(v) < parse("5.11"), f"{v} is outside verl's allowed range [5.5.3, 5.11)"
    assert v != "5.6.0", "verl explicitly excludes 5.6.0"
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401

    return f"{v} | Qwen3VL importable"


def verl_check():
    import verl
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop  # noqa: F401
    from verl.models.transformers.qwen3_vl import fast_pos_embed_interpolate  # noqa: F401
    from verl.tools.base_tool import BaseTool  # noqa: F401
    from verl.utils.attention_utils import index_first_axis  # noqa: F401

    return f"{getattr(verl, '__version__', md.version('verl'))} | qwen3_vl patch + ToolAgentLoop + tools ok"


def video_check():
    """Decode for real -- ENVIRONMENT.md 7.

    Import alone proves nothing: torchcodec imports fine and only fails when it
    dlopens its ffmpeg core, so a smoke decode is the only honest check. Also
    asserts qwen-vl-utils picked torchcodec: falling through to torchvision is
    silent here but fatal later (torchvision 0.26 removed io.read_video).
    """
    import os
    import tempfile

    import av
    import numpy as np
    from qwen_vl_utils import process_vision_info  # noqa: F401
    from qwen_vl_utils.vision_process import get_video_reader_backend
    from torchcodec.decoders import VideoDecoder

    backend = get_video_reader_backend()
    assert backend == "torchcodec", f"qwen-vl-utils picked {backend!r}, not torchcodec"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "probe.mp4")
        with av.open(path, mode="w") as out:
            stream = out.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
            for i in range(10):
                plane = np.full((64, 64, 3), i * 25, dtype=np.uint8)
                out.mux(stream.encode(av.VideoFrame.from_ndarray(plane, format="rgb24")))
            out.mux(stream.encode(None))
        frame = VideoDecoder(path)[0]          # <- the dlopen happens here
    assert frame.shape == (3, 64, 64), frame.shape

    preload = os.environ.get("LD_PRELOAD", "")
    assert "libstdc++" in preload, f"LD_PRELOAD missing libstdc++ (got {preload!r})"
    return (f"qwen-vl-utils {md.version('qwen-vl-utils')} | torchcodec "
            f"{md.version('torchcodec')} decode ok | PyAV {md.version('av')}")


def misc_check():
    from packaging.version import parse

    td, ray_v = md.version("tensordict"), md.version("ray")
    assert parse("0.8.0") <= parse(td) <= parse("0.10.0") and td != "0.9.0", td
    assert parse(ray_v) >= parse("2.41.0"), ray_v
    return f"tensordict {td} | ray {ray_v} | pyarrow {md.version('pyarrow')} | numpy {md.version('numpy')}"


def mem_check():
    import torch

    free, total = torch.cuda.mem_get_info()
    return f"free {free / 2**30:.1f} GiB / total {total / 2**30:.1f} GiB"


for n, f in [
    ("torch + GPU", torch_check),
    ("flash-attn (FA2)", fa_check),
    ("vllm", vllm_check),
    ("transformers", transformers_check),
    ("verl", verl_check),
    ("video stack", video_check),
    ("tensordict/ray/arrow", misc_check),
    ("GPU memory", mem_check),
]:
    check(n, f)

print()
if failed:
    print(f"\033[1;31m{len(failed)} check(s) failed: {', '.join(failed)}\033[0m")
    sys.exit(1)
print("\033[1;32mAll checks passed.\033[0m")
