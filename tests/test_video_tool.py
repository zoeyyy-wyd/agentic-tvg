import asyncio

import av
import numpy as np
import pytest

from agentic_tvg.video_frames import get_video_duration, sample_frames

DURATION_S = 10
FPS = 4


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """Synthetic 10s clip whose frame color encodes time (red ramps up)."""
    path = tmp_path_factory.mktemp("vid") / "synthetic.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        for i in range(DURATION_S * FPS):
            arr = np.zeros((96, 128, 3), dtype=np.uint8)
            arr[..., 0] = int(255 * i / (DURATION_S * FPS))
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def test_duration(video):
    assert abs(get_video_duration(video) - DURATION_S) < 0.5


def test_sample_frames_count_and_timestamps(video):
    frames, ts = sample_frames(video, 2.0, 8.0, 8, max_pixels=150_528, min_pixels=3_136)
    assert len(frames) == 8 and len(ts) == 8
    assert ts == sorted(ts)
    assert abs(ts[0] - 2.0) < 0.5 and abs(ts[-1] - 8.0) < 0.5
    w, h = frames[0].size
    assert w % 32 == 0 and h % 32 == 0 and w * h <= 150_528


def test_sample_frames_full_video_and_past_eof(video):
    frames, ts = sample_frames(video, 0.0, DURATION_S + 5, 6, max_pixels=50_176, min_pixels=3_136)
    assert len(frames) == 6  # EOF targets repeat the last frame instead of vanishing


def test_upscale_to_min_pixels(video):
    frames, _ = sample_frames(video, 0.0, 5.0, 2, max_pixels=150_528, min_pixels=50_176)
    w, h = frames[0].size
    assert w * h >= 50_176 * 0.8  # near the floor after 32-alignment


def test_bad_interval_raises(video):
    with pytest.raises(ValueError):
        sample_frames(video, 5.0, 5.0, 4, 150_528, 3_136)


# ---------------- CropVideoTool lifecycle (requires verl) ----------------

verl = pytest.importorskip("verl")
from agentic_tvg.crop_video_tool import CropVideoTool  # noqa: E402


def _make_tool(**config):
    return CropVideoTool(config={"type": "native", "num_frames": 4, **config}, tool_schema=None)


def test_tool_schema():
    schema = _make_tool().get_openai_tool_schema()
    assert schema.function.name == "crop_video"
    assert set(schema.function.parameters.required) == {"start_time", "end_time"}


def test_tool_execute(video):
    async def run():
        tool = _make_tool()
        iid, _ = await tool.create(create_kwargs={"video_path": video, "duration": float(DURATION_S)})
        resp, reward, metrics = await tool.execute(iid, {"start_time": 2.0, "end_time": 8.0})
        assert resp.image and len(resp.image) == 4
        assert "2.0s to 8.0s" in resp.text and "timestamps" in resp.text
        assert metrics["window"] == [2.0, 8.0]
        await tool.release(iid)
        assert iid not in tool._instances

    asyncio.run(run())


def test_tool_clamps_and_expands(video):
    async def run():
        tool = _make_tool(min_crop_seconds=2.0)
        iid, _ = await tool.create(create_kwargs={"video_path": video, "duration": float(DURATION_S)})
        # out of range -> clamped to duration; narrow -> expanded
        resp, _, metrics = await tool.execute(iid, {"start_time": 9.5, "end_time": 99.0})
        assert metrics["window"][1] == DURATION_S and "expanded" in resp.text
        # invalid after clamp -> error text, no frames, no crash
        resp2, _, m2 = await tool.execute(iid, {"start_time": 50, "end_time": 60})
        assert resp2.image is None and "Error" in resp2.text

    asyncio.run(run())


def test_tool_bad_args_and_missing_video(video):
    async def run():
        tool = _make_tool()
        iid, _ = await tool.create(create_kwargs={"video_path": video, "duration": 10.0})
        resp, _, _ = await tool.execute(iid, {"start_time": "abc", "end_time": None})
        assert "Error" in resp.text
        iid2, _ = await tool.create(create_kwargs={})
        resp2, _, _ = await tool.execute(iid2, {"start_time": 0, "end_time": 5})
        assert "no video" in resp2.text

    asyncio.run(run())
