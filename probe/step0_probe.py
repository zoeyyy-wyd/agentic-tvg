#!/usr/bin/env python
"""Step-0 zero-shot agentic probe (plan §3): Direct / Tool-optional / Tool-forced.

Runs Qwen3-VL against a prepared TVG parquet through a vLLM OpenAI-compatible
server (serve_qwen3vl.sh), with the crop_video tool executed client-side
by the *same* sampling code the verl tool uses (agentic_tvg.video_frames).

Formatting note: here video frames are sent as base64 images each preceded by
a "[12.3s]" text label — in verl training the initial video goes through the
processor's native video path instead. Interaction protocol, budgets, and the
tool are identical; absolute mIoU may shift slightly between the two renderings.

Outputs under --out:
- results_<mode>.jsonl   one record per item (resumable: existing indices skipped)
- summary.csv / stdout   per-mode: mIoU, R@0.3/0.5/0.7, format rate,
                         tool-call rate, avg calls, window-revision gain

Usage:
    python probe/step0_probe.py --data data/processed/rl_val.parquet \
        --modes direct,tool_optional,tool_forced --limit 100 --out outputs/step0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_tvg.constants import (  # noqa: E402
    CROP_MAX_PIXELS,
    CROP_MIN_PIXELS,
    CROP_NUM_FRAMES,
    GLOBAL_MAX_PIXELS,
    GLOBAL_MIN_PIXELS,
    GLOBAL_NUM_FRAMES,
    MAX_TOOL_CALLS,
    TOOL_NAME,
)
from agentic_tvg.prompts import MODES, build_system_prompt  # noqa: E402
from agentic_tvg.span import parse_answer_span, temporal_iou  # noqa: E402
from agentic_tvg.video_frames import sample_frames  # noqa: E402

try:  # single source of truth for the tool schema; fall back if verl is absent
    from agentic_tvg.crop_video_tool import build_crop_video_schema

    TOOL_SCHEMA = build_crop_video_schema().model_dump(exclude_none=True)
except ImportError:
    TOOL_SCHEMA = None  # probe refuses tool modes in this case


# --------------------------------------------------------------------------
# message building
# --------------------------------------------------------------------------

def frames_content(path: str, start: float, end: float, n: int, max_px: int, min_px: int) -> tuple[list[dict], list[float]]:
    """Interleaved [timestamp text, image] content parts for one interval."""
    frames, ts = sample_frames(path, start, end, n, max_px, min_px)
    parts: list[dict] = []
    for t, img in zip(ts, frames):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        parts.append({"type": "text", "text": f"[{t:.1f}s]"})
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts, ts


def initial_messages(item: dict, mode: str) -> list[dict]:
    parts, _ = frames_content(
        item["video_path"], 0.0, item["duration"], GLOBAL_NUM_FRAMES, GLOBAL_MAX_PIXELS, GLOBAL_MIN_PIXELS
    )
    user_content = (
        [{"type": "text", "text": f"The video is {item['duration']:.1f} seconds long. Global frames:"}]
        + parts
        + [{"type": "text", "text": f'Query: "{item["query"]}"\nFind the time interval when this happens in the video.'}]
    )
    return [
        {"role": "system", "content": build_system_prompt(mode)},
        {"role": "user", "content": user_content},
    ]


def tool_result_messages(item: dict, args: dict, style: str) -> tuple[list[dict], dict]:
    """Execute crop_video locally; return messages to append + a trace record."""
    try:
        start, end = float(args["start_time"]), float(args["end_time"])
    except (KeyError, TypeError, ValueError):
        text = "Error: crop_video requires numeric start_time and end_time."
        msg = {"role": "tool", "content": text} if style == "tool" else {"role": "user", "content": f"<tool_response>{text}</tool_response>"}
        return [msg], {"error": "bad_args", "raw_args": args}

    start = min(max(start, 0.0), item["duration"])
    end = min(max(end, 0.0), item["duration"])
    if end <= start:
        text = f"Error: invalid interval [{start:.1f}, {end:.1f}]."
        msg = {"role": "tool", "content": text} if style == "tool" else {"role": "user", "content": f"<tool_response>{text}</tool_response>"}
        return [msg], {"error": "bad_window", "window": [start, end]}

    parts, _ = frames_content(item["video_path"], start, end, CROP_NUM_FRAMES, CROP_MAX_PIXELS, CROP_MIN_PIXELS)
    label = {"type": "text", "text": f"{CROP_NUM_FRAMES} frames from {start:.1f}s to {end:.1f}s:"}
    if style == "tool":
        msg = {"role": "tool", "content": [label] + parts}
    else:  # "user": wrap as a tool_response block in a user turn
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "<tool_response>"}, label] + parts + [{"type": "text", "text": "</tool_response>"}],
        }
    return [msg], {"window": [start, end]}


# --------------------------------------------------------------------------
# server client
# --------------------------------------------------------------------------

class Client:
    def __init__(self, base_url: str, model: str, temperature: float, timeout: float):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, messages: list[dict], tools: list | None, tool_choice: str | None, max_tokens: int = 1024) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        r = self.session.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]


# --------------------------------------------------------------------------
# one rollout
# --------------------------------------------------------------------------

def run_item(client: Client, item: dict, mode: str, max_calls: int, tool_style: str) -> dict:
    t0 = time.time()
    messages = initial_messages(item, mode)
    tools = [TOOL_SCHEMA] if mode != "direct" else None
    crop_windows: list[list[float]] = []
    n_calls = 0
    final_text = ""

    for turn in range(max_calls + 1):
        tool_choice = None
        if tools:
            # force at least one call in tool_forced mode; stop offering tools when exhausted
            if mode == "tool_forced" and n_calls == 0:
                tool_choice = "required"
            if n_calls >= max_calls:
                tools_now = None
            else:
                tools_now = tools
        else:
            tools_now = None

        try:
            msg = client.chat(messages, tools_now, tool_choice)
        except requests.HTTPError as e:
            if tool_choice == "required":  # server may not support it -> prompt-level forcing only
                msg = client.chat(messages, tools_now, None)
            else:
                raise e

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls or n_calls >= max_calls:
            final_text = msg.get("content") or ""
            break

        # keep the assistant message exactly as returned (needed for the template)
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        call = tool_calls[0]
        try:
            args = json.loads(call["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            args = {}
        new_msgs, rec = tool_result_messages(item, args, tool_style)
        if tool_style == "tool":
            new_msgs[0]["tool_call_id"] = call.get("id", f"call_{n_calls}")
        messages.extend(new_msgs)
        if "window" in rec and "error" not in rec:
            crop_windows.append(rec["window"])
        n_calls += 1
    else:
        final_text = ""

    parsed = parse_answer_span(final_text)
    gt = tuple(item["gt"])
    iou = temporal_iou(parsed.span if parsed.valid else None, gt)
    first_window_iou = temporal_iou(tuple(crop_windows[0]), gt) if crop_windows else None

    return {
        "index": item["index"],
        "mode": mode,
        "query": item["query"],
        "gt": list(gt),
        "duration": item["duration"],
        "pred": list(parsed.span) if parsed.span else None,
        "iou": round(iou, 4),
        "format_ok": parsed.format_ok,
        "answer_source": parsed.source,
        "n_tool_calls": n_calls,
        "crop_windows": crop_windows,
        "first_window_iou": round(first_window_iou, 4) if first_window_iou is not None else None,
        "final_text": final_text[-2000:],
        "wall_seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------
# dataset + metrics
# --------------------------------------------------------------------------

def load_items(parquet: Path, limit: int | None, seed: int) -> list[dict]:
    df = pd.read_parquet(parquet)
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).sort_index()
    items = []
    for _, r in df.iterrows():
        info = r["extra_info"]
        items.append(
            {
                "index": int(info["index"]),
                "video_path": info["video_path"],
                "duration": float(info["duration"]),
                "query": info["question"],
                "gt": [float(x) for x in r["reward_model"]["ground_truth"]],
            }
        )
    return items


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    ious = [r["iou"] for r in records]
    with_calls = [r for r in records if r["n_tool_calls"] > 0]
    revised = [r["iou"] - r["first_window_iou"] for r in with_calls if r.get("first_window_iou") is not None]
    return {
        "n": n,
        "mIoU": round(sum(ious) / n, 4),
        "R@0.3": round(sum(i >= 0.3 for i in ious) / n, 4),
        "R@0.5": round(sum(i >= 0.5 for i in ious) / n, 4),
        "R@0.7": round(sum(i >= 0.7 for i in ious) / n, 4),
        "format_rate": round(sum(r["format_ok"] for r in records) / n, 4),
        "answered_rate": round(sum(r["pred"] is not None for r in records) / n, 4),
        "tool_call_rate": round(len(with_calls) / n, 4),
        "avg_calls": round(sum(r["n_tool_calls"] for r in records) / n, 2),
        "revision_gain": round(sum(revised) / len(revised), 4) if revised else None,
        "avg_wall_s": round(sum(r["wall_seconds"] for r in records) / n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="prepared parquet (extract_rl.py / prepare_charades.py output)")
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3-vl-4b", help="served model name")
    ap.add_argument("--modes", default="direct,tool_optional,tool_forced")
    ap.add_argument("--limit", type=int, default=None, help="subsample this many items (fixed seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/step0"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--tool-msg-style", choices=["tool", "user"], default="tool",
                    help="how tool results are sent back: role=tool parts (verl-style) or a user turn wrapping <tool_response>")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r}")
    if any(m != "direct" for m in modes) and TOOL_SCHEMA is None:
        raise SystemExit("verl not importable -> tool schema unavailable; run inside the `verl` env")

    items = load_items(args.data, args.limit, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    client = Client(args.server, args.model, args.temperature, args.timeout)
    print(f"{len(items)} items from {args.data}, modes={modes}, tool_msg_style={args.tool_msg_style}")

    all_summaries = {}
    for mode in modes:
        out_file = args.out / f"results_{mode}.jsonl"
        done: set[int] = set()
        records: list[dict] = []
        if out_file.exists():
            for line in out_file.read_text().splitlines():
                rec = json.loads(line)
                if "error" not in rec:
                    done.add(rec["index"])
                    records.append(rec)
        todo = [it for it in items if it["index"] not in done]
        print(f"== mode={mode}: {len(done)} done, {len(todo)} to run")

        def _run(it, mode=mode):
            try:
                return run_item(client, it, mode, args.max_tool_calls, args.tool_msg_style)
            except Exception as e:
                traceback.print_exc()
                return {"index": it["index"], "mode": mode, "error": f"{type(e).__name__}: {e}"}

        with out_file.open("a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i, rec in enumerate(pool.map(_run, todo), 1):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if "error" not in rec:
                    records.append(rec)
                if i % 10 == 0 or i == len(todo):
                    print(f"   {mode}: {i}/{len(todo)}  (running mIoU {summarize(records).get('mIoU')})")

        all_summaries[mode] = summarize(records)
        print(f"== {mode}: {all_summaries[mode]}")

    summary_df = pd.DataFrame(all_summaries).T
    summary_df.to_csv(args.out / "summary.csv")
    print("\n" + summary_df.to_markdown())


if __name__ == "__main__":
    main()
