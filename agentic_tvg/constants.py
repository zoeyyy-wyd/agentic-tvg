"""Frame / token budget: the single-GPU contract from plan §2.

Everything that bounds context length lives here and is imported by the
prompts, the crop_video tool, the probe, and data prep — one source of truth.

Token math (Qwen3-VL: patch 16, spatial merge 2 => 1 token per 32x32 px block;
videos additionally merge 2 consecutive frames into one temporal group):

- global view : 128 frames @ <=50_176 px -> 64 groups x ~45-49 tok + stamps
                ~= 3,944 tok measured    -> prompt cap 4,608
- one crop    : 30 images @ <=150_528 px -> 30 x ~94-147 tok ~= 2.9-4.6K tok
  (tool frames return as *images*, so no temporal merge applies)
- 3 crops + reasoning fits the 16,384 response cap (worst ~14.7K + text);
  total context 4,608 + 16,384 = 20,992. Budgets and the sweep behind these
  numbers: FRAMES_SWEEP.md (2026-08-26).
"""

# --- global coarse view (initial prompt, rendered as native video) ---------
# 128 (2026-08-26, was 64): evidence-window coverage at 64 left 12.4% of RL
# questions with <3 global frames in-window; 128 cuts that to 1.7% (median 10
# in-window frames) at +23% step time. Our videos are short (median 205 s), so
# 128 ~= 0.62 fps -- denser than the paper's 512 frames on their long videos.
# Sweep evidence: FRAMES_SWEEP.md.
GLOBAL_NUM_FRAMES = 128
GLOBAL_MAX_PIXELS = 50_176  # 49 * 32*32: ~224x224 per frame, coarse by design
GLOBAL_MIN_PIXELS = 3_136   # 56x56 floor

# --- crop_video tool returns (rendered as images) --------------------------
# 30 (2026-08-26, was 16): LongVT's own tool samples crops at 1 fps (measured
# from selftrace: n_frames ~= window seconds, median window 31 s ~= 30 frames);
# fixed 30 matches that typical density while keeping the schema/prompt shape
# and the response budget static (3 crops x 30 x ~147 tok worst ~= 14.7K).
CROP_NUM_FRAMES = 30
CROP_MAX_PIXELS = 150_528   # 147 * 32*32: ~384x384, the "zoomed-in" view
CROP_MIN_PIXELS = 3_136
MIN_CROP_SECONDS = 2.0      # narrower requests are expanded symmetrically

# --- interaction protocol --------------------------------------------------
TOOL_NAME = "crop_video"
MAX_TOOL_CALLS = 3          # plan §2: max turns T = 3
MAX_CONTEXT_TOKENS = 20_992  # documentation: 4608 prompt + 16384 response (not enforced here)

# --- answer format ---------------------------------------------------------
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
