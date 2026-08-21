"""Frame / token budget: the single-GPU contract from plan §2.

Everything that bounds context length lives here and is imported by the
prompts, the crop_video tool, the probe, and data prep — one source of truth.

Token math (Qwen3-VL: patch 16, spatial merge 2 => 1 token per 32x32 px block;
videos additionally merge 2 consecutive frames into one temporal group):

- global view : 32 frames @ <=50_176 px  -> 16 groups x 49 tok  ~=  784 tok
- one crop    : 16 images @ <=150_528 px -> 16      x 147 tok   ~= 2352 tok
  (tool frames return as *images*, so no temporal merge applies)
- 3 crops + global + text/reasoning  ~= 784 + 7056 + ~2000  < 12K budget
"""

# --- global coarse view (initial prompt, rendered as native video) ---------
GLOBAL_NUM_FRAMES = 32
GLOBAL_MAX_PIXELS = 50_176  # 49 * 32*32: ~224x224 per frame, coarse by design
GLOBAL_MIN_PIXELS = 3_136   # 56x56 floor

# --- crop_video tool returns (rendered as images) --------------------------
CROP_NUM_FRAMES = 16
CROP_MAX_PIXELS = 150_528   # 147 * 32*32: ~384x384, the "zoomed-in" view
CROP_MIN_PIXELS = 3_136
MIN_CROP_SECONDS = 2.0      # narrower requests are expanded symmetrically

# --- interaction protocol --------------------------------------------------
TOOL_NAME = "crop_video"
MAX_TOOL_CALLS = 3          # plan §2: max turns T = 3
MAX_CONTEXT_TOKENS = 12_288  # documentation of the overall budget (not enforced here)

# --- answer format ---------------------------------------------------------
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
