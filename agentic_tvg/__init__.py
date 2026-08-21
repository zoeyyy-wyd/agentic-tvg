"""Agentic temporal video grounding with verifiable rewards.

Core library shared by the Step-0 probe, verl GRPO training, and evaluation:

- constants:  frame / token budget (single-GPU contract, see plan §2)
- span:       answer-span parsing and temporal IoU
- reward:     verl custom reward functions (vanilla and penalty-aware)
- prompts:    system / user prompt builders for all probe modes and training
- video_frames: PyAV interval frame sampling used by both the tool and the probe

`agentic_tvg.crop_video_tool` and `agentic_tvg.sft_dataset` are intentionally
not imported here: they depend on verl, while everything above stays importable
in any env. (The isolation mechanism is exactly this non-import — Python only
loads a module when something names it.)
"""

__version__ = "0.1.0"
