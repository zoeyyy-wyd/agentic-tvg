"""Regression tests for judge_v2's incremental, cross-process-visible cache
(2026-09-01: the first cold-cache val duplicated 47% of its judge calls
because each worker loaded the file once and never saw later appends)."""

import importlib
import json

import pytest


@pytest.fixture
def jv(tmp_path):
    """judge_v2 reloaded against a temp cache; restored for other tests after."""
    import agentic_tvg.judge_v2 as mod
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JUDGE_CACHE", str(tmp_path / "cache.jsonl"))
        mp.setenv("JUDGE_DISABLE", "1")   # never reach the API in tests
        yield importlib.reload(mod), tmp_path / "cache.jsonl"
    importlib.reload(mod)                 # back to the real cache path


def _row(mod, key, verdict, **over):
    rec = {"k": key, "v": verdict, "q": "q", "gt": "gt", "a": "a",
           "model": mod.JUDGE_MODEL, "rubric": "v2"} | over
    return json.dumps(rec) + "\n"


def test_sees_appends_made_after_first_load(jv):
    mod, path = jv
    with mod._lock:
        assert mod._load_cache() == {}            # first load, empty file
    key = mod._key("who?", "a red flag", "a red flag")
    with path.open("a") as f:                     # "another worker" lands one
        f.write(_row(mod, key, 1.0))
    with mod._lock:
        assert mod._load_cache().get(key) == 1.0  # refresh saw it -- no reload-from-scratch


def test_first_verdict_wins_over_later_duplicates(jv):
    mod, path = jv
    key = mod._key("q", "gt", "ans")
    with path.open("a") as f:
        f.write(_row(mod, key, 0.5))
        f.write(_row(mod, key, 1.0))              # racing duplicate, different draw
    with mod._lock:
        assert mod._load_cache()[key] == 0.5      # first append is the verdict, ever after


def test_partial_line_deferred_until_complete(jv):
    mod, path = jv
    k1, k2 = mod._key("q1", "g", "a"), mod._key("q2", "g", "a")
    line2 = _row(mod, k2, 1.0)
    with path.open("a") as f:                     # a writer caught mid-append
        f.write(_row(mod, k1, 1.0) + line2[:10])
    with mod._lock:
        cache = mod._load_cache()
        assert cache.get(k1) == 1.0 and k2 not in cache
    with path.open("a") as f:                     # append completes
        f.write(line2[10:])
    with mod._lock:
        assert mod._load_cache().get(k2) == 1.0   # tail consumed exactly once


def test_other_instruments_rows_never_load(jv):
    mod, path = jv
    key = mod._key("q", "gt", "ans")
    with path.open("a") as f:
        f.write(_row(mod, key, 1.0, model="claude-haiku-4-5", rubric=None))
        f.write(_row(mod, key, 1.0, rubric="v1"))
    with mod._lock:
        assert key not in mod._load_cache()
