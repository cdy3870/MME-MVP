"""Result cache (EVL-9): reuse answers by provider, model, request and replica."""

from __future__ import annotations

import asyncio
import os
import stat

import pytest

from migration_eval.cache import ResultCache, cache_key, cached
from migration_eval.models.record import ModelRequest
from migration_eval.replay import RateLimited, ReplayResult


def request(text="hi", temperature=None) -> ModelRequest:
    body = {"messages": [{"role": "user", "content": text}]}
    if temperature is not None:
        body["parameters"] = {"temperature": temperature}
    return ModelRequest.model_validate(body)


def result(text="ok", error=None, **kwargs) -> ReplayResult:
    return ReplayResult(text, kwargs.pop("tool_calls", None), 10, 5, 100.0, error, **kwargs)


# --- cache_key ----------------------------------------------------------------------------


def test_the_key_is_a_stable_sha256():
    key = cache_key("openai", "gpt-4o", request())
    assert key == cache_key("openai", "gpt-4o", request()) and len(key) == 64 and int(key, 16) >= 0


@pytest.mark.parametrize("other", [
    ("anthropic", "gpt-4o", request(), 0),
    ("openai", "gpt-4o-mini", request(), 0),
    ("openai", "gpt-4o", request("hello"), 0),
    ("openai", "gpt-4o", request(temperature=0.5), 0),
    ("openai", "gpt-4o", request(), 1),
])  # fmt: skip
def test_any_change_to_provider_model_request_settings_or_replica_changes_the_key(other):
    assert cache_key("openai", "gpt-4o", request()) != cache_key(*other)


def test_the_replica_defaults_to_zero():
    assert cache_key("openai", "m", request()) == cache_key("openai", "m", request(), replica=0)


# --- ResultCache --------------------------------------------------------------------------


def test_a_stored_result_comes_back_equal(tmp_path):
    cache = ResultCache(tmp_path / "c")
    stored = result("answer", tool_calls=[{"name": "f", "arguments": {"a": 1}}], converted=True, unconverted_fields=("x: y",))
    cache.put("abc123", stored)
    assert cache.get("abc123") == stored


def test_a_miss_is_none(tmp_path):
    assert ResultCache(tmp_path).get("nothere") is None


def test_the_directory_is_created_and_results_survive_a_new_cache_object(tmp_path):
    ResultCache(tmp_path / "a" / "b").put("k1", result("kept"))
    assert ResultCache(tmp_path / "a" / "b").get("k1").text == "kept"


def test_failed_results_are_not_stored(tmp_path):
    cache = ResultCache(tmp_path)
    cache.put("k1", result(None, error="HTTP 500"))
    assert cache.get("k1") is None and list(tmp_path.iterdir()) == []


def test_a_damaged_file_is_a_miss_not_a_crash(tmp_path):
    cache = ResultCache(tmp_path)
    (tmp_path / "k1.json").write_text("{not json")
    (tmp_path / "k2.json").write_text('{"text": "x", "surprise": 1}')  # written by another version
    assert cache.get("k1") is None and cache.get("k2") is None


def test_a_later_put_replaces_the_earlier_one(tmp_path):
    cache = ResultCache(tmp_path)
    cache.put("k1", result("first"))
    cache.put("k1", result("second"))
    assert cache.get("k1").text == "second"


def test_no_temporary_files_are_left_behind(tmp_path):
    cache = ResultCache(tmp_path)
    for i in range(5):
        cache.put(f"k{i}", result())
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json"] * 5


@pytest.mark.parametrize("key", ["../outside", "a/b", "", "a.b"])
def test_a_key_cannot_name_another_path(tmp_path, key):
    cache = ResultCache(tmp_path / "c")
    with pytest.raises(ValueError):
        cache.put(key, result())
    with pytest.raises(ValueError):
        cache.get(key)


@pytest.mark.skipif(os.name != "posix", reason="file modes")
def test_only_the_owner_can_read_the_cached_partner_data(tmp_path):
    cache = ResultCache(tmp_path / "c")
    cache.put("k1", result())
    assert stat.S_IMODE((tmp_path / "c").stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE((tmp_path / "c" / "k1.json").stat().st_mode) & 0o077 == 0


# --- cached -------------------------------------------------------------------------------


def counting(texts=None):
    calls = []

    async def complete(provider, model, req):
        calls.append(req.messages[0].content)
        return result(texts[len(calls) - 1] if texts else f"answer {len(calls)}")

    return complete, calls


def run(complete, req, provider="openai", model="m"):
    return asyncio.run(complete(provider, model, req))


def test_the_second_identical_call_does_not_reach_the_model(tmp_path):
    complete, calls = counting()
    wrapped = cached(complete, ResultCache(tmp_path))
    first, second = run(wrapped, request()), run(wrapped, request())
    assert first == second and len(calls) == 1


def test_a_different_request_model_or_provider_misses(tmp_path):
    complete, calls = counting()
    wrapped = cached(complete, ResultCache(tmp_path))
    run(wrapped, request("a"))
    run(wrapped, request("b"))
    run(wrapped, request("a"), model="other")
    run(wrapped, request("a"), provider="anthropic")
    assert len(calls) == 4


def test_a_rerun_with_a_new_wrapper_still_hits_the_stored_answers(tmp_path):
    complete, calls = counting()
    run(cached(complete, ResultCache(tmp_path)), request())
    run(cached(complete, ResultCache(tmp_path)), request())
    assert len(calls) == 1


def test_replicas_keep_the_incumbent_runs_independent_so_the_noise_floor_is_not_hidden(tmp_path):
    complete, calls = counting()
    cache = ResultCache(tmp_path)
    answers = [run(cached(complete, cache, replica=i), request()).text for i in range(3)]
    assert answers == ["answer 1", "answer 2", "answer 3"] and len(calls) == 3
    # Running the same three replicas again reuses each one's own answer.
    again = [run(cached(complete, cache, replica=i), request()).text for i in range(3)]
    assert again == answers and len(calls) == 3


def test_failed_results_are_retried_not_cached(tmp_path):
    outcomes = iter([result(None, error="HTTP 500"), result("fine")])

    async def complete(provider, model, req):
        return next(outcomes)

    wrapped = cached(complete, ResultCache(tmp_path))
    assert run(wrapped, request()).error == "HTTP 500"
    assert run(wrapped, request()).text == "fine"
    assert run(wrapped, request()).text == "fine"  # now cached


def test_a_rate_limit_passes_through_and_stores_nothing(tmp_path):
    async def complete(provider, model, req):
        raise RateLimited(3.0)

    with pytest.raises(RateLimited):
        run(cached(complete, ResultCache(tmp_path)), request())
    assert list(tmp_path.iterdir()) == []


def test_the_cached_result_keeps_its_conversion_flags(tmp_path):
    async def complete(provider, model, req):
        return result("t", converted=True, unconverted_fields=("parameters.seed: no seed",))

    wrapped = cached(complete, ResultCache(tmp_path))
    run(wrapped, request())
    hit = run(wrapped, request())
    assert hit.converted is True and hit.unconverted_fields == ("parameters.seed: no seed",)
