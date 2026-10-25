"""Stage 1 candidate filter (CAT-3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from migration_eval.candidates import (
    Constraints,
    check_allow_list,
    check_context,
    check_hosting,
    check_incumbent,
    check_region,
    check_tools,
    stage_one,
)
from migration_eval.catalog import Catalog, load_catalog
from migration_eval.profile import Incumbent, Profile

ROOT = Path(__file__).resolve().parent.parent


def offering(provider="openai", model="gpt-4o", context=128_000, tools=True, regions=("us", "eu")):
    return {
        "provider": provider,
        "model": model,
        "context_window": context,
        "tool_support": tools,
        "regions": list(regions),
        "prices_usd_per_million_tokens": {"input": 1, "output": 2},
        "as_of": "2026-10-24",
        "source_url": "https://example.com/p",
    }


def catalog(*offerings: dict) -> Catalog:
    return Catalog.model_validate({"offerings": list(offerings)})


def profile(max_context=1000, uses_tools=False, incumbent=None) -> Profile:
    return Profile("support", 10, max_context, uses_tools, incumbent)


def constraints(allow=("openai", "anthropic", "gemini"), hosting="api", region=None) -> Constraints:
    return Constraints(allow_list=frozenset(allow), hosting=hosting, region=region)


def one(**kwargs):
    return catalog(offering(**kwargs)).offerings[0]


# --- each rule ----------------------------------------------------------------------------


def test_context_window_too_small_is_dropped_with_both_numbers():
    assert check_context(one(context=128_000), profile(max_context=190_000)) == (
        "needs 190,000 tokens, window is 128,000"
    )


def test_a_window_exactly_equal_to_the_need_passes():
    assert check_context(one(context=1000), profile(max_context=1000)) is None


def test_unknown_context_need_cannot_drop_anything():
    assert check_context(one(context=1), profile(max_context=None)) is None


def test_tools_needed_but_not_supported_is_dropped():
    assert "does not support them" in check_tools(one(tools=False), profile(uses_tools=True))


def test_tools_not_needed_passes_even_without_support():
    assert check_tools(one(tools=False), profile(uses_tools=False)) is None
    assert check_tools(one(tools=True), profile(uses_tools=True)) is None


def test_required_region_the_offering_does_not_serve_is_dropped():
    reason = check_region(one(regions=("us",)), constraints(region="eu"))
    assert "'eu'" in reason and "serves us" in reason


def test_no_region_requirement_passes_everything():
    assert check_region(one(regions=("us",)), constraints(region=None)) is None


def test_a_served_region_passes():
    assert check_region(one(regions=("us", "eu")), constraints(region="eu")) is None


def test_api_hosting_passes_and_any_other_hosting_option_is_dropped():
    assert check_hosting(one(), constraints(hosting="api")) is None
    assert "'self-hosted'" in check_hosting(one(), constraints(hosting="self-hosted"))


def test_provider_not_on_the_allow_list_is_dropped():
    assert check_allow_list(one(provider="gemini"), constraints(allow=("openai",))) == (
        "provider 'gemini' is not on the allow-list"
    )
    assert check_allow_list(one(provider="openai"), constraints(allow=("openai",))) is None


def test_the_incumbent_is_dropped_but_other_models_of_the_same_provider_are_not():
    me = profile(incumbent=Incumbent("openai", "gpt-4o"))
    assert check_incumbent(one(provider="openai", model="gpt-4o"), me) is not None
    assert check_incumbent(one(provider="openai", model="gpt-4o-mini"), me) is None
    assert check_incumbent(one(provider="gemini", model="gpt-4o"), me) is None


def test_no_incumbent_drops_nothing():
    assert check_incumbent(one(), profile(incumbent=None)) is None


# --- stage_one ----------------------------------------------------------------------------


def test_an_offering_that_passes_every_check_is_kept():
    cat = catalog(offering())
    kept, dropped = stage_one(cat, profile(), constraints())
    assert [o.model for o in kept] == ["gpt-4o"] and dropped == []


def test_an_empty_allow_list_drops_everything():
    cat = catalog(offering(), offering(provider="anthropic", model="c"), offering(provider="gemini", model="g"))
    kept, dropped = stage_one(cat, profile(), constraints(allow=()))
    assert kept == [] and len(dropped) == 3
    assert all("not on the allow-list" in reasons[0] for _, reasons in dropped)


def test_every_failing_reason_is_kept_in_a_fixed_order():
    small_no_tools = offering(provider="gemini", model="tiny", context=100, tools=False, regions=("us",))
    kept, dropped = stage_one(
        catalog(small_no_tools),
        profile(max_context=5000, uses_tools=True),
        constraints(allow=("openai",), region="eu", hosting="self-hosted"),
    )
    (_, reasons), = dropped
    assert [r.split(" ")[0] for r in reasons] == ["needs", "the", "it", "hosting", "provider"]
    assert len(reasons) == 5 and kept == []


def test_kept_and_dropped_follow_catalog_order_and_cover_every_offering():
    cat = catalog(
        offering(model="a"),
        offering(model="b", context=10),
        offering(model="c"),
        offering(model="d", tools=False),
    )
    kept, dropped = stage_one(cat, profile(max_context=1000, uses_tools=True), constraints())
    assert [o.model for o in kept] == ["a", "c"]
    assert [o.model for o, _ in dropped] == ["b", "d"]
    assert len(kept) + len(dropped) == len(cat.offerings)


def test_the_incumbent_is_removed_from_the_candidates():
    cat = catalog(offering(model="current"), offering(model="other"))
    kept, dropped = stage_one(cat, profile(incumbent=Incumbent("openai", "current")), constraints())
    assert [o.model for o in kept] == ["other"]
    assert dropped[0][1] == ["it is the model the traffic already uses"]


def test_the_real_catalog_is_filtered_by_allow_list():
    cat = load_catalog(ROOT / "catalog" / "catalog.yaml")
    kept, dropped = stage_one(cat, profile(), constraints(allow=("anthropic",)))
    assert kept and all(o.provider == "anthropic" for o in kept)
    assert dropped and all(o.provider != "anthropic" for o, _ in dropped)
    assert len(kept) + len(dropped) == len(cat.offerings)


@pytest.mark.parametrize("need", [1, 128_000, 128_001])
def test_the_context_boundary_in_the_filter(need):
    kept, dropped = stage_one(catalog(offering(context=128_000)), profile(max_context=need), constraints())
    assert bool(kept) == (need <= 128_000)
