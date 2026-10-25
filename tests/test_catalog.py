"""The catalog loader (CAT-1): the real file, and every way a hand-edited file can go wrong."""

from __future__ import annotations

from pathlib import Path

import pytest

from migration_eval.catalog import MAX_PROBLEMS_SHOWN, CatalogError, load_catalog

ROOT = Path(__file__).resolve().parent.parent
REAL_CATALOG = ROOT / "catalog" / "catalog.yaml"

OFFERING = """\
  - provider: openai
    model: gpt-4o
    context_window: 128000
    tool_support: true
    regions: [us, eu]
    prices_usd_per_million_tokens:
      input: 2.50
      output: 10.00
    as_of: "2026-10-24"
    source_url: https://example.com/pricing
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text(text)
    return path


def catalog_text(*offerings: str) -> str:
    return "offerings:\n" + "".join(offerings)


def with_change(old: str, new: str, offering: str = OFFERING) -> str:
    assert old in offering, f"{old!r} not in the test offering"
    return offering.replace(old, new)


def problems_of(tmp_path: Path, text: str) -> str:
    with pytest.raises(CatalogError) as error:
        load_catalog(write(tmp_path, text))
    return str(error.value)


# --- the real catalog ---------------------------------------------------------------------


def test_the_committed_catalog_is_valid():
    catalog = load_catalog(REAL_CATALOG)
    assert len(catalog.offerings) >= 1
    assert {o.provider for o in catalog.offerings} <= {"anthropic", "openai", "gemini"}


# --- a valid file loads into typed values -------------------------------------------------


def test_a_valid_offering_loads_with_the_right_types(tmp_path):
    (offering,) = load_catalog(write(tmp_path, catalog_text(OFFERING))).offerings
    assert (offering.provider, offering.model) == ("openai", "gpt-4o")
    assert offering.context_window == 128000 and offering.tool_support is True
    assert offering.prices_usd_per_million_tokens.input == 2.5
    assert offering.prices_usd_per_million_tokens.cached_input is None
    assert offering.as_of == "2026-10-24" and isinstance(offering.as_of, str)


def test_regions_are_plain_strings_so_membership_checks_work(tmp_path):
    (offering,) = load_catalog(write(tmp_path, catalog_text(OFFERING))).offerings
    assert offering.regions == ["us", "eu"]
    assert "us" in offering.regions and "asia" not in offering.regions


def test_an_unquoted_date_is_read_as_text_not_a_date_object(tmp_path):
    text = catalog_text(with_change('"2026-10-24"', "2026-10-24"))
    (offering,) = load_catalog(write(tmp_path, text)).offerings
    assert offering.as_of == "2026-10-24"


def test_the_same_model_name_under_two_providers_is_allowed(tmp_path):
    other = with_change("provider: openai", "provider: gemini")
    assert len(load_catalog(write(tmp_path, catalog_text(OFFERING, other))).offerings) == 2


def test_an_optional_cached_price_is_read(tmp_path):
    text = catalog_text(with_change("      output: 10.00\n", "      output: 10.00\n      cached_input: 1.25\n"))
    (offering,) = load_catalog(write(tmp_path, text)).offerings
    assert offering.prices_usd_per_million_tokens.cached_input == 1.25


# --- CAT-2: missing as-of dates and source links name the offering ------------------------


def test_a_missing_as_of_names_the_offering_and_the_field(tmp_path):
    text = catalog_text(with_change('    as_of: "2026-10-24"\n', ""))
    assert "offerings[0] (openai/gpt-4o): as_of is missing" in problems_of(tmp_path, text)


def test_a_missing_source_url_names_the_offering_and_the_field(tmp_path):
    text = catalog_text(with_change("    source_url: https://example.com/pricing\n", ""))
    assert "offerings[0] (openai/gpt-4o): source_url is missing" in problems_of(tmp_path, text)


def test_an_empty_as_of_and_source_url_are_rejected(tmp_path):
    for old, new, expect in [
        ('"2026-10-24"', '""', "as_of must be a date written YYYY-MM-DD"),
        ("https://example.com/pricing", '""', "source_url must be an https:// link"),
    ]:
        assert expect in problems_of(tmp_path, catalog_text(with_change(old, new)))


def test_a_null_value_is_rejected(tmp_path):
    assert "as_of" in problems_of(tmp_path, catalog_text(with_change('"2026-10-24"', "null")))


# --- bad values -----------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["24/10/2026", "2026-10-4", "October 24", "2026-13-01", "2026-00-10", "2026-10-32"])
def test_a_badly_formatted_date_is_rejected(tmp_path, bad):
    message = problems_of(tmp_path, catalog_text(with_change('"2026-10-24"', f'"{bad}"')))
    assert "as_of must be a date written YYYY-MM-DD" in message and bad in message


def test_a_date_that_does_not_exist_is_rejected(tmp_path):
    message = problems_of(tmp_path, catalog_text(with_change('"2026-10-24"', '"2026-02-30"')))
    assert "as_of '2026-02-30' is not a real date" in message


@pytest.mark.parametrize("bad", ["http://example.com/p", "example.com/p", "see the pricing page"])
def test_a_source_that_is_not_an_https_link_is_rejected(tmp_path, bad):
    message = problems_of(tmp_path, catalog_text(with_change("https://example.com/pricing", bad)))
    assert "source_url must be an https:// link" in message


def test_a_negative_price_is_rejected(tmp_path):
    message = problems_of(tmp_path, catalog_text(with_change("input: 2.50", "input: -1")))
    assert "prices_usd_per_million_tokens.input" in message


def test_a_zero_price_is_allowed_for_free_models(tmp_path):
    text = catalog_text(with_change("input: 2.50", "input: 0").replace("output: 10.00", "output: 0"))
    assert load_catalog(write(tmp_path, text)).offerings[0].prices_usd_per_million_tokens.input == 0


def test_a_missing_price_is_rejected(tmp_path):
    message = problems_of(tmp_path, catalog_text(with_change("      output: 10.00\n", "")))
    assert "prices_usd_per_million_tokens.output is missing" in message


@pytest.mark.parametrize("bad", ["0", "-5", "lots", "1.5"])
def test_a_bad_context_window_is_rejected(tmp_path, bad):
    assert "context_window" in problems_of(tmp_path, catalog_text(with_change("128000", bad)))


def test_tool_support_must_be_a_real_boolean(tmp_path):
    assert "tool_support" in problems_of(tmp_path, catalog_text(with_change("true", "maybe")))


@pytest.mark.parametrize("regions", ["[US]", "[]", "[us, us]", "[us eu]"])
def test_bad_regions_are_rejected(tmp_path, regions):
    assert "regions" in problems_of(tmp_path, catalog_text(with_change("[us, eu]", regions)))


def test_a_repeated_region_names_the_offering_and_the_region(tmp_path):
    message = problems_of(tmp_path, catalog_text(with_change("[us, eu]", "[us, eu, us]")))
    assert "offerings[0] (openai/gpt-4o): regions lists 'us' more than once" in message


def test_a_bad_region_says_what_a_valid_one_looks_like(tmp_path):
    message = problems_of(tmp_path, catalog_text(with_change("[us, eu]", "[us, US]")))
    assert "regions.1 must be a lowercase identifier such as us or eu (got 'US')" in message


def test_the_extra_checks_run_once_the_schema_is_satisfied(tmp_path):
    """Schema problems come first; fixing them can reveal date and duplicate problems."""
    bad_schema = with_change('    as_of: "2026-10-24"\n', "")
    bad_date = with_change('"2026-10-24"', '"2026-02-30"')
    first = problems_of(tmp_path, catalog_text(bad_schema, bad_date))
    assert "as_of is missing" in first and "not a real date" not in first

    second = problems_of(tmp_path, catalog_text(bad_date, bad_date))
    assert "not a real date" in second and "listed more than once" in second  # both reported together


def test_an_uppercase_provider_is_rejected(tmp_path):
    assert "provider" in problems_of(tmp_path, catalog_text(with_change("provider: openai", "provider: OpenAI")))


def test_a_retirement_date_is_not_a_known_field_in_the_mvp(tmp_path):
    text = catalog_text(with_change("    model: gpt-4o\n", "    model: gpt-4o\n    retirement_date: 2027-01-01\n"))
    assert "retirement_date is not a known field" in problems_of(tmp_path, text)


# --- structure -------------------------------------------------------------------------------


def test_a_repeated_provider_and_model_pair_is_rejected(tmp_path):
    other = with_change("context_window: 128000", "context_window: 64000")
    message = problems_of(tmp_path, catalog_text(OFFERING, other))
    assert "openai/gpt-4o is listed more than once (offerings[0], offerings[1])" in message


def test_a_key_written_twice_is_an_error_not_last_one_wins(tmp_path):
    text = catalog_text(with_change("    tool_support: true\n", "    tool_support: true\n    tool_support: false\n"))
    assert "duplicate key 'tool_support'" in problems_of(tmp_path, text)


def test_an_empty_offerings_list_is_rejected(tmp_path):
    assert "at least 1" in problems_of(tmp_path, "offerings: []\n")


def test_a_file_without_offerings_is_rejected(tmp_path):
    assert "offerings" in problems_of(tmp_path, "something_else: 1\n")


def test_an_unknown_top_level_key_is_rejected(tmp_path):
    assert "version is not a known field" in problems_of(tmp_path, catalog_text(OFFERING) + "version: 2\n")


@pytest.mark.parametrize("text", ["", "- just\n- a list\n", "42\n"])
def test_a_file_that_is_not_a_catalog_mapping_is_rejected(tmp_path, text):
    assert "expected a mapping" in problems_of(tmp_path, text)


def test_invalid_yaml_is_rejected_with_a_clear_message(tmp_path):
    assert "not valid YAML" in problems_of(tmp_path, "offerings: [unclosed\n")


def test_a_missing_file_raises_the_normal_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "nope.yaml")


# --- reporting -------------------------------------------------------------------------------


def test_every_problem_is_reported_at_once_with_a_count(tmp_path):
    first = with_change('    as_of: "2026-10-24"\n', "")
    second = with_change("model: gpt-4o", "model: gpt-4o-mini").replace("    source_url: https://example.com/pricing\n", "")
    message = problems_of(tmp_path, catalog_text(first, second))
    assert "has 2 problems" in message
    assert "offerings[0] (openai/gpt-4o): as_of is missing" in message
    assert "offerings[1] (openai/gpt-4o-mini): source_url is missing" in message


def test_one_problem_is_worded_in_the_singular(tmp_path):
    assert "has 1 problem:" in problems_of(tmp_path, catalog_text(with_change('    as_of: "2026-10-24"\n', "")))


def test_a_huge_number_of_problems_is_capped(tmp_path):
    broken = [with_change("model: gpt-4o", f"model: m{i}").replace('    as_of: "2026-10-24"\n', "") for i in range(MAX_PROBLEMS_SHOWN + 5)]
    message = problems_of(tmp_path, catalog_text(*broken))
    assert f"has {MAX_PROBLEMS_SHOWN + 5} problems" in message
    assert "... and 5 more" in message
    assert message.count("as_of is missing") == MAX_PROBLEMS_SHOWN
