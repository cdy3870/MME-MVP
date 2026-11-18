"""GPU-8: measured vs estimated labels."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from migration_eval.gpu.labels import UNTESTED_NOTICE, Estimate


def test_estimate_is_labelled_and_has_a_midpoint():
    e = Estimate(low=10, high=30, unit="%", assumptions=["typical gain"])
    assert e.basis == "estimated"
    assert e.midpoint == 20


def test_estimate_rejects_reversed_range():
    with pytest.raises(ValidationError, match="above high"):
        Estimate(low=5, high=1, unit="%")


def test_untested_notice_says_untested():
    assert "not been tested" in UNTESTED_NOTICE
