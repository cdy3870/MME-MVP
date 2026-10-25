"""Pricing: what a number of tokens costs on a catalog offering (CAT-4).

Stage 2 uses it to predict a month's cost from logged traffic. The run estimate (RUN-2) and
the spend meter (RUN-5) will need the same arithmetic, which is why it lives here and not
inside the candidate filter.

Open decisions (see the notes on CAT-4):
    * Cached tokens. `usage.input_tokens` is stored as each provider reports it, and they
      differ: Anthropic's excludes cached tokens, OpenAI's includes them. Pricing input and
      cached tokens separately would double count for OpenAI. Starting point: price
      input_tokens and output_tokens only, and ignore cached discounts.
    * How a window of logged traffic becomes a month. See `predicted_monthly_cost`.
    * Tokenizers differ, so a candidate will see somewhat different counts than the
      incumbent logged. This predicts from the logged counts, as the ERD says.

Every function here is a stub: implement them one at a time.
"""

from __future__ import annotations

from datetime import timedelta

from migration_eval.catalog import Offering
from migration_eval.models import CallRecord

TOKENS_PER_PRICE_UNIT = 1_000_000  # catalog prices are USD per million tokens
DAYS_PER_MONTH = 30


def request_cost(offering: Offering, input_tokens: int, output_tokens: int) -> float:
    """USD for one call: input tokens at the input price plus output tokens at the output price.

    Prices are per million tokens (`offering.prices_usd_per_million_tokens`).
    """
    prices = offering.prices_usd_per_million_tokens
    return (input_tokens * prices.input + output_tokens * prices.output) / TOKENS_PER_PRICE_UNIT


def predicted_monthly_cost(
    offering: Offering,
    records: list[CallRecord],
    window: timedelta,
) -> float | None:
    """What this traffic would cost a month on `offering`, from the logged token counts.

    Add up `request_cost` over the records that have both token counts, then scale from the
    period they cover to a month: `total * DAYS_PER_MONTH / window_in_days`. `window` is the
    period the records represent (the caller knows, e.g. the export's `--since 7d`); raise
    ValueError if it is not positive.

    Returns None when no record has both counts: there is nothing to predict from.
    """
    if window <= timedelta(0):
        raise ValueError("window must be positive")

    no_records_contain = True
    total_cost = 0
    for record in records:
        usage = record.usage
        if usage is not None and usage.input_tokens is not None and usage.output_tokens is not None:
            no_records_contain = False
            total_cost += request_cost(offering, record.usage.input_tokens, record.usage.output_tokens)
            

    if no_records_contain:
        return None

    monthly_cost = total_cost * DAYS_PER_MONTH / (window.total_seconds() / 86400)

    return monthly_cost
        

