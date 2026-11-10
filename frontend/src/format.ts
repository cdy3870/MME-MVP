import type { Verdict } from "../../ts/report.v0";

export const pct = (x: number | null, digits = 1): string => (x == null ? "—" : `${(x * 100).toFixed(digits)}%`);

export function ms(x: number | null): string {
  if (x == null) return "—";
  return x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${Math.round(x)} ms`;
}

/** Cost per request is a few hundredths of a cent, which nobody can read; show it per 1,000 requests. */
export const usdPer1k = (perRequest: number | null): string => (perRequest == null ? "—" : `$${(perRequest * 1000).toFixed(2)}`);

/** Relative change from `base` to `value` (0.1 is 10% more); null when either is missing or the base is zero. */
export function change(value: number | null, base: number | null): number | null {
  if (value == null || base == null || base === 0) return null;
  return value / base - 1;
}

export function changeText(c: number | null): string {
  if (c == null) return "—";
  const rounded = Math.round(c * 100);
  return rounded === 0 ? "same" : `${rounded > 0 ? "+" : "−"}${Math.abs(rounded)}%`;
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  safe_to_swap: "Safe to swap",
  swap_with_caveats: "Swap with caveats",
  do_not_swap: "Do not swap",
  insufficient_data: "Insufficient data",
};

export const VERDICT_ORDER: Verdict[] = ["safe_to_swap", "swap_with_caveats", "do_not_swap", "insufficient_data"];

export const CHECK_LABEL: Record<string, string> = {
  error: "The call failed",
  output_schema: "Output did not match the requested schema",
  tool_calls: "Tool calls differed from the current model's",
  refusal: "Refused when the current model did not",
};

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
