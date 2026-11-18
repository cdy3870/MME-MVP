import { describe, expect, it } from "vitest";
import type { Estimate } from "../live/gpu-types";
import { formatEstimate } from "./GpuReportView";

const est = (low: number, high: number, unit: string): Estimate => ({ basis: "estimated", low, high, unit, assumptions: [] });

describe("formatEstimate", () => {
  it("shows a range, or one number when both ends match", () => {
    expect(formatEstimate(est(10, 30, "%"))).toBe("+10–30%");
    expect(formatEstimate(est(1, 1, "GPUs"))).toBe("1 GPU");
    expect(formatEstimate(est(1, 2, "GPUs"))).toBe("1–2 GPUs");
  });

  it("formats money, keeping a negative low end", () => {
    expect(formatEstimate(est(-500, 1200, "USD/month"))).toBe(`−$500–$${(1200).toLocaleString()} a month`);
    expect(formatEstimate(est(0.5, 1.25, "USD per 1M tokens"))).toBe("$0.500–$1.25 per 1M tokens");
  });
});
