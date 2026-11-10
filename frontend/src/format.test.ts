import { describe, expect, it } from "vitest";
import { change, changeText, ms, pct, usdPer1k } from "./format";
import { agreementDomain, position, ticks } from "./scale";

describe("format", () => {
  it("formats rates, with a dash for missing values", () => {
    expect(pct(0.9733)).toBe("97.3%");
    expect(pct(0.9, 0)).toBe("90%");
    expect(pct(null)).toBe("—");
  });

  it("formats milliseconds and seconds", () => {
    expect(ms(820)).toBe("820 ms");
    expect(ms(2100)).toBe("2.1 s");
    expect(ms(null)).toBe("—");
  });

  it("shows cost per 1,000 requests", () => {
    expect(usdPer1k(0.0075)).toBe("$7.50");
    expect(usdPer1k(0.0004)).toBe("$0.40");
    expect(usdPer1k(null)).toBe("—");
  });

  it("computes relative change, and refuses to divide by nothing", () => {
    expect(change(410, 820)).toBeCloseTo(-0.5);
    expect(change(null, 820)).toBeNull();
    expect(change(5, 0)).toBeNull();
    expect(changeText(-0.5)).toBe("−50%");
    expect(changeText(0.3)).toBe("+30%");
    expect(changeText(0.001)).toBe("same");
    expect(changeText(null)).toBe("—");
  });
});

describe("agreement scale", () => {
  it("starts the axis just under the lowest value, in whole tenths", () => {
    expect(agreementDomain([0.7467, 0.97, 0.92])).toEqual([0.7, 1]);
    expect(agreementDomain([0.5, 0.6, 0.92])).toEqual([0.4, 1]);
  });

  it("never zooms in on a sliver and never goes below zero", () => {
    expect(agreementDomain([0.99, 1])).toEqual([0.8, 1]);
    expect(agreementDomain([0.9467, 0.97, 0.92])).toEqual([0.8, 1]); // would have been [0.9, 1]
    expect(agreementDomain([0.01, 0.5])).toEqual([0, 1]);
  });

  it("falls back to the full range when there is nothing to plot", () => {
    expect(agreementDomain([null, null])).toEqual([0, 1]);
    expect(agreementDomain([])).toEqual([0, 1]);
  });

  it("ticks every 5 points on a narrow axis and every 10 on a wide one", () => {
    expect(ticks([0.8, 1])).toEqual([0.8, 0.85, 0.9, 0.95, 1]);
    expect(ticks([0.4, 1])).toEqual([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]);
  });

  it("positions values along the axis and clamps to it", () => {
    expect(position(0.9, [0.8, 1])).toBeCloseTo(50);
    expect(position(0.5, [0.8, 1])).toBe(0);
    expect(position(1.2, [0.8, 1])).toBe(100);
  });
});
