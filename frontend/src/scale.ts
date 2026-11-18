/** The x scale of the agreement charts: an interval plot may start above zero, and the axis says where. */
export type Domain = [number, number];

export function agreementDomain(values: (number | null)[]): Domain {
  const present = values.filter((v): v is number => v != null);
  if (present.length === 0) return [0, 1];
  let start = Math.max(0, Math.floor((Math.min(...present) - 0.02) * 10 + 1e-9) / 10);
  if (1 - start < 0.2) start = 0.8; // never zoom in on a sliver
  return [start, 1];
}

export function ticks([start, end]: Domain): number[] {
  const step = end - start <= 0.25 ? 0.05 : 0.1;
  const count = Math.round((end - start) / step);
  return Array.from({ length: count + 1 }, (_, i) => Math.round((start + i * step) * 1000) / 1000);
}

/** Where a value sits along the axis, as a percentage of its width (clamped to the axis). */
export function position(x: number, [start, end]: Domain): number {
  return Math.min(1, Math.max(0, (x - start) / (end - start))) * 100;
}
