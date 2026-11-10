import type { Report } from "../../ts/report.v0";

// The example reports in schema/examples/report/valid, bundled so the prototype works with no backend.
const modules = import.meta.glob("@examples/*.json", { eager: true, import: "default" }) as Record<string, Report>;

export const EXAMPLES: { name: string; report: Report }[] = Object.entries(modules)
  .map(([path, report]) => ({ name: path.split("/").pop()!.replace(/\.json$/, ""), report }))
  .sort((a, b) => a.name.localeCompare(b.name));

/** A light guard for a report loaded from disk. The Python side validates against the schema. */
export function looksLikeReport(value: unknown): value is Report {
  const r = value as Partial<Report> | null;
  return (
    !!r &&
    typeof r === "object" &&
    r.schema_version === "v0" &&
    Array.isArray(r.candidates) &&
    Array.isArray(r.dropped_candidates) &&
    typeof r.component === "string" &&
    !!r.noise_floor &&
    !!r.incumbent
  );
}
