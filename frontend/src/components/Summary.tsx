import type { Report, Verdict } from "../../../ts/report.v0";
import { VERDICT_LABEL, VERDICT_ORDER, change, changeText, ms, pct, usdPer1k } from "../format";
import { VerdictIcon } from "./VerdictBadge";

function Tile({ label, value, note }: { label: string; value: React.ReactNode; note?: React.ReactNode }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {note && <div className="tile-note">{note}</div>}
    </div>
  );
}

export function VerdictCounts({ report }: { report: Report }) {
  const counts = Object.fromEntries(VERDICT_ORDER.map((v) => [v, 0])) as Record<Verdict, number>;
  for (const c of report.candidates) counts[c.verdict] += 1;
  const total = report.candidates.length;

  return (
    <section aria-label="Verdicts at a glance">
      <h2 className="visually-hidden">Verdicts at a glance</h2>
      <p className="lede">
        {total === 0 ? (
          "No candidates were replayed."
        ) : (
          <>
            <strong>{counts.safe_to_swap}</strong> of {total} {total === 1 ? "candidate is" : "candidates are"} safe to swap.
          </>
        )}
      </p>
      <div className="tiles four">
        {VERDICT_ORDER.map((v) => (
          <div className="tile compact" key={v}>
            <div className="tile-label">
              <VerdictIcon verdict={v} /> {VERDICT_LABEL[v]}
            </div>
            <div className="tile-value">{counts[v]}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function Baseline({ report }: { report: Report }) {
  const { incumbent, noise_floor: floor, sample } = report;
  return (
    <section aria-labelledby="baseline-h">
      <h2 id="baseline-h">The current model, measured now</h2>
      <p className="section-note">
        {floor.runs === 0 ? (
          <>{incumbent.provider}/{incumbent.model} could not be measured, so there is no baseline to compare candidates with.</>
        ) : (
          <>
            {incumbent.provider}/{incumbent.model} replayed against itself {floor.runs} times on {floor.requests} requests. Candidates are
            compared with these numbers, not with perfect agreement.
          </>
        )}
      </p>
      <div className="tiles four">
        <Tile
          label="Noise floor"
          value={floor.value == null ? "Not measured" : pct(floor.value)}
          note="How often the current model agrees with itself. A candidate cannot be expected to do better."
        />
        <Tile label="Median latency" value={ms(incumbent.latency.p50_ms)} note={`${incumbent.latency.n} calls`} />
        <Tile label="Slowest 5%" value={ms(incumbent.latency.p95_ms)} note="95th percentile" />
        <Tile
          label="Cost per 1,000 requests"
          value={usdPer1k(incumbent.cost_per_request_usd)}
          note={`Replayed at ${sample.replay_speed}× the logged pace`}
        />
      </div>
    </section>
  );
}

export function Delta({ value, base, format, lowerIsBetter }: { value: number | null; base: number | null; format: (x: number | null) => string; lowerIsBetter: boolean }) {
  const c = change(value, base);
  const better = c == null || Math.round(c * 100) === 0 ? null : lowerIsBetter ? c < 0 : c > 0;
  return (
    <span className="delta-cell">
      <span className="delta-value">{format(value)}</span>
      <span className={`delta-change${better === true ? " better" : better === false ? " worse" : ""}`}>
        {c == null ? "" : `${c < 0 ? "▼ " : c > 0 ? "▲ " : ""}${changeText(c)}`}
      </span>
    </span>
  );
}
