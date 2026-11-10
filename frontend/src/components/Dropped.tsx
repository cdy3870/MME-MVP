import type { DroppedCandidate } from "../../../ts/report.v0";

const STAGE: Record<number, string> = { 1: "Does not fit the traffic or your rules", 2: "Too expensive" };

export function Dropped({ dropped }: { dropped: DroppedCandidate[] }) {
  return (
    <section aria-labelledby="dropped-h">
      <h2 id="dropped-h">Models that were not tested</h2>
      {dropped.length === 0 ? (
        <p className="empty">Every model in the catalog was tested.</p>
      ) : (
        <div className="card">
          <ul className="dropped">
            {dropped.map((d) => (
              <li key={`${d.provider}/${d.model}`}>
                <div className="dropped-head">
                  <strong>{d.provider}/{d.model}</strong>
                  <span className="flag">{STAGE[d.stage] ?? `stage ${d.stage}`}</span>
                </div>
                <ul className="plain">
                  {d.reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
