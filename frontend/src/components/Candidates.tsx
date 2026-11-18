import type { Candidate, Report } from "../../../ts/report.v0";
import { pct } from "../format";
import { IntervalChart, type IntervalRow } from "./IntervalChart";
import { Delta } from "./Summary";
import { VerdictBadge } from "./VerdictBadge";

export const candidateKey = (c: Candidate) => `${c.provider}/${c.model}`;

function Flags({ c }: { c: Candidate }) {
  return (
    <>
      {c.stopped_early && <span className="flag">stopped early</span>}
      {c.conversion.converted_requests > 0 && <span className="flag">converted requests</span>}
    </>
  );
}

export function Candidates({ report, selectedKey, onSelect }: { report: Report; selectedKey: string | undefined; onSelect: (key: string) => void }) {
  const { candidates, noise_floor, incumbent } = report;

  if (candidates.length === 0) {
    return (
      <section aria-labelledby="cand-h">
        <h2 id="cand-h">Candidates</h2>
        <p className="empty">Nothing was replayed, so there is nothing to compare. See below for why models were dropped.</p>
      </section>
    );
  }

  const rows: IntervalRow[] = candidates.map((c) => ({
    key: candidateKey(c),
    label: candidateKey(c),
    detail: `${c.requests_run} requests`,
    agreement: c.agreement,
  }));

  return (
    <section aria-labelledby="cand-h">
      <h2 id="cand-h">Agreement with the current model</h2>
      <p className="section-note">
        The share of requests where the candidate gave an equivalent answer. The dot is the result, the line is the range it could
        plausibly be (95% interval). A candidate is only as good as the noise floor if its interval reaches that line.
      </p>

      <div className="card">
        <IntervalChart rows={rows} noiseFloor={noise_floor.value} selectedKey={selectedKey} onSelect={onSelect} caption="Agreement by candidate" />
        <p className="key" aria-hidden="true">
          <span className="key-dot" /> agreement <span className="key-line" /> 95% interval <span className="key-floor" /> noise floor
        </p>
      </div>

      <h2 id="table-h">All candidates</h2>
      <div className="card table-wrap">
        <table aria-labelledby="table-h">
          <thead>
            <tr>
              <th scope="col">Candidate</th>
              <th scope="col">Verdict</th>
              <th scope="col" className="num">Agreement</th>
              <th scope="col" className="num">Median latency</th>
              <th scope="col" className="num">Cost per 1,000</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((c) => {
              const key = candidateKey(c);
              return (
                <tr key={key} className={key === selectedKey ? "selected" : ""} onClick={() => onSelect(key)}>
                  <th scope="row">
                    <button type="button" className="link" onClick={() => onSelect(key)} aria-pressed={key === selectedKey}>
                      {key}
                    </button>
                    <Flags c={c} />
                  </th>
                  <td><VerdictBadge verdict={c.verdict} /></td>
                  <td className="num">
                    {pct(c.agreement.rate)}
                    <span className="sub">{pct(c.agreement.ci_low)} to {pct(c.agreement.ci_high)}</span>
                  </td>
                  <td className="num">
                    <Delta value={c.latency.p50_ms} base={incumbent.latency.p50_ms} format={(x) => (x == null ? "—" : x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${Math.round(x)} ms`)} lowerIsBetter />
                  </td>
                  <td className="num">
                    <Delta
                      value={c.cost.per_request_usd}
                      base={incumbent.cost_per_request_usd}
                      format={(x) => (x == null ? "—" : `$${(x * 1000).toFixed(2)}`)}
                      lowerIsBetter
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
