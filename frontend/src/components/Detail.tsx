import type { Candidate, Report } from "../../../ts/report.v0";
import { CHECK_LABEL, ms, pct, usdPer1k } from "../format";
import { candidateKey } from "./Candidates";
import { IntervalChart } from "./IntervalChart";
import { Delta } from "./Summary";
import { VerdictBadge } from "./VerdictBadge";

export function Detail({ report, candidate: c }: { report: Report; candidate: Candidate }) {
  const { incumbent, noise_floor: floor, thresholds, sample } = report;
  const conv = c.conversion;
  const rows = c.by_category.map((cat) => ({
    key: cat.category,
    label: cat.category,
    detail: cat.agreement.n < thresholds.min_category_requests ? `${cat.agreement.n} requests, too few to count` : `${cat.agreement.n} requests`,
    agreement: cat.agreement,
  }));

  return (
    <section aria-labelledby="detail-h" className="detail">
      <h2 id="detail-h">
        {candidateKey(c)} <VerdictBadge verdict={c.verdict} />
      </h2>
      <p className="explanation">{c.explanation}</p>

      {c.stopped_early && (
        <div className="notice">
          <strong>Stopped early.</strong> It ran {c.requests_run} of the {sample.requests_replayed} sampled requests. Its agreement was already
          clearly below the noise floor, so the rest were not run.
        </div>
      )}

      {conv.converted_requests > 0 && (
        <div className="notice caution">
          <strong>Converted requests.</strong> {conv.converted_requests} of {conv.requests} requests were translated from {incumbent.provider}&rsquo;s
          format into this provider&rsquo;s, so the agreement reflects the translation as well as the model.
          {conv.unconverted.length > 0 ? " These settings could not be carried over:" : " Everything was carried over."}
          {conv.unconverted.length > 0 && (
            <ul>
              {conv.unconverted.map((u) => (
                <li key={u.field}>
                  {u.field} <span className="muted">({u.requests} requests)</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="grid-2">
        <div>
          <h3>By request category</h3>
          <div className="card">
            <IntervalChart rows={rows} noiseFloor={floor.value} caption={`Agreement by request category for ${candidateKey(c)}`} />
          </div>
        </div>

        <div>
          <h3>Speed and cost</h3>
          <div className="card compare">
            <table>
              <thead>
                <tr>
                  <th scope="col"><span className="visually-hidden">Measure</span></th>
                  <th scope="col" className="num">{candidateKey(c)}</th>
                  <th scope="col" className="num">Current</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Median latency</th>
                  <td className="num"><Delta value={c.latency.p50_ms} base={incumbent.latency.p50_ms} format={ms} lowerIsBetter /></td>
                  <td className="num">{ms(incumbent.latency.p50_ms)}</td>
                </tr>
                <tr>
                  <th scope="row">Slowest 5%</th>
                  <td className="num"><Delta value={c.latency.p95_ms} base={incumbent.latency.p95_ms} format={ms} lowerIsBetter /></td>
                  <td className="num">{ms(incumbent.latency.p95_ms)}</td>
                </tr>
                <tr>
                  <th scope="row">Cost per 1,000</th>
                  <td className="num"><Delta value={c.cost.per_request_usd} base={incumbent.cost_per_request_usd} format={usdPer1k} lowerIsBetter /></td>
                  <td className="num">{usdPer1k(incumbent.cost_per_request_usd)}</td>
                </tr>
                <tr>
                  <th scope="row">Failed calls</th>
                  <td className="num">{c.latency.failed}</td>
                  <td className="num">{incumbent.latency.failed}</td>
                </tr>
              </tbody>
            </table>
            <p className="fineprint">
              {sample.replay_speed === 1
                ? "Replayed at the logged pace, so latency is comparable."
                : `Replayed at ${sample.replay_speed}× the logged pace, so latency is not directly comparable.`}
            </p>
          </div>

          <h3>Automatic checks that failed</h3>
          <div className="card">
            {c.check_failures.length === 0 ? (
              <p className="muted-text">No automatic check failed.</p>
            ) : (
              <ul className="plain">
                {c.check_failures.map((f) => (
                  <li key={f.check}>
                    <strong>{f.requests}</strong> {CHECK_LABEL[f.check] ?? f.check}
                  </li>
                ))}
              </ul>
            )}
            <p className="fineprint">A request that fails a check counts as different without asking a judge.</p>
          </div>
        </div>
      </div>

      <details className="rules">
        <summary>How this verdict is decided</summary>
        <ul>
          <li><strong>Insufficient data</strong> if fewer than {thresholds.min_requests} requests ran.</li>
          <li><strong>Do not swap</strong> if it stopped early, or the top of its interval is more than {pct(thresholds.caveat_margin, 0)} points below the noise floor.</li>
          <li>
            <strong>Safe to swap</strong> if the bottom of its interval reaches the noise floor{thresholds.safe_margin > 0 ? ` (within ${pct(thresholds.safe_margin, 0)} points)` : ""}, no
            category of at least {thresholds.min_category_requests} requests is clearly below it, and it was sent the exact logged requests.
          </li>
          <li><strong>Swap with caveats</strong> otherwise: it is close, but one of those conditions is not met.</li>
        </ul>
      </details>
    </section>
  );
}
