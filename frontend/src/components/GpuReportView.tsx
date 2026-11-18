import type {
  ChecklistItem,
  CostEstimates,
  Estimate,
  Finding,
  GpuReport,
  GpuUsage,
  Recommendation,
  Severity,
  Stage,
  Stat,
  Throttling,
} from "../live/gpu-types";

// GPU-8: every number says where it came from. Measured = the partner's own capture; estimated = ours.
function Measured() {
  return <span className="basis measured" title="Read from your own capture">Measured</span>;
}

function Estimated({ estimate }: { estimate?: Estimate }) {
  const title = estimate && estimate.assumptions.length > 0 ? `Assumes: ${estimate.assumptions.join("; ")}` : "Our estimate, not measured";
  return <span className="basis estimated" title={title}>Estimate</span>;
}

const whole = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const money = (x: number): string => `${x < 0 ? "−" : ""}$${whole.format(Math.abs(x))}`;
const cents = (x: number): string => `$${x.toFixed(x < 1 ? 3 : 2)}`;

/** An estimate as words: "+10–30%", "$1,200–2,400 a month", "1–2 GPUs". One number when low equals high. */
export function formatEstimate(e: Estimate): string {
  const span = (f: (x: number) => string) => (e.low === e.high ? f(e.low) : `${f(e.low)}–${f(e.high)}`);
  switch (e.unit) {
    case "%":
      return `+${span((x) => whole.format(x))}%`;
    case "GPUs":
      return `${span((x) => whole.format(x))} GPU${e.high === 1 ? "" : "s"}`;
    case "USD/month":
      return `${span(money)} a month`;
    case "USD/year":
      return `${span(money)} a year`;
    case "USD/hour":
      return `${span(cents)} an hour`;
    case "USD per 1M tokens":
      return `${span(cents)} per 1M tokens`;
    default:
      return `${span((x) => String(x))} ${e.unit}`;
  }
}

const SEVERITY_LABEL: Record<Severity, string> = { critical: "Critical", warning: "Warning", info: "Note" };
const STAGE_LABEL: Record<Stage, string> = {
  fix: "1. Fix what is wrong now",
  runtime: "2. Try runtime changes in staging",
  capacity: "3. Change capacity",
};
const STAGES: Stage[] = ["fix", "runtime", "capacity"];
const INTERCONNECT: Record<string, string> = {
  nvlink: "NVLink",
  pcie: "PCIe",
  mixed: "NVLink and PCIe",
  single_gpu: "Single GPU",
  unknown: "Not collected",
};

const num = (x: number | null | undefined, digits = 0): string => (x == null ? "—" : x.toFixed(digits));
const share = (x: number | undefined): string => (x == null ? "—" : `${Math.round(x * 100)}%`);

function Tile({ label, value, note, badge }: { label: string; value: React.ReactNode; note?: React.ReactNode; badge?: React.ReactNode }) {
  return (
    <div className="tile">
      <div className="tile-label">
        {label} {badge}
      </div>
      <div className="tile-value">{value}</div>
      {note && <div className="tile-note">{note}</div>}
    </div>
  );
}

function Overview({ report }: { report: GpuReport }) {
  const a = report.assessment;
  const names = [...new Set(a.usage.map((u) => u.name))].join(", ");
  const counts = { critical: 0, warning: 0, info: 0 };
  for (const f of a.findings) counts[f.severity] += 1;
  const utils = a.usage.map((u) => u.utilization_pct?.mean).filter((x): x is number => x != null);
  const meanUtil = utils.length ? utils.reduce((s, x) => s + x, 0) / utils.length : null;
  return (
    <section aria-label="Overview">
      <div className="tiles four">
        <Tile label="GPUs" value={a.gpu_count} note={names} />
        <Tile label="Mean utilization" value={meanUtil == null ? "—" : `${Math.round(meanUtil)}%`} badge={<Measured />} note="Across all GPUs, over the capture" />
        <Tile label="Interconnect" value={INTERCONNECT[a.topology.interconnect] ?? a.topology.interconnect} note={Object.values(a.topology.pcie)[0]} />
        <Tile
          label="Findings"
          value={a.findings.filter((f) => f.kind !== "missing_data").length}
          note={`${counts.critical} critical, ${counts.warning} warnings`}
        />
      </div>
    </section>
  );
}

function Findings({ findings }: { findings: Finding[] }) {
  const shown = findings.filter((f) => f.kind !== "missing_data");
  return (
    <section aria-labelledby="gpu-findings-h">
      <h2 id="gpu-findings-h">
        What the capture shows <Measured />
      </h2>
      {shown.length === 0 ? (
        <p className="empty">Nothing stood out: no throttling, imbalance, placement or headroom problems in this capture.</p>
      ) : (
        <ul className="findings">
          {shown.map((f, i) => (
            <li key={`${f.kind}-${i}`} className={`finding ${f.severity}`}>
              <span className={`severity ${f.severity}`}>{SEVERITY_LABEL[f.severity]}</span>
              <span>{f.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function statCell(s: Stat | null, digits = 0, unit = "") {
  return s == null ? "—" : `${s.mean.toFixed(digits)}${unit}`;
}

function Readings({ usage, throttling }: { usage: GpuUsage[]; throttling: Throttling[] }) {
  const slowed = new Map(throttling.map((t) => [t.index, t] as const));
  return (
    <section aria-labelledby="gpu-readings-h">
      <h2 id="gpu-readings-h">
        Per-GPU readings <Measured />
      </h2>
      <div className="card table-wrap gpu-table">
        <table>
          <thead>
            <tr>
              <th scope="col">GPU</th>
              <th scope="col" className="num">Utilization (mean / p95)</th>
              <th scope="col" className="num">Memory peak</th>
              <th scope="col" className="num">Power p95</th>
              <th scope="col" className="num">Temp max</th>
              <th scope="col" className="num">Slowed</th>
            </tr>
          </thead>
          <tbody>
            {usage.map((u) => {
              const t = slowed.get(u.index);
              return (
                <tr key={u.index}>
                  <th scope="row">
                    GPU {u.index}
                    <span className="sub">{u.name}</span>
                  </th>
                  <td className="num">
                    {statCell(u.utilization_pct, 0, "%")} / {u.utilization_pct ? `${u.utilization_pct.p95.toFixed(0)}%` : "—"}
                  </td>
                  <td className="num">
                    {u.memory_used_mib ? `${whole.format(u.memory_used_mib.max)} MiB` : "—"}
                    <span className="sub">{u.memory_headroom_pct == null ? "" : `${u.memory_headroom_pct.toFixed(0)}% free of ${whole.format(u.memory_total_mib ?? 0)}`}</span>
                  </td>
                  <td className="num">
                    {u.power_draw_w ? `${u.power_draw_w.p95.toFixed(0)} W` : "—"}
                    <span className="sub">{u.power_limit_w == null ? "" : `limit ${num(u.power_limit_w)} W`}</span>
                  </td>
                  <td className="num">{u.temperature_c ? `${u.temperature_c.max.toFixed(0)} °C` : "—"}</td>
                  <td className="num">{share(t?.slowdown_fraction)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="fineprint">&ldquo;Slowed&rdquo; is the share of samples with a power or heat slowdown active. An idle GPU is not counted as slowed.</p>
    </section>
  );
}

function Recommendations({ recommendations }: { recommendations: Recommendation[] }) {
  return (
    <section aria-labelledby="gpu-recs-h">
      <h2 id="gpu-recs-h">
        Recommended changes <Estimated />
      </h2>
      <p className="section-note">Ranked by expected gain. Each is an experiment to run, with what to measure before and after.</p>
      {recommendations.length === 0 ? (
        <p className="empty">No configuration changes to suggest from this capture.</p>
      ) : (
        <ol className="recs">
          {recommendations.map((r) => (
            <li key={r.id} className="card rec">
              <div className="rec-head">
                <h3>{r.title}</h3>
                <span className="rec-gain">
                  {formatEstimate(r.expected_gain)} <Estimated estimate={r.expected_gain} />
                </span>
              </div>
              <p className="rec-meta">
                Effort: {r.effort} · Risk: {r.risk} · Stage: {r.stage}
              </p>
              <p>{r.action}</p>
              <p className="muted-text">
                <strong>How to check:</strong> {r.experiment}
              </p>
              {r.expected_gain.assumptions.length > 0 && (
                <details className="rules">
                  <summary>Assumptions behind the gain</summary>
                  <ul>
                    {r.expected_gain.assumptions.map((a) => (
                      <li key={a}>{a}</li>
                    ))}
                  </ul>
                </details>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function EstimateTile({ label, estimate, note }: { label: string; estimate: Estimate | null | undefined; note?: string }) {
  return <Tile label={label} value={estimate ? formatEstimate(estimate) : "—"} note={note} badge={estimate ? <Estimated estimate={estimate} /> : undefined} />;
}

function Costs({ costs }: { costs: CostEstimates | null }) {
  return (
    <section aria-labelledby="gpu-costs-h">
      <h2 id="gpu-costs-h">
        Cost estimates <Estimated />
      </h2>
      {costs == null ? (
        <p className="empty">
          No GPU price list was found, so costs are left out. Add prices to <code>catalog/gpu_prices.yaml</code> and upload the profile again.
        </p>
      ) : (
        <>
          <div className="tiles four">
            <EstimateTile label="Current GPU cost" estimate={costs.current_monthly_cost} />
            <EstimateTile
              label="GPUs the load needs"
              estimate={costs.consolidation?.needed_gpus}
              note={costs.consolidation ? `Today: ${costs.consolidation.current_gpus}; limited by ${costs.consolidation.limited_by}` : undefined}
            />
            <EstimateTile label="Savings" estimate={costs.monthly_savings} note={costs.annual_savings ? formatEstimate(costs.annual_savings) : undefined} />
            <EstimateTile label="Cost per 1M tokens" estimate={costs.cost_per_million_tokens} />
          </div>
          {costs.sizing && (
            <p className="section-note" style={{ marginTop: 12 }}>
              Cheapest GPU that fits the measured memory peak: <strong>{costs.sizing.gpu}</strong>, {formatEstimate(costs.sizing.count)} at{" "}
              {formatEstimate(costs.sizing.monthly_cost)}.
            </p>
          )}
          {costs.missing.length > 0 && (
            <div className="notice">
              Not estimated:
              <ul>
                {costs.missing.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function Checklist({ items }: { items: ChecklistItem[] }) {
  return (
    <section aria-labelledby="gpu-checklist-h">
      <h2 id="gpu-checklist-h">Your checklist</h2>
      <p className="section-note">Work top to bottom: fixes first, then runtime changes in staging, then capacity.</p>
      {items.length === 0 ? (
        <p className="empty">Nothing to do from this capture.</p>
      ) : (
        STAGES.filter((s) => items.some((i) => i.stage === s)).map((stage) => (
          <div key={stage}>
            <h3>{STAGE_LABEL[stage]}</h3>
            <ol className="checklist">
              {items
                .filter((i) => i.stage === stage)
                .map((i) => (
                  <li key={i.step} value={i.step}>
                    <strong>{i.title}</strong>
                    <span className="sub">
                      Expected: {formatEstimate(i.expected_gain)} <Estimated estimate={i.expected_gain} /> · Risk: {i.risk}
                    </span>
                  </li>
                ))}
            </ol>
          </div>
        ))
      )}
    </section>
  );
}

/** The GPU report: what was measured, what to change, what it may save, and in which order. */
export function GpuReportView({ report }: { report: GpuReport }) {
  const missing = [...report.assessment.missing, ...report.assessment.findings.filter((f) => f.kind === "missing_data").map((f) => f.message)];
  return (
    <main>
      <Overview report={report} />
      <Findings findings={report.assessment.findings} />
      <Readings usage={report.assessment.usage} throttling={report.assessment.throttling} />
      <p className="notice caution" role="note">
        {report.untested_notice}
      </p>
      <Recommendations recommendations={report.recommendations} />
      <Costs costs={report.costs} />
      <Checklist items={report.checklist} />
      {missing.length > 0 && (
        <details className="rules">
          <summary>What this report could not see ({missing.length})</summary>
          <ul>
            {missing.map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        </details>
      )}
    </main>
  );
}
