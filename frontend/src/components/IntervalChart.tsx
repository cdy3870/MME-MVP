import { useState, type CSSProperties } from "react";
import type { Agreement } from "../../../ts/report.v0";
import { pct } from "../format";
import { agreementDomain, position, ticks } from "../scale";

export interface IntervalRow {
  key: string;
  label: string;
  detail?: string; // a second line under the label, e.g. "150 requests"
  agreement: Agreement;
}

interface Props {
  rows: IntervalRow[];
  noiseFloor: number | null;
  selectedKey?: string;
  onSelect?: (key: string) => void;
  caption: string; // what the chart shows, also the accessible name of the list
}

interface Hover {
  row: IntervalRow;
  x: number;
  y: number;
}

function verdictOnFloor(a: Agreement, floor: number | null): string {
  if (floor == null || a.ci_low == null || a.ci_high == null) return "No noise floor to compare with.";
  if (a.ci_low >= floor) return "Even the low end of the interval reaches the noise floor.";
  if (a.ci_high < floor) return "Even the high end of the interval is below the noise floor.";
  return "The interval straddles the noise floor.";
}

/** An interval plot: a dot for the agreement, a thin whisker for its 95% interval, a line for the noise floor. */
export function IntervalChart({ rows, noiseFloor, selectedKey, onSelect, caption }: Props) {
  const [hover, setHover] = useState<Hover | null>(null);

  const domain = agreementDomain([...rows.flatMap((r) => [r.agreement.ci_low, r.agreement.rate]), noiseFloor]);
  const axis = ticks(domain);
  const floorAt = noiseFloor == null ? null : position(noiseFloor, domain);

  const show = (row: IntervalRow, el: HTMLElement, e?: React.PointerEvent) => {
    const box = el.getBoundingClientRect();
    setHover({ row, x: e ? e.clientX : box.left + box.width / 2, y: e ? e.clientY : box.top });
  };

  return (
    <div className="interval-chart" role="list" aria-label={caption}>
      {floorAt != null && (
        <div className="floor-row">
          <span />
          <span className="plot floor-label-slot">
            <span className={`floor-label${floorAt > 80 ? " right" : ""}`} style={{ left: `${floorAt}%` }}>
              Noise floor {pct(noiseFloor)}
            </span>
          </span>
          <span />
        </div>
      )}

      {rows.map((row) => {
        const { rate, ci_low, ci_high, n } = row.agreement;
        const selected = row.key === selectedKey;
        return (
          <button
            key={row.key}
            type="button"
            role="listitem"
            className={`interval-row${selected ? " selected" : ""}`}
            aria-pressed={onSelect ? selected : undefined}
            onClick={() => onSelect?.(row.key)}
            onPointerMove={(e) => show(row, e.currentTarget, e)}
            onPointerLeave={() => setHover(null)}
            onFocus={(e) => show(row, e.currentTarget)}
            onBlur={() => setHover(null)}
          >
            <span className="row-label">
              <span className="row-name">{row.label}</span>
              {row.detail && <span className="row-detail">{row.detail}</span>}
            </span>

            <span className="plot">
              {axis.map((t) => (
                <span key={t} className="gridline" style={{ left: `${position(t, domain)}%` }} />
              ))}
              {floorAt != null && <span className="floor-line" style={{ left: `${floorAt}%` }} />}
              {rate != null && ci_low != null && ci_high != null ? (
                <>
                  <span
                    className="whisker"
                    style={{ left: `${position(ci_low, domain)}%`, width: `${Math.max(position(ci_high, domain) - position(ci_low, domain), 0)}%` }}
                  />
                  <span className="dot" style={{ left: `${position(rate, domain)}%` }} />
                </>
              ) : (
                <span className="no-data">no requests</span>
              )}
            </span>

            <span className="row-values">
              <span className="row-rate">{pct(rate)}</span>
              <span className="row-ci">{ci_low == null || ci_high == null ? "" : `${pct(ci_low)} to ${pct(ci_high)}`}</span>
              <span className="visually-hidden">{` over ${n} requests`}</span>
            </span>
          </button>
        );
      })}

      <div className="axis-row">
        <span />
        <span className="plot axis">
          {axis.map((t) => (
            <span key={t} className="tick" style={{ left: `${position(t, domain)}%` }}>
              {Math.round(t * 100)}%
            </span>
          ))}
        </span>
        <span />
      </div>

      {hover && (
        <div
          className={`tooltip${hover.y < 170 ? " below" : ""}`}
          role="tooltip"
          style={{ "--x": `${hover.x}px`, "--y": `${hover.y}px` } as CSSProperties}
        >
          <div className="tooltip-title">{hover.row.label}</div>
          <div className="tooltip-value">{pct(hover.row.agreement.rate)}</div>
          <div className="tooltip-line">
            95% interval {pct(hover.row.agreement.ci_low)} to {pct(hover.row.agreement.ci_high)}
          </div>
          <div className="tooltip-line">
            {hover.row.agreement.agreed} of {hover.row.agreement.n} requests agreed
          </div>
          <div className="tooltip-line">{verdictOnFloor(hover.row.agreement, noiseFloor)}</div>
        </div>
      )}
    </div>
  );
}
