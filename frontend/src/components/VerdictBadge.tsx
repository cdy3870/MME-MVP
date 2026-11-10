import type { Verdict } from "../../../ts/report.v0";
import { VERDICT_LABEL } from "../format";

// The color is never the only signal: every verdict also has its own icon shape and its label.
const ICON: Record<Verdict, { color: string; shape: JSX.Element }> = {
  safe_to_swap: {
    color: "var(--good)",
    shape: <path d="M5 8.4l2.1 2.1L11 6.2" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />,
  },
  swap_with_caveats: {
    color: "var(--warning)",
    shape: <path d="M8 5.2v3.6M8 11.2v.1" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />,
  },
  do_not_swap: {
    color: "var(--critical)",
    shape: <path d="M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />,
  },
  insufficient_data: {
    color: "var(--muted)",
    shape: <path d="M5.4 8h5.2" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />,
  },
};

export function VerdictIcon({ verdict }: { verdict: Verdict }) {
  const { color, shape } = ICON[verdict];
  const outline =
    verdict === "swap_with_caveats" ? <path d="M8 1.6l6.6 11.6H1.4z" fill={color} stroke={color} strokeWidth="1.4" strokeLinejoin="round" /> : <circle cx="8" cy="8" r="7" fill={color} />;
  return (
    <svg className="verdict-icon" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
      {outline}
      <g style={{ color: verdict === "swap_with_caveats" ? "#0b0b0b" : "#ffffff" }}>{shape}</g>
    </svg>
  );
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <span className="badge">
      <VerdictIcon verdict={verdict} />
      <span>{VERDICT_LABEL[verdict]}</span>
    </span>
  );
}
