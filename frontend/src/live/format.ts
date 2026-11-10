export function formatBytes(size: number): string {
  if (size < 1024) return `${size} bytes`;
  const units = ["KB", "MB", "GB"];
  let value = size;
  for (const unit of units) {
    value /= 1024;
    if (Math.round(value * 10) / 10 < 1024 || unit === "GB") return `${value.toFixed(1)} ${unit}`;
  }
  return `${size} bytes`;
}

/** "45 seconds", "12 minutes", "3.5 hours", "6 days": a rough length of time for a sentence. */
export function formatDuration(seconds: number): string {
  const units: [string, number][] = [["day", 86400], ["hour", 3600], ["minute", 60], ["second", 1]];
  for (const [name, size] of units) {
    if (seconds >= size || name === "second") {
      const value = seconds / size;
      const text = value >= 10 || Number.isInteger(value) ? String(Math.round(value)) : value.toFixed(1);
      return `${text} ${name}${text === "1" ? "" : "s"}`;
    }
  }
  return `${seconds} seconds`;
}

export const REPLAY_BUDGET_SECONDS = 600; // aim for a replay pass of about ten minutes

/** The replay speed that fits a stretch of logged traffic into the budget; 1 when it is short already. */
export function suggestedSpeed(spanSeconds: number): number {
  return Math.max(1, Math.ceil(spanSeconds / REPLAY_BUDGET_SECONDS));
}

export const usd = (value: number): string => `$${value.toFixed(value < 1 ? 4 : 2)}`;
