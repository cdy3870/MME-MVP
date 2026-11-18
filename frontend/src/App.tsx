import { useState } from "react";
import type { Report } from "../../ts/report.v0";
import { ReportHeader, ReportView } from "./components/ReportView";
import { EXAMPLES, looksLikeReport } from "./examples";
import { urlParams, useTheme, type Theme } from "./theme";

const UPLOADED = "uploaded";

/** The example-report viewer: pick a bundled example or open a report file. (`?live=1` is the live mode.) */
export function App() {
  const [theme, setTheme] = useTheme();
  const [source, setSource] = useState(
    EXAMPLES.find((e) => e.name === urlParams.get("report"))?.name ??
      EXAMPLES.find((e) => e.name === "support_bot_all_verdicts")?.name ??
      EXAMPLES[0]?.name ??
      "",
  );
  const [uploaded, setUploaded] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  const report: Report | undefined = source === UPLOADED && uploaded ? uploaded : EXAMPLES.find((e) => e.name === source)?.report;

  async function onFile(file: File | undefined) {
    if (!file) return;
    try {
      const parsed: unknown = JSON.parse(await file.text());
      if (!looksLikeReport(parsed)) throw new Error("This file is not a report (schema v0).");
      setUploaded(parsed);
      setSource(UPLOADED);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not read that file.");
    }
  }

  return (
    <div className="page">
      <ReportHeader report={report}>
        <div className="controls">
          <label>
            Report
            <select value={source} onChange={(e) => setSource(e.target.value)}>
              {EXAMPLES.map((e) => (
                <option key={e.name} value={e.name}>
                  Example: {e.name.replace(/_/g, " ")}
                </option>
              ))}
              {uploaded && <option value={UPLOADED}>Uploaded file</option>}
            </select>
          </label>
          <label className="file">
            Open a report file
            <input type="file" accept="application/json,.json" onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
          <label>
            Theme
            <select value={theme} onChange={(e) => setTheme(e.target.value as Theme)}>
              <option value="system">Match system</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </label>
          <a className="btn" href="?live=1">
            Live mode
          </a>
        </div>
      </ReportHeader>

      {error && <p className="error" role="alert">{error}</p>}
      {report && <ReportView report={report} />}
    </div>
  );
}
