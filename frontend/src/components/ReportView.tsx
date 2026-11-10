import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { Report } from "../../../ts/report.v0";
import { formatDate } from "../format";
import { Candidates, candidateKey } from "./Candidates";
import { Detail } from "./Detail";
import { Dropped } from "./Dropped";
import { Errors } from "./Errors";
import { Baseline, VerdictCounts } from "./Summary";

/** The title and the one-line facts about a report. `children` sit on the right (controls, navigation). */
export function ReportHeader({ report, children }: { report: Report | undefined; children?: ReactNode }) {
  return (
    <header className="top">
      <div>
        <p className="eyebrow">Model migration report</p>
        <h1>{report ? report.component : "No report"}</h1>
        {report && (
          <p className="meta">
            Generated {formatDate(report.generated_at)} · {report.sample.requests_replayed} of {report.sample.records_available} logged requests sampled · catalog
            as of {report.catalog.as_of} (<abbr title={report.catalog.sha256}>{report.catalog.sha256.slice(0, 8)}</abbr>)
          </p>
        )}
      </div>
      {children}
    </header>
  );
}

/** Everything under the header: the verdicts, the baseline, the candidates and their details. */
export function ReportView({ report }: { report: Report }) {
  const [selected, setSelected] = useState<string | undefined>(undefined);

  // Show the first candidate's details by default, and again whenever a different report is opened.
  const firstKey = report.candidates[0] ? candidateKey(report.candidates[0]) : undefined;
  useEffect(() => setSelected(firstKey), [report, firstKey]);
  const candidate = useMemo(() => report.candidates.find((c) => candidateKey(c) === selected), [report, selected]);

  return (
    <main>
      <VerdictCounts report={report} />
      <Baseline report={report} />
      <Candidates report={report} selectedKey={selected} onSelect={setSelected} />
      {candidate && <Detail report={report} candidate={candidate} />}
      <Errors errors={report.errors} />
      <Dropped dropped={report.dropped_candidates} />
    </main>
  );
}
