import { useEffect, useState } from "react";
import type { Report } from "../../../ts/report.v0";
import { ReportHeader, ReportView } from "../components/ReportView";
import { urlParams, useTheme, type Theme } from "../theme";
import { GpuReportView } from "../components/GpuReportView";
import { formatDate } from "../format";
import { ApiError, getGpuReport, getReport, getRun } from "./api";
import type { RunStatus, RunSummary, UploadInfo } from "./api-types";
import { ConfigureScreen } from "./ConfigureScreen";
import type { GpuReport } from "./gpu-types";
import { GpuScreen } from "./GpuScreen";
import { RunScreen } from "./RunScreen";
import { RunsScreen } from "./RunsScreen";
import { UploadScreen } from "./UploadScreen";

// A run whose report is ready to read: finished, or stopped with what it had.
const READABLE = ["done", "cancelled"];

// Where the user is in the flow. The report step shows the existing viewer, so it does not change.
export type Step =
  | { name: "upload" }
  | { name: "configure"; upload: UploadInfo; component: string }
  | { name: "run"; run: RunStatus }
  | { name: "report"; run: RunStatus; report: Report }
  | { name: "runs" }
  | { name: "gpu" }
  | { name: "gpuReport"; id: string; report: GpuReport; createdAt?: string };

// Keep `?live=1` (and the theme) in the address bar, plus `?run=<id>` while a run is open or
// `?gpu=<id>` while a GPU report is, so a refresh or a shared link comes back to the same place.
function setAddress(open?: { run?: string; gpu?: string }) {
  const next = new URLSearchParams({ live: "1" });
  const theme = urlParams.get("theme");
  if (theme) next.set("theme", theme);
  if (open?.run) next.set("run", open.run);
  if (open?.gpu) next.set("gpu", open.gpu);
  window.history.replaceState(null, "", `?${next.toString()}`);
}

/**
 * The live mode: upload an export, run it, read the report. Open it with `?live=1`; `?run=<id>` opens
 * a particular run (its report when it has one, its progress otherwise).
 */
export function LiveApp() {
  const [theme, setTheme] = useTheme();
  const [step, setStep] = useState<Step>({ name: "upload" });
  const [error, setError] = useState<string | null>(null);

  async function openRun(id: string) {
    try {
      const run = await getRun(id);
      if (run.has_report && READABLE.includes(run.state)) {
        setStep({ name: "report", run, report: await getReport(id) });
      } else {
        setStep({ name: "run", run });
      }
      setAddress({ run: id });
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not open that run.");
      setStep({ name: "upload" });
      setAddress();
    }
  }

  async function openGpuReport(id: string, createdAt?: string) {
    try {
      setStep({ name: "gpuReport", id, report: await getGpuReport(id), createdAt });
      setAddress({ gpu: id });
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not open that GPU report.");
      setStep({ name: "gpu" });
      setAddress();
    }
  }

  useEffect(() => {
    const run = urlParams.get("run");
    const gpu = urlParams.get("gpu");
    if (run) void openRun(run);
    else if (gpu) void openGpuReport(gpu);
    else setAddress();
  }, []);

  function go(next: Step) {
    setError(null);
    setStep(next);
    if (next.name === "run" || next.name === "report") setAddress({ run: next.run.id });
    else if (next.name === "gpuReport") setAddress({ gpu: next.id });
    else setAddress();
  }

  const nav = (
    <div className="controls">
      <button type="button" className="btn" onClick={() => go({ name: "upload" })}>New upload</button>
      <button type="button" className="btn" onClick={() => go({ name: "runs" })}>Past runs</button>
      <button type="button" className="btn" onClick={() => go({ name: "gpu" })}>GPU report</button>
      <label>
        Theme
        <select value={theme} onChange={(e) => setTheme(e.target.value as Theme)}>
          <option value="system">Match system</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
    </div>
  );

  return (
    <div className="page">
      {step.name === "report" ? (
        <ReportHeader report={step.report}>{nav}</ReportHeader>
      ) : step.name === "gpu" || step.name === "gpuReport" ? (
        <header className="top">
          <div>
            <p className="eyebrow">GPU report</p>
            <h1>{step.name === "gpuReport" ? step.report.hostname : "Your GPU serving setup"}</h1>
            <p className="meta">
              {step.name === "gpuReport"
                ? `${step.report.assessment.gpu_count} GPU${step.report.assessment.gpu_count === 1 ? "" : "s"}${step.createdAt ? `, report built ${formatDate(step.createdAt)}` : ""}`
                : "Performance, configuration changes and cost, from one capture on your GPU machine."}
            </p>
          </div>
          {nav}
        </header>
      ) : (
        <header className="top">
          <div>
            <p className="eyebrow">Model migration</p>
            <h1>Evaluate a model swap</h1>
            <p className="meta">Upload your logged calls, replay them against other models, and read the report.</p>
          </div>
          {nav}
        </header>
      )}

      {error && <p className="error" role="alert">{error}</p>}

      {step.name === "upload" && (
        <UploadScreen onChoose={(upload, component) => go({ name: "configure", upload, component })} onShowRuns={() => go({ name: "runs" })} />
      )}
      {step.name === "configure" && (
        <ConfigureScreen
          upload={step.upload}
          component={step.component}
          onStarted={(run) => go({ name: "run", run })}
          onBack={() => go({ name: "upload" })}
        />
      )}
      {step.name === "run" && <RunScreen key={step.run.id} run={step.run} onViewReport={(run) => void openRun(run.id)} />}
      {step.name === "runs" && <RunsScreen onOpen={(run: RunSummary) => void openRun(run.id)} onBack={() => go({ name: "upload" })} />}
      {step.name === "gpu" && <GpuScreen onOpen={(info) => void openGpuReport(info.id, info.created_at)} />}
      {step.name === "gpuReport" && <GpuReportView report={step.report} />}
      {step.name === "report" && (
        <>
          {step.run.state === "cancelled" && <p className="notice">This run was stopped early, so candidates that had not finished are marked &ldquo;insufficient data&rdquo;.</p>}
          <ReportView report={step.report} />
        </>
      )}
    </div>
  );
}
