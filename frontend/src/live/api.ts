// The browser's side of the server's API (see src/migration_eval/server/app.py for the routes).
// Every call goes to a relative /api path: the Vite dev server proxies it to the Python server, and a
// built frontend is served by that same server, so there is one origin and no CORS.

import type { Report } from "../../../ts/report.v0";
import type { ProviderStatus, RunRequest, RunStatus, RunSummary, UploadInfo } from "./api-types";

/** A failed request. `message` is the server's `detail`, already worded for the user. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

// The one place requests are made: send `init`, return the parsed JSON, and on a non-2xx answer throw
// an ApiError built from the body's `detail` (fall back to the status text when there is none). A 204
// has no body. Nothing but the path and the user's own input goes into a request.
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("Could not reach the server. Is `migration-eval serve` running?", 0);
  }

  if (!response.ok) {
    let message = response.statusText || `The request failed (${response.status}).`;
    try {
      const body: { detail?: unknown } = await response.json();
      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body.detail)) {
        // FastAPI's own validation errors: a list of {loc, msg}
        message = body.detail.map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? "invalid"}`).join("; ");
      }
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(message, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const post = (body?: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

/** GET /api/providers: which providers have a key set on the server. */
export const getProviders = (): Promise<ProviderStatus[]> => request("/api/providers");

/** POST /api/uploads with the file as multipart form data. Rejects with the reason when the archive is refused. */
export function uploadArchive(file: File): Promise<UploadInfo> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/uploads", { method: "POST", body: form }); // the browser sets the multipart boundary
}

/** GET /api/uploads/{id}. */
export const getUpload = (id: string): Promise<UploadInfo> => request(`/api/uploads/${id}`);

/** DELETE /api/uploads/{id}. */
export const deleteUpload = (id: string): Promise<void> => request(`/api/uploads/${id}`, { method: "DELETE" });

/** POST /api/runs. The server refuses it unless `confirmed` is true. */
export const startRun = (body: RunRequest): Promise<RunStatus> => request("/api/runs", post(body));

/** GET /api/runs: past runs, newest first. */
export const listRuns = (): Promise<RunSummary[]> => request("/api/runs");

/** GET /api/runs/{id}: poll this while a run is queued, running or cancelling. */
export const getRun = (id: string): Promise<RunStatus> => request(`/api/runs/${id}`);

/** POST /api/runs/{id}/cancel. */
export const cancelRun = (id: string): Promise<RunStatus> => request(`/api/runs/${id}/cancel`, post());

/** GET /api/runs/{id}/report: the report the viewer draws. */
export const getReport = (id: string): Promise<Report> => request(`/api/runs/${id}/report`);

/** DELETE /api/runs/{id}: removes the run and its report from the server. */
export const deleteRun = (id: string): Promise<void> => request(`/api/runs/${id}`, { method: "DELETE" });
