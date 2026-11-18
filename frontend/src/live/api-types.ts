// The server's request and response shapes, mirrored by hand from src/migration_eval/server/models.py.
// FastAPI publishes the real ones at /api/openapi.json, so these can be generated from that instead
// (an open decision); keep the two in step until then. The report itself uses ../../../ts/report.v0.

export type RunState = "queued" | "running" | "cancelling" | "done" | "failed" | "cancelled";
export type Tier = "smoke" | "full";

export interface ProviderStatus {
  name: string; // "openai", "anthropic", "gemini"
  key_variable: string; // the environment variable the server reads, e.g. "ANTHROPIC_API_KEY"
  key_configured: boolean; // never the key itself
}

export interface ComponentInfo {
  name: string;
  records: number;
  replay_grade: number;
  incumbent_provider: string | null;
  incumbent_model: string | null;
}

export interface UploadInfo {
  id: string;
  filename: string;
  size_bytes: number;
  record_count: number;
  first_started_at: string | null;
  last_started_at: string | null;
  components: ComponentInfo[];
}

export interface RunRequest {
  upload_id: string;
  component: string;
  tier: Tier;
  spending_cap_usd: number;
  allow: string[]; // at least one provider
  cost_cap_usd: number | null;
  replay_speed: number; // 1 is the logged pace; higher compresses the gaps (latency is only comparable at 1)
  confirmed: boolean; // true only after the user has seen what will be sent and where
}

export interface RunStatus {
  id: string;
  upload_id: string;
  component: string;
  state: RunState;
  stage: string | null; // "prepare", "candidates", "baseline", "candidate <provider>/<model>", "report"
  stages_done: string[];
  spent_usd: number | null; // read live while the run is going
  spending_cap_usd: number | null;
  started_at: string;
  finished_at: string | null;
  error: string | null;
  has_report: boolean;
}

export interface RunSummary {
  id: string;
  component: string;
  state: RunState;
  started_at: string;
  verdicts: Record<string, number> | null;
}
