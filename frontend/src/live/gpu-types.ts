// The GPU report (GPU-5 to GPU-9), mirrored by hand from src/migration_eval/gpu/report.py and the
// models it uses (assessment.py, recommendations.py, costs.py, checklist.py, labels.py), plus
// GpuReportInfo from src/migration_eval/server/models.py. Keep them in step.

export type Severity = "info" | "warning" | "critical";
export type Level = "low" | "medium" | "high";
export type Stage = "fix" | "runtime" | "capacity";

/** A reading from the partner's own capture (GPU-8). */
export interface Stat {
  basis: "measured";
  mean: number;
  p50: number;
  p95: number;
  min: number;
  max: number;
  samples: number;
}

/** Our estimate, always a range with its assumptions (GPU-8). */
export interface Estimate {
  basis: "estimated";
  low: number;
  high: number;
  unit: string; // "%", "GPUs", "USD/month", "USD/year", "USD per 1M tokens"
  assumptions: string[];
}

export interface GpuUsage {
  index: number;
  name: string;
  utilization_pct: Stat | null;
  memory_used_mib: Stat | null;
  power_draw_w: Stat | null;
  temperature_c: Stat | null;
  sm_clock_mhz: Stat | null;
  memory_total_mib: number | null;
  power_limit_w: number | null;
  memory_headroom_pct: number | null;
  power_headroom_pct: number | null;
}

export interface Imbalance {
  basis: "measured";
  spread_pct: number;
  coefficient_of_variation: number;
  busiest: number;
  idlest: number;
}

export interface Throttling {
  basis: "measured";
  index: number;
  fraction_by_reason: Record<string, number>;
  slowdown_fraction: number;
  thermal_fraction: number;
  power_fraction: number;
  sm_clock_p50_when_slowed_mhz: number | null;
}

export interface TopologySummary {
  interconnect: "nvlink" | "pcie" | "mixed" | "single_gpu" | "unknown";
  link_counts: Record<string, number>;
  cross_socket_pairs: [number, number][];
  gpus_by_numa_node: Record<string, number[]>;
  active_nvlinks_by_gpu: Record<string, number>;
  pcie: Record<string, string>;
  serving_cpus_outside_gpu_numa: number[];
}

export interface Finding {
  kind: string;
  severity: Severity;
  gpus: number[];
  message: string;
  basis: "measured";
}

export interface GpuAssessment {
  gpu_count: number;
  usage: GpuUsage[];
  imbalance: Imbalance | null;
  throttling: Throttling[];
  topology: TopologySummary;
  findings: Finding[];
  missing: string[];
}

export interface Recommendation {
  id: string;
  title: string;
  area: string;
  stage: Stage;
  action: string;
  experiment: string;
  expected_gain: Estimate;
  effort: Level;
  risk: Level;
  gpus: number[];
  based_on: string[];
}

export interface Consolidation {
  current_gpus: number;
  needed_gpus: Estimate;
  limited_by: string;
}

export interface Sizing {
  gpu: string;
  count: Estimate;
  monthly_cost: Estimate;
}

export interface CostEstimates {
  current_monthly_cost: Estimate | null;
  consolidation: Consolidation | null;
  sizing: Sizing | null;
  cost_per_million_tokens: Estimate | null;
  monthly_savings: Estimate | null;
  annual_savings: Estimate | null;
  missing: string[];
}

export interface ChecklistItem {
  step: number;
  stage: Stage;
  title: string;
  action: string;
  expected_gain: Estimate;
  risk: Level;
  source: "recommendation" | "cost";
  source_id: string;
}

export interface GpuReport {
  hostname: string;
  assessment: GpuAssessment;
  recommendations: Recommendation[];
  costs: CostEstimates | null;
  checklist: ChecklistItem[];
  untested_notice: string;
}

export interface GpuReportInfo {
  id: string;
  filename: string;
  hostname: string;
  created_at: string;
  gpu_count: number;
  gpu_names: string[];
  recommendations: number;
  has_costs: boolean;
  upload_id: string | null;
}
