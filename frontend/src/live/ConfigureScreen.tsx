import { useEffect, useState } from "react";
import { ApiError, getProviders, startRun } from "./api";
import type { ProviderStatus, RunStatus, Tier, UploadInfo } from "./api-types";
import { formatDuration, suggestedSpeed, usd } from "./format";

export interface ConfigureScreenProps {
  upload: UploadInfo;
  component: string;
  onStarted: (run: RunStatus) => void; // called once the server has started the run
  onBack: () => void;
}

// How many different prompts each tier replays at most. These mirror run.TIER_SAMPLE_SIZE.
const TIER_PROMPTS: Record<Tier, number> = { smoke: 50, full: 200 };

/**
 * Step 2: set up the run. Provider keys are never typed here: a provider with no key on the server is
 * disabled and names the environment variable to set. The current model's provider is always listed
 * as "will be contacted" because the baseline replays it. "Start" opens a confirmation that says in
 * plain words what will be sent and where; only its confirm button starts the run (`confirmed: true`).
 */
export function ConfigureScreen({ upload, component, onStarted, onBack }: ConfigureScreenProps) {
  const info = upload.components.find((c) => c.name === component);
  const span =
    upload.first_started_at && upload.last_started_at
      ? Math.max(0, (new Date(upload.last_started_at).getTime() - new Date(upload.first_started_at).getTime()) / 1000)
      : 0;
  const suggestion = suggestedSpeed(span);

  const [providers, setProviders] = useState<ProviderStatus[] | null>(null);
  const [tier, setTier] = useState<Tier>("smoke");
  const [cap, setCap] = useState("1.00");
  const [costCap, setCostCap] = useState("");
  const [speed, setSpeed] = useState(String(suggestion));
  const [allow, setAllow] = useState<string[]>([]);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getProviders().then(setProviders, (e) => setError(e instanceof ApiError ? e.message : "Could not load the providers."));
  }, []);

  if (!info) return <p className="error">That component is not in the upload.</p>;

  const incumbent = info.incumbent_provider;
  const contacted = [...new Set([...allow, ...(incumbent ? [incumbent] : [])])].sort();
  const keyFor = (name: string) => providers?.find((p) => p.name === name);
  const incumbentKey = incumbent ? keyFor(incumbent) : undefined;
  const capValue = Number(cap);
  const speedValue = Number(speed);
  const prompts = Math.min(info.replay_grade, TIER_PROMPTS[tier]);

  const problem =
    providers === null ? "Loading the providers…"
    : allow.length === 0 ? "Choose at least one provider candidates may be called on."
    : !(capValue > 0) ? "The spending cap must be more than 0."
    : costCap !== "" && !(Number(costCap) > 0) ? "The monthly cost cap must be more than 0 (or left empty)."
    : !(speedValue > 0) ? "The replay speed must be more than 0."
    : incumbentKey && !incumbentKey.key_configured ? `The current model's provider needs a key too: set ${incumbentKey.key_variable} where the server runs.`
    : null;

  function toggle(name: string, on: boolean) {
    setAllow((current) => (on ? [...current, name] : current.filter((n) => n !== name)));
  }

  async function start() {
    setBusy(true);
    setError(null);
    try {
      onStarted(
        await startRun({
          upload_id: upload.id, component, tier, spending_cap_usd: capValue, allow,
          cost_cap_usd: costCap === "" ? null : Number(costCap), replay_speed: speedValue, confirmed: true,
        }),
      );
    } catch (e) {
      setConfirming(false);
      setError(e instanceof ApiError ? e.message : "Could not start the run.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <section aria-labelledby="config-h">
        <h2 id="config-h">2. Set up the run for {component}</h2>
        <p className="section-note">
          {info.records} logged calls, {info.replay_grade} can be replayed. Current model: <strong>{incumbent ? `${incumbent}/${info.incumbent_model}` : "unknown"}</strong>.
        </p>

        <form className="card form" onSubmit={(e) => { e.preventDefault(); if (!problem) setConfirming(true); }}>
          <label>
            Size
            <select value={tier} onChange={(e) => setTier(e.target.value as Tier)}>
              <option value="smoke">Smoke test (up to {TIER_PROMPTS.smoke} prompts)</option>
              <option value="full">Full (up to {TIER_PROMPTS.full} prompts)</option>
            </select>
          </label>

          <label>
            Most to spend on provider calls (USD)
            <input type="number" min="0" step="0.01" value={cap} onChange={(e) => setCap(e.target.value)} />
            <span className="hint">The run stops sending when the next call could pass this.</span>
          </label>

          <label>
            Drop candidates that would cost more than this a month (USD, optional)
            <input type="number" min="0" step="1" value={costCap} onChange={(e) => setCostCap(e.target.value)} />
          </label>

          <label>
            Replay speed
            <input type="number" min="0" step="any" value={speed} onChange={(e) => setSpeed(e.target.value)} />
            <span className="hint">
              {span > 0
                ? `Your logs span ${formatDuration(span)}, and 1× replays at the logged pace, so one pass would take that long. ${suggestion}× fits a pass into about ten minutes.`
                : "1× replays at the logged pace."}{" "}
              Latency numbers are only comparable at 1×.
            </span>
          </label>

          <fieldset>
            <legend>Providers candidates may be called on</legend>
            {providers === null && <p className="muted-text">Loading…</p>}
            {providers?.map((p) => (
              <label key={p.name} className="check">
                <input type="checkbox" disabled={!p.key_configured} checked={allow.includes(p.name)} onChange={(e) => toggle(p.name, e.target.checked)} />
                <span>
                  {p.name}
                  {!p.key_configured && <span className="hint"> — no key: set {p.key_variable} where the server runs</span>}
                </span>
              </label>
            ))}
            <p className="hint">Keys are read from the server&rsquo;s environment. They are never typed here and never leave the server.</p>
          </fieldset>

          {incumbent && !allow.includes(incumbent) && (
            <p className="notice">
              The current model&rsquo;s provider, <strong>{incumbent}</strong>, will be contacted too: the run replays the current model to see how much it varies on its own.
            </p>
          )}

          {error && <p className="error" role="alert">{error}</p>}
          {problem && !error && <p className="hint">{problem}</p>}

          <div className="actions">
            <button type="button" className="btn" onClick={onBack}>Back</button>
            <button type="submit" className="btn primary" disabled={problem !== null || busy}>Review and start…</button>
          </div>
        </form>
      </section>

      {confirming && (
        <div className="modal-backdrop" role="presentation" onKeyDown={(e) => e.key === "Escape" && setConfirming(false)}>
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="confirm-h">
            <h2 id="confirm-h">Send these prompts to the providers?</h2>
            <p>
              This will send up to <strong>{prompts}</strong> different {prompts === 1 ? "prompt" : "prompts"} from <strong>{component}</strong> to{" "}
              <strong>{contacted.join(", ")}</strong>, each one to several models.
            </p>
            <p>
              Spending is capped at <strong>{usd(capValue)}</strong>. Once sent, a prompt cannot be taken back.
            </p>
            <div className="actions">
              <button type="button" className="btn" autoFocus onClick={() => setConfirming(false)}>Cancel</button>
              <button type="button" className="btn primary" disabled={busy} onClick={() => void start()}>
                {busy ? "Starting…" : "Send and start"}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
