import { useRef, useState } from "react";
import { formatDate } from "../format";
import { ApiError, deleteUpload, uploadArchive } from "./api";
import type { UploadInfo } from "./api-types";
import { formatBytes } from "./format";

export interface UploadScreenProps {
  onChoose: (upload: UploadInfo, component: string) => void; // the user picked a component to evaluate
  onShowRuns: () => void;
}

/**
 * Step 1: pick the export. A drop zone and a file button accept the `.tar.gz` that `migration-eval
 * export` writes. On accept the screen shows what the server found: the record count, the time range,
 * and a table of components (records, replay-grade share, current model) with a button to evaluate
 * each. When the server refuses the file, its reason is shown and another file can be picked. The file
 * stays on this machine and can be removed.
 */
export function UploadScreen({ onChoose, onShowRuns }: UploadScreenProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [upload, setUpload] = useState<UploadInfo | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function send(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setUpload(await uploadArchive(file));
    } catch (e) {
      setUpload(null);
      setError(e instanceof ApiError ? e.message : "The upload failed.");
    } finally {
      setBusy(false);
      if (input.current) input.current.value = ""; // so the same file can be picked again
    }
  }

  async function remove() {
    if (!upload) return;
    try {
      await deleteUpload(upload.id);
      setUpload(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not remove the file.");
    }
  }

  return (
    <main>
      <section aria-labelledby="upload-h">
        <h2 id="upload-h">1. Upload your export</h2>
        <p className="section-note">
          Create it in a terminal with <code>migration-eval export</code>, then drop the <code>.tar.gz</code> here. The file stays on this machine
          and you can remove it afterwards.
        </p>

        <div
          className={`dropzone${dragging ? " over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void send(e.dataTransfer.files[0]);
          }}
        >
          <p>{busy ? "Checking the file…" : "Drop the export file here"}</p>
          <button type="button" className="btn" disabled={busy} onClick={() => input.current?.click()}>
            Choose a file
          </button>
          <input ref={input} type="file" accept=".gz,.tgz,application/gzip" hidden onChange={(e) => void send(e.target.files?.[0])} />
        </div>

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </section>

      {upload && (
        <section aria-labelledby="found-h">
          <h2 id="found-h">What is in {upload.filename}</h2>
          <p className="section-note">
            {upload.record_count} records, {formatBytes(upload.size_bytes)}
            {upload.first_started_at && upload.last_started_at ? `, from ${formatDate(upload.first_started_at)} to ${formatDate(upload.last_started_at)}` : ""}. Pick the
            component to evaluate.
          </p>
          <div className="card table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Component</th>
                  <th scope="col" className="num">Records</th>
                  <th scope="col" className="num">Can be replayed</th>
                  <th scope="col">Current model</th>
                  <th scope="col"><span className="visually-hidden">Action</span></th>
                </tr>
              </thead>
              <tbody>
                {upload.components.map((c) => (
                  <tr key={c.name} style={{ cursor: "default" }}>
                    <th scope="row">{c.name}</th>
                    <td className="num">{c.records}</td>
                    <td className="num">{c.replay_grade}</td>
                    <td>{c.incumbent_provider ? `${c.incumbent_provider}/${c.incumbent_model}` : "unknown"}</td>
                    <td className="num">
                      <button type="button" className="btn primary" disabled={c.replay_grade === 0} onClick={() => onChoose(upload, c.name)}>
                        Evaluate
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fineprint">
            <button type="button" className="link" onClick={() => void remove()}>
              Remove this file from the server
            </button>
          </p>
        </section>
      )}

      <p>
        <button type="button" className="link" onClick={onShowRuns}>
          Past runs
        </button>
      </p>
    </main>
  );
}
