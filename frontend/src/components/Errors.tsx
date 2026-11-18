import type { CallFailure } from "../../../ts/report.v0";

/** Why calls failed, so a run where something went wrong says what. Shown only when there were failures. */
export function Errors({ errors }: { errors: CallFailure[] }) {
  if (errors.length === 0) return null;
  return (
    <section aria-labelledby="errors-h">
      <h2 id="errors-h">Calls that failed</h2>
      <p className="section-note">
        The most common error messages from the provider. A failed call counts as a disagreement, so a model whose calls mostly fail looks
        worse than it is.
      </p>
      <div className="card">
        <ul className="plain">
          {errors.map((e) => (
            <li key={`${e.where}|${e.message}`}>
              <strong>{e.count}</strong> × <span className="muted-text">{e.where}</span>: <code>{e.message}</code>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
