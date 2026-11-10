import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { LiveApp } from "./live/LiveApp";
import { urlParams } from "./theme";
import "./styles.css";

// `?live=1` is the live mode (upload, run, report against the local server); otherwise the example viewer.
createRoot(document.getElementById("root")!).render(<StrictMode>{urlParams.get("live") === "1" ? <LiveApp /> : <App />}</StrictMode>);
