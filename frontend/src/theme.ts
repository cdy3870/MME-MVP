import { useEffect, useState } from "react";

export type Theme = "system" | "light" | "dark";

// ?theme=light|dark is honoured on load, handy for linking and screenshots.
const params = new URLSearchParams(window.location.search);
export const urlParams = params;

export function useTheme(): [Theme, (t: Theme) => void] {
  const initial = params.get("theme");
  const [theme, setTheme] = useState<Theme>(initial === "light" || initial === "dark" ? initial : "system");
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
  }, [theme]);
  return [theme, setTheme];
}
