// The page is always dark. ?theme=light is still honoured on load, for screenshots and printing.
const params = new URLSearchParams(window.location.search);
export const urlParams = params;

export function applyTheme(): void {
  document.documentElement.setAttribute("data-theme", params.get("theme") === "light" ? "light" : "dark");
}
