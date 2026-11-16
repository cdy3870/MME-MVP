"""Temporary: plot the samples in one or more gpu_profile.json files.

Six panels against seconds since each capture started: GPU utilization, memory used, power
draw, temperature, SM clock, and which throttle reasons were active. Pass several files to
compare them (e.g. idle vs under load).

Needs matplotlib, which is not a project dependency:

    pip install matplotlib
    python scripts/plot_gpu_profile.py gpu_profile_test.json gpu_profile_load.json
    python scripts/plot_gpu_profile.py gpu_profile_load.json --out load.png --show
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# nvidia-smi clocks_event_reasons bits. 0x1 (idle) means "no work", not a slowdown.
THROTTLE_BITS = {
    0x1: "GPU idle",
    0x2: "App clocks setting",
    0x4: "SW power cap",
    0x8: "HW slowdown",
    0x10: "Sync boost",
    0x20: "SW thermal",
    0x40: "HW thermal",
    0x80: "HW power brake",
    0x100: "Display clock",
}

# (sample field, panel title, y label, matching limit field in the sample, or None)
PANELS = (
    ("utilization_gpu_pct", "GPU utilization", "%", None),
    ("memory_used_mib", "Memory used", "MiB", "memory_total_mib"),
    ("power_draw_w", "Power draw", "W", "power_limit_w"),
    ("temperature_c", "Temperature", "°C", None),
    ("sm_clock_mhz", "SM clock", "MHz", None),
)


def load(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not profile.get("samples"):
        sys.exit(f"{path} has no samples.")
    return profile


def by_gpu(profile: dict) -> dict[int, list[dict]]:
    """Samples grouped by GPU index, each with `t` = seconds since the capture's first sample."""
    start = min(datetime.fromisoformat(s["taken_at"]) for s in profile["samples"])
    groups: dict[int, list[dict]] = {}
    for s in profile["samples"]:
        t = (datetime.fromisoformat(s["taken_at"]) - start).total_seconds()
        groups.setdefault(s["index"], []).append({**s, "t": t})
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiles", nargs="+", type=Path, help="gpu_profile.json files.")
    parser.add_argument("--out", type=Path, help="PNG to write. Default: <first file>.png")
    parser.add_argument("--show", action="store_true", help="Also open a window.")
    args = parser.parse_args()

    try:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")  # no window needed, so don't require a GUI backend
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is not installed: pip install matplotlib")

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    panels = list(axes.flat)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    titles = []
    series = 0

    for path in args.profiles:
        profile = load(path)
        gpus = {g["index"]: g["name"] for g in profile["snapshot"]["gpus"]}
        titles.append(f"{path.name}: {', '.join(sorted(set(gpus.values())))} (driver {profile['snapshot']['driver_version']})")

        for index, rows in sorted(by_gpu(profile).items()):
            color = colors[series % len(colors)]
            label = f"{path.stem} · GPU{index}"
            ts = [r["t"] for r in rows]

            for ax, (field, title, unit, limit_field) in zip(panels, PANELS):
                ys = [r[field] for r in rows]
                ax.plot(ts, [y if y is not None else float("nan") for y in ys], color=color, label=label, lw=1.5)
                if limit_field:
                    limits = [r[limit_field] for r in rows if r[limit_field] is not None]
                    if limits:
                        ax.axhline(max(limits), color=color, ls="--", lw=0.8, alpha=0.6)
                ax.set_title(title + (" (dashed = limit)" if limit_field else ""))
                ax.set_ylabel(unit)

            # Throttle panel: one row per reason, a dot wherever that bit was set.
            ax = panels[5]
            offset = (series % 4) * 0.12 - 0.18  # nudge files apart so dots don't overlap
            for row_number, (bit, _) in enumerate(THROTTLE_BITS.items()):
                hits = [r["t"] for r in rows if r["throttle_reasons"] is not None and r["throttle_reasons"] & bit]
                if hits:
                    ax.scatter(hits, [row_number + offset] * len(hits), color=color, s=14, label=None)
            series += 1

    ax = panels[5]
    ax.set_title("Throttle reasons active")
    ax.set_yticks(range(len(THROTTLE_BITS)), list(THROTTLE_BITS.values()))
    ax.set_ylim(-0.6, len(THROTTLE_BITS) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    panels[0].set_ylim(0, 105)
    for ax in panels[:5]:
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("seconds since capture start")
    panels[0].legend(loc="best", fontsize=8)

    fig.suptitle("\n".join(titles), fontsize=10)
    fig.tight_layout()

    out = args.out or args.profiles[0].with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
