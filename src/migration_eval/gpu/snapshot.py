"""The one-time snapshot: what hardware is there and how it is connected.

Two commands:

    nvidia-smi -q -x     XML. GPU model and count, memory total, driver and CUDA version,
                         PCIe generation and width, power limit (current and max), max clocks, persistence mode.
    nvidia-smi topo -m   The GPU-to-GPU link matrix (NV#, PIX, PXB, PHB, NODE, SYS) plus
                         CPU affinity and NUMA affinity columns.

`topo -m` has no machine-readable form, so it is the one exception to "parse CSV/XML only".
Parse it by header names, never by column position, and keep the raw text in the profile so
a parsing gap can be fixed later without asking the partner to re-run.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from pydantic import BaseModel

from migration_eval.gpu.nvidia_smi import NvidiaSmiError, Runner

XML_ARGS = ("-q", "-x")
TOPO_ARGS = ("topo", "-m")
NVLINK_ARGS = ("nvlink", "-s")


class GpuInfo(BaseModel):
    """One GPU, from `-q -x`."""

    index: int
    uuid: str
    name: str  # e.g. "NVIDIA H100 80GB HBM3"
    pci_bus_id: str
    memory_total_mib: int
    power_limit_w: float | None = None
    max_power_limit_w: float | None = None  # the most the limit can be raised to (`nvidia-smi -pl`)
    max_sm_clock_mhz: int | None = None
    max_mem_clock_mhz: int | None = None
    pcie_gen_max: int | None = None
    pcie_width_max: int | None = None
    # Persistence mode keeps the driver loaded with no GPU process running, so a new process
    # doesn't pay driver start-up. None when not reported ("N/A" on Windows, which has no such mode).
    persistence_mode: bool | None = None


class Topology(BaseModel):
    """From `topo -m`. Keys are GPU names as `topo -m` prints them ("GPU0", "GPU1", ...)."""

    links: dict[str, dict[str, str]]  # links["GPU0"]["GPU1"] == "NV12"
    cpu_affinity: dict[str, str]  # "GPU0" -> "0-31,64-95"
    numa_affinity: dict[str, str]  # "GPU0" -> "0"; "N/A" on single-socket machines


class NvLink(BaseModel):
    """One NVLink on one GPU, from `nvidia-smi nvlink -s`."""

    link: int
    speed_gbps: float | None = None  # GB/s per direction; None when "<inactive>"
    active: bool


class Snapshot(BaseModel):
    driver_version: str
    cuda_version: str | None = None
    gpus: list[GpuInfo]
    topology: Topology | None = None  # None when `topo -m` couldn't be run or parsed
    nvlink: dict[int, list[NvLink]] | None = None  # GPU index -> links; None when not run/parsed
    raw_xml: str
    raw_topo: str
    raw_nvlink: str = ""


# --- nvidia-smi -q -x --------------------------------------------------------------------------

# Where each value lives, in the order to try. Element names have moved between drivers.
POWER_LIMIT_PATHS = (
    "gpu_power_readings/current_power_limit",
    "power_readings/current_power_limit",
    "power_readings/power_limit",
)
MAX_POWER_LIMIT_PATHS = ("gpu_power_readings/max_power_limit", "power_readings/max_power_limit")


def _text(element: ET.Element, *paths: str) -> str | None:
    """The first non-missing, non-"N/A" text found at `paths`, stripped."""
    for path in paths:
        value = element.findtext(path)
        if value is None:
            continue
        value = value.strip()
        if value and value not in ("N/A", "[N/A]") and not value.startswith("Not "):
            return value
    return None


def _number(element: ET.Element, *paths: str) -> float | None:
    """A value like "81559 MiB", "700.00 W" or "16x", without its unit."""
    value = _text(element, *paths)
    if value is None:
        return None
    match = re.match(r"[-+]?\d+(\.\d+)?", value)
    if match is None:
        return None
    return float(match.group())


def _int(element: ET.Element, *paths: str) -> int | None:
    value = _number(element, *paths)
    return None if value is None else int(value)


def _enabled(element: ET.Element, *paths: str) -> bool | None:
    """"Enabled" -> True, "Disabled" -> False, anything else (missing, "N/A") -> None."""
    value = (_text(element, *paths) or "").lower()
    return {"enabled": True, "disabled": False}.get(value)


def parse_query_xml(xml_text: str) -> tuple[str, str | None, list[GpuInfo]]:
    """Parse `nvidia-smi -q -x` into (driver_version, cuda_version, gpus).

    GPUs are indexed in the order the XML lists them, which is nvidia-smi's own index order
    (the one `--query-gpu=index` reports). `minor_number` is not used: it is the device-node
    number, can differ from the index, and is "N/A" on Windows.

    Raise ValueError if the text is not XML, has no driver version, lists no GPUs, or a GPU
    has no UUID or memory total.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"not valid XML ({exc})") from None

    driver_version = _text(root, "driver_version")
    if driver_version is None:
        raise ValueError("no <driver_version> element")
    cuda_version = _text(root, "cuda_version")

    gpus: list[GpuInfo] = []
    for index, gpu in enumerate(root.findall("gpu")):
        uuid = _text(gpu, "uuid")
        memory_total = _int(gpu, "fb_memory_usage/total")
        if uuid is None or memory_total is None:
            raise ValueError(f"GPU {index} has no UUID or memory total")
        link = "pci/pci_gpu_link_info"
        gpus.append(
            GpuInfo(
                index=index,
                uuid=uuid,
                name=_text(gpu, "product_name") or "unknown",
                pci_bus_id=gpu.get("id") or _text(gpu, "pci/pci_bus_id") or "unknown",
                memory_total_mib=memory_total,
                power_limit_w=_number(gpu, *POWER_LIMIT_PATHS),
                max_power_limit_w=_number(gpu, *MAX_POWER_LIMIT_PATHS),
                max_sm_clock_mhz=_int(gpu, "max_clocks/sm_clock"),
                max_mem_clock_mhz=_int(gpu, "max_clocks/mem_clock"),
                pcie_gen_max=_int(gpu, f"{link}/pcie_gen/max_link_gen"),
                pcie_width_max=_int(gpu, f"{link}/link_widths/max_link_width"),
                persistence_mode=_enabled(gpu, "persistence_mode"),
            )
        )
    if not gpus:
        raise ValueError("no <gpu> elements")
    return driver_version, cuda_version, gpus


# --- nvidia-smi topo -m ------------------------------------------------------------------------

# topo -m underlines its header with terminal escape codes, even when piped on some drivers.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_GPU_NAME = re.compile(r"GPU\d+")
CPU_AFFINITY = "CPU Affinity"
NUMA_AFFINITY = "NUMA Affinity"


def _cells(line: str) -> list[str]:
    """Split a topo line on tabs, dropping empty cells.

    Dropping empties matters: rows have an extra tab before "GPU NUMA ID" that the header
    doesn't, so keeping them would shift every later column by one.
    """
    return [cell.strip() for cell in _ANSI.sub("", line).split("\t") if cell.strip()]


def parse_topo(topo_text: str) -> Topology:
    """Parse `nvidia-smi topo -m`.

    The header is the first line that names "CPU Affinity". It lists the device columns
    (GPU0, GPU1, and NIC0... on machines with NICs), then the affinity columns. Each row
    starts with its device name. Read rows until the blank line before "Legend:". NIC rows
    are skipped; NIC columns are kept in `links`, since GPU-to-NIC placement matters too.

    Raise ValueError if there's no such header, it names no GPUs, or a GPU row is short.
    """
    lines = topo_text.splitlines()
    header_at = next((i for i, line in enumerate(lines) if CPU_AFFINITY in line), None)
    if header_at is None:
        raise ValueError(f"no header line naming '{CPU_AFFINITY}'")
    header = _cells(lines[header_at])
    if not any(_GPU_NAME.fullmatch(name) for name in header):
        raise ValueError("the header names no GPU columns")
    device_columns = header[: header.index(CPU_AFFINITY)]

    links: dict[str, dict[str, str]] = {}
    cpu_affinity: dict[str, str] = {}
    numa_affinity: dict[str, str] = {}
    for line in lines[header_at + 1 :]:
        if not line.strip() or line.strip().startswith("Legend"):
            break
        cells = _cells(line)
        name, values = cells[0], cells[1:]
        if not _GPU_NAME.fullmatch(name):
            continue  # a NIC row
        # Trailing columns (NUMA Affinity, GPU NUMA ID) may be missing; up to CPU Affinity may not.
        if len(values) < len(device_columns) + 1:
            raise ValueError(f"row {name} has {len(values)} values for {len(header)} columns")
        row = dict(zip(header, values))
        links[name] = {col: row[col] for col in device_columns if col != name and col in row}
        cpu_affinity[name] = row[CPU_AFFINITY]
        if NUMA_AFFINITY in row:
            numa_affinity[name] = row[NUMA_AFFINITY]

    if not links:
        raise ValueError("no GPU rows")
    return Topology(links=links, cpu_affinity=cpu_affinity, numa_affinity=numa_affinity)


# --- nvidia-smi nvlink -s ----------------------------------------------------------------------


def parse_nvlink_status(text: str) -> dict[int, list[NvLink]]:
    """Parse `nvidia-smi nvlink -s` (GPU-1).

    Blocks of "GPU 0: <name> (UUID: ...)" followed by indented "Link 0: 26.562 GB/s" or
    "Link 1: <inactive>" lines. GPUs without NVLink print no Link lines (or "NVML: Not
    Supported"): give them an empty list. Raise ValueError if there are no "GPU N:" lines.
    """
    links: dict[int, list[NvLink]] = {}
    current: list[NvLink] | None = None
    for line in _ANSI.sub("", text).splitlines():
        if m := _NVLINK_GPU.match(line):
            current = links.setdefault(int(m.group(1)), [])
            continue
        if current is None:
            continue
        if m := _NVLINK_LINK.match(line):
            speed = re.match(r"([\d.]+)\s*GB/s", m.group(2).strip())
            if speed:
                current.append(NvLink(link=int(m.group(1)), speed_gbps=float(speed.group(1)), active=True))
            else:  # "<inactive>"
                current.append(NvLink(link=int(m.group(1)), active=False))
    if not links:
        raise ValueError('no "GPU N:" lines')
    return links


_NVLINK_GPU = re.compile(r"^GPU (\d+):")
_NVLINK_LINK = re.compile(r"^\s*Link (\d+):\s*(.*)$")


# --- all of them -------------------------------------------------------------------------------


def take_snapshot(run: Runner) -> Snapshot:
    """Run both commands and parse them.

    `topo -m` is not fatal: if it fails to run (unsupported in some VMs and containers) or
    to parse, set `topology=None` and keep whatever raw text there was. `-q -x` is fatal,
    since without it we don't know what GPUs there are: raise NvidiaSmiError.
    """
    raw_xml = run(XML_ARGS)
    try:
        driver_version, cuda_version, gpus = parse_query_xml(raw_xml)
    except ValueError as exc:
        raise NvidiaSmiError(f"Could not read `nvidia-smi -q -x` output: {exc}.") from None

    try:
        raw_topo = run(TOPO_ARGS)
    except NvidiaSmiError:
        raw_topo = ""
    try:
        topology: Topology | None = parse_topo(raw_topo)
    except ValueError:
        topology = None

    try:
        raw_nvlink = run(NVLINK_ARGS)
    except NvidiaSmiError:
        raw_nvlink = ""
    try:
        nvlink: dict[int, list[NvLink]] | None = parse_nvlink_status(raw_nvlink)
    except ValueError:
        nvlink = None

    return Snapshot(
        driver_version=driver_version,
        cuda_version=cuda_version,
        gpus=gpus,
        topology=topology,
        nvlink=nvlink,
        raw_xml=raw_xml,
        raw_topo=raw_topo,
        raw_nvlink=raw_nvlink,
    )
