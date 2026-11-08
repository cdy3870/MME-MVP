"""The one-time snapshot: what hardware is there and how it is connected.

Two commands:

    nvidia-smi -q -x     XML. GPU model and count, memory total, driver and CUDA version,
                         PCIe generation and width, power limit, max clocks.
    nvidia-smi topo -m   The GPU-to-GPU link matrix (NV#, PIX, PXB, PHB, NODE, SYS) plus
                         CPU affinity and NUMA affinity columns.

`topo -m` has no machine-readable form, so it is the one exception to "parse CSV/XML only".
Parse it by header names, never by column position, and keep the raw text in the profile so
a parsing gap can be fixed later without asking the partner to re-run.
"""

from __future__ import annotations

from pydantic import BaseModel

from migration_eval.gpu.nvidia_smi import Runner

XML_ARGS = ("-q", "-x")
TOPO_ARGS = ("topo", "-m")


class GpuInfo(BaseModel):
    """One GPU, from `-q -x`."""

    index: int
    uuid: str
    name: str  # e.g. "NVIDIA H100 80GB HBM3"
    pci_bus_id: str
    memory_total_mib: int
    power_limit_w: float | None = None
    max_sm_clock_mhz: int | None = None
    max_mem_clock_mhz: int | None = None
    pcie_gen_max: int | None = None
    pcie_width_max: int | None = None


class Topology(BaseModel):
    """From `topo -m`. Keys are GPU names as `topo -m` prints them ("GPU0", "GPU1", ...)."""

    links: dict[str, dict[str, str]]  # links["GPU0"]["GPU1"] == "NV12"
    cpu_affinity: dict[str, str]  # "GPU0" -> "0-31,64-95"
    numa_affinity: dict[str, str]  # "GPU0" -> "0"; "N/A" on single-socket machines


class Snapshot(BaseModel):
    driver_version: str
    cuda_version: str | None = None
    gpus: list[GpuInfo]
    topology: Topology | None = None  # None when `topo -m` couldn't be parsed
    raw_xml: str
    raw_topo: str


def parse_query_xml(xml_text: str) -> tuple[str, str | None, list[GpuInfo]]:
    """Parse `nvidia-smi -q -x` into (driver_version, cuda_version, gpus).

    Use `xml.etree.ElementTree`. Values come with units ("81559 MiB", "700.00 W") or "N/A";
    strip the unit and map "N/A" to None. Element names have moved between driver versions
    (e.g. `power_readings` vs `gpu_power_readings`), so look in both places where known.
    """
    raise NotImplementedError


def parse_topo(topo_text: str) -> Topology:
    """Parse `nvidia-smi topo -m`.

    The first line is the header: GPU names, then "CPU Affinity", "NUMA Affinity" and sometimes
    "GPU NUMA ID". Columns are separated by tabs. Stop at the blank line before the "Legend:"
    block. Raise ValueError if the header doesn't contain the expected columns.
    """
    raise NotImplementedError


def take_snapshot(run: Runner) -> Snapshot:
    """Run both commands and parse them.

    A `topo -m` parse failure is not fatal: set `topology=None` and keep `raw_topo`. A
    `-q -x` parse failure is fatal, since without it we don't know what GPUs there are.
    """
    raise NotImplementedError
