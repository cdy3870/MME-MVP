"""GPU-1 snapshot: parsing `-q -x` XML and `topo -m`, with no GPU."""

from __future__ import annotations

import pytest

from migration_eval.gpu.snapshot import parse_query_xml, parse_topo, take_snapshot

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="GPU-1 skeleton")

# Trimmed `nvidia-smi -q -x`: only the elements the parser reads.
QUERY_XML = """<?xml version="1.0" ?>
<nvidia_smi_log>
  <driver_version>550.54.15</driver_version>
  <cuda_version>12.4</cuda_version>
  <attached_gpus>2</attached_gpus>
  <gpu id="00000000:18:00.0">
    <product_name>NVIDIA H100 80GB HBM3</product_name>
    <uuid>GPU-aaaa</uuid>
    <minor_number>0</minor_number>
    <fb_memory_usage><total>81559 MiB</total></fb_memory_usage>
    <pci><pci_gpu_link_info>
      <pcie_gen><max_link_gen>5</max_link_gen></pcie_gen>
      <link_widths><max_link_width>16x</max_link_width></link_widths>
    </pci_gpu_link_info></pci>
    <gpu_power_readings><current_power_limit>700.00 W</current_power_limit></gpu_power_readings>
    <max_clocks><sm_clock>1980 MHz</sm_clock><mem_clock>2619 MHz</mem_clock></max_clocks>
  </gpu>
  <gpu id="00000000:2A:00.0">
    <product_name>NVIDIA H100 80GB HBM3</product_name>
    <uuid>GPU-bbbb</uuid>
    <minor_number>1</minor_number>
    <fb_memory_usage><total>81559 MiB</total></fb_memory_usage>
    <gpu_power_readings><current_power_limit>N/A</current_power_limit></gpu_power_readings>
  </gpu>
</nvidia_smi_log>
"""

TOPO = (
    "\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
    "GPU0\t X \tNV18\t0-31,64-95\t0\t\tN/A\n"
    "GPU1\tNV18\t X \t32-63,96-127\t1\t\tN/A\n"
    "\n"
    "Legend:\n"
    "\n"
    "  X    = Self\n"
    "  NV#  = Connection traversing a bonded set of # NVLinks\n"
)


def test_parse_query_xml_reads_driver_and_gpus():
    driver, cuda, gpus = parse_query_xml(QUERY_XML)
    assert (driver, cuda) == ("550.54.15", "12.4")
    assert [g.uuid for g in gpus] == ["GPU-aaaa", "GPU-bbbb"]
    assert gpus[0].memory_total_mib == 81559
    assert gpus[0].power_limit_w == 700.0
    assert gpus[0].pcie_gen_max == 5 and gpus[0].pcie_width_max == 16


def test_parse_query_xml_maps_na_and_missing_to_none():
    _, _, gpus = parse_query_xml(QUERY_XML)
    assert gpus[1].power_limit_w is None
    assert gpus[1].max_sm_clock_mhz is None


def test_parse_topo_links_and_affinity():
    topo = parse_topo(TOPO)
    assert topo.links["GPU0"]["GPU1"] == "NV18"
    assert topo.cpu_affinity["GPU1"] == "32-63,96-127"
    assert topo.numa_affinity == {"GPU0": "0", "GPU1": "1"}


def test_parse_topo_rejects_unknown_layout():
    with pytest.raises(ValueError):
        parse_topo("something else entirely\n")


def test_take_snapshot_keeps_raw_text_when_topo_unparseable():
    def run(args):
        return QUERY_XML if list(args) == ["-q", "-x"] else "garbled"

    snap = take_snapshot(run)
    assert snap.topology is None
    assert snap.raw_topo == "garbled"
