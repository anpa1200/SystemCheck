#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linux System Check — colorful hardware report with storage deep-dive.

Features
- System: vendor/model/version/serial, chassis; Motherboard: vendor/name/version/serial
- OS: pretty name, version/ID, kernel, architecture
- CPU: model/vendor/arch, cores, freq, utilization
- Memory: totals, usage, swap  (per-DIMM modules intentionally not collected)
- GPU: adapters with vendor/model/driver; VRAM (and usage if NVIDIA tools available)
- Storage HW: per-disk model, vendor, size, transport, block sizes, TYPE (HDD/SSD/NVMe)
- Filesystems: per-mount capacity/usage + largest directories (depth=1) and largest files
- Output modes: TTY colored text (default), JSON (--json), Beautiful HTML (--html)
- Safety: timeouts for deep scans and configurable limits
"""

import argparse
import datetime as _dt
import html
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Optional live metrics
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


# ---------------- Helpers ----------------
def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def sh(cmd: List[str], timeout: Optional[int] = None, input_text: Optional[str] = None) -> str:
    try:
        out = subprocess.check_output(
            cmd,
            input=input_text.encode() if input_text else None,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return out.decode(errors="replace")
    except Exception:
        return ""


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def shlex_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def parse_size_to_bytes(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = s.strip()
    if s.lower() in ("unknown", "no module installed"):
        return None
    m = re.match(r"(?i)\s*([\d.]+)\s*(b|kb|mb|gb|tb|pb)\s*$", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    factor = {"b":1,"kb":1024,"mb":1024**2,"gb":1024**3,"tb":1024**4,"pb":1024**5}[unit]
    return int(val * factor)


# ---------------- Colors for TTY ----------------
class Color:
    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdout.isatty()

    def code(self, s: str) -> str:
        return s if self.enabled else ""

    @property
    def R(self): return self.code("\033[31m")  # red
    @property
    def G(self): return self.code("\033[32m")  # green
    @property
    def Y(self): return self.code("\033[33m")  # yellow
    @property
    def B(self): return self.code("\033[34m")  # blue
    @property
    def M(self): return self.code("\033[35m")  # magenta
    @property
    def C(self): return self.code("\033[36m")  # cyan
    @property
    def W(self): return self.code("\033[37m")  # white
    @property
    def BOLD(self): return self.code("\033[1m")
    @property
    def DIM(self): return self.code("\033[2m")
    @property
    def RESET(self): return self.code("\033[0m")


# ---------------- Progress bar (TTY) ----------------
def progress_bar(pct: Optional[float], color: Color, width: int = 28) -> str:
    c = color
    if pct is None:
        return f"Usage: {c.DIM}N/A{c.RESET}"
    p = max(0.0, min(100.0, float(pct)))
    filled = int(round((p / 100.0) * width))
    empty = width - filled
    col = c.G if p < 60 else (c.Y if p < 85 else c.R)
    return f"[{col}{'■'*filled}{c.RESET}{'·'*empty}] {p:5.1f}%"


# ---------------- OS / System ----------------
def parse_os_release() -> Dict[str, str]:
    d: Dict[str, str] = {}
    path = "/etc/os-release"
    if not os.path.exists(path):
        return d
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            d[k] = v
    return d


def get_os_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "name": None, "pretty_name": None, "version": None, "id": None, "id_like": None,
        "kernel": None, "architecture": None,
    }
    osr = parse_os_release()
    if osr:
        info["pretty_name"] = osr.get("PRETTY_NAME")
        info["name"] = osr.get("NAME")
        info["version"] = osr.get("VERSION") or osr.get("VERSION_ID")
        info["id"] = osr.get("ID")
        info["id_like"] = osr.get("ID_LIKE")

    if which("hostnamectl"):
        txt = sh(["hostnamectl"], timeout=3)
        if txt:
            m = re.search(r"Operating System:\s*(.+)", txt)
            if m and not info["pretty_name"]:
                info["pretty_name"] = m.group(1).strip()
            m = re.search(r"Kernel:\s*(.+)", txt)
            if m:
                info["kernel"] = m.group(1).strip()
            m = re.search(r"Architecture:\s*(.+)", txt)
            if m:
                info["architecture"] = m.group(1).strip()

    if which("lsb_release") and not info.get("pretty_name"):
        out = sh(["lsb_release", "-a"], timeout=3)
        if out:
            desc = re.search(r"Description:\s*(.+)", out)
            if desc:
                info["pretty_name"] = desc.group(1).strip()
            rel = re.search(r"Release:\s*(.+)", out)
            if rel and not info["version"]:
                info["version"] = rel.group(1).strip()
            dist = re.search(r"Distributor ID:\s*(.+)", out)
            if dist and not info["name"]:
                info["name"] = dist.group(1).strip()

    try:
        u = os.uname()
        if not info["kernel"]:
            info["kernel"] = f"{u.sysname} {u.release}"
        if not info["architecture"]:
            info["architecture"] = u.machine
    except Exception:
        pass

    return info


def read_dmi_file(name: str) -> Optional[str]:
    path = f"/sys/devices/virtual/dmi/id/{name}"
    try:
        if os.path.exists(path):
            val = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
            return val if val and val != "None" else None
    except Exception:
        pass
    return None


def parse_dmidecode_sections(typ: str) -> List[str]:
    txt = sh(["dmidecode", "-t", typ], timeout=5) if which("dmidecode") else ""
    if not txt:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", txt) if "Information" in b or "Device" in b]
    return blocks


def get_system_info() -> Dict[str, Any]:
    sys_vendor = read_dmi_file("sys_vendor")
    product_name = read_dmi_file("product_name")
    product_version = read_dmi_file("product_version")
    product_serial = read_dmi_file("product_serial")
    chassis_type = read_dmi_file("chassis_type")
    chassis_vendor = read_dmi_file("chassis_vendor")

    board_vendor = read_dmi_file("board_vendor")
    board_name = read_dmi_file("board_name")
    board_version = read_dmi_file("board_version")
    board_serial = read_dmi_file("board_serial")

    if not (sys_vendor and product_name):
        for b in parse_dmidecode_sections("system"):
            if not sys_vendor:
                m = re.search(r"Manufacturer:\s*(.+)", b)
                if m: sys_vendor = m.group(1).strip()
            if not product_name:
                m = re.search(r"Product Name:\s*(.+)", b)
                if m: product_name = m.group(1).strip()
            if not product_version:
                m = re.search(r"Version:\s*(.+)", b)
                if m: product_version = m.group(1).strip()
            if not product_serial:
                m = re.search(r"Serial Number:\s*(.+)", b)
                if m: product_serial = m.group(1).strip()

    if not (board_vendor and board_name):
        for b in parse_dmidecode_sections("baseboard"):
            if not board_vendor:
                m = re.search(r"Manufacturer:\s*(.+)", b)
                if m: board_vendor = m.group(1).strip()
            if not board_name:
                m = re.search(r"Product Name:\s*(.+)", b)
                if m: board_name = m.group(1).strip()
            if not board_version:
                m = re.search(r"Version:\s*(.+)", b)
                if m: board_version = m.group(1).strip()
            if not board_serial:
                m = re.search(r"Serial Number:\s*(.+)", b)
                if m: board_serial = m.group(1).strip()

    if which("lshw") and (not sys_vendor or not product_name):
        out = sh(["lshw", "-class", "system", "-json"], timeout=5)
        try:
            data = json.loads(out) if out else None
        except Exception:
            data = None
        if isinstance(data, dict):
            sys_vendor = sys_vendor or data.get("vendor")
            product_name = product_name or data.get("product")
            product_version = product_version or data.get("version")
            product_serial = product_serial or data.get("serial")

    return {
        "vendor": sys_vendor,
        "product_name": product_name,
        "product_version": product_version,
        "serial": product_serial,
        "chassis_vendor": chassis_vendor,
        "chassis_type": chassis_type,
        "motherboard": {
            "vendor": board_vendor,
            "name": board_name,
            "version": board_version,
            "serial": board_serial,
        },
    }


# ---------------- CPU ----------------
def get_cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model": None, "architecture": None, "vendor": None,
        "logical_cores": None, "physical_cores": None,
        "base_freq_hz": None, "max_freq_hz": None,
        "current_utilization_percent": None,
    }

    if which("lscpu"):
        out = sh(["lscpu", "-J"])
        if out:
            try:
                data = json.loads(out)
                kv = {i["field"].strip(":"): i["data"] for i in data.get("lscpu", [])}
                info["model"] = kv.get("Model name") or kv.get("Model Name")
                info["architecture"] = kv.get("Architecture")
                info["vendor"] = kv.get("Vendor ID")
                try:
                    info["logical_cores"] = int(kv.get("CPU(s)")) if kv.get("CPU(s)") else None
                except Exception:
                    pass
                try:
                    sockets = int(kv.get("Socket(s)") or 1)
                    cores_per_socket = int(kv.get("Core(s) per socket") or 0)
                    info["physical_cores"] = sockets * cores_per_socket or None
                except Exception:
                    pass

                def parse_freq(s: Optional[str]) -> Optional[int]:
                    if not s: return None
                    m = re.search(r"([\d.]+)\s*(MHz|GHz)", s)
                    if not m: return None
                    val = float(m.group(1))
                    return int(val * (1_000_000 if m.group(2) == "MHz" else 1_000_000_000))

                info["base_freq_hz"] = parse_freq(kv.get("CPU MHz") or kv.get("CPU max MHz"))
                info["max_freq_hz"] = parse_freq(kv.get("CPU max MHz"))
            except Exception:
                pass

    if (info["model"] is None) and os.path.exists("/proc/cpuinfo"):
        try:
            txt = open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore").read()
            m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
            v = re.search(r"^vendor_id\s*:\s*(.+)$", txt, re.M)
            info["model"] = info["model"] or (m.group(1).strip() if m else None)
            info["vendor"] = info["vendor"] or (v.group(1).strip() if v else None)
            if info["logical_cores"] is None:
                info["logical_cores"] = len(re.findall(r"^processor\s*:\s*\d+", txt, re.M))
        except Exception:
            pass

    if psutil:
        try: info["current_utilization_percent"] = psutil.cpu_percent(interval=0.5)
        except Exception: pass
        try:
            freq = psutil.cpu_freq()
            if freq:
                if not info["base_freq_hz"] and freq.min: info["base_freq_hz"] = int(freq.min * 1_000_000)
                if not info["max_freq_hz"] and freq.max: info["max_freq_hz"] = int(freq.max * 1_000_000)
        except Exception: pass
        try:
            if info["physical_cores"] is None: info["physical_cores"] = psutil.cpu_count(logical=False)
            if info["logical_cores"] is None: info["logical_cores"] = psutil.cpu_count(logical=True)
        except Exception: pass

    return info


# ---------------- Memory (totals only) ----------------
def get_memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "total_bytes": None, "available_bytes": None, "used_bytes": None, "free_bytes": None,
        "usage_percent": None, "swap_total_bytes": None, "swap_used_bytes": None,
        "swap_free_bytes": None, "swap_usage_percent": None,
    }
    if psutil:
        try:
            vm = psutil.virtual_memory()
            info.update(dict(
                total_bytes=int(vm.total), available_bytes=int(vm.available),
                used_bytes=int(vm.used), free_bytes=int(vm.free), usage_percent=float(vm.percent)
            ))
        except Exception: pass
        try:
            sm = psutil.swap_memory()
            info.update(dict(
                swap_total_bytes=int(sm.total), swap_used_bytes=int(sm.used),
                swap_free_bytes=int(sm.free), swap_usage_percent=float(sm.percent)
            ))
        except Exception: pass

    if info["total_bytes"] is None and os.path.exists("/proc/meminfo"):
        kv: Dict[str, int] = {}
        for line in open("/proc/meminfo", "r", encoding="utf-8", errors="ignore"):
            parts = line.split(":")
            if len(parts) != 2: continue
            m = re.search(r"(\d+)\s*kB", parts[1])
            if m: kv[parts[0].strip()] = int(m.group(1)) * 1024
        total = kv.get("MemTotal")
        avail = kv.get("MemAvailable", kv.get("MemFree"))
        used = total - avail if total and avail else None
        pct = (used / total * 100.0) if used and total else None
        info.update(dict(
            total_bytes=total, available_bytes=avail, used_bytes=used, free_bytes=kv.get("MemFree"),
            usage_percent=pct, swap_total_bytes=kv.get("SwapTotal"), swap_free_bytes=kv.get("SwapFree"),
            swap_used_bytes=(kv.get("SwapTotal", 0)-kv.get("SwapFree", 0)) if kv.get("SwapTotal") else None,
            swap_usage_percent=((kv.get("SwapTotal", 0)-kv.get("SwapFree", 0))/kv.get("SwapTotal")*100.0)
            if kv.get("SwapTotal") else None
        ))
    return info


# ---------------- GPU ----------------
def parse_nvidia_smi() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    if not which("nvidia-smi"):
        return gpus
    fmt = "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,pci.bus_id --format=csv,noheader,nounits"
    out = sh(["bash", "-lc", f"nvidia-smi {fmt}"], timeout=4)
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        name, driver, mem_total, mem_used, mem_free, bus = parts[:6]
        def mib_to_bytes(x: str) -> Optional[int]:
            try: return int(float(x)) * 1024 * 1024
            except Exception: return None
        total_b = mib_to_bytes(mem_total)
        used_b = mib_to_bytes(mem_used)
        free_b = mib_to_bytes(mem_free)
        pct = None
        if total_b and used_b is not None:
            try: pct = used_b / total_b * 100.0
            except Exception: pct = None
        gpus.append({
            "vendor": "NVIDIA",
            "model": name or None,
            "driver": driver or None,
            "bus": bus or None,
            "vram_total_bytes": total_b,
            "vram_used_bytes": used_b,
            "vram_free_bytes": free_b,
            "vram_usage_percent": pct,
            "source": "nvidia-smi",
        })
    return gpus


def parse_lshw_display() -> List[Dict[str, Any]]:
    if not which("lshw"):
        return []
    out = sh(["lshw", "-class", "display", "-json"], timeout=6)
    gpus: List[Dict[str, Any]] = []
    try:
        data = json.loads(out) if out else None
    except Exception:
        data = None
    def walk(n):
        if isinstance(n, dict):
            yield n
            for ch in n.get("children", []) or []:
                yield from walk(ch)
        elif isinstance(n, list):
            for it in n:
                yield from walk(it)
    if data:
        for node in walk(data):
            if not isinstance(node, dict): continue
            if node.get("class") != "display": continue
            prod = node.get("product") or node.get("description")
            vend = node.get("vendor")
            drv = None
            conf = node.get("configuration") or {}
            if isinstance(conf, dict):
                drv = conf.get("driver") or conf.get("driverversion")
            size_b = node.get("size")
            gpus.append({
                "vendor": vend,
                "model": prod,
                "driver": drv,
                "bus": node.get("businfo"),
                "vram_total_bytes": int(size_b) if isinstance(size_b, int) else None,
                "vram_used_bytes": None,
                "vram_free_bytes": None,
                "vram_usage_percent": None,
                "source": "lshw",
            })
    return gpus


def parse_lspci_display() -> List[Dict[str, Any]]:
    if not which("lspci"):
        return []
    out = sh(["bash", "-lc", "lspci -mm -nn | egrep -i 'VGA|3D|Display'"], timeout=4)
    gpus: List[Dict[str, Any]] = []
    for line in out.splitlines():
        m = re.search(r'^\S+\s+"[^"]+"\s+"([^"]+)"\s+"([^"]+)"', line)
        vend = prod = None
        if m:
            vend = m.group(1)
            prod = m.group(2)
        else:
            parts = line.split(": ", 1)
            if len(parts) == 2:
                prod = parts[1]
        gpus.append({
            "vendor": vend,
            "model": prod,
            "driver": None,
            "bus": None,
            "vram_total_bytes": None,
            "vram_used_bytes": None,
            "vram_free_bytes": None,
            "vram_usage_percent": None,
            "source": "lspci",
        })
    return gpus


def get_gpu_info() -> List[Dict[str, Any]]:
    gpus = parse_nvidia_smi()
    if not gpus:
        gpus = parse_lshw_display()
    if not gpus:
        gpus = parse_lspci_display()
    return gpus


# ---------------- Storage hardware & FS ----------------
def classify_disk(entry: Dict[str, Any]) -> str:
    name = (entry.get("name") or "").lower()
    tran = (entry.get("transport") or "").lower()
    rota = entry.get("rota")
    typ = (entry.get("type") or "").lower()

    if name.startswith("nvme") or tran == "nvme":
        return "NVMe SSD"
    if typ == "rom":
        return "Optical"
    if rota == 0:
        return "SSD"
    if rota == 1:
        return "HDD"
    return "Disk"


def get_lsblk() -> Dict[str, Any]:
    if not which("lsblk"): return {}
    out = sh(["lsblk", "-J", "-O"])
    try:
        return json.loads(out) if out else {}
    except Exception:
        return {}


def normalize_devices(lsblk_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    for b in lsblk_json.get("blockdevices", []) or []:
        if b.get("type") != "disk":
            continue
        dev = {
            "name": b.get("name"),
            "type": b.get("type"),
            "model": b.get("model"),
            "serial": b.get("serial"),
            "vendor": b.get("vendor"),
            "transport": b.get("tran"),
            "rota": b.get("rota"),
            "logical_block_size": b.get("log-sec") or b.get("LOG-SEC"),
            "physical_block_size": b.get("phy-sec") or b.get("PHY-SEC"),
            "size_bytes": b.get("size") if isinstance(b.get("size"), int) else None,
            "children": [],
        }
        for c in b.get("children", []) or []:
            dev["children"].append({
                "name": c.get("name"),
                "type": c.get("type"),
                "fstype": c.get("fstype"),
                "mountpoint": c.get("mountpoint"),
                "size_bytes": c.get("size") if isinstance(c.get("size"), int) else None,
            })
        dev["class"] = classify_disk(dev)
        devices.append(dev)
    return devices


def get_mounts() -> List[Dict[str, Any]]:
    mounts: List[Dict[str, Any]] = []
    if psutil:
        try:
            for p in psutil.disk_partitions(all=False):
                if p.fstype in ("tmpfs", "devtmpfs", "squashfs", "overlay"):
                    continue
                try:
                    usage = psutil.disk_usage(p.mountpoint)
                except Exception:
                    continue
                mounts.append(dict(
                    mountpoint=p.mountpoint, device=p.device, fstype=p.fstype,
                    total_bytes=int(usage.total), used_bytes=int(usage.used),
                    free_bytes=int(usage.free), usage_percent=float(usage.percent)
                ))
        except Exception:
            pass

    if not mounts:
        out = sh(["df", "-P", "-B1"])
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 6: continue
            dev, total, used, avail, pct, mnt = parts[:6]
            if dev.startswith("tmpfs") or dev.startswith("devtmpfs"):
                continue
            try:
                mounts.append(dict(
                    mountpoint=mnt, device=dev, fstype=None,
                    total_bytes=int(total), used_bytes=int(used),
                    free_bytes=int(avail),
                    usage_percent=float(pct.strip("%")) if pct.endswith("%") else None
                ))
            except Exception:
                continue
    return mounts


# ---------------- Deep usage scan (largest dirs/files) ----------------
def largest_dirs(mountpoint: str, top: int, timeout: int) -> List[Tuple[int, str]]:
    if not which("du"): return []
    out = sh(["du", "-x", "-B1", "-d1", mountpoint], timeout=timeout)
    rows: List[Tuple[int, str]] = []
    for line in out.splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) != 2:
            parts = line.strip().split(None, 1)
            if len(parts) != 2: continue
        try:
            size = int(parts[0])
            path = parts[1]
            if path.rstrip("/") == mountpoint.rstrip("/"):
                continue
            rows.append((size, path))
        except Exception:
            continue
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:top]


def largest_files(mountpoint: str, top: int, timeout: int) -> List[Tuple[int, str]]:
    if not which("find") or not which("sort") or not which("head"): return []
    cmd = f"find {shlex_quote(mountpoint)} -xdev -type f -printf '%s\\t%p\\n' 2>/dev/null | sort -nr | head -n {int(top)}"
    try:
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.DEVNULL, timeout=timeout)
        text = out.decode(errors="replace")
    except Exception:
        return []
    rows: List[Tuple[int, str]] = []
    for line in text.splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) != 2: continue
        try:
            size = int(parts[0]); path = parts[1]
            rows.append((size, path))
        except Exception:
            continue
    return rows


# ---------------- Collect & Render (Text + JSON) ----------------
def collect_report(deep: bool, top_n: int, max_mounts: int, per_mount_timeout: int) -> Dict[str, Any]:
    os_info = get_os_info()
    sys_info = get_system_info()
    storage_hw_raw = get_lsblk()
    devices = normalize_devices(storage_hw_raw) if storage_hw_raw else []
    mounts = get_mounts()
    gpus = get_gpu_info()

    scanned = 0
    for m in mounts:
        if deep and scanned < max_mounts:
            fstype = (m.get("fstype") or "").lower()
            dev = (m.get("device") or "").lower()
            if fstype.startswith("nfs") or fstype.startswith("cifs") or fstype in ("smbfs", "fuse.sshfs") or dev.startswith("overlay"):
                pass
            else:
                m["largest_dirs"] = [
                    {"path": p, "size_bytes": int(sz)}
                    for sz, p in largest_dirs(m["mountpoint"], top_n, per_mount_timeout)
                ]
                m["largest_files"] = [
                    {"path": p, "size_bytes": int(sz)}
                    for sz, p in largest_files(m["mountpoint"], top_n, per_mount_timeout)
                ]
                scanned += 1

    return {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "os": os_info,
        "system": sys_info,
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": gpus,
        "storage": {"devices": devices, "mounts": mounts},
    }


def render_text(report: Dict[str, Any], color: Color, top_n: int) -> str:
    c = color
    lines: List[str] = []
    H = lambda s: f"{c.BOLD}{c.C}{s}{c.RESET}"

    # Header timestamp
    ts = report.get("generated_at")
    if ts:
        lines.append(f"{c.DIM}Generated at: {ts}{c.RESET}\n")

    # System
    sysi = report.get("system", {}) or {}
    lines.append(H("Computer — System"))
    lines.append(f"{c.BOLD}Vendor/Model:{c.RESET} {sysi.get('vendor') or 'N/A'} {sysi.get('product_name') or ''}".rstrip())
    lines.append(f"{c.BOLD}Version:{c.RESET} {sysi.get('product_version') or 'N/A'}   {c.DIM}Serial:{c.RESET} {sysi.get('serial') or 'N/A'}")
    chv = sysi.get("chassis_vendor"); cht = sysi.get("chassis_type")
    if chv or cht:
        lines.append(f"{c.DIM}Chassis:{c.RESET} {chv or '—'}  type={cht or '—'}")
    mb = sysi.get("motherboard") or {}
    lines.append(f"{c.BOLD}Motherboard:{c.RESET} {mb.get('vendor') or 'N/A'} {mb.get('name') or ''}".rstrip())
    lines.append(f"  {c.DIM}Version:{c.RESET} {mb.get('version') or 'N/A'}   {c.DIM}Serial:{c.RESET} {mb.get('serial') or 'N/A'}\n")

    # OS
    osi = report.get("os", {}) or {}
    lines.append(H("OS"))
    pn = osi.get("pretty_name") or f"{osi.get('name') or 'N/A'} {osi.get('version') or ''}".strip()
    lines.append(f"{c.BOLD}{pn}{c.RESET}")
    lines.append(f"{c.DIM}Kernel:{c.RESET} {osi.get('kernel') or 'N/A'}   {c.DIM}Arch:{c.RESET} {osi.get('architecture') or 'N/A'}\n")

    # CPU
    cpu = report.get("cpu", {})
    lines.append(H("CPU"))
    lines.append(f"{c.BOLD}Model:{c.RESET} {cpu.get('model') or 'N/A'}")
    lines.append(f"{c.BOLD}Vendor:{c.RESET} {cpu.get('vendor') or 'N/A'}   {c.DIM}Arch:{c.RESET} {cpu.get('architecture') or 'N/A'}")
    lines.append(f"{c.BOLD}Cores:{c.RESET} {cpu.get('physical_cores') or 'N/A'} phys / {cpu.get('logical_cores') or 'N/A'} logical")
    base = human_bytes(cpu.get("base_freq_hz")) if cpu.get("base_freq_hz") else "N/A"
    mx = human_bytes(cpu.get("max_freq_hz")) if cpu.get("max_freq_hz") else "N/A"
    lines.append(f"{c.BOLD}Freq:{c.RESET} base≈{base}  max≈{mx}")
    util = cpu.get("current_utilization_percent")
    util_s = f"{util:.1f}%" if isinstance(util, (int, float)) else "N/A"
    util_color = c.G if isinstance(util, (int, float)) and util < 60 else (c.Y if isinstance(util, (int, float)) and util < 85 else c.R)
    lines.append(f"{c.BOLD}Util:{c.RESET} {util_color}{util_s}{c.RESET}\n")

    # Memory (totals only)
    mem = report.get("memory", {})
    lines.append(H("Memory"))
    lines.append(f"Total {c.BOLD}{human_bytes(mem.get('total_bytes'))}{c.RESET}  "
                 f"Used {c.Y}{human_bytes(mem.get('used_bytes'))}{c.RESET}  "
                 f"Free {c.G}{human_bytes(mem.get('free_bytes'))}{c.RESET}  "
                 f"Avail {c.G}{human_bytes(mem.get('available_bytes'))}{c.RESET}")
    mp = mem.get("usage_percent")
    mp_s = f"{mp:.1f}%" if isinstance(mp, (int, float)) else "N/A"
    lines.append(f"Usage: {c.BOLD}{mp_s}{c.RESET}")
    lines.append(f"{c.DIM}Swap:{c.RESET} total {human_bytes(mem.get('swap_total_bytes'))}, "
                 f"used {human_bytes(mem.get('swap_used_bytes'))}, "
                 f"free {human_bytes(mem.get('swap_free_bytes'))}\n")

    # GPU
    lines.append(H("GPU"))
    gpus = report.get("gpu") or []
    if not gpus:
        lines.append(f"{c.DIM}(No GPU adapters detected; consider installing lshw or NVIDIA drivers){c.RESET}\n")
    else:
        for g in gpus:
            vram_total = human_bytes(g.get("vram_total_bytes"))
            vram_used = human_bytes(g.get("vram_used_bytes"))
            vram_free = human_bytes(g.get("vram_free_bytes"))
            pct = g.get("vram_usage_percent")
            usage = f"{pct:.1f}%" if isinstance(pct, (int, float)) else None
            bar = progress_bar(pct, color) if isinstance(pct, (int, float)) else None
            lines.append(f"- {c.BOLD}{g.get('vendor') or '—'} {g.get('model') or '—'}{c.RESET}")
            lines.append(f"  driver={g.get('driver') or '—'}  bus={g.get('bus') or '—'}  source={g.get('source') or '—'}")
            if g.get("vram_total_bytes") is not None:
                lines.append(f"  VRAM: total={vram_total}  used={vram_used}  free={vram_free}" + (f"  ({usage})" if usage else ""))
                if bar:
                    lines.append(f"  {bar}")
            lines.append("")

    # Storage hardware
    lines.append(H("Storage — Hardware"))
    devs = report.get("storage", {}).get("devices", []) or []
    if not devs:
        lines.append(f"{c.DIM}(lsblk not available or no disks detected){c.RESET}\n")
    else:
        for d in devs:
            tclass = d.get("class") or "Disk"
            color_map = {"NVMe SSD": c.M, "SSD": c.C, "HDD": c.Y, "Disk": c.W}
            tcol = color_map.get(tclass, c.W)
            lines.append(f"{tcol}{c.BOLD}/dev/{d.get('name')}{c.RESET}  [{tclass}]  "
                         f"{c.DIM}{d.get('transport') or ''}{' ' if d.get('transport') else ''}({d.get('vendor') or '—'}){c.RESET}")
            lines.append(f"  Model: {d.get('model') or 'N/A'}   Serial: {d.get('serial') or 'N/A'}")
            lines.append(f"  Size:  {human_bytes(d.get('size_bytes'))}   "
                         f"Block: logical={d.get('logical_block_size') or 'N/A'}, physical={d.get('physical_block_size') or 'N/A'}")
            ch = d.get("children") or []
            if ch:
                lines.append(f"  {c.DIM}Partitions/LVs:{c.RESET}")
                for cpart in ch:
                    mp = cpart.get("mountpoint") or "—"
                    lines.append(f"    - {c.BOLD}{cpart.get('name')}{c.RESET}  "
                                 f"fs={cpart.get('fstype') or 'N/A'}  size={human_bytes(cpart.get('size_bytes'))}  mount={mp}")
            lines.append("")

    # Filesystems with deep usage
    lines.append(H("Storage — Filesystems & Usage"))
    mnts = report.get("storage", {}).get("mounts", []) or []
    if not mnts:
        lines.append(f"{c.DIM}(no mounts detected){c.RESET}")
    for m in sorted(mnts, key=lambda x: x.get("mountpoint") or ""):
        used_pct = m.get("usage_percent")
        bar = progress_bar(used_pct if isinstance(used_pct, (int, float)) else None, color)
        head = f"{c.BOLD}{m.get('mountpoint')}{c.RESET}  " \
               f"dev={m.get('device')}  fs={m.get('fstype') or 'N/A'}\n" \
               f"  total={human_bytes(m.get('total_bytes'))}  used={human_bytes(m.get('used_bytes'))}  free={human_bytes(m.get('free_bytes'))}\n" \
               f"  {bar}"
        lines.append(head)

        ldirs = m.get("largest_dirs")
        if isinstance(ldirs, list) and ldirs:
            lines.append(f"  {c.DIM}Largest directories (depth=1):{c.RESET}")
            for entry in ldirs:
                lines.append(f"    - {human_bytes(entry['size_bytes']).rjust(9)}  {entry['path']}")
        lf = m.get("largest_files")
        if isinstance(lf, list) and lf:
            lines.append(f"  {c.DIM}Largest files:{c.RESET}")
            for entry in lf:
                lines.append(f"    - {human_bytes(entry['size_bytes']).rjust(9)}  {entry['path']}")
        lines.append("")

    return "\n".join(lines)


# ---------------- HTML Rendering ----------------
def _pct(v: Optional[float]) -> Optional[float]:
    if v is None: return None
    try:
        return max(0.0, min(100.0, float(v)))
    except Exception:
        return None


def _usage_bar_html(pct: Optional[float]) -> str:
    if pct is None:
        return '<div class="bar bar-na" title="N/A">N/A</div>'
    p = _pct(pct) or 0.0
    # Color thresholds: green <60, amber <85, red otherwise
    cls = "ok" if p < 60 else ("warn" if p < 85 else "bad")
    return f'''
<div class="bar {cls}" aria-valuenow="{p:.1f}" aria-valuemin="0" aria-valuemax="100">
  <div class="bar-fill" style="width:{p:.1f}%"></div>
  <div class="bar-label">{p:.1f}%</div>
</div>'''.strip()


def escape(s: Any) -> str:
    if s is None:
        return "—"
    return html.escape(str(s))


def render_html(report: Dict[str, Any], top_n: int) -> str:
    ts = escape(report.get("generated_at"))
    osi = report.get("os", {}) or {}
    sysi = report.get("system", {}) or {}
    cpu = report.get("cpu", {}) or {}
    mem = report.get("memory", {}) or {}
    gpus = report.get("gpu") or []
    devs = (report.get("storage", {}) or {}).get("devices", []) or []
    mnts = (report.get("storage", {}) or {}).get("mounts", []) or []

    def human(n): return escape(human_bytes(n))

    # Build device cards
    dev_cards = []
    for d in devs:
        parts = []
        parts.append(f'<div class="kv"><span>Device</span><span>/dev/{escape(d.get("name"))}</span></div>')
        parts.append(f'<div class="kv"><span>Type</span><span>{escape(d.get("class") or "Disk")}</span></div>')
        parts.append(f'<div class="kv"><span>Transport</span><span>{escape(d.get("transport"))}</span></div>')
        parts.append(f'<div class="kv"><span>Vendor</span><span>{escape(d.get("vendor"))}</span></div>')
        parts.append(f'<div class="kv"><span>Model</span><span>{escape(d.get("model"))}</span></div>')
        parts.append(f'<div class="kv"><span>Serial</span><span>{escape(d.get("serial"))}</span></div>')
        parts.append(f'<div class="kv"><span>Capacity</span><span>{human(d.get("size_bytes"))}</span></div>')
        parts.append(f'<div class="kv"><span>Block Sizes</span><span>logical={escape(d.get("logical_block_size"))}, physical={escape(d.get("physical_block_size"))}</span></div>')
        # partitions
        ch = d.get("children") or []
        if ch:
            rows = []
            for c in ch:
                rows.append(f"<tr><td>{escape(c.get('name'))}</td><td>{escape(c.get('fstype'))}</td><td>{human(c.get('size_bytes'))}</td><td>{escape(c.get('mountpoint'))}</td></tr>")
            parts.append(f'''
<table class="mini">
  <thead><tr><th>Partition</th><th>FS</th><th>Size</th><th>Mount</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>''')
        dev_cards.append(f'<div class="card">{"".join(parts)}</div>')

    # Build filesystem cards
    fs_cards = []
    for m in sorted(mnts, key=lambda x: x.get("mountpoint") or ""):
        used_pct = m.get("usage_percent")
        bar = _usage_bar_html(used_pct if isinstance(used_pct, (int, float)) else None)
        parts = []
        parts.append(f'<div class="kv title"><span>{escape(m.get("mountpoint"))}</span><span>fs={escape(m.get("fstype"))} • dev={escape(m.get("device"))}</span></div>')
        parts.append(f'<div class="kv"><span>Total</span><span>{human(m.get("total_bytes"))}</span></div>')
        parts.append(f'<div class="kv"><span>Used</span><span>{human(m.get("used_bytes"))}</span></div>')
        parts.append(f'<div class="kv"><span>Free</span><span>{human(m.get("free_bytes"))}</span></div>')
        parts.append(bar)

        ldirs = m.get("largest_dirs")
        if isinstance(ldirs, list) and ldirs:
            rows = []
            for e in ldirs:
                rows.append(f"<tr><td>{human(e.get('size_bytes'))}</td><td>{escape(e.get('path'))}</td></tr>")
            parts.append(f'''
<div class="table-wrap">
  <div class="table-title">Largest directories (depth=1)</div>
  <table class="mini">
    <thead><tr><th>Size</th><th>Path</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>''')

        lf = m.get("largest_files")
        if isinstance(lf, list) and lf:
            rows = []
            for e in lf:
                rows.append(f"<tr><td>{human(e.get('size_bytes'))}</td><td>{escape(e.get('path'))}</td></tr>")
            parts.append(f'''
<div class="table-wrap">
  <div class="table-title">Largest files</div>
  <table class="mini">
    <thead><tr><th>Size</th><th>Path</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>''')

        fs_cards.append(f'<div class="card">{"".join(parts)}</div>')

    # GPU cards
    gpu_cards = []
    for g in gpus:
        parts = []
        parts.append(f'<div class="kv title"><span>{escape(g.get("vendor"))} {escape(g.get("model"))}</span><span>{escape(g.get("source"))}</span></div>')
        parts.append(f'<div class="kv"><span>Driver</span><span>{escape(g.get("driver"))}</span></div>')
        parts.append(f'<div class="kv"><span>Bus</span><span>{escape(g.get("bus"))}</span></div>')
        if g.get("vram_total_bytes") is not None:
            parts.append(f'<div class="kv"><span>VRAM</span><span>total {human(g.get("vram_total_bytes"))} • used {human(g.get("vram_used_bytes"))} • free {human(g.get("vram_free_bytes"))}</span></div>')
            parts.append(_usage_bar_html(g.get("vram_usage_percent")))
        gpu_cards.append(f'<div class="card">{"".join(parts)}</div>')

    # HTML template
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>System Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {{
  --bg: #0f1222;
  --panel: #171a2f;
  --text: #e8eaf6;
  --muted: #a5acc7;
  --accent: #7c83ff;
  --ok: #2ecc71;
  --warn: #f1c40f;
  --bad: #e74c3c;
  --border: #2a2e4b;
  --chip: #232748;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin:0; padding:0; background:var(--bg); color:var(--text); font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial, sans-serif; }}
.wrapper {{ max-width: 1100px; margin: 28px auto 48px; padding: 0 16px; }}
h1 {{ font-size: 28px; margin: 0 0 6px; }}
.sub {{ color: var(--muted); margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }}
.section {{ grid-column: 1/-1; }}
.section h2 {{ font-size: 18px; letter-spacing:.3px; color: #c9ceff; margin: 24px 0 12px; }}
.card {{ background: var(--panel); border:1px solid var(--border); border-radius: 14px; padding: 14px 14px 10px; box-shadow: 0 10px 24px rgba(0,0,0,.25); }}
.kv {{ display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-bottom: 1px dashed var(--border); }}
.kv:last-child {{ border-bottom: none; }}
.kv.title span:first-child {{ font-weight: 700; }}
.kv > span:first-child {{ color: var(--muted); min-width: 140px; }}
.mini {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
.mini th, .mini td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
.table-title {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
.row {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:16px; }}
@media (max-width: 880px) {{ .row {{ grid-template-columns: 1fr; }} }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; background:var(--chip); color:var(--text); font-size:12px; }}
/* Bars */
.bar {{ position: relative; height: 24px; background:#0b0e1e; border:1px solid var(--border); border-radius: 10px; overflow:hidden; margin-top:8px; }}
.bar .bar-fill {{ position:absolute; inset:0 0 0 0; width:0%; background: linear-gradient(90deg, rgba(255,255,255,.15), rgba(255,255,255,.05)); }}
.bar.ok .bar-fill {{ background-color: var(--ok); }}
.bar.warn .bar-fill {{ background-color: var(--warn); }}
.bar.bad .bar-fill {{ background-color: var(--bad); }}
.bar .bar-label {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-weight:600; color:#111; mix-blend-mode:screen; }}
.bar-na {{ display:flex; align-items:center; justify-content:center; color: var(--muted); height:24px; border:1px dashed var(--border); background:transparent; border-radius:10px; margin-top:8px; }}
/* chips row */
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }}
.sep {{ height: 6px; }}
</style>
</head>
<body>
<div class="wrapper">
  <h1>System Report</h1>
  <div class="sub">Generated at <span class="badge">{ts}</span></div>

  <div class="section">
    <h2>Computer</h2>
    <div class="row">
      <div class="card">
        <div class="kv title"><span>System</span><span></span></div>
        <div class="kv"><span>Vendor/Model</span><span>{escape(sysi.get('vendor'))} {escape(sysi.get('product_name'))}</span></div>
        <div class="kv"><span>Version</span><span>{escape(sysi.get('product_version'))}</span></div>
        <div class="kv"><span>Serial</span><span>{escape(sysi.get('serial'))}</span></div>
        <div class="kv"><span>Chassis</span><span>{escape(sysi.get('chassis_vendor'))} • type {escape(sysi.get('chassis_type'))}</span></div>
      </div>
      <div class="card">
        <div class="kv title"><span>Motherboard</span><span></span></div>
        <div class="kv"><span>Vendor/Name</span><span>{escape((sysi.get('motherboard') or {{}}).get('vendor'))} {escape((sysi.get('motherboard') or {{}}).get('name'))}</span></div>
        <div class="kv"><span>Version</span><span>{escape((sysi.get('motherboard') or {{}}).get('version'))}</span></div>
        <div class="kv"><span>Serial</span><span>{escape((sysi.get('motherboard') or {{}}).get('serial'))}</span></div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>OS</h2>
    <div class="row">
      <div class="card">
        <div class="kv"><span>Pretty</span><span>{escape(osi.get('pretty_name') or ((osi.get('name') or '') + ' ' + (osi.get('version') or '')))}</span></div>
        <div class="kv"><span>Kernel</span><span>{escape(osi.get('kernel'))}</span></div>
        <div class="kv"><span>Architecture</span><span>{escape(osi.get('architecture'))}</span></div>
        <div class="kv"><span>ID</span><span>{escape(osi.get('id'))} (like {escape(osi.get('id_like'))})</span></div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>CPU & Memory</h2>
    <div class="row">
      <div class="card">
        <div class="kv title"><span>CPU</span><span></span></div>
        <div class="kv"><span>Model</span><span>{escape(cpu.get('model'))}</span></div>
        <div class="kv"><span>Vendor</span><span>{escape(cpu.get('vendor'))}</span></div>
        <div class="kv"><span>Arch</span><span>{escape(cpu.get('architecture'))}</span></div>
        <div class="kv"><span>Cores</span><span>{escape(cpu.get('physical_cores'))} phys / {escape(cpu.get('logical_cores'))} logical</span></div>
        <div class="kv"><span>Base / Max</span><span>{human(cpu.get('base_freq_hz'))} / {human(cpu.get('max_freq_hz'))}</span></div>
        <div class="kv"><span>Utilization</span><span style="flex:1"></span></div>
        {_usage_bar_html(cpu.get('current_utilization_percent'))}
      </div>
      <div class="card">
        <div class="kv title"><span>Memory</span><span></span></div>
        <div class="kv"><span>Total</span><span>{human(mem.get('total_bytes'))}</span></div>
        <div class="kv"><span>Used</span><span>{human(mem.get('used_bytes'))}</span></div>
        <div class="kv"><span>Free</span><span>{human(mem.get('free_bytes'))}</span></div>
        <div class="kv"><span>Available</span><span>{human(mem.get('available_bytes'))}</span></div>
        <div class="kv"><span>Usage</span><span style="flex:1"></span></div>
        {_usage_bar_html(mem.get('usage_percent'))}
        <div class="sep"></div>
        <div class="kv"><span>Swap</span><span>total {human(mem.get('swap_total_bytes'))} • used {human(mem.get('swap_used_bytes'))} • free {human(mem.get('swap_free_bytes'))}</span></div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>GPU</h2>
    <div class="row">
      {''.join(gpu_cards) if gpu_cards else '<div class="card"><div class="kv">No GPU adapters detected.</div></div>'}
    </div>
  </div>

  <div class="section">
    <h2>Storage — Hardware</h2>
    <div class="row">
      {''.join(dev_cards) if dev_cards else '<div class="card"><div class="kv">No disks detected (lsblk unavailable?)</div></div>'}
    </div>
  </div>

  <div class="section">
    <h2>Storage — Filesystems & Usage</h2>
    <div class="row">
      {''.join(fs_cards) if fs_cards else '<div class="card"><div class="kv">No mounts detected.</div></div>'}
    </div>
  </div>

</div>
</body>
</html>"""
    return html_doc


# ---------------- Output orchestration ----------------
def collect_and_render(json_mode: bool, html_mode: bool, html_file: Optional[str],
                       deep: bool, top: int, max_mounts: int, per_mount_timeout: int, color: Color) -> None:
    report = collect_report(
        deep=deep,
        top_n=max(1, top),
        max_mounts=max(1, max_mounts),
        per_mount_timeout=max(10, per_mount_timeout),
    )
    if json_mode:
        print(json.dumps(report, indent=2))
        return
    if html_mode:
        doc = render_html(report, top_n=top)
        if html_file:
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(doc)
            print(f"HTML report written to: {html_file}")
        else:
            # write to stdout
            sys.stdout.write(doc)
        return
    # default TTY text
    print(render_text(report, color=color, top_n=top))


def main() -> None:
    ap = argparse.ArgumentParser(description="Colorful Linux system check with storage hot-spots")
    out = ap.add_argument_group("Output options")
    out.add_argument("--json", action="store_true", help="Output JSON only")
    out.add_argument("--html", action="store_true", help="Output a professional HTML report to stdout (or --html-file)")
    out.add_argument("--html-file", type=str, help="Write HTML report to this file instead of stdout")
    ap.add_argument("--no-color", action="store_true", help="Disable colored text output")
    ap.add_argument("--deep", dest="deep", action="store_true", help="Force deep scan of mounts (largest dirs/files)")
    ap.add_argument("--no-deep", dest="no_deep", action="store_true", help="Skip deep scan of mounts")
    ap.add_argument("--top", type=int, default=5, help="Top N directories & files per scanned mount (default: 5)")
    ap.add_argument("--max-mounts", type=int, default=5, help="Max mounts to deep scan (default: 5)")
    ap.add_argument("--per-mount-timeout", type=int, default=60, help="Timeout seconds for each mount scan (default: 60)")
    args = ap.parse_args()

    # Mode selection
    deep = False if (args.json or args.html) else (not args.no_deep)
    if args.deep:
        deep = True

    color = Color(enabled=not args.no_color and not args.json and not args.html)

    if sys.version_info < (3, 8):
        print("Please run with Python 3.8 or newer.", file=sys.stderr)
        sys.exit(1)

    collect_and_render(
        json_mode=args.json,
        html_mode=args.html,
        html_file=args.html_file,
        deep=deep,
        top=args.top,
        max_mounts=args.max_mounts,
        per_mount_timeout=args.per_mount_timeout,
        color=color,
    )


if __name__ == "__main__":
    main()