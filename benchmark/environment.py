import csv
import ctypes
import platform
import shutil
import subprocess
import sys

from benchmark.models import (
    BenchmarkWarning,
    EnvironmentInfo,
    GPUInfo,
)


BYTES_PER_GIGABYTE = 1024 ** 3
MEGABYTES_PER_GIGABYTE = 1024


class MEMORYSTATUSEX(ctypes.Structure):
    """Windows memory status structure."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def collect_ram_gb() -> float:
    """Collect total physical RAM in gigabytes."""

    if sys.platform == "win32":
        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)

        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(memory_status)
        )

        if not success:
            raise OSError("GlobalMemoryStatusEx failed")

        return memory_status.ullTotalPhys / BYTES_PER_GIGABYTE

    if hasattr(__import__("os"), "sysconf"):
        import os

        total_pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")

        return (total_pages * page_size) / BYTES_PER_GIGABYTE

    raise NotImplementedError("RAM collection is unsupported on this platform")


def collect_gpu_info() -> GPUInfo | None:
    """Collect NVIDIA GPU information using nvidia-smi."""

    executable = shutil.which("nvidia-smi")

    if executable is None:
        return None

    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )

    rows = list(csv.reader(completed.stdout.splitlines()))

    if not rows:
        raise ValueError("nvidia-smi returned no GPU information")

    model, total_vram_mb, driver_version = [
        value.strip() for value in rows[0]
    ]

    return GPUInfo(
        model=model,
        total_vram_gb=float(total_vram_mb) / MEGABYTES_PER_GIGABYTE,
        driver_version=driver_version,
    )


def collect_environment() -> tuple[EnvironmentInfo, list[BenchmarkWarning]]:
    """Collect static environment metadata without failing the benchmark."""

    warnings = []

    environment = EnvironmentInfo(
        operating_system=platform.platform(),
        python_version=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
    )

    try:
        environment.ram_gb = collect_ram_gb()

    except Exception as exc:
        warnings.append(
            BenchmarkWarning(
                component="environment",
                message="Unable to collect system RAM information",
                details=str(exc),
            )
        )

    try:
        environment.gpu = collect_gpu_info()

    except Exception as exc:
        warnings.append(
            BenchmarkWarning(
                component="environment",
                message="Unable to collect NVIDIA GPU information",
                details=str(exc),
            )
        )

    return environment, warnings
