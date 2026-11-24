import threading
import time
from datetime import datetime, timezone

import psutil
import pynvml

from benchmark.models import ResourceSample, ResourceStatistics


DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1
BYTES_PER_MEGABYTE = 1024 ** 2


class ResourceMonitor:
    """Collects system and GPU resource measurements during inference."""

    def __init__(
        self,
        sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        gpu_index: int = 0,
    ):
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than 0")

        self.sample_interval_seconds = sample_interval_seconds
        self.gpu_index = gpu_index

        self.samples: list[ResourceSample] = []
        self.errors: list[str] = []

        self._gpu_handle = None
        self._start_time: float | None = None

        self._stop_event = threading.Event()
        self._sampling_thread: threading.Thread | None = None

        self._nvml_initialized = False
        self._running = False

    def _initialize_gpu(self) -> None:
        """Initialize NVIDIA Management Library and select the target GPU."""

        pynvml.nvmlInit()
        self._nvml_initialized = True

        try:
            device_count = pynvml.nvmlDeviceGetCount()

            if self.gpu_index < 0 or self.gpu_index >= device_count:
                raise ValueError(
                    f"GPU index {self.gpu_index} is invalid. "
                    f"Detected {device_count} GPU(s)."
                )

            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.gpu_index
            )

        except Exception:
            self._shutdown_gpu()
            raise

    def _shutdown_gpu(self) -> None:
        """Release NVIDIA Management Library resources."""

        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:
                self.errors.append(
                    f"GPU shutdown failed: {type(exc).__name__}: {exc}"
                )
            finally:
                self._nvml_initialized = False
                self._gpu_handle = None

    def _collect_sample(self) -> ResourceSample:
        """Collect one snapshot of system and GPU resource usage."""

        if self._start_time is None:
            raise RuntimeError("Resource monitor has not been started")

        captured_at_utc = datetime.now(timezone.utc)
        elapsed_seconds = time.perf_counter() - self._start_time

        gpu_memory_used_mb = None
        gpu_utilization_percent = None

        if self._gpu_handle is not None:
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(
                    self._gpu_handle
                )

                gpu_memory_used_mb = (
                    memory_info.used / BYTES_PER_MEGABYTE
                )
                gpu_utilization_percent = utilization.gpu

            except Exception as exc:
                self.errors.append(
                    f"GPU monitoring failed: {type(exc).__name__}: {exc}"
                )
                self._shutdown_gpu()

        memory_info = psutil.virtual_memory()

        return ResourceSample(
            timestamp_seconds=elapsed_seconds,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            system_ram_used_mb=(
                memory_info.used / BYTES_PER_MEGABYTE
            ),
            system_ram_available_mb=(
                memory_info.available / BYTES_PER_MEGABYTE
            ),
            captured_at_utc=captured_at_utc,
        )

    def _sampling_loop(self) -> None:
        """Continuously collect resource samples until stopped."""

        while not self._stop_event.wait(self.sample_interval_seconds):
            try:
                sample = self._collect_sample()
                self.samples.append(sample)

            except Exception as exc:
                self.errors.append(
                    f"{type(exc).__name__}: {exc}"
                )

    def capture_snapshot(self) -> ResourceSample:
        """Capture a one-time resource snapshot without starting background monitoring."""

        if self._running:
            raise RuntimeError(
                "Cannot capture a standalone snapshot while monitoring is running"
            )

        captured_at_utc = datetime.now(timezone.utc)

        gpu_handle = None
        nvml_initialized = False

        gpu_memory_used_mb = None
        gpu_utilization_percent = None

        try:
            pynvml.nvmlInit()
            nvml_initialized = True

            device_count = pynvml.nvmlDeviceGetCount()

            if self.gpu_index < 0 or self.gpu_index >= device_count:
                raise ValueError(
                    f"GPU index {self.gpu_index} is invalid. "
                    f"Detected {device_count} GPU(s)."
                )

            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.gpu_index
            )

            memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)

            gpu_memory_used_mb = (
                memory_info.used / BYTES_PER_MEGABYTE
            )
            gpu_utilization_percent = utilization.gpu

        except Exception as exc:
            self.errors.append(
                f"GPU snapshot failed: {type(exc).__name__}: {exc}"
            )

        finally:
            if nvml_initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception as exc:
                    self.errors.append(
                        f"GPU shutdown failed: {type(exc).__name__}: {exc}"
                    )

        memory_info = psutil.virtual_memory()

        return ResourceSample(
            timestamp_seconds=0.0,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            system_ram_used_mb=(
                memory_info.used / BYTES_PER_MEGABYTE
            ),
            system_ram_available_mb=(
                memory_info.available / BYTES_PER_MEGABYTE
            ),
            captured_at_utc=captured_at_utc,
        )

    def start(self) -> None:
        """Initialize monitoring and start the background sampling thread."""

        if self._running:
            raise RuntimeError("Resource monitor is already running")

        self.samples.clear()
        self.errors.clear()
        self._stop_event.clear()

        try:
            self._initialize_gpu()
        except Exception as exc:
            self.errors.append(
                f"GPU initialization failed: {type(exc).__name__}: {exc}"
            )

        self._start_time = time.perf_counter()

        try:
            self.samples.append(self._collect_sample())
        except Exception as exc:
            self.errors.append(
                f"Initial resource sampling failed: {type(exc).__name__}: {exc}"
            )

        self._sampling_thread = threading.Thread(
            target=self._sampling_loop,
            name="ResourceMonitor",
            daemon=True,
        )

        self._running = True

        try:
            self._sampling_thread.start()
        except Exception:
            self._running = False
            self._sampling_thread = None
            self._shutdown_gpu()
            raise

    def stop(self) -> list[ResourceSample]:
        """Stop sampling, collect a final sample, and release GPU resources."""

        if not self._running:
            return self.samples

        self._stop_event.set()

        try:
            if self._sampling_thread is not None:
                self._sampling_thread.join()

            # Capture one final snapshot after inference has completed.
            try:
                self.samples.append(self._collect_sample())
            except Exception as exc:
                self.errors.append(
                    f"Final resource sampling failed: {type(exc).__name__}: {exc}"
                )

        finally:
            self._sampling_thread = None
            self._running = False
            self._shutdown_gpu()

        return self.samples

    def _calculate_statistics(self) -> ResourceStatistics:
        """Calculate aggregate resource statistics from collected samples."""

        if not self.samples:
            return ResourceStatistics(sample_count=0)

        def get_values(attribute: str) -> list[float]:
            return [
                value
                for sample in self.samples
                if (value := getattr(sample, attribute)) is not None
            ]

        gpu_memory = get_values("gpu_memory_used_mb")
        gpu_utilization = get_values("gpu_utilization_percent")
        system_ram = get_values("system_ram_used_mb")

        return ResourceStatistics(
            sample_count=len(self.samples),

            gpu_memory_baseline_mb=(
                gpu_memory[0] if gpu_memory else None
            ),
            gpu_memory_peak_mb=(
                max(gpu_memory) if gpu_memory else None
            ),
            gpu_memory_average_mb=(
                sum(gpu_memory) / len(gpu_memory) if gpu_memory else None
            ),

            gpu_memory_peak_delta_mb=(
                max(gpu_memory) - gpu_memory[0]
                if gpu_memory else None
            ),

            gpu_utilization_average_percent=(
                sum(gpu_utilization) / len(gpu_utilization)
                if gpu_utilization else None
            ),
            gpu_utilization_peak_percent=(
                max(gpu_utilization) if gpu_utilization else None
            ),

            system_ram_baseline_mb=(
                system_ram[0] if system_ram else None
            ),
            system_ram_peak_mb=(
                max(system_ram) if system_ram else None
            ),

            system_ram_peak_delta_mb=(
                max(system_ram) - system_ram[0]
                if system_ram else None
            ),
        )
