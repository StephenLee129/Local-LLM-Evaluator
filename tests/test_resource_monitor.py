from datetime import timedelta, timezone

import pytest

from benchmark.models import ResourceSample
from benchmark.monitoring import ResourceMonitor


def test_calculate_statistics_computes_resource_deltas():
    monitor = ResourceMonitor()

    monitor.samples = [
        ResourceSample(
            timestamp_seconds=0.0,
            gpu_memory_used_mb=1000.0,
            system_ram_used_mb=8000.0,
        ),
        ResourceSample(
            timestamp_seconds=0.1,
            gpu_memory_used_mb=1100.0,
            system_ram_used_mb=8100.0,
        ),
        ResourceSample(
            timestamp_seconds=0.2,
            gpu_memory_used_mb=1050.0,
            system_ram_used_mb=8200.0,
        ),
    ]

    statistics = monitor._calculate_statistics()

    assert statistics.gpu_memory_baseline_mb == 1000.0
    assert statistics.gpu_memory_peak_mb == 1100.0
    assert statistics.gpu_memory_peak_delta_mb == 100.0

    assert statistics.system_ram_baseline_mb == 8000.0
    assert statistics.system_ram_peak_mb == 8200.0
    assert statistics.system_ram_peak_delta_mb == 200.0


def test_calculate_statistics_handles_missing_gpu_measurements():
    monitor = ResourceMonitor()

    monitor.samples = [
        ResourceSample(
            timestamp_seconds=0.0,
            gpu_memory_used_mb=None,
            gpu_utilization_percent=None,
            system_ram_used_mb=8000.0,
        ),
        ResourceSample(
            timestamp_seconds=0.1,
            gpu_memory_used_mb=None,
            gpu_utilization_percent=None,
            system_ram_used_mb=8200.0,
        ),
    ]

    statistics = monitor._calculate_statistics()

    # GPU metrics should remain unavailable.
    assert statistics.gpu_memory_baseline_mb is None
    assert statistics.gpu_memory_peak_mb is None
    assert statistics.gpu_memory_peak_delta_mb is None
    assert statistics.gpu_memory_average_mb is None
    assert statistics.gpu_utilization_average_percent is None

    # System RAM monitoring should still work.
    assert statistics.system_ram_baseline_mb == 8000.0
    assert statistics.system_ram_peak_mb == 8200.0
    assert statistics.system_ram_peak_delta_mb == 200.0


def test_calculate_statistics_handles_missing_ram_measurements():
    monitor = ResourceMonitor()

    monitor.samples = [
        ResourceSample(
            timestamp_seconds=0.0,
            gpu_memory_used_mb=1000.0,
            gpu_utilization_percent=40.0,
            system_ram_used_mb=None,
        ),
        ResourceSample(
            timestamp_seconds=0.1,
            gpu_memory_used_mb=1100.0,
            gpu_utilization_percent=80.0,
            system_ram_used_mb=None,
        ),
    ]

    statistics = monitor._calculate_statistics()

    # GPU metrics should still work.
    assert statistics.gpu_memory_baseline_mb == 1000.0
    assert statistics.gpu_memory_peak_mb == 1100.0
    assert statistics.gpu_memory_peak_delta_mb == 100.0
    assert statistics.gpu_utilization_average_percent == 60.0

    # System RAM metrics should remain unavailable.
    assert statistics.system_ram_baseline_mb is None
    assert statistics.system_ram_peak_mb is None
    assert statistics.system_ram_peak_delta_mb is None


def test_capture_snapshot_does_not_start_monitoring():
    monitor = ResourceMonitor()

    snapshot = monitor.capture_snapshot()

    assert isinstance(snapshot, ResourceSample)
    assert monitor._running is False
    assert monitor._sampling_thread is None


def test_capture_snapshot_records_utc_timestamp():
    monitor = ResourceMonitor()

    snapshot = monitor.capture_snapshot()

    assert snapshot.captured_at_utc is not None
    assert snapshot.captured_at_utc.tzinfo == timezone.utc
    assert snapshot.captured_at_utc.utcoffset() == timedelta(0)


def test_collect_sample_records_utc_timestamp():
    monitor = ResourceMonitor()

    # Set up the minimum state needed for a sample.
    # GPU monitoring is skipped because no GPU handle is assigned.
    import time

    monitor._start_time = time.perf_counter()

    sample = monitor._collect_sample()

    assert sample.captured_at_utc is not None
    assert sample.captured_at_utc.tzinfo == timezone.utc
    assert sample.captured_at_utc.utcoffset() == timedelta(0)


def test_capture_snapshot_rejects_running_monitor():
    monitor = ResourceMonitor()

    monitor._running = True

    try:
        with pytest.raises(RuntimeError):
            monitor.capture_snapshot()
    finally:
        monitor._running = False
