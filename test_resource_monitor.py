import time

from benchmark.monitoring import ResourceMonitor


def main():
    monitor = ResourceMonitor(sample_interval_seconds=0.1)

    print("Starting resource monitor...")
    monitor.start()

    try:
        time.sleep(2)
    finally:
        samples = monitor.stop()

    print(f"\nCollected {len(samples)} samples")
    print(f"Monitoring errors: {monitor.errors}")

    if samples:
        print("\nFirst sample:")
        print(samples[0])

        print("\nLast sample:")
        print(samples[-1])

        print("\nSample timestamps:")
        print([round(sample.timestamp_seconds, 3) for sample in samples])

    statistics = monitor._calculate_statistics()

    print("\nResource statistics:")
    print(statistics)


if __name__ == "__main__":
    main()
