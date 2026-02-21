"""
Time-series metrics generator for simulated FCT microservices.

Produces realistic operational metrics for each service on a configurable
interval (default: every 5 seconds). Metrics are modelled as Gaussian random
walks around service-specific baselines so the anomaly detectors see plausible
"normal" variance before fault injection.

Metrics produced per service:
  - cpu_percent       : CPU utilisation (0–100)
  - memory_mb         : Resident memory in megabytes
  - latency_ms        : P99 request latency in milliseconds
  - error_rate        : Errors per second (float)
  - request_rate      : Requests per second (float)
  - active_connections: Current open connections (int)

The FaultInjector calls `apply_fault_profile()` to shift baselines and variance
during an active fault, creating the anomalous signal that the detection layer
is expected to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator


@dataclass
class MetricSnapshot:
    """A single point-in-time metrics snapshot for one service."""

    timestamp: datetime
    service: str
    cpu_percent: float
    memory_mb: float
    latency_ms: float
    error_rate: float
    request_rate: float
    active_connections: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        raise NotImplementedError


@dataclass
class ServiceBaseline:
    """
    Normal operating baseline for one service.

    All metrics are modelled as Gaussian: N(mean, std).
    FaultInjector temporarily replaces these with fault profiles.
    """

    cpu_mean: float
    cpu_std: float
    memory_mean: float
    memory_std: float
    latency_mean: float
    latency_std: float
    error_rate_mean: float
    error_rate_std: float
    request_rate_mean: float
    request_rate_std: float
    connections_mean: float
    connections_std: float


class MetricsGenerator:
    """
    Generates a continuous stream of MetricSnapshot objects for all services.

    Maintains per-service baselines and applies random walk noise to simulate
    realistic metric fluctuation. Baselines can be temporarily overridden by
    the FaultInjector to produce anomalous readings.
    """

    def __init__(self, simulator_config: dict[str, Any]) -> None:
        """
        Initialise the generator from the [simulator] section of config.yaml.

        Args:
            simulator_config: Dict containing 'services' list and
                              'metrics_interval_seconds'.
        """
        raise NotImplementedError

    def generate(self) -> Generator[MetricSnapshot, None, None]:
        """
        Yield one MetricSnapshot per service per interval tick.

        Yields:
            MetricSnapshot instances in service definition order.
        """
        raise NotImplementedError

    def apply_fault_profile(self, service: str, fault_type: str) -> None:
        """
        Override the baseline for a service to simulate anomalous metrics.

        Called by FaultInjector when a fault is activated. The overridden
        baseline persists until `restore_baseline()` is called.

        Args:
            service:    Name of the affected service.
            fault_type: One of: 'crash', 'latency_spike', 'connection_failure',
                        'memory_leak', 'oom'.
        """
        raise NotImplementedError

    def restore_baseline(self, service: str) -> None:
        """
        Restore a service's metrics baseline to its normal operating values.

        Args:
            service: Name of the service to restore.
        """
        raise NotImplementedError

    def get_latest_snapshot(self, service: str) -> MetricSnapshot | None:
        """
        Return the most recently generated snapshot for a service.

        Args:
            service: Service name.

        Returns:
            Latest MetricSnapshot, or None if no snapshot has been generated yet.
        """
        raise NotImplementedError

    def _sample_metric(self, mean: float, std: float, min_val: float = 0.0) -> float:
        """
        Sample a metric value from N(mean, std), clipped at min_val.

        Args:
            mean:    Distribution mean.
            std:     Distribution standard deviation.
            min_val: Lower bound for the sampled value.

        Returns:
            Sampled metric value.
        """
        raise NotImplementedError

    def _build_default_baselines(self) -> dict[str, ServiceBaseline]:
        """
        Construct hardcoded normal baselines for each FCT service.

        Returns:
            Dict mapping service name to its ServiceBaseline.
        """
        raise NotImplementedError
