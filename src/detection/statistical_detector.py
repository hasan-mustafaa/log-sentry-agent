"""
Statistical anomaly detector using Z-score and moving average.

Provides fast, single-metric anomaly detection as the lightweight component
of the ensemble. Operates directly on raw metric time-series (not feature
vectors) so it can emit low-latency signals before the ML model has enough
data to be reliable.

Detection strategy:
  1. Maintain a rolling window (default: 60 s) of each metric per service.
  2. Compute the rolling mean (μ) and rolling standard deviation (σ).
  3. For each new data point x:
         z = (x - μ) / σ
  4. If |z| > z_score_threshold (default: 3.0), the metric is flagged.
  5. A service is declared anomalous if ANY monitored metric is flagged.

The returned anomaly score is normalised to [0, 1] by clipping and scaling
the max absolute z-score across all metrics, so it can be combined with the
Isolation Forest score in the ensemble.

Monitored metrics:
  - cpu_percent
  - memory_mb
  - latency_ms
  - error_rate
  - active_connections
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.simulator.metrics_generator import MetricSnapshot


@dataclass
class StatDetectionResult:
    """Result of a single statistical anomaly check."""

    service: str
    is_anomaly: bool
    anomaly_score: float               # Normalised to [0, 1]
    triggered_metrics: list[str]       # Names of metrics that exceeded threshold
    z_scores: dict[str, float]         # Per-metric z-scores for inspection
    snapshot: MetricSnapshot | None = None


class StatisticalDetector:
    """
    Rolling Z-score anomaly detector for operational metrics.

    Maintains independent rolling windows per service per metric.
    Safe to call every metrics tick; handles the cold-start period
    (insufficient window data) by returning a neutral score.
    """

    MONITORED_METRICS: tuple[str, ...] = (
        "cpu_percent",
        "memory_mb",
        "latency_ms",
        "error_rate",
        "active_connections",
    )

    def __init__(self, detection_config: dict[str, Any]) -> None:
        """
        Initialise the detector from the [detection] section of config.yaml.

        Args:
            detection_config: Must contain 'window_size_seconds' and
                              'z_score_threshold'.
        """
        raise NotImplementedError

    def update(self, snapshot: MetricSnapshot) -> StatDetectionResult:
        """
        Ingest a new MetricSnapshot and return the anomaly verdict.

        Appends each metric value to its rolling window, recomputes z-scores,
        and determines if any metric exceeds the threshold.

        Args:
            snapshot: Latest MetricSnapshot from the MetricsGenerator.

        Returns:
            StatDetectionResult with is_anomaly flag, score, and diagnostics.
        """
        raise NotImplementedError

    def get_rolling_stats(
        self, service: str, metric: str
    ) -> tuple[float, float] | None:
        """
        Return the current (mean, std) for a specific service metric window.

        Args:
            service: Service name.
            metric:  One of MONITORED_METRICS.

        Returns:
            Tuple of (mean, std), or None if the window is not yet populated.
        """
        raise NotImplementedError

    def reset(self, service: str | None = None) -> None:
        """
        Clear rolling windows for one or all services.

        Args:
            service: Service name to reset, or None to reset all.
        """
        raise NotImplementedError

    def _compute_z_score(self, value: float, mean: float, std: float) -> float:
        """
        Compute the z-score for a single observation.

        Returns 0.0 if std is zero to avoid division-by-zero.

        Args:
            value: Observed metric value.
            mean:  Rolling window mean.
            std:   Rolling window standard deviation.

        Returns:
            Z-score (signed float).
        """
        raise NotImplementedError

    def _normalise_score(self, max_abs_z: float) -> float:
        """
        Map the maximum absolute z-score to a [0, 1] anomaly score.

        Uses sigmoid-like scaling so scores around the threshold map to ~0.5.

        Args:
            max_abs_z: Maximum absolute z-score across all monitored metrics.

        Returns:
            Normalised anomaly score in [0, 1].
        """
        raise NotImplementedError
