"""
Feature extractor for the anomaly detection pipeline.

Aggregates streams of ParsedLog objects and MetricSnapshot objects over a
sliding time window (default: 60 seconds) into fixed-size feature vectors
suitable for input to both the StatisticalDetector and the MLDetector.

Feature vector composition (per service, per window):
  Log-derived features:
    - log_count_total        : total log lines emitted
    - log_count_error        : ERROR + CRITICAL log count
    - log_count_warning      : WARNING log count
    - error_rate             : log_count_error / log_count_total
    - unique_templates       : number of distinct Drain cluster IDs seen
    - top_template_frequency : fraction of logs matching the most common template

  Metrics-derived features (mean over window):
    - cpu_percent_mean
    - cpu_percent_std
    - memory_mb_mean
    - memory_mb_std
    - latency_ms_mean
    - latency_ms_p99         : approximated from window samples
    - metric_error_rate_mean
    - request_rate_mean
    - active_connections_mean

All features are normalised to [0, 1] or z-scored before being passed to the
detectors (controlled by the 'normalise' flag).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.detection.log_parser import ParsedLog
from src.simulator.metrics_generator import MetricSnapshot


@dataclass
class FeatureVector:
    """
    A fixed-size numerical feature vector for one service over one time window.

    Attributes:
        service:    The service this vector describes.
        window_start: Start timestamp of the aggregation window.
        window_end:   End timestamp of the aggregation window.
        features:   Numpy array of feature values (length = FEATURE_DIM).
        feature_names: Ordered list of feature names matching the array.
        ensemble_score: Combined anomaly score populated by the detectors (0–1).
        is_anomaly:   Final anomaly verdict after ensemble scoring.
    """

    service: str
    window_start: datetime
    window_end: datetime
    features: np.ndarray
    feature_names: list[str]
    ensemble_score: float = 0.0
    is_anomaly: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary (converts ndarray to list)."""
        raise NotImplementedError


class FeatureExtractor:
    """
    Maintains a sliding time window of parsed logs and metric snapshots per
    service, and emits a FeatureVector at the end of each window.

    Also responsible for fusing the statistical and ML detector scores into
    the final ensemble score using configurable weights.
    """

    FEATURE_DIM: int = 15  # Number of features in each vector

    def __init__(self, detection_config: dict[str, Any]) -> None:
        """
        Initialise the extractor with window and ensemble weight settings.

        Args:
            detection_config: The [detection] section of config.yaml, which
                              provides window_size_seconds and ensemble_weights.
        """
        raise NotImplementedError

    def ingest_log(self, parsed_log: ParsedLog) -> None:
        """
        Add a parsed log to the current window buffer for its service.

        Args:
            parsed_log: A ParsedLog produced by LogParser.
        """
        raise NotImplementedError

    def ingest_metric(self, snapshot: MetricSnapshot) -> None:
        """
        Add a metrics snapshot to the current window buffer for its service.

        Args:
            snapshot: A MetricSnapshot produced by MetricsGenerator.
        """
        raise NotImplementedError

    def extract(self, service: str) -> FeatureVector | None:
        """
        Extract a FeatureVector from the current window for the given service.

        Returns None if the window has insufficient data (cold-start period).

        Args:
            service: Service name to extract features for.

        Returns:
            FeatureVector or None.
        """
        raise NotImplementedError

    def flush_window(self, service: str) -> None:
        """
        Slide the window forward for a service, discarding data older than
        window_size_seconds.

        Args:
            service: Service name to flush.
        """
        raise NotImplementedError

    def compute_ensemble_score(
        self,
        stat_score: float,
        ml_score: float,
    ) -> tuple[float, bool]:
        """
        Combine statistical and ML anomaly scores into a single ensemble score.

        Args:
            stat_score: Anomaly signal from StatisticalDetector (0–1).
            ml_score:   Anomaly signal from MLDetector (0–1).

        Returns:
            Tuple of (ensemble_score: float, is_anomaly: bool).
        """
        raise NotImplementedError

    def _build_log_features(self, service: str) -> dict[str, float]:
        """
        Compute log-derived feature values from the current window buffer.

        Args:
            service: Service name.

        Returns:
            Dict of feature_name → value.
        """
        raise NotImplementedError

    def _build_metric_features(self, service: str) -> dict[str, float]:
        """
        Compute metrics-derived feature values from the current window buffer.

        Args:
            service: Service name.

        Returns:
            Dict of feature_name → value.
        """
        raise NotImplementedError
