"""
ML-based anomaly detection using Isolation Forest.

Trains on normal operational data (feature vectors from the warm-up period)
to learn baseline behaviour, then scores new observations to detect anomalies.
Anomalies — being rare and structurally different from normal data — are
isolated in fewer random splits, yielding a shorter path length and thus a
lower (more negative) Isolation Forest score.

The raw IF score is mapped to a normalised [0, 1] anomaly score so it can
be combined with the statistical detector output in the ensemble.

Training strategy:
  - The model is trained once after a configurable warm-up period
    (enough windows of normal data).
  - It can be retrained periodically on a rolling buffer to adapt to
    gradual baseline drift (concept drift).
  - Training is skipped if active faults are present in the warm-up data
    (controlled by FaultInjector state passed in at train time).

Hyperparameters (from config.yaml → detection.isolation_forest):
  - contamination:  expected proportion of anomalies (default: 0.1)
  - n_estimators:   number of isolation trees (default: 100)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from src.detection.feature_extractor import FeatureVector


@dataclass
class MLDetectionResult:
    """Result of a single Isolation Forest anomaly check."""

    service: str
    is_anomaly: bool
    anomaly_score: float          # Normalised to [0, 1]; higher = more anomalous
    raw_if_score: float           # Raw sklearn decision_function output
    feature_vector: FeatureVector | None = None
    model_trained: bool = False   # False during warm-up (score is unreliable)


class MLAnomalyDetector:
    """
    Isolation Forest based anomaly detector for operational metrics.

    Maintains one trained model per service so each service's normal behaviour
    is captured independently.
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100) -> None:
        """
        Initialise the detector with Isolation Forest hyperparameters.

        Args:
            contamination: Expected proportion of anomalies in training data.
                           Passed directly to sklearn IsolationForest.
            n_estimators:  Number of isolation trees to build.
        """
        raise NotImplementedError

    def train(self, features: np.ndarray, service: str) -> None:
        """
        Train (or retrain) the Isolation Forest model for a specific service.

        Args:
            features: 2D numpy array of shape (n_samples, n_features) containing
                      FeatureVector arrays from the warm-up / training window.
            service:  Name of the service this model will be used for.

        Raises:
            ValueError: If features has fewer than min_training_samples rows.
        """
        raise NotImplementedError

    def detect(self, feature_vector: FeatureVector) -> MLDetectionResult:
        """
        Score a single FeatureVector and return an anomaly verdict.

        If the model for the given service has not yet been trained
        (warm-up phase), returns a result with model_trained=False and
        a neutral score of 0.0.

        Args:
            feature_vector: FeatureVector produced by FeatureExtractor.

        Returns:
            MLDetectionResult with is_anomaly flag and normalised score.
        """
        raise NotImplementedError

    def detect_batch(
        self, feature_vectors: list[FeatureVector]
    ) -> list[MLDetectionResult]:
        """
        Score a list of FeatureVectors (must all be for the same service).

        Args:
            feature_vectors: List of FeatureVector objects.

        Returns:
            Corresponding list of MLDetectionResult objects.

        Raises:
            ValueError: If vectors span more than one service.
        """
        raise NotImplementedError

    def is_trained(self, service: str) -> bool:
        """
        Return True if the model for this service has been trained.

        Args:
            service: Service name.

        Returns:
            True if a trained IsolationForest model exists for the service.
        """
        raise NotImplementedError

    def get_model(self, service: str) -> Optional[IsolationForest]:
        """
        Return the raw sklearn IsolationForest model for a service.

        Args:
            service: Service name.

        Returns:
            Trained IsolationForest, or None if not yet trained.
        """
        raise NotImplementedError

    def _normalise_score(self, raw_score: float) -> float:
        """
        Map a raw Isolation Forest decision_function score to [0, 1].

        sklearn's IsolationForest.decision_function() returns negative values
        for anomalies. This method inverts and scales to [0, 1] where 1 means
        highly anomalous.

        Args:
            raw_score: Output of IsolationForest.decision_function() for one sample.

        Returns:
            Normalised anomaly score in [0, 1].
        """
        raise NotImplementedError
