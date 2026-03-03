"""Tests for MLAnomalyDetector (Isolation Forest)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.detection.feature_extractor import FEATURE_NAMES, FeatureVector
from src.detection.ml_detector import MIN_TRAINING_SAMPLES, MLAnomalyDetector


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_feature_vector(
    service: str = "title-search-service",
    features: np.ndarray | None = None,
) -> FeatureVector:
    if features is None:
        features = np.random.default_rng(42).normal(loc=50, scale=5, size=15)
    return FeatureVector(
        service=service,
        window_start=datetime.now(),
        window_end=datetime.now(),
        features=features,
        feature_names=list(FEATURE_NAMES),
    )


def _make_normal_training_data(n: int = 50, seed: int = 42) -> np.ndarray:
    """Generate n samples of 'normal' data around a known baseline."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=50, scale=5, size=(n, 15))


@pytest.fixture
def detector() -> MLAnomalyDetector:
    return MLAnomalyDetector(contamination=0.1, n_estimators=10)


@pytest.fixture
def trained_detector(detector: MLAnomalyDetector) -> MLAnomalyDetector:
    """Detector with a trained model for title-search-service."""
    data = _make_normal_training_data()
    detector.train(data, "title-search-service")
    return detector


# ── Before training ──────────────────────────────────────────────────────────

class TestBeforeTraining:

    def test_is_trained_false(self, detector: MLAnomalyDetector) -> None:
        assert detector.is_trained("title-search-service") is False

    def test_get_model_returns_none(self, detector: MLAnomalyDetector) -> None:
        assert detector.get_model("title-search-service") is None

    def test_detect_returns_model_trained_false(self, detector: MLAnomalyDetector) -> None:
        fv = _make_feature_vector()
        result = detector.detect(fv)
        assert result.model_trained is False
        assert result.anomaly_score == 0.0
        assert result.is_anomaly is False


# ── Training ─────────────────────────────────────────────────────────────────

class TestTraining:

    def test_is_trained_true_after_train(self, detector: MLAnomalyDetector) -> None:
        data = _make_normal_training_data()
        detector.train(data, "title-search-service")
        assert detector.is_trained("title-search-service") is True

    def test_get_model_returns_model_after_train(self, detector: MLAnomalyDetector) -> None:
        data = _make_normal_training_data()
        detector.train(data, "title-search-service")
        model = detector.get_model("title-search-service")
        assert model is not None

    def test_train_requires_minimum_samples(self, detector: MLAnomalyDetector) -> None:
        too_few = np.zeros((MIN_TRAINING_SAMPLES - 1, 15))
        with pytest.raises(ValueError, match="at least"):
            detector.train(too_few, "title-search-service")

    def test_train_accepts_exact_minimum(self, detector: MLAnomalyDetector) -> None:
        data = _make_normal_training_data(n=MIN_TRAINING_SAMPLES)
        detector.train(data, "title-search-service")
        assert detector.is_trained("title-search-service") is True


# ── Detection ────────────────────────────────────────────────────────────────

class TestDetection:

    def test_normal_features_score_below_0_5(self, trained_detector: MLAnomalyDetector) -> None:
        """Features drawn from the training distribution should score low."""
        rng = np.random.default_rng(99)
        normal = rng.normal(loc=50, scale=5, size=15)
        fv = _make_feature_vector(features=normal)
        result = trained_detector.detect(fv)
        assert result.model_trained is True
        assert result.anomaly_score < 0.5

    def test_anomalous_features_score_above_0_5(
        self, trained_detector: MLAnomalyDetector
    ) -> None:
        """Features far from the training distribution should score high."""
        anomalous = np.full(15, 99999.0)
        fv = _make_feature_vector(features=anomalous)
        result = trained_detector.detect(fv)
        assert result.anomaly_score > 0.5

    def test_normalized_score_in_unit_interval(
        self, trained_detector: MLAnomalyDetector
    ) -> None:
        for val in [0.0, 50.0, 99999.0]:
            fv = _make_feature_vector(features=np.full(15, val))
            result = trained_detector.detect(fv)
            assert 0.0 <= result.anomaly_score <= 1.0

    def test_detect_returns_model_trained_true(
        self, trained_detector: MLAnomalyDetector
    ) -> None:
        fv = _make_feature_vector()
        result = trained_detector.detect(fv)
        assert result.model_trained is True


# ── Batch detection ──────────────────────────────────────────────────────────

class TestBatchDetection:

    def test_batch_length_matches_input(self, trained_detector: MLAnomalyDetector) -> None:
        vectors = [_make_feature_vector() for _ in range(5)]
        results = trained_detector.detect_batch(vectors)
        assert len(results) == 5

    def test_batch_matches_sequential(self, trained_detector: MLAnomalyDetector) -> None:
        rng = np.random.default_rng(123)
        vectors = [
            _make_feature_vector(features=rng.normal(50, 5, size=15))
            for _ in range(3)
        ]
        batch_results = trained_detector.detect_batch(vectors)
        sequential_results = [trained_detector.detect(fv) for fv in vectors]

        for b, s in zip(batch_results, sequential_results):
            assert b.anomaly_score == pytest.approx(s.anomaly_score)


# ── Service isolation ────────────────────────────────────────────────────────

class TestServiceIsolation:

    def test_untrained_service_returns_neutral(
        self, trained_detector: MLAnomalyDetector
    ) -> None:
        """Model trained for title-search-service; fraud-check should be untrained."""
        fv = _make_feature_vector(service="fraud-check-service")
        result = trained_detector.detect(fv)
        assert result.model_trained is False
        assert result.anomaly_score == 0.0
