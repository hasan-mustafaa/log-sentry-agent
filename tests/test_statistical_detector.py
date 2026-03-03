"""Tests for StatisticalDetector."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.detection.statistical_detector import StatDetectionResult, StatisticalDetector
from src.simulator.metrics_generator import MetricSnapshot


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_snapshot(
    service: str = "title-search-service",
    cpu: float = 30.0,
    memory: float = 512.0,
    latency: float = 45.0,
    error_rate: float = 0.01,
    request_rate: float = 150.0,
    connections: int = 50,
) -> MetricSnapshot:
    return MetricSnapshot(
        timestamp=datetime.now(), service=service, cpu_percent=cpu,
        memory_mb=memory, latency_ms=latency, error_rate=error_rate,
        request_rate=request_rate, active_connections=connections,
    )


@pytest.fixture
def config() -> dict:
    return {"window_size_seconds": 60, "z_score_threshold": 3.0}


@pytest.fixture
def detector(config: dict) -> StatisticalDetector:
    return StatisticalDetector(config)


# ── Cold start ───────────────────────────────────────────────────────────────

class TestColdStart:

    def test_returns_neutral_score_with_few_samples(self, detector: StatisticalDetector) -> None:
        """Before MIN_WINDOW_SAMPLES, z-scores should be 0 → neutral score."""
        result = detector.update(_make_snapshot())
        assert result.anomaly_score < 0.1
        assert result.is_anomaly is False

    def test_no_triggered_metrics_during_cold_start(self, detector: StatisticalDetector) -> None:
        result = detector.update(_make_snapshot())
        assert result.triggered_metrics == []


# ── Normal operation ─────────────────────────────────────────────────────────

class TestNormalOperation:

    def _warm_up(self, detector: StatisticalDetector, n: int = 20) -> None:
        """Feed n normal snapshots to fill the rolling window."""
        for _ in range(n):
            detector.update(_make_snapshot())

    def test_normal_metrics_not_flagged(self, detector: StatisticalDetector) -> None:
        self._warm_up(detector)
        result = detector.update(_make_snapshot())
        assert result.is_anomaly is False
        assert result.triggered_metrics == []

    def test_extreme_cpu_triggers_anomaly(self, detector: StatisticalDetector) -> None:
        self._warm_up(detector)
        # CPU baseline is 30.0; inject a massive spike
        result = detector.update(_make_snapshot(cpu=99999.0))
        assert result.is_anomaly is True
        assert "cpu_percent" in result.triggered_metrics

    def test_extreme_latency_triggers_anomaly(self, detector: StatisticalDetector) -> None:
        self._warm_up(detector)
        result = detector.update(_make_snapshot(latency=99999.0))
        assert result.is_anomaly is True
        assert "latency_ms" in result.triggered_metrics

    def test_extreme_error_rate_triggers_anomaly(self, detector: StatisticalDetector) -> None:
        self._warm_up(detector)
        result = detector.update(_make_snapshot(error_rate=99999.0))
        assert result.is_anomaly is True
        assert "error_rate" in result.triggered_metrics


# ── Score properties ─────────────────────────────────────────────────────────

class TestScoreProperties:

    def _warm_up(self, detector: StatisticalDetector, n: int = 20) -> None:
        for _ in range(n):
            detector.update(_make_snapshot())

    def test_anomaly_score_in_unit_interval(self, detector: StatisticalDetector) -> None:
        self._warm_up(detector)
        result = detector.update(_make_snapshot(cpu=99999.0))
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_normal_score_below_0_5(self, detector: StatisticalDetector) -> None:
        self._warm_up(detector)
        result = detector.update(_make_snapshot())
        assert result.anomaly_score < 0.5

    def test_anomalous_score_above_0_5(self, detector: StatisticalDetector) -> None:
        self._warm_up(detector)
        result = detector.update(_make_snapshot(cpu=99999.0))
        assert result.anomaly_score > 0.5


# ── Z-scores dict ────────────────────────────────────────────────────────────

class TestZScores:

    def test_z_scores_has_all_monitored_metrics(self, detector: StatisticalDetector) -> None:
        result = detector.update(_make_snapshot())
        for metric in StatisticalDetector.MONITORED_METRICS:
            assert metric in result.z_scores

    def test_z_scores_zero_during_cold_start(self, detector: StatisticalDetector) -> None:
        result = detector.update(_make_snapshot())
        for z in result.z_scores.values():
            assert z == 0.0


# ── Rolling stats ────────────────────────────────────────────────────────────

class TestRollingStats:

    def test_returns_none_for_unknown_service(self, detector: StatisticalDetector) -> None:
        assert detector.get_rolling_stats("nonexistent", "cpu_percent") is None

    def test_returns_mean_std_after_updates(self, detector: StatisticalDetector) -> None:
        for _ in range(10):
            detector.update(_make_snapshot(cpu=30.0))
        stats = detector.get_rolling_stats("title-search-service", "cpu_percent")
        assert stats is not None
        mean, std = stats
        assert mean == pytest.approx(30.0)
        assert std == pytest.approx(0.0)


# ── Reset ────────────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_clears_rolling_window(self, detector: StatisticalDetector) -> None:
        detector.update(_make_snapshot())
        detector.reset("title-search-service")
        assert detector.get_rolling_stats("title-search-service", "cpu_percent") is None

    def test_reset_all(self, detector: StatisticalDetector) -> None:
        detector.update(_make_snapshot(service="svc-a"))
        detector.update(_make_snapshot(service="svc-b"))
        detector.reset()
        assert detector.get_rolling_stats("svc-a", "cpu_percent") is None
        assert detector.get_rolling_stats("svc-b", "cpu_percent") is None

    def test_reset_one_preserves_others(self, detector: StatisticalDetector) -> None:
        detector.update(_make_snapshot(service="svc-a"))
        detector.update(_make_snapshot(service="svc-b"))
        detector.reset("svc-a")
        assert detector.get_rolling_stats("svc-a", "cpu_percent") is None
        assert detector.get_rolling_stats("svc-b", "cpu_percent") is not None
