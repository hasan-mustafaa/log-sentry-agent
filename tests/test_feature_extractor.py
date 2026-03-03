"""Tests for FeatureExtractor."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from src.detection.feature_extractor import FEATURE_NAMES, FeatureExtractor, FeatureVector
from src.detection.log_parser import ParsedLog
from src.simulator.log_generator import LogEntry
from src.simulator.metrics_generator import MetricSnapshot


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_parsed_log(
    service: str = "title-search-service",
    level: str = "INFO",
    cluster_id: int = 1,
    template: str = "GET /search <*> <*>ms",
) -> ParsedLog:
    entry = LogEntry(
        timestamp=datetime.now(), level=level, service=service,
        message="GET /search 200 42ms",
    )
    return ParsedLog(original=entry, template=template, cluster_id=cluster_id)


def _make_metric(
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
    return {
        "window_size_seconds": 60,
        "ensemble_weights": {"statistical": 0.4, "ml": 0.6},
    }


@pytest.fixture
def extractor(config: dict) -> FeatureExtractor:
    return FeatureExtractor(config)


# ── Extract ──────────────────────────────────────────────────────────────────

class TestExtract:

    def test_returns_none_on_empty_window(self, extractor: FeatureExtractor) -> None:
        assert extractor.extract("title-search-service") is None

    def test_feature_vector_has_correct_shape(self, extractor: FeatureExtractor) -> None:
        extractor.ingest_log(_make_parsed_log())
        extractor.ingest_metric(_make_metric())
        fv = extractor.extract("title-search-service")
        assert fv is not None
        assert len(fv.features) == FeatureExtractor.FEATURE_DIM
        assert fv.feature_names == FEATURE_NAMES

    def test_returns_feature_vector_with_logs_only(self, extractor: FeatureExtractor) -> None:
        extractor.ingest_log(_make_parsed_log())
        fv = extractor.extract("title-search-service")
        assert fv is not None
        assert fv.service == "title-search-service"

    def test_returns_feature_vector_with_metrics_only(self, extractor: FeatureExtractor) -> None:
        extractor.ingest_metric(_make_metric())
        fv = extractor.extract("title-search-service")
        assert fv is not None


# ── Log features ─────────────────────────────────────────────────────────────

class TestLogFeatures:

    def test_error_rate_increases_with_errors(self, extractor: FeatureExtractor) -> None:
        for _ in range(8):
            extractor.ingest_log(_make_parsed_log(level="INFO"))
        for _ in range(2):
            extractor.ingest_log(_make_parsed_log(level="ERROR"))

        fv = extractor.extract("title-search-service")
        assert fv is not None
        error_rate_idx = FEATURE_NAMES.index("error_rate")
        assert fv.features[error_rate_idx] == pytest.approx(0.2)

    def test_unique_templates_counted(self, extractor: FeatureExtractor) -> None:
        extractor.ingest_log(_make_parsed_log(cluster_id=1))
        extractor.ingest_log(_make_parsed_log(cluster_id=2))
        extractor.ingest_log(_make_parsed_log(cluster_id=3))

        fv = extractor.extract("title-search-service")
        unique_idx = FEATURE_NAMES.index("unique_templates")
        assert fv.features[unique_idx] == 3.0

    def test_top_template_frequency(self, extractor: FeatureExtractor) -> None:
        # 3 logs with cluster 1, 1 log with cluster 2 → top freq = 0.75
        for _ in range(3):
            extractor.ingest_log(_make_parsed_log(cluster_id=1))
        extractor.ingest_log(_make_parsed_log(cluster_id=2))

        fv = extractor.extract("title-search-service")
        top_idx = FEATURE_NAMES.index("top_template_frequency")
        assert fv.features[top_idx] == pytest.approx(0.75)


# ── Metric features ─────────────────────────────────────────────────────────

class TestMetricFeatures:

    def test_cpu_mean(self, extractor: FeatureExtractor) -> None:
        extractor.ingest_metric(_make_metric(cpu=20.0))
        extractor.ingest_metric(_make_metric(cpu=40.0))

        fv = extractor.extract("title-search-service")
        cpu_idx = FEATURE_NAMES.index("cpu_percent_mean")
        assert fv.features[cpu_idx] == pytest.approx(30.0)

    def test_latency_p99(self, extractor: FeatureExtractor) -> None:
        for lat in range(1, 101):
            extractor.ingest_metric(_make_metric(latency=float(lat)))

        fv = extractor.extract("title-search-service")
        p99_idx = FEATURE_NAMES.index("latency_ms_p99")
        assert fv.features[p99_idx] >= 99.0  # p99 of 1..100

    def test_single_metric_std_is_zero(self, extractor: FeatureExtractor) -> None:
        extractor.ingest_metric(_make_metric(cpu=50.0))

        fv = extractor.extract("title-search-service")
        std_idx = FEATURE_NAMES.index("cpu_percent_std")
        assert fv.features[std_idx] == 0.0


# ── Ensemble scoring ─────────────────────────────────────────────────────────

class TestEnsembleScore:

    def test_weighted_average(self, extractor: FeatureExtractor) -> None:
        score, _ = extractor.compute_ensemble_score(stat_score=1.0, ml_score=0.0)
        assert score == pytest.approx(0.4)  # 0.4 * 1.0 + 0.6 * 0.0

    def test_is_anomaly_false_for_low_scores(self, extractor: FeatureExtractor) -> None:
        _, is_anomaly = extractor.compute_ensemble_score(stat_score=0.0, ml_score=0.0)
        assert is_anomaly is False

    def test_is_anomaly_true_for_high_scores(self, extractor: FeatureExtractor) -> None:
        _, is_anomaly = extractor.compute_ensemble_score(stat_score=1.0, ml_score=1.0)
        assert is_anomaly is True

    def test_score_clamped_to_unit_interval(self, extractor: FeatureExtractor) -> None:
        score, _ = extractor.compute_ensemble_score(stat_score=1.5, ml_score=1.5)
        assert 0.0 <= score <= 1.0


# ── Serialization ────────────────────────────────────────────────────────────

class TestSerialization:

    def test_to_dict_is_json_serializable(self, extractor: FeatureExtractor) -> None:
        extractor.ingest_log(_make_parsed_log())
        extractor.ingest_metric(_make_metric())
        fv = extractor.extract("title-search-service")
        d = fv.to_dict()
        # Should not raise
        json.dumps(d)
        assert d["service"] == "title-search-service"
        assert isinstance(d["features"], list)


# ── Flush window ─────────────────────────────────────────────────────────────

class TestFlushWindow:

    def test_flush_clears_old_data(self, extractor: FeatureExtractor) -> None:
        extractor.ingest_log(_make_parsed_log())
        extractor.ingest_metric(_make_metric())
        # With window=60s, data just ingested should survive flush
        extractor.flush_window("title-search-service")
        fv = extractor.extract("title-search-service")
        assert fv is not None

    def test_extract_none_after_total_flush(self) -> None:
        # Window of 0s means everything is "old"
        ext = FeatureExtractor({"window_size_seconds": 0, "ensemble_weights": {}})
        ext.ingest_log(_make_parsed_log())
        ext.ingest_metric(_make_metric())
        ext.flush_window("title-search-service")
        assert ext.extract("title-search-service") is None
