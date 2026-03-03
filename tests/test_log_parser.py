"""Standalone tests for LogParser (Drain algorithm).

Isolated from the rest of the detection package so it can run without
FeatureExtractor, StatisticalDetector, or MLDetector being implemented.
"""

from __future__ import annotations

from datetime import datetime

import pytest

# Import directly from the module to avoid detection/__init__.py import chain.
from src.detection.log_parser import LogParser, ParsedLog
from src.simulator.log_generator import LogEntry


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_entry(
    service: str = "title-search-service",
    level: str = "INFO",
    message: str = "GET /search 200 42ms",
) -> LogEntry:
    return LogEntry(timestamp=datetime.now(), level=level, service=service, message=message)


@pytest.fixture
def parser() -> LogParser:
    return LogParser({"window_size_seconds": 60})


# ── Basic parsing ────────────────────────────────────────────────────────────

class TestParse:

    def test_returns_parsed_log(self, parser: LogParser) -> None:
        result = parser.parse(_make_entry())
        assert isinstance(result, ParsedLog)
        assert result.template  # non-empty

    def test_preserves_original_entry(self, parser: LogParser) -> None:
        entry = _make_entry(service="fraud-check-service", level="ERROR")
        result = parser.parse(entry)
        assert result.original is entry
        assert result.service == "fraud-check-service"
        assert result.level == "ERROR"

    def test_cluster_id_is_positive_int(self, parser: LogParser) -> None:
        result = parser.parse(_make_entry())
        assert isinstance(result.cluster_id, int)
        assert result.cluster_id >= 1


# ── Template generalization ──────────────────────────────────────────────────

class TestTemplateGeneralization:

    def test_variable_tokens_are_wildcarded(self, parser: LogParser) -> None:
        """Two messages differing only in numbers should share a template."""
        parser.parse(_make_entry(message="Connection to 10.0.0.1 timed out after 5000ms"))
        p2 = parser.parse(_make_entry(message="Connection to 192.168.1.5 timed out after 3000ms"))
        assert "<*>" in p2.template

    def test_parameters_extracted(self, parser: LogParser) -> None:
        parser.parse(_make_entry(message="Connection to 10.0.0.1 timed out after 5000ms"))
        p2 = parser.parse(_make_entry(message="Connection to 192.168.1.5 timed out after 3000ms"))
        # Should extract the variable parts
        assert len(p2.parameters) > 0

    def test_same_message_same_cluster(self, parser: LogParser) -> None:
        msg = "Title record not found for TXN-12345: index may be stale"
        p1 = parser.parse(_make_entry(message=msg))
        p2 = parser.parse(_make_entry(message=msg))
        assert p1.cluster_id == p2.cluster_id


# ── Batch parsing ────────────────────────────────────────────────────────────

class TestParseBatch:

    def test_output_length_matches_input(self, parser: LogParser) -> None:
        entries = [_make_entry(message=f"Request {i} completed") for i in range(5)]
        results = parser.parse_batch(entries)
        assert len(results) == 5

    def test_batch_matches_sequential(self, parser: LogParser) -> None:
        entries = [
            _make_entry(message="GET /api/v1/users 200 15ms"),
            _make_entry(message="POST /api/v1/orders 201 42ms"),
        ]
        batch_results = parser.parse_batch(entries)

        parser2 = LogParser({"window_size_seconds": 60})
        sequential_results = [parser2.parse(e) for e in entries]

        for b, s in zip(batch_results, sequential_results):
            assert b.template == s.template


# ── Service isolation ────────────────────────────────────────────────────────

class TestServiceIsolation:

    def test_no_cross_service_bleed(self, parser: LogParser) -> None:
        """Templates from service A should not appear in service B's templates."""
        parser.parse(_make_entry(service="auth-svc", message="Login succeeded for user admin"))
        parser.parse(_make_entry(service="db-svc", message="Query completed in 42ms"))

        auth_templates = parser.get_templates("auth-svc")
        db_templates = parser.get_templates("db-svc")

        # Each should have their own templates, not the other's.
        assert len(auth_templates) > 0
        assert len(db_templates) > 0
        assert set(auth_templates.keys()).isdisjoint(set(db_templates.keys()))

    def test_same_message_different_services_different_clusters(self, parser: LogParser) -> None:
        msg = "Connection refused: 3 retries exhausted"
        p1 = parser.parse(_make_entry(service="svc-a", message=msg))
        p2 = parser.parse(_make_entry(service="svc-b", message=msg))
        assert p1.cluster_id != p2.cluster_id


# ── get_templates ────────────────────────────────────────────────────────────

class TestGetTemplates:

    def test_returns_known_clusters(self, parser: LogParser) -> None:
        parser.parse(_make_entry(message="GET /search 200 42ms"))
        templates = parser.get_templates("title-search-service")
        assert len(templates) >= 1

    def test_unknown_service_returns_empty(self, parser: LogParser) -> None:
        assert parser.get_templates("nonexistent-service") == {}

    def test_no_filter_aggregates_all(self, parser: LogParser) -> None:
        parser.parse(_make_entry(service="svc-a", message="hello world"))
        parser.parse(_make_entry(service="svc-b", message="goodbye world"))
        all_templates = parser.get_templates()
        assert len(all_templates) >= 2


# ── Reset ────────────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_clears_templates(self, parser: LogParser) -> None:
        parser.parse(_make_entry(message="GET /search 200 42ms"))
        assert len(parser.get_templates("title-search-service")) > 0

        parser.reset("title-search-service")
        assert parser.get_templates("title-search-service") == {}

    def test_reset_one_preserves_others(self, parser: LogParser) -> None:
        parser.parse(_make_entry(service="svc-a", message="request ok"))
        parser.parse(_make_entry(service="svc-b", message="request ok"))

        parser.reset("svc-a")
        assert parser.get_templates("svc-a") == {}
        assert len(parser.get_templates("svc-b")) > 0

    def test_reset_all(self, parser: LogParser) -> None:
        parser.parse(_make_entry(service="svc-a", message="msg a"))
        parser.parse(_make_entry(service="svc-b", message="msg b"))

        parser.reset()
        assert parser.get_templates() == {}

    def test_parser_works_after_reset(self, parser: LogParser) -> None:
        parser.parse(_make_entry(message="first message"))
        parser.reset()
        result = parser.parse(_make_entry(message="second message"))
        assert isinstance(result, ParsedLog)
        assert result.template
