"""
Tests for the simulator package.

Covers:
  - LogGenerator:     correct schema, level distribution, fault message injection
  - MetricsGenerator: baseline sampling, fault profile application, baseline restore
  - FaultInjector:    fault activation, expiry, cascade across dependent services,
                      clear / clear_all behaviour
"""

from __future__ import annotations

import pytest

from src.simulator.log_generator import LogEntry, LogGenerator
from src.simulator.metrics_generator import MetricSnapshot, MetricsGenerator
from src.simulator.fault_injector import FaultInjector, FaultScenario


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simulator_config() -> dict:
    """Minimal simulator config mirroring config.yaml structure."""
    return {
        "services": [
            {"name": "transaction-validator",  "port": 8001, "dependencies": ["fraud-check-service", "document-processor"]},
            {"name": "fraud-check-service",    "port": 8002, "dependencies": ["title-search-service"]},
            {"name": "document-processor",     "port": 8003, "dependencies": ["title-search-service"]},
            {"name": "title-search-service",   "port": 8004, "dependencies": []},
        ],
        "log_interval_seconds": 1,
        "metrics_interval_seconds": 5,
    }


@pytest.fixture
def log_generator(simulator_config: dict) -> LogGenerator:
    """Create a LogGenerator from the test config."""
    raise NotImplementedError


@pytest.fixture
def metrics_generator(simulator_config: dict) -> MetricsGenerator:
    """Create a MetricsGenerator from the test config."""
    raise NotImplementedError


@pytest.fixture
def fault_injector(
    simulator_config: dict,
    log_generator: LogGenerator,
    metrics_generator: MetricsGenerator,
) -> FaultInjector:
    """Create a FaultInjector wired to the test generators."""
    raise NotImplementedError


# ── LogGenerator tests ────────────────────────────────────────────────────────

class TestLogGenerator:

    def test_generate_yields_one_entry_per_service(self, log_generator: LogGenerator) -> None:
        """generate() should yield exactly one LogEntry per configured service per tick."""
        raise NotImplementedError

    def test_log_entry_schema(self, log_generator: LogGenerator) -> None:
        """Each LogEntry must have non-empty timestamp, level, service, trace_id, message."""
        raise NotImplementedError

    def test_log_level_is_valid(self, log_generator: LogGenerator) -> None:
        """Log levels must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."""
        raise NotImplementedError

    def test_service_names_match_config(self, log_generator: LogGenerator) -> None:
        """LogEntry.service must match one of the configured service names."""
        raise NotImplementedError

    def test_inject_fault_message_returns_error_or_critical(
        self, log_generator: LogGenerator
    ) -> None:
        """inject_fault_message() must return a log at ERROR or CRITICAL level."""
        raise NotImplementedError

    def test_inject_fault_message_includes_fault_type(
        self, log_generator: LogGenerator
    ) -> None:
        """Fault message content or metadata should reference the fault type."""
        raise NotImplementedError

    def test_trace_ids_are_unique_across_entries(self, log_generator: LogGenerator) -> None:
        """Each generated log entry should have a distinct trace_id UUID."""
        raise NotImplementedError

    def test_to_dict_is_json_serialisable(self, log_generator: LogGenerator) -> None:
        """LogEntry.to_dict() should return a dict serialisable by json.dumps."""
        raise NotImplementedError


# ── MetricsGenerator tests ────────────────────────────────────────────────────

class TestMetricsGenerator:

    def test_generate_yields_one_snapshot_per_service(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """generate() should yield one MetricSnapshot per configured service per tick."""
        raise NotImplementedError

    def test_snapshot_values_within_plausible_range(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """cpu_percent in [0,100], memory_mb > 0, latency_ms > 0, error_rate >= 0."""
        raise NotImplementedError

    def test_apply_fault_profile_increases_latency(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """After apply_fault_profile('latency_spike'), mean latency should increase."""
        raise NotImplementedError

    def test_apply_fault_profile_crash_drops_metrics(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """After apply_fault_profile('crash'), metrics should reflect service unavailability."""
        raise NotImplementedError

    def test_restore_baseline_reverts_fault_profile(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """restore_baseline() should return metrics to their normal operating range."""
        raise NotImplementedError

    def test_get_latest_snapshot_returns_none_before_first_tick(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """get_latest_snapshot() should return None before any metrics are generated."""
        raise NotImplementedError

    def test_snapshot_to_dict_is_json_serialisable(
        self, metrics_generator: MetricsGenerator
    ) -> None:
        """MetricSnapshot.to_dict() should return a dict serialisable by json.dumps."""
        raise NotImplementedError


# ── FaultInjector tests ───────────────────────────────────────────────────────

class TestFaultInjector:

    def test_inject_returns_fault_scenario(self, fault_injector: FaultInjector) -> None:
        """inject() should return a FaultScenario with is_active=True."""
        raise NotImplementedError

    def test_active_faults_contains_injected_fault(
        self, fault_injector: FaultInjector
    ) -> None:
        """active_faults() should include a fault immediately after inject()."""
        raise NotImplementedError

    def test_fault_expires_after_duration(self, fault_injector: FaultInjector) -> None:
        """After tick() advances past duration_seconds, the fault should be resolved."""
        raise NotImplementedError

    def test_clear_removes_specific_service_fault(
        self, fault_injector: FaultInjector
    ) -> None:
        """clear(service) should deactivate all faults on that service."""
        raise NotImplementedError

    def test_clear_all_removes_every_fault(self, fault_injector: FaultInjector) -> None:
        """clear_all() should leave active_faults() empty."""
        raise NotImplementedError

    def test_inject_invalid_service_raises_value_error(
        self, fault_injector: FaultInjector
    ) -> None:
        """inject() with an unknown service name should raise ValueError."""
        raise NotImplementedError

    def test_inject_invalid_fault_type_raises_value_error(
        self, fault_injector: FaultInjector
    ) -> None:
        """inject() with an unknown fault type should raise ValueError."""
        raise NotImplementedError

    def test_fault_history_accumulates_resolved_faults(
        self, fault_injector: FaultInjector
    ) -> None:
        """fault_history() should include both active and resolved FaultScenarios."""
        raise NotImplementedError
