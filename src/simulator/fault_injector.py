"""
Fault injector for the simulated FCT microservice mesh.

Periodically introduces realistic fault scenarios into the running simulation
so the detection and remediation layers have real anomalies to respond to.

Supported fault types:
  - crash              : Service stops responding; metrics drop to zero, logs go silent
                         then emit CRITICAL restart messages.
  - latency_spike      : P99 latency multiplies by 5–20x for a configurable duration.
  - connection_failure : Active connections drop suddenly; ERROR logs emit
                         "connection refused / timeout" messages.
  - memory_leak        : Memory grows monotonically over time until OOM or remediation.
  - oom                : Sudden memory exhaustion; service emits OOM CRITICAL log and
                         restarts.

Faults can target any of the four FCT services. Injecting into
title-search-service (the shared leaf dependency) will naturally cascade
upstream through fraud-check-service and document-processor, which is the
most interesting scenario for testing the agent's dependency-graph reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.simulator.log_generator import LogGenerator
from src.simulator.metrics_generator import MetricsGenerator


FAULT_TYPES = frozenset(
    {"crash", "latency_spike", "connection_failure", "memory_leak", "oom"}
)


@dataclass
class FaultScenario:
    """Describes a single injected fault event."""

    fault_id: str                      # UUID for tracking
    service: str                       # Target service name
    fault_type: str                    # One of FAULT_TYPES
    started_at: datetime
    duration_seconds: float            # How long the fault persists; -1 = until cleared
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Return True if the fault is still ongoing."""
        raise NotImplementedError


class FaultInjector:
    """
    Orchestrates fault scenario injection into LogGenerator and MetricsGenerator.

    Runs as a background thread (or async task) alongside the generators.
    On each tick, it checks whether a new fault should be triggered, updates
    active faults, and clears faults that have expired.
    """

    def __init__(
        self,
        simulator_config: dict[str, Any],
        log_generator: LogGenerator,
        metrics_generator: MetricsGenerator,
    ) -> None:
        """
        Initialise the injector with references to the active generators.

        Args:
            simulator_config:  [simulator] section of config.yaml.
            log_generator:     Running LogGenerator instance to inject error logs into.
            metrics_generator: Running MetricsGenerator to apply fault profiles to.
        """
        raise NotImplementedError

    def inject(
        self,
        service: str,
        fault_type: str,
        duration_seconds: float = 60.0,
    ) -> FaultScenario:
        """
        Activate a fault scenario immediately.

        Args:
            service:          Name of the target service.
            fault_type:       One of FAULT_TYPES.
            duration_seconds: How long the fault should persist. Pass -1 for
                              indefinite (must be cleared manually).

        Returns:
            The created FaultScenario.

        Raises:
            ValueError: If service or fault_type is invalid.
        """
        raise NotImplementedError

    def clear(self, service: str, fault_type: str | None = None) -> None:
        """
        Deactivate all active faults on a service (or a specific fault type).

        Args:
            service:    Target service name.
            fault_type: If provided, only clear faults of this type;
                        otherwise clear all faults on the service.
        """
        raise NotImplementedError

    def clear_all(self) -> None:
        """Deactivate every active fault across all services."""
        raise NotImplementedError

    def tick(self) -> None:
        """
        Advance the injector by one time step.

        - Checks if any active faults have expired and resolves them.
        - Randomly decides whether to inject a new fault (based on config
          fault probability settings).

        Called by the main pipeline loop on each simulation tick.
        """
        raise NotImplementedError

    def active_faults(self) -> list[FaultScenario]:
        """Return a list of all currently active FaultScenario objects."""
        raise NotImplementedError

    def fault_history(self) -> list[FaultScenario]:
        """Return the full ordered history of all faults (active + resolved)."""
        raise NotImplementedError

    def _resolve_fault(self, scenario: FaultScenario) -> None:
        """
        Mark a fault as resolved and restore normal operation.

        Args:
            scenario: The FaultScenario to resolve.
        """
        raise NotImplementedError

    def _validate_fault(self, service: str, fault_type: str) -> None:
        """
        Raise ValueError if the service or fault_type is not recognised.

        Args:
            service:    Service name to validate.
            fault_type: Fault type to validate.
        """
        raise NotImplementedError
