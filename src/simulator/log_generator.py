"""
Structured log generator for simulated FCT microservices.

Generates realistic, structured JSON log lines for each service defined in
config.yaml. Each log line conforms to a consistent schema (timestamp, level,
service, trace_id, message, metadata) so the downstream log parser always
receives well-formed input.

Log levels and their approximate emission rates during normal operation:
  - DEBUG:    10%
  - INFO:     75%
  - WARNING:  12%
  - ERROR:     3%

Services simulated:
  - transaction-validator  (upstream orchestrator)
  - fraud-check-service    (depends on title-search-service)
  - document-processor     (depends on title-search-service)
  - title-search-service   (leaf dependency, no upstream deps)

The FaultInjector calls into this generator to inject error-level messages
and anomalous patterns when a fault scenario is active.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator


@dataclass
class LogEntry:
    """A single structured log record emitted by a simulated service."""

    timestamp: datetime
    level: str                    # DEBUG | INFO | WARNING | ERROR | CRITICAL
    service: str                  # e.g. "transaction-validator"
    trace_id: str                 # UUID4 for correlating a request across services
    message: str                  # Human-readable log message
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        raise NotImplementedError

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        raise NotImplementedError


class LogGenerator:
    """
    Generates a continuous stream of structured log entries for all simulated
    microservices.

    Each call to `generate()` yields one LogEntry per registered service,
    simulating a synchronous polling interval. The generator can be put into
    a fault mode (by the FaultInjector) to emit elevated error rates and
    specific fault-related messages.
    """

    def __init__(self, simulator_config: dict[str, Any]) -> None:
        """
        Initialise the generator from the [simulator] section of config.yaml.

        Args:
            simulator_config: Dict containing 'services' list and
                              'log_interval_seconds'.
        """
        raise NotImplementedError

    def generate(self) -> Generator[LogEntry, None, None]:
        """
        Yield one LogEntry per service per interval tick.

        Yields:
            LogEntry instances in chronological order across services.
        """
        raise NotImplementedError

    def inject_fault_message(
        self,
        service: str,
        fault_type: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> LogEntry:
        """
        Construct and return a fault-specific ERROR or CRITICAL log entry.

        Called by FaultInjector when a fault scenario is activated.

        Args:
            service:        Name of the affected service.
            fault_type:     One of: 'crash', 'latency_spike', 'connection_failure',
                            'memory_leak', 'oom'.
            extra_metadata: Optional additional fields to include in the log entry.

        Returns:
            A LogEntry with level ERROR or CRITICAL and a fault-specific message.
        """
        raise NotImplementedError

    def _select_log_level(self, fault_active: bool) -> str:
        """
        Randomly select a log level, weighted by normal or fault-mode distribution.

        Args:
            fault_active: Whether a fault is currently injected for this service.

        Returns:
            Log level string.
        """
        raise NotImplementedError

    def _build_message(self, service: str, level: str, fault_type: str | None) -> str:
        """
        Choose a realistic log message template for the given service and level.

        Args:
            service:    Service name.
            level:      Log level.
            fault_type: Active fault type, or None for normal operation.

        Returns:
            Formatted log message string.
        """
        raise NotImplementedError

    def _new_trace_id(self) -> str:
        """Generate a fresh UUID4 trace ID string."""
        raise NotImplementedError
