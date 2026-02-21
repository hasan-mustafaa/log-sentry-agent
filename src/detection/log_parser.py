"""
Log parser using the Drain algorithm.

Converts a stream of raw, unstructured log message strings into structured
log templates using the Drain3 library. Each parsed log is enriched with:
  - The extracted template (e.g., "Connection to <*> timed out after <*> ms")
  - Variable token values captured in the <*> wildcards
  - A cluster ID that groups logs sharing the same template

This structured representation is then fed to the FeatureExtractor, which
counts template frequencies per time window to build feature vectors.

Why Drain?
  - Online / streaming: no batch processing or pre-defined rules required
  - Adapts automatically to new log patterns as services emit them
  - Low memory footprint even with thousands of distinct templates

Configuration (from config.yaml → detection):
  - window_size_seconds: indirectly controls how often template counts reset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.simulator.log_generator import LogEntry


@dataclass
class ParsedLog:
    """
    A log entry enriched with Drain-extracted template information.

    Attributes:
        original:    The source LogEntry from the simulator.
        template:    The generalised log template string (e.g., "GET <*> 200 <*>ms").
        cluster_id:  Integer ID of the Drain cluster this log belongs to.
        parameters:  Ordered list of variable values extracted from <*> slots.
    """

    original: LogEntry
    template: str
    cluster_id: int
    parameters: list[str] = field(default_factory=list)

    @property
    def service(self) -> str:
        """Convenience accessor for the originating service name."""
        return self.original.service

    @property
    def level(self) -> str:
        """Convenience accessor for the log level."""
        return self.original.level


class LogParser:
    """
    Online log parser wrapping the Drain3 TemplateMiner.

    Maintains a single Drain instance per service so that each service's
    log vocabulary is learnt independently (avoids cross-service template
    pollution).
    """

    def __init__(self, detection_config: dict[str, Any]) -> None:
        """
        Initialise one Drain TemplateMiner per configured service.

        Args:
            detection_config: The [detection] section of config.yaml.
        """
        raise NotImplementedError

    def parse(self, log_entry: LogEntry) -> ParsedLog:
        """
        Parse a single log entry and return its enriched ParsedLog.

        Passes the log message through the appropriate per-service Drain
        instance and extracts template + parameters.

        Args:
            log_entry: Raw LogEntry from the LogGenerator.

        Returns:
            ParsedLog with template, cluster_id, and extracted parameters.
        """
        raise NotImplementedError

    def parse_batch(self, log_entries: list[LogEntry]) -> list[ParsedLog]:
        """
        Parse a list of log entries in order.

        Args:
            log_entries: Ordered list of raw LogEntry objects.

        Returns:
            Corresponding list of ParsedLog objects.
        """
        raise NotImplementedError

    def get_templates(self, service: str | None = None) -> dict[int, str]:
        """
        Return all known templates, optionally filtered to a single service.

        Args:
            service: If provided, return only templates learnt from this service.

        Returns:
            Dict mapping cluster_id → template string.
        """
        raise NotImplementedError

    def reset(self, service: str | None = None) -> None:
        """
        Reset the Drain state for one or all services.

        Useful between test runs to prevent template bleed-over.

        Args:
            service: Service name to reset, or None to reset all.
        """
        raise NotImplementedError

    def _get_miner(self, service: str) -> Any:
        """
        Retrieve (or lazily create) the Drain TemplateMiner for a service.

        Args:
            service: Service name.

        Returns:
            drain3.TemplateMiner instance.
        """
        raise NotImplementedError
