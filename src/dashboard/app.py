"""
Streamlit dashboard for the LogSentry Agent.

Renders four main panels that refresh every config.dashboard.refresh_interval_seconds:

  1. Service Health Overview
       - Colour-coded status cards per service (healthy / degraded / down).
       - Current metric values (CPU %, memory MB, latency ms, error rate).

  2. Metrics Charts
       - Rolling time-series line charts for each metric per service (Plotly).
       - Anomaly score overlay on each chart so spikes correlate with detections.

  3. Log Stream
       - Live scrolling table of the last config.dashboard.max_log_display log entries.
       - Colour-coded rows: grey (DEBUG/INFO), yellow (WARNING), red (ERROR/CRITICAL).
       - Drain template shown alongside the raw message.

  4. Agent Reasoning & Remediation
       - Anomaly alert cards with service, score, triggered metrics, detected_at.
       - Expandable ReAct trace: each step shows Thought → Action → Observation.
       - Final RCA report rendered as a formatted JSON block.
       - Remediation log table: action, target, outcome, guardrail decision, timestamp.

State management:
  The dashboard reads from a shared in-memory state object (DashboardState)
  that is populated by the main pipeline. Streamlit's session_state is used to
  persist state across page re-renders without restarting the pipeline.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DashboardState:
    """
    Shared state container populated by the pipeline and read by the dashboard.

    All fields are append-only lists or dicts to allow safe concurrent writes
    from the pipeline thread and reads from the Streamlit render thread.

    Attributes:
        metric_history:    Per-service list of MetricSnapshot dicts (capped at N).
        log_buffer:        Ring buffer of the last max_log_display log entry dicts.
        anomaly_events:    Ordered list of anomaly detection event dicts.
        agent_results:     Ordered list of AgentResult dicts from completed runs.
        remediation_log:   Ordered list of ExecutionResult dicts.
        service_status:    Dict mapping service name → 'healthy'|'degraded'|'down'.
    """

    metric_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    log_buffer: list[dict[str, Any]] = field(default_factory=list)
    anomaly_events: list[dict[str, Any]] = field(default_factory=list)
    agent_results: list[dict[str, Any]] = field(default_factory=list)
    remediation_log: list[dict[str, Any]] = field(default_factory=list)
    service_status: dict[str, str] = field(default_factory=dict)
    pipeline_started_at: datetime | None = None


def run_dashboard(state: DashboardState | None = None) -> None:
    """
    Entry point for the Streamlit dashboard.

    When called from the pipeline (src/main.py), a pre-populated DashboardState
    is passed in. When called directly via `streamlit run`, a new empty state is
    created and the dashboard shows a "Waiting for pipeline..." message.

    Args:
        state: Shared DashboardState from the running pipeline, or None.
    """
    raise NotImplementedError


def render_header() -> None:
    """
    Render the dashboard title, subtitle, and pipeline status badge.

    Shows whether the pipeline is running, paused, or waiting to start.
    """
    raise NotImplementedError


def render_service_health(state: DashboardState) -> None:
    """
    Render the service health overview row.

    Displays one status card per FCT service using Streamlit columns.
    Each card shows: service name, status badge, and current key metrics.

    Args:
        state: Current DashboardState.
    """
    raise NotImplementedError


def render_metrics_charts(state: DashboardState) -> None:
    """
    Render rolling time-series charts for each service using Plotly.

    Creates a tab per service, each containing subplots for:
    CPU %, memory MB, latency ms, error rate, and anomaly score.

    Args:
        state: Current DashboardState containing metric_history.
    """
    raise NotImplementedError


def render_log_stream(state: DashboardState) -> None:
    """
    Render the live scrolling log stream table.

    Displays the most recent log entries from state.log_buffer in a
    colour-coded dataframe. ERROR/CRITICAL rows are highlighted in red,
    WARNING in yellow, INFO/DEBUG in default styling.

    Args:
        state: Current DashboardState containing log_buffer.
    """
    raise NotImplementedError


def render_anomaly_alerts(state: DashboardState) -> None:
    """
    Render active anomaly alert cards.

    Each card shows: service, ensemble anomaly score, triggered metrics,
    detection timestamp, and a link to the corresponding agent trace.

    Args:
        state: Current DashboardState containing anomaly_events.
    """
    raise NotImplementedError


def render_agent_trace(state: DashboardState) -> None:
    """
    Render the ReAct agent reasoning trace for the most recent run.

    Uses Streamlit expanders to show each Thought → Action → Observation
    step, followed by the final structured RCA report as a formatted JSON block.

    Args:
        state: Current DashboardState containing agent_results.
    """
    raise NotImplementedError


def render_remediation_log(state: DashboardState) -> None:
    """
    Render the remediation action history as a sortable table.

    Columns: timestamp, service, action_type, outcome, guardrail_blocked, reason.

    Args:
        state: Current DashboardState containing remediation_log.
    """
    raise NotImplementedError


def _status_badge(status: str) -> str:
    """
    Return a coloured emoji badge for a service status string.

    Args:
        status: One of 'healthy', 'degraded', 'down'.

    Returns:
        Emoji string: green circle, yellow circle, or red circle.
    """
    raise NotImplementedError


def _cap_history(
    history: list[dict[str, Any]], max_points: int = 500
) -> list[dict[str, Any]]:
    """
    Trim a history list to the most recent max_points entries.

    Args:
        history:    List of time-ordered snapshot dicts.
        max_points: Maximum number of entries to retain.

    Returns:
        Trimmed list.
    """
    raise NotImplementedError
