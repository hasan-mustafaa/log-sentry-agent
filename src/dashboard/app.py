"""Streamlit dashboard for the LogSentry Agent.

Renders four panels that auto-refresh every N seconds:
  1. Service Health  — colour-coded status card per service.
  2. Metrics Charts  — Plotly rolling time-series, one tab per service.
  3. Log Stream      — scrolling colour-coded log table.
  4. Agent & Remediation — ReAct trace, RCA report, action history.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────

_REFRESH_INTERVAL = 2  # seconds; updated from config at startup if available
_MAX_CHART_POINTS = 500
_STATE_FILE = Path("data/dashboard_state.json")
_METRIC_COLS = ["cpu_percent", "memory_mb", "latency_ms", "error_rate", "request_rate"]
_METRIC_LABELS: dict[str, str] = {
    "cpu_percent":  "CPU (%)",
    "memory_mb":    "Memory (MB)",
    "latency_ms":   "Latency (ms)",
    "error_rate":   "Error Rate (err/s)",
    "request_rate": "Request Rate (req/s)",
}
_LEVEL_BG: dict[str, str] = {
    "CRITICAL": "#ff4444",
    "ERROR":    "#ffcccc",
    "WARNING":  "#fff3cd",
}


# ── Shared state ──────────────────────────────────────────────────────────────

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


# ── Public render functions ───────────────────────────────────────────────────

def _load_state_from_file() -> DashboardState:
    """Read DashboardState from the JSON file written by the pipeline process."""
    if not _STATE_FILE.exists():
        return DashboardState()
    try:
        with open(_STATE_FILE) as f:
            data = json.load(f)
        state = DashboardState(
            metric_history=data.get("metric_history", {}),
            log_buffer=data.get("log_buffer", []),
            anomaly_events=data.get("anomaly_events", []),
            agent_results=data.get("agent_results", []),
            remediation_log=data.get("remediation_log", []),
            service_status=data.get("service_status", {}),
        )
        if data.get("pipeline_started_at"):
            state.pipeline_started_at = datetime.fromisoformat(data["pipeline_started_at"])
        return state
    except Exception:
        return DashboardState()


def run_dashboard(state: DashboardState | None = None) -> None:
    """
    Entry point for the Streamlit dashboard.

    When called from the pipeline (src/main.py), a pre-populated DashboardState
    is passed in. When called directly via `streamlit run`, a new empty state is
    created and the dashboard shows a "Waiting for pipeline..." message.

    Args:
        state: Shared DashboardState from the running pipeline, or None.
    """
    st.set_page_config(
        page_title="LogSentry Agent",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # In standalone mode (state=None), reload from the file on every render
    # so the dashboard reflects the latest data written by the pipeline process.
    # When called directly from the pipeline, use the passed state object.
    if state is not None:
        st.session_state.dashboard_state = state
    else:
        st.session_state.dashboard_state = _load_state_from_file()

    _state: DashboardState = st.session_state.dashboard_state

    render_header()

    if not _state.metric_history:
        st.info("Waiting for pipeline data. Start the pipeline with: `python -m src.main`")
        time.sleep(_REFRESH_INTERVAL)
        st.rerun()
        return

    render_service_health(_state)
    st.divider()
    render_metrics_charts(_state)
    st.divider()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_log_stream(_state)
    with col_right:
        render_anomaly_alerts(_state)

    st.divider()
    render_agent_trace(_state)
    st.divider()
    render_remediation_log(_state)

    time.sleep(_REFRESH_INTERVAL)
    st.rerun()


def render_header() -> None:
    """
    Render the dashboard title, subtitle, and pipeline status badge.

    Shows whether the pipeline is running, paused, or waiting to start.
    """
    st.title("LogSentry Agent Dashboard")
    st.caption("AIOps pipeline — real-time anomaly detection and remediation")

    _state: DashboardState | None = st.session_state.get("dashboard_state")
    if _state is None or _state.pipeline_started_at is None:
        st.markdown(":orange[**Pipeline: Waiting**]")
    else:
        uptime = datetime.now(timezone.utc) - _state.pipeline_started_at
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)
        st.markdown(
            f":green[**Pipeline: Running**] — uptime: "
            f"{hours:02d}:{mins:02d}:{secs:02d}"
        )


def render_service_health(state: DashboardState) -> None:
    """
    Render the service health overview row.

    Displays one status card per FCT service using Streamlit columns.
    Each card shows: service name, status badge, and current key metrics.

    Args:
        state: Current DashboardState.
    """
    st.subheader("Service Health")
    services = list(state.service_status.keys()) or list(state.metric_history.keys())
    if not services:
        st.info("No service data yet.")
        return

    cols = st.columns(len(services))
    for col, service in zip(cols, services):
        with col:
            status = state.service_status.get(service, "healthy")
            badge = _status_badge(status)
            history = state.metric_history.get(service, [])
            latest = history[-1] if history else {}

            with st.container(border=True):
                st.markdown(f"**{badge} {service}**")
                st.caption(status.upper())
                if latest:
                    st.metric("CPU", f"{latest.get('cpu_percent', 0):.1f}%")
                    st.metric("Memory", f"{latest.get('memory_mb', 0):.0f} MB")
                    st.metric("Latency", f"{latest.get('latency_ms', 0):.1f} ms")
                    st.metric("Errors/s", f"{latest.get('error_rate', 0):.3f}")
                else:
                    st.caption("No snapshots yet")


def render_metrics_charts(state: DashboardState) -> None:
    """
    Render rolling time-series charts for each service using Plotly.

    Creates a tab per service, each containing subplots for:
    CPU %, memory MB, latency ms, error rate, and anomaly score.

    Args:
        state: Current DashboardState containing metric_history.
    """
    st.subheader("Metrics")
    services = list(state.metric_history.keys())
    if not services:
        st.info("No metric history yet.")
        return

    tabs = st.tabs(services)
    for tab, service in zip(tabs, services):
        with tab:
            history = _cap_history(state.metric_history.get(service, []))
            if not history:
                st.info(f"No data for {service}.")
                continue

            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            n_rows = len(_METRIC_COLS)
            fig = make_subplots(
                rows=n_rows,
                cols=1,
                shared_xaxes=True,
                subplot_titles=list(_METRIC_LABELS.values()),
                vertical_spacing=0.06,
            )

            for row, (col_name, label) in enumerate(_METRIC_LABELS.items(), start=1):
                if col_name in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df[col_name],
                            name=label,
                            mode="lines",
                            line=dict(width=1.5),
                        ),
                        row=row,
                        col=1,
                    )

            # Overlay anomaly markers on every subplot row.
            service_anomalies = [
                a for a in state.anomaly_events if a.get("service") == service
            ]
            if service_anomalies:
                anomaly_times = [a.get("detected_at") for a in service_anomalies]
                anomaly_scores = [a.get("anomaly_score", 0.0) for a in service_anomalies]
                for row in range(1, n_rows + 1):
                    # Use anomaly_score as y-value only on the last row for the overlay.
                    y_vals = anomaly_scores if row == n_rows else [None] * len(anomaly_times)
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_times,
                            y=y_vals,
                            mode="markers",
                            marker=dict(color="red", size=8, symbol="x"),
                            name="Anomaly",
                            showlegend=(row == 1),
                        ),
                        row=row,
                        col=1,
                    )

            fig.update_layout(height=700, showlegend=True, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)


def render_log_stream(state: DashboardState) -> None:
    """
    Render the live scrolling log stream table.

    Displays the most recent log entries from state.log_buffer in a
    colour-coded dataframe. ERROR/CRITICAL rows are highlighted in red,
    WARNING in yellow, INFO/DEBUG in default styling.

    Args:
        state: Current DashboardState containing log_buffer.
    """
    st.subheader("Log Stream")
    if not state.log_buffer:
        st.info("No log entries yet.")
        return

    display_cols = [
        c for c in ["timestamp", "service", "level", "message", "template"]
        if any(c in entry for entry in state.log_buffer[:1])
    ]
    df = pd.DataFrame(state.log_buffer)
    if display_cols:
        existing = [c for c in display_cols if c in df.columns]
        df = df[existing]

    def _color_rows(row: pd.Series) -> list[str]:
        level = row.get("level", "INFO") if "level" in row.index else "INFO"
        bg = _LEVEL_BG.get(str(level), "")
        style = f"background-color: {bg}" if bg else ""
        return [style] * len(row)

    styled = df.style.apply(_color_rows, axis=1)
    st.dataframe(styled, use_container_width=True, height=300)


def render_anomaly_alerts(state: DashboardState) -> None:
    """
    Render active anomaly alert cards.

    Each card shows: service, ensemble anomaly score, triggered metrics,
    detection timestamp, and a link to the corresponding agent trace.

    Args:
        state: Current DashboardState containing anomaly_events.
    """
    st.subheader("Anomaly Alerts")
    if not state.anomaly_events:
        st.success("No anomalies detected.")
        return

    # Show the most recent 10 anomalies, newest first.
    recent = state.anomaly_events[-10:]
    for event in reversed(recent):
        service = event.get("service", "unknown")
        score = event.get("anomaly_score", 0.0)
        triggered = event.get("triggered_metrics", [])
        detected_at = event.get("detected_at", "")

        severity = "error" if score >= 0.8 else "warning"
        fn = st.error if severity == "error" else st.warning
        fn(
            f"**{service}** — Score: `{score:.3f}`  \n"
            f"Triggered: `{', '.join(triggered)}`  \n"
            f"Detected: `{detected_at}`"
        )


def render_agent_trace(state: DashboardState) -> None:
    """
    Render the ReAct agent reasoning trace for the most recent run.

    Uses Streamlit expanders to show each Thought → Action → Observation
    step, followed by the final structured RCA report as a formatted JSON block.

    Args:
        state: Current DashboardState containing agent_results.
    """
    st.subheader("Agent Reasoning & Remediation")
    if not state.agent_results:
        st.info("No agent runs yet.")
        return

    result = state.agent_results[-1]
    context = result.get("context", {})
    trace = result.get("reasoning_trace", [])
    rca = result.get("rca_report", {})

    resolved = result.get("resolved", False)
    escalated = result.get("escalated", False)
    status_str = ":green[Resolved]" if resolved else (":red[Escalated]" if escalated else ":orange[Unresolved]")

    st.markdown(
        f"**Service:** `{context.get('service', '?')}` | "
        f"**Score:** `{context.get('anomaly_score', 0.0):.3f}` | "
        f"**Status:** {status_str}"
    )

    if trace:
        st.markdown("**Reasoning Trace:**")
        for step in trace:
            step_num = step.get("step_number", "?")
            with st.expander(f"Step {step_num}", expanded=False):
                thought = step.get("thought", "")
                if thought:
                    st.markdown(f"**Thought:** {thought}")

                action = step.get("action")
                if action:
                    st.markdown("**Action:**")
                    st.json(action)

                observation = step.get("observation", "")
                if observation:
                    st.markdown(f"**Observation:** {observation}")

    if rca:
        st.markdown("**RCA Report:**")
        st.json(rca)


def render_remediation_log(state: DashboardState) -> None:
    """
    Render the remediation action history as a sortable table.

    Columns: timestamp, service, action_type, outcome, guardrail_blocked, reason.

    Args:
        state: Current DashboardState containing remediation_log.
    """
    st.subheader("Remediation Log")
    if not state.remediation_log:
        st.info("No remediation actions yet.")
        return

    rows = []
    for entry in state.remediation_log:
        action = entry.get("action", {})
        rows.append({
            "timestamp":        entry.get("executed_at", ""),
            "service":          action.get("target", ""),
            "action_type":      action.get("action", ""),
            "outcome":          "SUCCESS" if entry.get("success") else "FAILED",
            "guardrail_blocked": entry.get("blocked_by_guardrail", False),
            "reason":           entry.get("message", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    """
    Return a coloured emoji badge for a service status string.

    Args:
        status: One of 'healthy', 'degraded', 'down'.

    Returns:
        Emoji string: green circle, yellow circle, or red circle.
    """
    return {"healthy": "🟢", "degraded": "🟡", "down": "🔴"}.get(status, "⚪")


def _cap_history(
    history: list[dict[str, Any]], max_points: int = _MAX_CHART_POINTS
) -> list[dict[str, Any]]:
    """
    Trim a history list to the most recent max_points entries.

    Args:
        history:    List of time-ordered snapshot dicts.
        max_points: Maximum number of entries to retain.

    Returns:
        Trimmed list.
    """
    return history[-max_points:] if len(history) > max_points else history


# ── Streamlit entry point ─────────────────────────────────────────────────────
# Executed on every Streamlit render cycle when run via `streamlit run`.

run_dashboard()
