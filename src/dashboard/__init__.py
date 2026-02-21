"""
Dashboard package.

Provides a live Streamlit web dashboard that visualises the LogSentry Agent
pipeline in real time:

  - Metrics charts  : rolling time-series plots of CPU, memory, latency,
                      and error rate per service (Plotly).
  - Log stream      : scrolling display of the most recent N log entries,
                      colour-coded by level.
  - Anomaly alerts  : highlighted cards for each active anomaly with its
                      ensemble score and triggered metrics.
  - Agent trace     : expandable Thought / Action / Observation steps from
                      the most recent ReAct run, with the final RCA report.
  - Remediation log : table of all executed actions with timestamps, outcomes,
                      and guardrail decisions.

Run with:
    streamlit run src/dashboard/app.py
"""

from src.dashboard.app import run_dashboard

__all__ = ["run_dashboard"]
