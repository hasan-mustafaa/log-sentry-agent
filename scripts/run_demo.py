"""
End-to-end demo script for LogSentry Agent.

Runs the full pipeline — simulator → detection → ReAct agent → remediation —
for a fixed duration and prints a rich-formatted summary to the terminal.

This script is intentionally self-contained and does NOT start the Streamlit
dashboard. It is designed for:
  - Quick functional verification after installation
  - CI/CD smoke tests (with --dry-run and --no-llm flags)
  - Live demos in a terminal session

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --duration 120 --fault latency_spike --service title-search-service
    python scripts/run_demo.py --dry-run          # actions logged, not executed
    python scripts/run_demo.py --no-llm           # uses rule-based fallback instead of LLM

Arguments:
    --duration   INT     Total demo run time in seconds (default: 60)
    --fault      STR     Fault type to inject: crash | latency_spike |
                         connection_failure | memory_leak | oom (default: latency_spike)
    --service    STR     Service to inject the fault into (default: title-search-service)
    --dry-run            Log remediation actions without executing them
    --no-llm             Skip LLM calls; use deterministic rule-based fallback agent
    --config     PATH    Path to config.yaml (default: config/config.yaml)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the demo script.

    Returns:
        Parsed argparse.Namespace with demo configuration.
    """
    raise NotImplementedError


def print_banner() -> None:
    """
    Print the LogSentry Agent ASCII banner to the terminal using Rich.

    Includes the project name, version, and a one-line description.
    """
    raise NotImplementedError


def print_pipeline_summary(results: dict[str, Any]) -> None:
    """
    Print a formatted summary table of the demo run to the terminal.

    Displays:
      - Fault injected (service + type + duration)
      - Anomaly detected (time to detection, ensemble score)
      - Agent reasoning (number of steps, actions taken)
      - Resolution status (resolved / escalated / unresolved)
      - RCA report excerpt

    Args:
        results: Dict assembled at the end of the demo run with all key metrics.
    """
    raise NotImplementedError


def run_demo(
    duration: int = 60,
    fault_type: str = "latency_spike",
    target_service: str = "title-search-service",
    dry_run: bool = False,
    no_llm: bool = False,
    config_path: Path = Path("config/config.yaml"),
) -> dict[str, Any]:
    """
    Execute the end-to-end LogSentry demo pipeline.

    Steps:
      1. Load config and initialise all pipeline components.
      2. Run the simulator for a warm-up period to collect baseline data.
      3. Train the Isolation Forest model on the warm-up feature vectors.
      4. Inject the specified fault into the target service.
      5. Run the detection loop until an anomaly is flagged (or timeout).
      6. Invoke the ReAct agent with the anomaly context.
      7. Execute the agent's recommended actions (subject to guardrails).
      8. Continue monitoring for post-remediation recovery signal.
      9. Return a results dict for print_pipeline_summary().

    Args:
        duration:        Total demo duration in seconds.
        fault_type:      Fault scenario to inject.
        target_service:  Service to inject the fault into.
        dry_run:         If True, actions are logged but not applied.
        no_llm:          If True, skip LLM calls (useful for offline testing).
        config_path:     Path to config.yaml.

    Returns:
        Dict with keys: fault_injected, detected_at, time_to_detection_s,
                        anomaly_score, agent_steps, actions_taken,
                        resolution_status, rca_summary.
    """
    raise NotImplementedError


def main() -> None:
    """
    Entry point: parse args, print banner, run demo, print summary.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
