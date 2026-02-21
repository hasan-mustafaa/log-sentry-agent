"""
Entry point for the LogSentry Agent pipeline.

Orchestrates the full system lifecycle:
  1. Load configuration from config/config.yaml and environment variables.
  2. Start the Simulator (log generator + metrics generator + fault injector).
  3. Start the Detection pipeline (log parser → feature extractor → ensemble detector).
  4. On anomaly detection, invoke the ReAct Agent for root cause analysis.
  5. Pass the agent's action plan to the Remediation executor (with guardrail checks).
  6. Optionally launch the Streamlit dashboard as a subprocess.

Run with:
    python -m src.main
    python -m src.main --no-dashboard
    python -m src.main --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the main entry point."""
    raise NotImplementedError


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """
    Load and validate configuration from YAML file and .env overrides.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Merged configuration dictionary.
    """
    raise NotImplementedError


def build_pipeline(config: dict, dry_run: bool = False) -> "Pipeline":
    """
    Instantiate and wire all pipeline components from config.

    Args:
        config:  Loaded configuration dictionary.
        dry_run: If True, remediation executor logs actions without applying them.

    Returns:
        A fully wired Pipeline object ready to run.
    """
    raise NotImplementedError


def main(config_path: Optional[Path] = None) -> None:
    """
    Main entry point. Parses args, loads config, builds and runs the pipeline.

    Args:
        config_path: Override config path (used in tests / programmatic invocation).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
