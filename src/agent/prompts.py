"""
LLM prompt templates for the LogSentry ReAct agent.

All prompts are assembled here as functions (not bare strings) so that
dynamic context — service topology, anomaly details, recent logs — can be
cleanly injected at runtime without string fragmentation throughout the codebase.

Prompt structure overview:
  - SYSTEM_PROMPT       : Defines the agent's role, capabilities, and the
                          FCT service topology. Injected once per agent session.
  - build_observe_prompt: Constructs the initial "Observation" turn describing
                          the detected anomaly (service, metrics, log snippets).
  - build_think_prompt  : Wraps the LLM's Thought step instructions — asks it to
                          reason over the dependency graph before acting.
  - build_action_prompt : Instructs the LLM to emit a JSON action block from
                          the allowed action schema.
  - build_rca_report_prompt: Asks the LLM to produce the final structured RCA
                              report after remediation.

Supported LLM providers:
  - OpenAI  (openai.ChatCompletion format)
  - Anthropic (anthropic.messages format)

The active provider is controlled by config.yaml → agent.llm_provider.
"""

from __future__ import annotations

from typing import Any

# ── Service topology injected into the system prompt ──────────────────────────
# Kept here so the agent knows which services depend on which.
FCT_SERVICE_TOPOLOGY: str = """
Services and their upstream dependencies:
  transaction-validator (port 8001)
    └── fraud-check-service (port 8002)
          └── title-search-service (port 8004)
    └── document-processor (port 8003)
          └── title-search-service (port 8004)
  title-search-service (port 8004) — leaf service, no dependencies
"""

# ── Available remediation actions (injected into prompts) ────────────────────
AVAILABLE_ACTIONS: str = """
Available actions (respond with exactly one JSON block per action):
  {"action": "restart_service",  "target": "<service-name>", "reason": "<str>"}
  {"action": "scale_service",    "target": "<service-name>", "replicas": <int>, "reason": "<str>"}
  {"action": "rollback_service", "target": "<service-name>", "reason": "<str>"}
  {"action": "alert_on_call",    "target": "<service-name>", "severity": "P1|P2|P3", "message": "<str>"}
  {"action": "no_action",        "reason": "<str>"}
"""


def build_system_prompt() -> str:
    """
    Build the static system prompt that establishes the agent's role and context.

    This prompt is sent as the 'system' role message at the start of every
    ReAct session. It should NOT change between reasoning steps.

    Returns:
        System prompt string.
    """
    raise NotImplementedError


def build_observe_prompt(
    service: str,
    anomaly_score: float,
    triggered_metrics: list[str],
    recent_logs: list[dict[str, Any]],
    metric_snapshot: dict[str, Any],
) -> str:
    """
    Build the initial Observation prompt describing the detected anomaly.

    Args:
        service:           Name of the anomalous service.
        anomaly_score:     Ensemble anomaly score (0–1).
        triggered_metrics: List of metric names that exceeded thresholds.
        recent_logs:       Last N log entries for the service (as dicts).
        metric_snapshot:   Latest MetricSnapshot as a dict.

    Returns:
        Formatted observation string to inject as a user/human turn.
    """
    raise NotImplementedError


def build_think_prompt(step: int, max_steps: int) -> str:
    """
    Build the instruction prompt that asks the LLM to emit a Thought.

    Args:
        step:      Current reasoning step number (1-indexed).
        max_steps: Maximum allowed steps (from config).

    Returns:
        Formatted prompt string requesting a Thought from the LLM.
    """
    raise NotImplementedError


def build_action_prompt() -> str:
    """
    Build the prompt instructing the LLM to emit a structured JSON action.

    Returns:
        Formatted prompt string requesting a JSON action block.
    """
    raise NotImplementedError


def build_observation_from_action(
    action: dict[str, Any],
    execution_result: dict[str, Any],
) -> str:
    """
    Build the Observation turn that follows an executed action.

    Injects the result of the executor (success/failure, new metric readings,
    post-action log snippet) so the LLM can ground its next Thought in real data.

    Args:
        action:           The action dict that was executed.
        execution_result: Result dict returned by the executor.

    Returns:
        Formatted observation string.
    """
    raise NotImplementedError


def build_rca_report_prompt(
    service: str,
    reasoning_trace: list[dict[str, Any]],
) -> str:
    """
    Build the final prompt requesting a structured RCA (Root Cause Analysis) report.

    The report is returned as JSON and stored in the incident record and dashboard.

    Args:
        service:         The primary affected service.
        reasoning_trace: Full list of Thought/Action/Observation dicts from the run.

    Returns:
        Formatted prompt string requesting a JSON RCA report.
    """
    raise NotImplementedError


def format_messages_openai(
    system: str,
    turns: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Format a system prompt + conversation turns into OpenAI ChatCompletion format.

    Args:
        system: System prompt string.
        turns:  List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        List of message dicts ready for openai.chat.completions.create().
    """
    raise NotImplementedError


def format_messages_anthropic(
    system: str,
    turns: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """
    Format a system prompt + conversation turns into Anthropic Messages format.

    Args:
        system: System prompt string.
        turns:  List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        Tuple of (system_string, messages_list) for anthropic.messages.create().
    """
    raise NotImplementedError
