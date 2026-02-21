"""
ReAct agent — the core LLM reasoning loop for root cause analysis.

Implements the ReAct (Reasoning + Acting) pattern:

    Observe → Think → Act → Observe → Think → Act → ... → Final Answer

Each iteration ("step") of the loop:
  1. OBSERVE  : the agent receives an anomaly context or the result of its
                last action as an Observation.
  2. THINK    : the LLM emits a Thought explaining its reasoning.
  3. ACT      : the LLM emits a structured JSON action block.
                ActionPlanner validates it; the executor runs it.
  4. OBSERVE  : post-action metrics and logs are fed back as the next Observation.

The loop terminates when:
  - The anomaly score drops below config.agent.confidence_threshold (resolved), or
  - A "no_action" is returned (agent judges no intervention needed), or
  - max_reasoning_steps (default: 5) is reached (escalation triggered).

After the loop, the agent emits a structured RCA report (JSON) summarising:
  root_cause, affected_services, actions_taken, confidence, resolution_status.

LLM provider support:
  - OpenAI   (config.yaml → agent.llm_provider: "openai")
  - Anthropic (config.yaml → agent.llm_provider: "anthropic")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agent.action_planner import Action, ActionPlanner
from src.detection.feature_extractor import FeatureVector


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """A single Thought / Action / Observation triplet from one ReAct iteration."""

    step_number: int
    thought: str
    action: Action | None
    observation: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentContext:
    """
    All context passed to the agent when an anomaly is detected.

    Assembled by main.py from the detection layer outputs before invoking
    the agent.
    """

    service: str                          # Primary anomalous service
    anomaly_score: float                  # Ensemble score (0–1)
    triggered_metrics: list[str]          # Metric names that exceeded thresholds
    feature_vector: FeatureVector         # Full feature vector for the window
    recent_logs: list[dict[str, Any]]     # Last N log entries as dicts
    metric_snapshot: dict[str, Any]       # Latest MetricSnapshot as dict
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentResult:
    """
    The full output of a completed ReAct agent run.

    Attributes:
        context:          The AgentContext that triggered this run.
        reasoning_trace:  Ordered list of ReasoningStep objects (the full chain).
        actions_taken:    Flattened list of all Action objects that were executed.
        rca_report:       Structured root cause analysis dict (from final LLM call).
        resolved:         True if the anomaly score dropped below threshold.
        escalated:        True if max steps were reached without resolution.
        completed_at:     Timestamp when the run finished.
    """

    context: AgentContext
    reasoning_trace: list[ReasoningStep]
    actions_taken: list[Action]
    rca_report: dict[str, Any]
    resolved: bool
    escalated: bool
    completed_at: datetime = field(default_factory=datetime.utcnow)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReActAgent:
    """
    LLM-powered ReAct agent for AIOps root cause analysis and remediation.

    Orchestrates the full Observe → Think → Act loop, calling the LLM,
    parsing its output with ActionPlanner, dispatching actions to the executor,
    and feeding results back as observations.
    """

    def __init__(self, agent_config: dict[str, Any]) -> None:
        """
        Initialise the agent from the [agent] section of config.yaml.

        Reads llm_provider, model, max_reasoning_steps, and
        confidence_threshold. Initialises the LLM client and ActionPlanner.

        Args:
            agent_config: The [agent] section of config.yaml.
        """
        raise NotImplementedError

    def run(
        self,
        context: AgentContext,
        executor: Any | None = None,
    ) -> AgentResult:
        """
        Execute the full ReAct loop for a detected anomaly.

        Args:
            context:  AgentContext assembled from the detection layer.
            executor: Optional Executor instance to dispatch actions against.
                      If None, actions are logged but not executed (dry-run).

        Returns:
            AgentResult with the full reasoning trace, actions taken, and RCA report.
        """
        raise NotImplementedError

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """
        Send a messages list to the configured LLM and return the response text.

        Handles both OpenAI and Anthropic API formats based on llm_provider config.

        Args:
            messages: List of {"role": ..., "content": ...} message dicts.

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If the LLM call fails after retries.
        """
        raise NotImplementedError

    def _build_initial_messages(self, context: AgentContext) -> list[dict[str, str]]:
        """
        Construct the initial message list for the first LLM call.

        Includes the system prompt and the first Observation turn.

        Args:
            context: AgentContext for the current anomaly.

        Returns:
            List of message dicts for the LLM.
        """
        raise NotImplementedError

    def _append_step(
        self,
        messages: list[dict[str, str]],
        step: ReasoningStep,
    ) -> list[dict[str, str]]:
        """
        Append a completed ReasoningStep (Thought + Action + Observation) to
        the message history for the next LLM call.

        Args:
            messages: Current message list.
            step:     Completed reasoning step.

        Returns:
            Updated message list.
        """
        raise NotImplementedError

    def _generate_rca_report(
        self,
        context: AgentContext,
        trace: list[ReasoningStep],
    ) -> dict[str, Any]:
        """
        Make a final LLM call to produce a structured JSON RCA report.

        Args:
            context: Original AgentContext.
            trace:   Full list of reasoning steps from this run.

        Returns:
            Dict with keys: root_cause, affected_services, actions_taken,
                            confidence, resolution_status, summary.
        """
        raise NotImplementedError

    def _check_resolved(self, service: str) -> bool:
        """
        Query current anomaly score for a service to check if it resolved.

        Args:
            service: Service name to check.

        Returns:
            True if the current ensemble score is below the confidence threshold.
        """
        raise NotImplementedError

    def _init_llm_client(self) -> Any:
        """
        Initialise and return the LLM client based on agent_config.llm_provider.

        Returns:
            openai.OpenAI or anthropic.Anthropic client instance.

        Raises:
            ValueError: If llm_provider is not 'openai' or 'anthropic'.
            RuntimeError: If the required API key is not set in environment.
        """
        raise NotImplementedError
