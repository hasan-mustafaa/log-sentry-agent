"""
Remediation executor for the simulated FCT microservice mesh.

Translates validated Action objects (from ActionPlanner) into concrete
operations against the simulated services. Since LogSentry operates on a
simulated environment (no real Kubernetes / Docker), all actions manipulate
the in-memory state of the simulator components:

  - restart_service:    Clears active faults, resets metrics to baseline,
                        emits a "service restarted" log entry, and introduces
                        a brief unavailability window (configurable).
  - scale_service:      Adjusts the replica count in SimulatedService state,
                        which the MetricsGenerator uses to redistribute load.
  - rollback_service:   Reverts to the previous service "version" (simulated
                        by restoring a snapshot of its baseline metrics).
  - alert_on_call:      Logs the alert to the dashboard and Rich console with
                        severity formatting; does not call a real paging system.
  - no_action:          No-op; records the decision in the execution log.

Every execution returns an ExecutionResult that the ReActAgent uses as its
next Observation, closing the feedback loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agent.action_planner import (
    Action,
    AlertAction,
    NoAction,
    RestartAction,
    RollbackAction,
    ScaleAction,
)
from src.remediation.guardrails import Guardrails


@dataclass
class ExecutionResult:
    """
    Result of a single remediation action execution.

    Attributes:
        action:          The Action that was executed.
        success:         True if the action completed without error.
        blocked_by_guardrail: True if Guardrails prevented execution.
        message:         Human-readable outcome description.
        post_metrics:    MetricSnapshot dict taken after the action (for feedback).
        post_logs:       Recent log entries after the action (for feedback).
        executed_at:     Timestamp of execution.
        duration_ms:     Wall-clock time taken to execute (simulated delay).
    """

    action: Action
    success: bool
    blocked_by_guardrail: bool
    message: str
    post_metrics: dict[str, Any] = field(default_factory=dict)
    post_logs: list[dict[str, Any]] = field(default_factory=list)
    executed_at: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary for LLM observation injection."""
        raise NotImplementedError


class Executor:
    """
    Executes remediation actions against the simulated service mesh.

    Checks each action against Guardrails before execution and records all
    attempts (successful or blocked) in an ordered execution log.
    """

    def __init__(
        self,
        remediation_config: dict[str, Any],
        guardrails: Guardrails,
        simulator_state: Any | None = None,
    ) -> None:
        """
        Initialise the executor.

        Args:
            remediation_config: The [remediation] section of config.yaml.
            guardrails:         Guardrails instance to validate actions against.
            simulator_state:    Reference to the running simulator (LogGenerator +
                                MetricsGenerator + FaultInjector) so the executor
                                can manipulate service state. May be None in tests.
        """
        raise NotImplementedError

    def execute(self, action: Action) -> ExecutionResult:
        """
        Validate and execute a single remediation action.

        Checks the action against Guardrails first. If allowed, dispatches
        to the appropriate private handler method. Records the result.

        Args:
            action: A validated Action object from ActionPlanner.

        Returns:
            ExecutionResult describing the outcome.
        """
        raise NotImplementedError

    def execution_log(self) -> list[ExecutionResult]:
        """Return the full ordered list of all execution attempts."""
        raise NotImplementedError

    def _execute_restart(self, action: RestartAction) -> ExecutionResult:
        """
        Simulate a service restart.

        Clears active faults on the target, resets its metrics baseline,
        emits a restart log entry, and records a brief unavailability window.

        Args:
            action: Validated RestartAction.

        Returns:
            ExecutionResult with post-restart metrics and logs.
        """
        raise NotImplementedError

    def _execute_scale(self, action: ScaleAction) -> ExecutionResult:
        """
        Simulate scaling a service to a new replica count.

        Updates the SimulatedService replica count and adjusts the metrics
        baseline to reflect the new capacity (lower CPU/latency per replica).

        Args:
            action: Validated ScaleAction.

        Returns:
            ExecutionResult with post-scale metrics and logs.
        """
        raise NotImplementedError

    def _execute_rollback(self, action: RollbackAction) -> ExecutionResult:
        """
        Simulate rolling back a service to its previous deployment.

        Restores the service's metrics baseline to a stored pre-fault snapshot.

        Args:
            action: Validated RollbackAction.

        Returns:
            ExecutionResult with post-rollback metrics and logs.
        """
        raise NotImplementedError

    def _execute_alert(self, action: AlertAction) -> ExecutionResult:
        """
        Emit an on-call alert to the dashboard and terminal.

        Args:
            action: Validated AlertAction.

        Returns:
            ExecutionResult confirming the alert was dispatched.
        """
        raise NotImplementedError

    def _execute_no_action(self, action: NoAction) -> ExecutionResult:
        """
        Record a no-action decision without modifying any state.

        Args:
            action: Validated NoAction.

        Returns:
            ExecutionResult confirming the decision was recorded.
        """
        raise NotImplementedError

    def _collect_post_action_state(self, service: str) -> dict[str, Any]:
        """
        Gather current metrics and recent logs for a service post-execution.

        Used to build the Observation fed back to the ReAct agent.

        Args:
            service: Service name to gather state for.

        Returns:
            Dict with 'metrics' and 'logs' keys.
        """
        raise NotImplementedError
