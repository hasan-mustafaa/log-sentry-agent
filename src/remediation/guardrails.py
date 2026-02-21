"""
Safety guardrails for automated remediation actions.

Prevents the ReAct agent from taking actions that could make a situation
worse, create restart loops, or overwhelm already-degraded services.

Guardrail rules (configurable via config.yaml → remediation):

  1. Max restarts per service (default: 3):
       If a service has been restarted max_restarts_per_service times within
       the current incident window, further restart actions are blocked and
       an escalation is triggered instead.

  2. Restart cooldown (default: 300 s):
       A minimum interval must elapse between consecutive restarts of the same
       service. Prevents rapid restart loops that could worsen cascade failures.

  3. Auto-escalate after N consecutive failures (default: 2):
       If the same remediation action has been attempted N times without the
       anomaly score improving, block further attempts and raise an alert.

  4. Action schema validation:
       Every action must pass Pydantic validation (handled by ActionPlanner)
       before reaching the guardrail layer. The guardrail layer performs
       additional semantic checks (e.g., cannot scale a crashed service).

  5. Dry-run mode:
       When dry_run=True, all actions are allowed but not executed; the
       guardrail simply logs what would have been blocked.

All decisions are recorded in a GuardrailLog for audit and dashboard display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agent.action_planner import Action, RestartAction


@dataclass
class GuardrailDecision:
    """
    The outcome of a guardrail check for a single action.

    Attributes:
        action:    The action that was checked.
        allowed:   True if the action is permitted to proceed.
        reason:    Human-readable explanation (especially useful when blocked).
        checked_at: Timestamp of the check.
    """

    action: Action
    allowed: bool
    reason: str
    checked_at: datetime = field(default_factory=datetime.utcnow)


class GuardrailViolation(Exception):
    """Raised when an action is blocked by a guardrail rule."""


class Guardrails:
    """
    Enforces safety constraints on remediation actions before execution.

    The Executor calls `check()` before executing any action. If the check
    returns a GuardrailDecision with allowed=False, the Executor creates a
    blocked ExecutionResult and does not call the action handler.
    """

    def __init__(
        self,
        remediation_config: dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        """
        Initialise the guardrails from the [remediation] section of config.yaml.

        Args:
            remediation_config: Must contain max_restarts_per_service,
                                restart_cooldown_seconds, and
                                auto_escalate_after_failures.
            dry_run:            If True, all checks pass but decisions are logged.
        """
        raise NotImplementedError

    def check(self, action: Action) -> GuardrailDecision:
        """
        Evaluate all applicable guardrail rules for the given action.

        Dispatches to the appropriate rule method(s) based on action type,
        then returns a combined decision.

        Args:
            action: The Action to evaluate (RestartAction, ScaleAction, etc.).

        Returns:
            GuardrailDecision with allowed flag and human-readable reason.
        """
        raise NotImplementedError

    def record_execution(self, action: Action, success: bool) -> None:
        """
        Update internal state after an action is executed.

        Called by the Executor after execution so guardrails can track
        cumulative restart counts, last restart timestamps, and failure streaks.

        Args:
            action:  The action that was executed.
            success: Whether the action achieved the desired effect.
        """
        raise NotImplementedError

    def reset_service(self, service: str) -> None:
        """
        Reset all guardrail counters and timestamps for a specific service.

        Called after a successful incident resolution so the service starts
        fresh in the next incident window.

        Args:
            service: Service name to reset.
        """
        raise NotImplementedError

    def reset_all(self) -> None:
        """Reset guardrail state for all services."""
        raise NotImplementedError

    def decision_log(self) -> list[GuardrailDecision]:
        """Return the full ordered log of all guardrail decisions."""
        raise NotImplementedError

    def _check_restart_limit(self, action: RestartAction) -> GuardrailDecision:
        """
        Enforce max_restarts_per_service for a RestartAction.

        Args:
            action: The RestartAction to check.

        Returns:
            GuardrailDecision allowing or blocking the restart.
        """
        raise NotImplementedError

    def _check_restart_cooldown(self, action: RestartAction) -> GuardrailDecision:
        """
        Enforce restart_cooldown_seconds between consecutive restarts.

        Args:
            action: The RestartAction to check.

        Returns:
            GuardrailDecision allowing or blocking the restart.
        """
        raise NotImplementedError

    def _check_failure_streak(self, action: Action) -> GuardrailDecision:
        """
        Block actions on a service that has failed auto_escalate_after_failures
        times in a row without improvement.

        Args:
            action: The action to check.

        Returns:
            GuardrailDecision allowing or blocking the action.
        """
        raise NotImplementedError
