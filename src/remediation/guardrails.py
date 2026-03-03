"""Safety guardrails for the remediation layer.

Enforces per-service limits on automated actions to prevent the agent from
making things worse (restart loops, thrashing, runaway scaling):

  - Max restarts per service within a cooldown window
  - Cooldown period between consecutive restarts
  - Auto-escalation after a configurable number of consecutive failures

check() is called before every execution. record_execution() is called after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.agent.action_planner import Action, RestartAction


@dataclass
class GuardrailViolation:
    """Records a single blocked action with the reason it was denied."""

    action: str
    reason: str
    blocked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Guardrails:
    """Enforces safety limits on automated remediation actions.

    All state is per-service and resets when reset() is called. Intended to be
    shared between the Executor and main pipeline so escalation thresholds are
    evaluated across the full session.
    """

    def __init__(self, remediation_config: dict[str, Any]) -> None:
        """Initialize from the [remediation] section of config.yaml."""
        self._max_restarts: int = remediation_config.get("max_restarts_per_service", 3)
        self._cooldown: float = remediation_config.get("restart_cooldown_seconds", 300)
        self._escalate_after: int = remediation_config.get("auto_escalate_after_failures", 2)

        self._restart_counts: dict[str, int] = {}
        self._last_restart: dict[str, datetime] = {}
        self._failure_counts: dict[str, int] = {}
        self._violations: list[GuardrailViolation] = []

    def check(self, action: Action) -> tuple[bool, str]:
        """Evaluate whether an action is permitted.

        Args:
            action: The action to evaluate.

        Returns:
            (True, "") if allowed, or (False, reason) if blocked.
        """
        if isinstance(action, RestartAction):
            return self._check_restart(action.target)
        # Scale, rollback, alert, and no_action are always permitted.
        return True, ""

    def record_execution(self, action: Action, success: bool) -> None:
        """Update internal state after an action has been executed.

        Args:
            action:  The action that was executed.
            success: Whether the execution succeeded.
        """
        if isinstance(action, RestartAction):
            service = action.target
            self._restart_counts[service] = self._restart_counts.get(service, 0) + 1
            self._last_restart[service] = datetime.now(timezone.utc)

        if not success:
            service = getattr(action, "target", "_global")
            self._failure_counts[service] = self._failure_counts.get(service, 0) + 1

    def should_escalate(self, service: str) -> bool:
        """Return True if the failure count for a service has hit the escalation threshold."""
        return self._failure_counts.get(service, 0) >= self._escalate_after

    def reset(self, service: str | None = None) -> None:
        """Clear guardrail state for one service, or all services if service is None."""
        if service is None:
            self._restart_counts.clear()
            self._last_restart.clear()
            self._failure_counts.clear()
        else:
            self._restart_counts.pop(service, None)
            self._last_restart.pop(service, None)
            self._failure_counts.pop(service, None)

    def get_restart_count(self, service: str) -> int:
        """Return the number of restarts recorded for a service."""
        return self._restart_counts.get(service, 0)

    def get_failure_count(self, service: str) -> int:
        """Return the number of failed executions recorded for a service."""
        return self._failure_counts.get(service, 0)

    def seconds_until_cooldown_expires(self, service: str) -> float:
        """Return seconds remaining in the restart cooldown for a service (0 if none)."""
        last = self._last_restart.get(service)
        if last is None:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return max(0.0, self._cooldown - elapsed)

    def _check_restart(self, service: str) -> tuple[bool, str]:
        """Validate a restart against count and cooldown limits."""
        count = self._restart_counts.get(service, 0)
        if count >= self._max_restarts:
            reason = (
                f"Restart limit reached for '{service}' "
                f"({count}/{self._max_restarts})."
            )
            self._violations.append(GuardrailViolation(action="restart_service", reason=reason))
            return False, reason

        remaining = self.seconds_until_cooldown_expires(service)
        if remaining > 0:
            reason = (
                f"Restart cooldown active for '{service}' — "
                f"{remaining:.0f}s remaining."
            )
            self._violations.append(GuardrailViolation(action="restart_service", reason=reason))
            return False, reason

        return True, ""
