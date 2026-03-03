"""Guardrails — safety checks that gate every remediation action.

Enforces per-service restart limits, cooldown windows between restarts,
and failure escalation thresholds. The Executor calls check() before
running any action and record_execution() afterwards to update state.

Config keys read from remediation section of config.yaml:
  max_restarts_per_service, restart_cooldown_seconds, auto_escalate_after_failures
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.agent.action_planner import Action, AlertAction, NoAction, RestartAction


@dataclass
class GuardrailViolation:
    """Describes why an action was blocked by guardrails."""

    action: Action
    reason: str
    blocked_at: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.blocked_at is None:
            self.blocked_at = datetime.now(timezone.utc)


class Guardrails:
    """Enforces safety limits on remediation actions.

    Tracks per-service restart counts, last-restart timestamps, and
    consecutive failure counts. Stateless actions (alert, no_action)
    always pass. ScaleAction and RollbackAction always pass by default
    since they don't carry crash-loop risk.
    """

    def __init__(self, remediation_config: dict[str, Any]) -> None:
        """Initialize limits from the [remediation] section of config.yaml."""
        self._max_restarts: int = remediation_config.get("max_restarts_per_service", 3)
        self._cooldown_seconds: int = remediation_config.get("restart_cooldown_seconds", 300)
        self._max_failures: int = remediation_config.get("auto_escalate_after_failures", 2)

        self._restart_counts: dict[str, int] = defaultdict(int)
        self._last_restart: dict[str, datetime] = {}
        self._failure_counts: dict[str, int] = defaultdict(int)

    def check(self, action: Action) -> tuple[bool, str]:
        """Check whether an action is permitted.

        Args:
            action: Validated Action object from ActionPlanner.

        Returns:
            (True, "") if allowed, or (False, reason_string) if blocked.
        """
        # Stateless actions always pass
        if isinstance(action, (AlertAction, NoAction)):
            return True, ""

        if isinstance(action, RestartAction):
            return self._check_restart(action.target)

        # ScaleAction and RollbackAction — no guardrails applied
        return True, ""

    def record_execution(self, action: Action, success: bool) -> None:
        """Update internal counters after an action has been executed.

        Args:
            action:  The action that was executed.
            success: Whether the execution succeeded.
        """
        if isinstance(action, RestartAction):
            if success:
                self._restart_counts[action.target] += 1
                self._last_restart[action.target] = datetime.now(timezone.utc)
            else:
                self._failure_counts[action.target] += 1
        elif not success:
            target = getattr(action, "target", None)
            if target:
                self._failure_counts[target] += 1

    def should_escalate(self, service: str) -> bool:
        """Return True if the failure count has hit the escalation threshold."""
        return self._failure_counts[service] >= self._max_failures

    def reset(self, service: str | None = None) -> None:
        """Clear guardrail state for one service or all services."""
        if service is not None:
            self._restart_counts.pop(service, None)
            self._last_restart.pop(service, None)
            self._failure_counts.pop(service, None)
        else:
            self._restart_counts.clear()
            self._last_restart.clear()
            self._failure_counts.clear()

    def get_restart_count(self, service: str) -> int:
        """Return the number of restarts recorded for a service."""
        return self._restart_counts[service]

    def get_failure_count(self, service: str) -> int:
        """Return the number of consecutive failures recorded for a service."""
        return self._failure_counts[service]

    def seconds_until_cooldown_expires(self, service: str) -> float:
        """Return remaining cooldown seconds for a service, or 0 if clear."""
        last = self._last_restart.get(service)
        if last is None:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        remaining = self._cooldown_seconds - elapsed
        return max(0.0, remaining)

    # ── Private ───────────────────────────────────────────────────────────────

    def _check_restart(self, service: str) -> tuple[bool, str]:
        """Apply restart-specific guardrails: max count and cooldown window."""
        count = self._restart_counts[service]
        if count >= self._max_restarts:
            return (
                False,
                f"Restart limit reached for '{service}' "
                f"({count}/{self._max_restarts}). Escalate instead.",
            )

        remaining = self.seconds_until_cooldown_expires(service)
        if remaining > 0:
            return (
                False,
                f"Restart cooldown active for '{service}' — "
                f"{remaining:.0f}s remaining.",
            )

        return True, ""
