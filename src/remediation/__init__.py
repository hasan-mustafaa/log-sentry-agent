"""
Remediation package.

Executes remediation actions selected by the ReAct agent against the simulated
FCT microservice mesh, subject to safety guardrails.

Components:
  - Executor:    Carries out restart, scale, rollback, and alert actions on
                 simulated services and returns structured execution results.
  - Guardrails:  Validates each action against safety constraints (rate limits,
                 cooldowns, escalation rules) before allowing execution.

Typical usage (orchestrated by ReActAgent):

    from src.remediation import Executor, Guardrails
    from src.agent.action_planner import RestartAction

    guardrails = Guardrails(config["remediation"])
    executor   = Executor(config["remediation"], guardrails)

    result = executor.execute(RestartAction(
        action="restart_service",
        target="title-search-service",
        reason="OOM detected"
    ))
"""

from src.remediation.executor import Executor
from src.remediation.guardrails import Guardrails

__all__ = ["Executor", "Guardrails"]
