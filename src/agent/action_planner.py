"""
Action planner — parses and validates LLM output into executable Action objects.

The LLM emits raw text containing JSON action blocks inside its response.
This module is responsible for:
  1. Extracting JSON blocks from the raw LLM text (regex or JSON boundary scan).
  2. Validating each block against the allowed action schema (via Pydantic).
  3. Returning a list of typed Action objects ready for the executor.
  4. Gracefully handling malformed LLM output (partial JSON, wrong keys, etc.)
     by raising ActionParseError rather than crashing the pipeline.

Action schemas (mirroring AVAILABLE_ACTIONS in prompts.py):
  - RestartAction    : {"action": "restart_service", "target": str, "reason": str}
  - ScaleAction      : {"action": "scale_service",   "target": str, "replicas": int, "reason": str}
  - RollbackAction   : {"action": "rollback_service","target": str, "reason": str}
  - AlertAction      : {"action": "alert_on_call",   "target": str, "severity": str, "message": str}
  - NoAction         : {"action": "no_action",       "reason": str}
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# ── Pydantic action models ────────────────────────────────────────────────────

VALID_SERVICES = frozenset(
    {
        "transaction-validator",
        "fraud-check-service",
        "document-processor",
        "title-search-service",
    }
)


class RestartAction(BaseModel):
    """Restart a simulated service."""

    action: Literal["restart_service"]
    target: str = Field(..., description="Name of the service to restart.")
    reason: str = Field(..., description="LLM-provided reason for the restart.")

    @field_validator("target")
    @classmethod
    def target_must_be_valid_service(cls, v: str) -> str:
        """Ensure target is one of the four FCT services."""
        raise NotImplementedError


class ScaleAction(BaseModel):
    """Scale a simulated service to a new replica count."""

    action: Literal["scale_service"]
    target: str = Field(..., description="Name of the service to scale.")
    replicas: int = Field(..., ge=1, le=10, description="Target replica count (1–10).")
    reason: str = Field(..., description="LLM-provided reason for scaling.")

    @field_validator("target")
    @classmethod
    def target_must_be_valid_service(cls, v: str) -> str:
        """Ensure target is one of the four FCT services."""
        raise NotImplementedError


class RollbackAction(BaseModel):
    """Roll back a simulated service to its previous deployment."""

    action: Literal["rollback_service"]
    target: str = Field(..., description="Name of the service to roll back.")
    reason: str = Field(..., description="LLM-provided reason for the rollback.")

    @field_validator("target")
    @classmethod
    def target_must_be_valid_service(cls, v: str) -> str:
        """Ensure target is one of the four FCT services."""
        raise NotImplementedError


class AlertAction(BaseModel):
    """Page on-call when automated remediation cannot resolve the issue."""

    action: Literal["alert_on_call"]
    target: str = Field(..., description="Affected service name.")
    severity: Literal["P1", "P2", "P3"] = Field(..., description="Alert severity.")
    message: str = Field(..., description="Alert message sent to on-call engineer.")

    @field_validator("target")
    @classmethod
    def target_must_be_valid_service(cls, v: str) -> str:
        """Ensure target is one of the four FCT services."""
        raise NotImplementedError


class NoAction(BaseModel):
    """Agent decides no remediation is required."""

    action: Literal["no_action"]
    reason: str = Field(..., description="LLM-provided explanation for no action.")


# Union type covering all possible action types
Action = Union[RestartAction, ScaleAction, RollbackAction, AlertAction, NoAction]


# ── Exception ─────────────────────────────────────────────────────────────────

class ActionParseError(Exception):
    """Raised when the LLM output cannot be parsed into a valid Action."""


# ── Planner ───────────────────────────────────────────────────────────────────

class ActionPlanner:
    """
    Parses raw LLM text output into validated Action objects.

    Used by ReActAgent after each LLM response to extract actionable
    instructions before passing them to the remediation executor.
    """

    def parse(self, llm_output: str) -> list[Action]:
        """
        Extract and validate all action JSON blocks from an LLM response.

        Scans the text for JSON objects, attempts to match each against the
        action schemas, and returns a list of validated Action objects.

        Args:
            llm_output: Raw text string returned by the LLM.

        Returns:
            List of validated Action objects (may be empty if only a Thought
            was emitted with no action block).

        Raises:
            ActionParseError: If a JSON block is found but fails schema validation.
        """
        raise NotImplementedError

    def parse_one(self, raw_dict: dict[str, Any]) -> Action:
        """
        Validate a single raw action dictionary against the action schemas.

        Uses Pydantic discriminated union on the 'action' field.

        Args:
            raw_dict: Dictionary parsed from a JSON block in the LLM output.

        Returns:
            Validated Action object.

        Raises:
            ActionParseError: If the dict does not match any known action schema.
        """
        raise NotImplementedError

    def _extract_json_blocks(self, text: str) -> list[dict[str, Any]]:
        """
        Scan text for JSON objects and return all found as dicts.

        Handles nested braces by tracking brace depth. Returns an empty list
        if no valid JSON is found rather than raising.

        Args:
            text: Raw LLM response text.

        Returns:
            List of parsed JSON dicts found in the text.
        """
        raise NotImplementedError
