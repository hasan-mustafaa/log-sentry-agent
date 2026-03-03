"""Action planner — parses and validates LLM output into executable Action objects.

Extracts JSON action blocks from raw LLM text, validates them against Pydantic
schemas, and returns typed Action objects ready for the executor. Raises
ActionParseError on malformed output so the ReAct loop can handle it gracefully.

Action schemas mirror AVAILABLE_ACTIONS in prompts.py:
  RestartAction, ScaleAction, RollbackAction, AlertAction, NoAction
"""

from __future__ import annotations

import json
from typing import Any, Literal, Union

import pydantic
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
        if v not in VALID_SERVICES:
            raise ValueError(f"Unknown service '{v}'. Must be one of {sorted(VALID_SERVICES)}")
        return v


class ScaleAction(BaseModel):
    """Scale a simulated service to a new replica count."""

    action: Literal["scale_service"]
    target: str = Field(..., description="Name of the service to scale.")
    replicas: int = Field(..., ge=1, le=10, description="Target replica count (1–10).")
    reason: str = Field(..., description="LLM-provided reason for scaling.")

    @field_validator("target")
    @classmethod
    def target_must_be_valid_service(cls, v: str) -> str:
        if v not in VALID_SERVICES:
            raise ValueError(f"Unknown service '{v}'. Must be one of {sorted(VALID_SERVICES)}")
        return v


class RollbackAction(BaseModel):
    """Roll back a simulated service to its previous deployment."""

    action: Literal["rollback_service"]
    target: str = Field(..., description="Name of the service to roll back.")
    reason: str = Field(..., description="LLM-provided reason for the rollback.")

    @field_validator("target")
    @classmethod
    def target_must_be_valid_service(cls, v: str) -> str:
        if v not in VALID_SERVICES:
            raise ValueError(f"Unknown service '{v}'. Must be one of {sorted(VALID_SERVICES)}")
        return v


class AlertAction(BaseModel):
    """Page on-call when automated remediation cannot resolve the issue."""

    action: Literal["alert_on_call"]
    target: str = Field(..., description="Affected service name.")
    severity: Literal["P1", "P2", "P3"] = Field(..., description="Alert severity.")
    message: str = Field(..., description="Alert message sent to on-call engineer.")

    @field_validator("target")
    @classmethod
    def target_must_be_valid_service(cls, v: str) -> str:
        if v not in VALID_SERVICES:
            raise ValueError(f"Unknown service '{v}'. Must be one of {sorted(VALID_SERVICES)}")
        return v


class NoAction(BaseModel):
    """Agent decides no remediation is required."""

    action: Literal["no_action"]
    reason: str = Field(..., description="LLM-provided explanation for no action.")


# Union type covering all possible action types
Action = Union[RestartAction, ScaleAction, RollbackAction, AlertAction, NoAction]

# Maps the 'action' discriminator field to the corresponding Pydantic model
_SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "restart_service": RestartAction,
    "scale_service": ScaleAction,
    "rollback_service": RollbackAction,
    "alert_on_call": AlertAction,
    "no_action": NoAction,
}


# ── Exception ─────────────────────────────────────────────────────────────────

class ActionParseError(Exception):
    """Raised when the LLM output cannot be parsed into a valid Action."""


# ── Planner ───────────────────────────────────────────────────────────────────

class ActionPlanner:
    """Parses raw LLM text output into validated Action objects.

    Used by ReActAgent after each LLM response to extract actionable
    instructions before passing them to the remediation executor.
    """

    def parse(self, llm_output: str) -> list[Action]:
        """Extract and validate all action JSON blocks from an LLM response.

        Returns an empty list if only a Thought was emitted with no action block.

        Args:
            llm_output: Raw text string returned by the LLM.

        Returns:
            List of validated Action objects.

        Raises:
            ActionParseError: If a JSON block is found but fails schema validation.
        """
        blocks = self._extract_json_blocks(llm_output)
        actions = []
        for block in blocks:
            # Skip JSON objects that aren't action blocks (e.g. nested data)
            if "action" in block:
                actions.append(self.parse_one(block))
        return actions

    def parse_one(self, raw_dict: dict[str, Any]) -> Action:
        """Validate a single raw action dict against the action schemas.

        Uses the 'action' field as a discriminator to select the right model.

        Args:
            raw_dict: Dictionary parsed from a JSON block in the LLM output.

        Returns:
            Validated Action object.

        Raises:
            ActionParseError: If the dict does not match any known action schema.
        """
        action_type = raw_dict.get("action")
        model_cls = _SCHEMA_MAP.get(action_type)  # type: ignore[arg-type]
        if model_cls is None:
            raise ActionParseError(
                f"Unknown action type: '{action_type}'. "
                f"Must be one of {list(_SCHEMA_MAP)}"
            )
        try:
            return model_cls.model_validate(raw_dict)  # type: ignore[return-value]
        except pydantic.ValidationError as exc:
            raise ActionParseError(f"Schema validation failed for '{action_type}': {exc}") from exc

    def _extract_json_blocks(self, text: str) -> list[dict[str, Any]]:
        """Scan text for JSON objects and return all found as dicts.

        Tracks brace depth to handle nested objects. Skips any candidate
        that fails json.loads rather than raising.

        Args:
            text: Raw LLM response text.

        Returns:
            List of parsed JSON dicts found in the text.
        """
        blocks: list[dict[str, Any]] = []
        depth = 0
        start = -1

        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            blocks.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    start = -1

        return blocks
