"""
Agent package.

Implements the LLM-powered reasoning layer that diagnoses root causes and
selects remediation actions when the detection layer raises an anomaly.

Components:
  - ReActAgent:     Drives the Observe → Think → Act → Observe loop.
  - prompts:        Prompt templates injected into the LLM at each step.
  - ActionPlanner:  Parses and validates the LLM's structured JSON output
                    into typed Action objects ready for the executor.

Typical usage (orchestrated by src/main.py):

    from src.agent import ReActAgent

    agent = ReActAgent(config["agent"])
    result = agent.run(anomaly_context)
    # result.actions → list of validated Action objects
    # result.reasoning_trace → full Thought/Action/Observation history
"""

from src.agent.react_agent import ReActAgent
from src.agent.action_planner import ActionPlanner

__all__ = ["ReActAgent", "ActionPlanner"]
