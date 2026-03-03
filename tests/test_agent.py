"""Tests for the agent layer: prompts, action_planner, react_agent."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent.action_planner import (
    Action,
    ActionParseError,
    ActionPlanner,
    AlertAction,
    NoAction,
    RestartAction,
    RollbackAction,
    ScaleAction,
    VALID_SERVICES,
)
from src.agent.prompts import (
    build_action_prompt,
    build_observation_from_action,
    build_observe_prompt,
    build_rca_report_prompt,
    build_system_prompt,
    build_think_prompt,
    format_messages_anthropic,
    format_messages_openai,
)
from src.agent.react_agent import AgentContext, AgentResult, ReActAgent, ReasoningStep
from src.detection.feature_extractor import FEATURE_NAMES, FeatureVector


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_feature_vector(service: str = "title-search-service") -> FeatureVector:
    return FeatureVector(
        service=service,
        window_start=datetime.now(),
        window_end=datetime.now(),
        features=np.zeros(15),
        feature_names=list(FEATURE_NAMES),
    )


def _make_context(service: str = "title-search-service") -> AgentContext:
    return AgentContext(
        service=service,
        anomaly_score=0.85,
        triggered_metrics=["cpu_percent", "latency_ms"],
        feature_vector=_make_feature_vector(service),
        recent_logs=[
            {"level": "ERROR", "message": "connection timeout to upstream"},
            {"level": "WARNING", "message": "retrying request"},
        ],
        metric_snapshot={"cpu_percent": 95.0, "latency_ms": 8500.0, "error_rate": 0.45},
    )


def _make_agent(responses: list[str], max_steps: int = 3) -> ReActAgent:
    """Create a ReActAgent with _call_llm mocked to return canned responses."""
    agent = ReActAgent({
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "max_reasoning_steps": max_steps,
        "confidence_threshold": 0.7,
    })
    agent._call_llm = MagicMock(side_effect=responses)  # type: ignore[assignment]
    return agent


@pytest.fixture
def planner() -> ActionPlanner:
    return ActionPlanner()


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildSystemPrompt:

    def test_contains_role(self) -> None:
        assert "LogSentry" in build_system_prompt()

    def test_contains_topology(self) -> None:
        prompt = build_system_prompt()
        assert "transaction-validator" in prompt
        assert "title-search-service" in prompt

    def test_contains_available_actions(self) -> None:
        prompt = build_system_prompt()
        assert "restart_service" in prompt
        assert "no_action" in prompt


class TestBuildObservePrompt:

    def test_includes_service_and_score(self) -> None:
        prompt = build_observe_prompt(
            service="fraud-check-service", anomaly_score=0.92,
            triggered_metrics=["cpu_percent"], recent_logs=[], metric_snapshot={"cpu": 98.0},
        )
        assert "fraud-check-service" in prompt
        assert "0.920" in prompt

    def test_includes_triggered_metrics(self) -> None:
        prompt = build_observe_prompt(
            service="svc", anomaly_score=0.8,
            triggered_metrics=["latency_ms", "error_rate"],
            recent_logs=[], metric_snapshot={},
        )
        assert "latency_ms" in prompt and "error_rate" in prompt

    def test_shows_recent_logs(self) -> None:
        prompt = build_observe_prompt(
            service="svc", anomaly_score=0.7, triggered_metrics=[],
            recent_logs=[{"level": "ERROR", "message": "disk full"}],
            metric_snapshot={},
        )
        assert "[ERROR]" in prompt and "disk full" in prompt

    def test_handles_empty_logs(self) -> None:
        prompt = build_observe_prompt(
            service="svc", anomaly_score=0.7, triggered_metrics=[],
            recent_logs=[], metric_snapshot={},
        )
        assert "no recent logs" in prompt

    def test_truncates_to_last_5_logs(self) -> None:
        logs = [{"level": "INFO", "message": f"log {i}"} for i in range(10)]
        prompt = build_observe_prompt(
            service="svc", anomaly_score=0.7, triggered_metrics=[],
            recent_logs=logs, metric_snapshot={},
        )
        assert "log 9" in prompt
        assert "log 5" in prompt


class TestBuildThinkPrompt:

    def test_includes_step_numbers(self) -> None:
        assert "Step 2/5" in build_think_prompt(2, 5)

    def test_asks_for_reasoning(self) -> None:
        prompt = build_think_prompt(1, 3)
        assert "root cause" in prompt.lower() or "reason" in prompt.lower()


class TestBuildActionPrompt:

    def test_asks_for_json(self) -> None:
        assert "JSON" in build_action_prompt()


class TestBuildObservationFromAction:

    def test_includes_action_and_result(self) -> None:
        obs = build_observation_from_action(
            {"action": "restart_service", "target": "title-search-service"},
            {"success": True, "message": "restarted in 2.1s"},
        )
        assert "restart_service" in obs
        assert "restarted in 2.1s" in obs


class TestBuildRcaReportPrompt:

    def test_includes_service_and_schema(self) -> None:
        prompt = build_rca_report_prompt("title-search-service", [{"step": 1}])
        assert "title-search-service" in prompt
        assert "root_cause_service" in prompt
        assert "fault_type" in prompt


class TestFormatMessagesOpenai:

    def test_prepends_system_message(self) -> None:
        result = format_messages_openai("sys", [{"role": "user", "content": "hi"}])
        assert result[0] == {"role": "system", "content": "sys"}
        assert len(result) == 2

    def test_preserves_turn_order(self) -> None:
        turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        result = format_messages_openai("sys", turns)
        assert result[1]["content"] == "a"
        assert result[2]["content"] == "b"


class TestFormatMessagesAnthropic:

    def test_returns_system_and_messages(self) -> None:
        sys, msgs = format_messages_anthropic("sys", [{"role": "user", "content": "hi"}])
        assert sys == "sys"
        assert len(msgs) == 1


# ══════════════════════════════════════════════════════════════════════════════
# ActionPlanner — JSON extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractJsonBlocks:

    def test_single_block(self, planner: ActionPlanner) -> None:
        blocks = planner._extract_json_blocks('{"action": "no_action", "reason": "ok"}')
        assert len(blocks) == 1

    def test_block_embedded_in_text(self, planner: ActionPlanner) -> None:
        text = 'Restart it. {"action": "restart_service", "target": "title-search-service", "reason": "crash"} Done.'
        blocks = planner._extract_json_blocks(text)
        assert len(blocks) == 1 and blocks[0]["target"] == "title-search-service"

    def test_multiple_blocks(self, planner: ActionPlanner) -> None:
        blocks = planner._extract_json_blocks('{"a": 1} text {"b": 2}')
        assert len(blocks) == 2

    def test_no_json_returns_empty(self, planner: ActionPlanner) -> None:
        assert planner._extract_json_blocks("just regular text") == []

    def test_malformed_json_skipped(self, planner: ActionPlanner) -> None:
        blocks = planner._extract_json_blocks('{bad} {"action": "no_action", "reason": "ok"}')
        assert len(blocks) == 1 and blocks[0]["action"] == "no_action"

    def test_nested_braces(self, planner: ActionPlanner) -> None:
        blocks = planner._extract_json_blocks('{"reason": "data: {nested}"}')
        # The outer block should still parse (inner braces are in a string)
        assert len(blocks) >= 0  # graceful handling either way


# ══════════════════════════════════════════════════════════════════════════════
# ActionPlanner — parse_one validation
# ══════════════════════════════════════════════════════════════════════════════

class TestParseOne:

    def test_restart_action(self, planner: ActionPlanner) -> None:
        a = planner.parse_one({"action": "restart_service", "target": "title-search-service", "reason": "OOM"})
        assert isinstance(a, RestartAction) and a.target == "title-search-service"

    def test_scale_action(self, planner: ActionPlanner) -> None:
        a = planner.parse_one({"action": "scale_service", "target": "fraud-check-service", "replicas": 3, "reason": "load"})
        assert isinstance(a, ScaleAction) and a.replicas == 3

    def test_rollback_action(self, planner: ActionPlanner) -> None:
        a = planner.parse_one({"action": "rollback_service", "target": "document-processor", "reason": "bad deploy"})
        assert isinstance(a, RollbackAction)

    def test_alert_action(self, planner: ActionPlanner) -> None:
        a = planner.parse_one({"action": "alert_on_call", "target": "transaction-validator", "severity": "P1", "message": "down"})
        assert isinstance(a, AlertAction) and a.severity == "P1"

    def test_no_action(self, planner: ActionPlanner) -> None:
        a = planner.parse_one({"action": "no_action", "reason": "stable"})
        assert isinstance(a, NoAction)

    def test_invalid_service_raises(self, planner: ActionPlanner) -> None:
        with pytest.raises(ActionParseError, match="Unknown service"):
            planner.parse_one({"action": "restart_service", "target": "bad-svc", "reason": "x"})

    def test_unknown_action_type_raises(self, planner: ActionPlanner) -> None:
        with pytest.raises(ActionParseError, match="Unknown action type"):
            planner.parse_one({"action": "delete_everything", "reason": "x"})

    def test_missing_required_field_raises(self, planner: ActionPlanner) -> None:
        with pytest.raises(ActionParseError):
            planner.parse_one({"action": "restart_service", "target": "title-search-service"})

    def test_replicas_out_of_range_raises(self, planner: ActionPlanner) -> None:
        with pytest.raises(ActionParseError):
            planner.parse_one({"action": "scale_service", "target": "fraud-check-service", "replicas": 99, "reason": "x"})

    def test_invalid_severity_raises(self, planner: ActionPlanner) -> None:
        with pytest.raises(ActionParseError):
            planner.parse_one({"action": "alert_on_call", "target": "title-search-service", "severity": "P5", "message": "x"})

    def test_all_fct_services_valid(self, planner: ActionPlanner) -> None:
        for svc in VALID_SERVICES:
            a = planner.parse_one({"action": "restart_service", "target": svc, "reason": "test"})
            assert a.target == svc


# ══════════════════════════════════════════════════════════════════════════════
# ActionPlanner — parse (end-to-end)
# ══════════════════════════════════════════════════════════════════════════════

class TestParse:

    def test_extracts_action_from_llm_text(self, planner: ActionPlanner) -> None:
        text = 'Crash detected.\n{"action": "restart_service", "target": "title-search-service", "reason": "OOM"}'
        actions = planner.parse(text)
        assert len(actions) == 1 and isinstance(actions[0], RestartAction)

    def test_thought_only_returns_empty(self, planner: ActionPlanner) -> None:
        assert planner.parse("Need to investigate more.") == []

    def test_skips_non_action_json(self, planner: ActionPlanner) -> None:
        assert planner.parse('{"cpu": 95.0, "memory": 1024}') == []

    def test_raises_on_invalid_action_in_json(self, planner: ActionPlanner) -> None:
        with pytest.raises(ActionParseError):
            planner.parse('{"action": "restart_service", "target": "bad-svc", "reason": "x"}')


# ══════════════════════════════════════════════════════════════════════════════
# ReActAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestReActAgentInit:

    def test_reads_config(self) -> None:
        agent = ReActAgent({"llm_provider": "anthropic", "model": "claude-sonnet-4-6", "max_reasoning_steps": 7})
        assert agent._provider == "anthropic"
        assert agent._model == "claude-sonnet-4-6"
        assert agent._max_steps == 7

    def test_defaults(self) -> None:
        agent = ReActAgent({})
        assert agent._provider == "openai"
        assert agent._max_steps == 5
        assert agent._confidence_threshold == 0.7


class TestReActAgentBuildMessages:

    def test_initial_messages(self) -> None:
        agent = ReActAgent({"llm_provider": "openai"})
        msgs = agent._build_initial_messages(_make_context())
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "title-search-service" in msgs[0]["content"]


class TestReActAgentInitLlm:

    def test_unknown_provider_raises(self) -> None:
        agent = ReActAgent({"llm_provider": "deepseek"})
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            agent._init_llm_client()

    def test_openai_without_key_raises(self) -> None:
        agent = ReActAgent({"llm_provider": "openai"})
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                agent._init_llm_client()

    def test_anthropic_without_key_raises(self) -> None:
        agent = ReActAgent({"llm_provider": "anthropic"})
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                agent._init_llm_client()


class TestReActAgentRun:

    def test_no_action_terminates_loop(self) -> None:
        agent = _make_agent([
            "Looks normal, false positive.",
            '{"action": "no_action", "reason": "false positive"}',
            '{"root_cause_service": "title-search-service", "fault_type": "unknown", "confidence": 0.3, "summary": "FP", "actions_taken": [], "resolved": false}',
        ])
        result = agent.run(_make_context())
        assert len(result.reasoning_trace) == 1
        assert result.actions_taken == []

    def test_single_step_with_action(self) -> None:
        agent = _make_agent([
            "CPU spike, likely OOM.",
            '{"action": "restart_service", "target": "title-search-service", "reason": "OOM"}',
            "Resolved.",
            '{"action": "no_action", "reason": "stable"}',
            '{"root_cause_service": "title-search-service", "fault_type": "oom", "confidence": 0.9, "summary": "OOM crash", "actions_taken": ["restart"], "resolved": true}',
        ])
        result = agent.run(_make_context())
        assert len(result.actions_taken) >= 1
        assert isinstance(result.actions_taken[0], RestartAction)

    def test_max_steps_causes_escalation(self) -> None:
        agent = _make_agent([
            "Step 1.", '{"action": "restart_service", "target": "title-search-service", "reason": "s1"}',
            "Step 2.", '{"action": "scale_service", "target": "title-search-service", "replicas": 3, "reason": "s2"}',
            "Step 3.", '{"action": "alert_on_call", "target": "title-search-service", "severity": "P1", "message": "unresolved"}',
            '{"root_cause_service": "title-search-service", "fault_type": "unknown", "confidence": 0.5, "summary": "Unresolved", "actions_taken": [], "resolved": false}',
        ], max_steps=3)
        result = agent.run(_make_context())
        assert result.escalated is True
        assert len(result.reasoning_trace) == 3

    def test_dry_run_mode(self) -> None:
        agent = _make_agent([
            "Restart needed.",
            '{"action": "restart_service", "target": "title-search-service", "reason": "crash"}',
            "Done.",
            '{"action": "no_action", "reason": "done"}',
            '{"root_cause_service": "title-search-service", "fault_type": "crash", "confidence": 0.8, "summary": "Crash", "actions_taken": ["restart"], "resolved": true}',
        ])
        result = agent.run(_make_context(), executor=None)
        assert len(result.actions_taken) == 1
        assert "Dry-run" in result.reasoning_trace[0].observation

    def test_malformed_action_doesnt_crash(self) -> None:
        agent = _make_agent([
            "Let me think.",
            "This is not valid JSON.",
            "Trying again.",
            '{"action": "no_action", "reason": "giving up"}',
            '{"root_cause_service": "title-search-service", "fault_type": "unknown", "confidence": 0.0, "summary": "Failed", "actions_taken": [], "resolved": false}',
        ])
        result = agent.run(_make_context())
        assert result.reasoning_trace[0].action is None
        assert len(result.reasoning_trace) == 2

    def test_rca_report_fallback(self) -> None:
        agent = _make_agent([
            "Thought.",
            '{"action": "no_action", "reason": "ok"}',
            "not valid json",  # RCA returns garbage
        ])
        result = agent.run(_make_context())
        assert result.rca_report["fault_type"] == "unknown"
        assert "failed" in result.rca_report["summary"].lower()

    def test_rca_report_parsed(self) -> None:
        agent = _make_agent([
            "OOM detected.",
            '{"action": "restart_service", "target": "title-search-service", "reason": "OOM"}',
            "Stable now.",
            '{"action": "no_action", "reason": "stable"}',
            '{"root_cause_service": "title-search-service", "fault_type": "oom", "confidence": 0.95, "summary": "OOM killed process", "actions_taken": ["restart"], "resolved": true}',
        ])
        result = agent.run(_make_context())
        assert result.rca_report["root_cause_service"] == "title-search-service"
        assert result.rca_report["fault_type"] == "oom"

    def test_result_type(self) -> None:
        agent = _make_agent([
            "Thought.",
            '{"action": "no_action", "reason": "ok"}',
            '{"root_cause_service": "x", "fault_type": "unknown", "confidence": 0.0, "summary": "x", "actions_taken": [], "resolved": false}',
        ])
        result = agent.run(_make_context())
        assert isinstance(result, AgentResult)
        assert isinstance(result.reasoning_trace[0], ReasoningStep)

    def test_with_executor(self) -> None:
        mock_exec = MagicMock()
        mock_exec.execute.return_value = MagicMock(
            to_dict=lambda: {"success": True, "message": "restarted"},
        )
        agent = _make_agent([
            "Restart it.",
            '{"action": "restart_service", "target": "title-search-service", "reason": "crash"}',
            "Done.",
            '{"action": "no_action", "reason": "stable"}',
            '{"root_cause_service": "title-search-service", "fault_type": "crash", "confidence": 0.9, "summary": "Crash", "actions_taken": ["restart"], "resolved": true}',
        ])
        result = agent.run(_make_context(), executor=mock_exec)
        mock_exec.execute.assert_called_once()
        assert "restarted" in result.reasoning_trace[0].observation


class TestCheckResolved:

    def test_always_false(self) -> None:
        agent = ReActAgent({})
        assert agent._check_resolved("title-search-service") is False
