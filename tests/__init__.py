"""
LogSentry Agent — test suite.

Tests are organised by pipeline component:
  - test_simulator.py   : LogGenerator, MetricsGenerator, FaultInjector
  - test_detection.py   : LogParser, FeatureExtractor, StatisticalDetector, MLDetector
  - test_agent.py       : ActionPlanner, ReActAgent (with LLM mocked)
  - test_remediation.py : Executor, Guardrails

Run all tests:
    pytest

Run a single module:
    pytest tests/test_detection.py -v

Run with coverage:
    pytest --cov=src --cov-report=term-missing
"""
