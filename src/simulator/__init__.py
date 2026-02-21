"""
Simulator package.

Provides three cooperating components that together create a realistic synthetic
microservice environment for LogSentry to monitor:

  - LogGenerator:     streams structured JSON log lines per service
  - MetricsGenerator: streams time-series CPU / memory / latency / error-rate data
  - FaultInjector:    injects fault scenarios into the running simulation

Typical usage (orchestrated by src/main.py):

    from src.simulator import LogGenerator, MetricsGenerator, FaultInjector

    log_gen     = LogGenerator(config["simulator"])
    metrics_gen = MetricsGenerator(config["simulator"])
    fault_inj   = FaultInjector(config["simulator"], log_gen, metrics_gen)
"""

from src.simulator.log_generator import LogGenerator
from src.simulator.metrics_generator import MetricsGenerator
from src.simulator.fault_injector import FaultInjector

__all__ = ["LogGenerator", "MetricsGenerator", "FaultInjector"]
