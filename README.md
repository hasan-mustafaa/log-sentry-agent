# LogSentry Agent

> AI-driven AIOps system that monitors simulated microservice logs and metrics, detects anomalies using ML, diagnoses root causes with a ReAct LLM agent, and automatically remediates issues.

---

## Overview

<!-- TODO: Fill in after MVP is complete -->
<!-- Describe what the system does end-to-end, the problem it solves, and why it matters. -->

LogSentry Agent is a self-healing observability pipeline built for simulated FCT (Financial Crime & Technology) microservices. It combines traditional statistical methods with modern ML and LLM-powered reasoning to detect, diagnose, and remediate operational incidents — all without human intervention.

**Core capabilities (to be implemented):**
- Real-time structured log generation across 4 interdependent microservices
- Fault injection (crashes, latency spikes, memory leaks, connection failures)
- Hybrid anomaly detection (Z-score + Isolation Forest ensemble)
- LLM-based root cause analysis via the ReAct reasoning pattern
- Automated remediation with safety guardrails
- Live Streamlit dashboard with agent reasoning traces

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

```
┌─────────────────────────────────────────────────────────────────┐
│                        LogSentry Agent                          │
│                                                                 │
│  [Simulator] → [Detection] → [ReAct Agent] → [Remediation]    │
│       ↓              ↓             ↓               ↓           │
│  [Dashboard ─────────────────────────────────────────────────] │
└─────────────────────────────────────────────────────────────────┘
```

<!-- TODO: Replace with full ASCII architecture diagram after implementation -->

---

## AI Techniques Used

See [AI_TECHNIQUES.md](AI_TECHNIQUES.md) for detailed explanations.

| Technique | Purpose |
|---|---|
| **Isolation Forest** | ML-based anomaly detection on feature vectors |
| **Z-Score / Moving Average** | Statistical baseline anomaly detection |
| **Ensemble Scoring** | Combines statistical + ML signals with tunable weights |
| **Drain Algorithm** | Structured log template mining from raw log lines |
| **ReAct Agent Pattern** | Observe → Think → Act LLM reasoning loop |
| **LLM Root Cause Analysis** | GPT-4o-mini / Claude reasons over service dependency graph |

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key (or Anthropic API key)
- `pip` or `uv` for dependency management

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd aiopsagent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running the Agent

```bash
# Run the full pipeline (simulator + detection + agent + remediation)
python -m src.main

# Or use the quick demo script
python scripts/run_demo.py
```

### Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
aiopsagent/
├── config/config.yaml          # Single source of truth for all config
├── src/
│   ├── main.py                 # Pipeline orchestrator
│   ├── simulator/              # Log & metrics generation, fault injection
│   ├── detection/              # Log parsing, feature extraction, anomaly detection
│   ├── agent/                  # ReAct LLM agent, prompts, action planner
│   ├── remediation/            # Action executor + safety guardrails
│   └── dashboard/              # Streamlit live dashboard
├── tests/                      # Unit tests per component
├── scripts/run_demo.py         # End-to-end demo
└── data/sample_logs/           # Persisted sample log files
```

---

## How It Works

<!-- TODO: Fill in after implementation with a numbered end-to-end flow description -->
<!-- Example:
1. The Simulator generates structured JSON logs and time-series metrics for 4 FCT microservices.
2. The FaultInjector periodically injects realistic faults into the service mesh.
3. The LogParser (Drain algorithm) extracts log templates from raw log streams.
4. The FeatureExtractor builds time-windowed feature vectors.
5. The Ensemble Detector (Z-score + Isolation Forest) scores each window.
6. On anomaly detection, the ReAct Agent is invoked with the anomaly context.
7. The Agent reasons over service dependencies and selects a remediation action.
8. The Executor applies the action (restart, scale, rollback) with guardrail checks.
9. The Dashboard reflects all state changes in real time.
-->

---

## Demo Video

<!-- TODO: Add demo video link after recording -->
> Demo video link: _coming soon_

---

## Assumptions & Future Improvements

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for a full list of assumptions, known limitations, and planned improvements.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML / Detection | scikit-learn, numpy, scipy, pandas |
| Log Parsing | drain3 |
| LLM | OpenAI GPT-4o-mini (or Anthropic Claude) |
| Dashboard | Streamlit + Plotly |
| Config | PyYAML + Pydantic |
| Testing | pytest + pytest-asyncio |
| Terminal UI | Rich |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
