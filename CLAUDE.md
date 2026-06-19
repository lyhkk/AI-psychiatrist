# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CBT-Discover is a multi-agent AI psychological counseling system built on LangGraph. It implements Cognitive Behavioral Therapy (CBT) assistance through a dual-agent architecture that decouples clinical diagnosis from therapeutic dialogue. The system includes a research/simulation pipeline for evaluating therapy effectiveness and a Flask web application for end-user interaction.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with API keys for THERAPIST_*, SUPERVISOR_*, JUDGE_*, BASELINE_* roles
```

## Running

**Web application:**
```bash
python webapp/app.py
# http://127.0.0.1:5000
```

**Research simulation pipeline:**
```bash
# Generate transcripts
python run_simulation.py --mode cbt-discover --turns 10 --psyqa-index 0-4

# Evaluate transcripts (IG-PQA, CTRS, Belief Conviction Decay)
python eval_pipeline.py --transcript results/sim/cbt-discover/ --output-dir results/eval/
```

**Comparison experiments (CBT-Discover vs single-model Baseline):**
```bash
python run_simulation.py --mode cbt-discover --psyqa-index 0 --output results/sim_cbt.json
python run_simulation.py --mode baseline --psyqa-index 0 --output results/sim_baseline.json
python eval_pipeline.py --transcript results/sim_cbt.json --output results/eval_cbt.json
python eval_pipeline.py --transcript results/sim_baseline.json --output results/eval_baseline.json
```

## Architecture

There is no traditional unit test suite — correctness is validated through the simulation + evaluation pipeline.

### Three Layers

**1. Agents Layer (`agents/`)**
- `DiagnosticianNode` — Silent background analyst using `SUPERVISOR_*` LLM config. Extracts CBT form fields (situation, emotion, automatic_thought, cognitive_distortion) as strict Pydantic-validated JSON. Never produces dialogue output.
- `TherapistNode` — Front-facing dialogue agent using `THERAPIST_*` config. Uses MDP-CoT (Memory-Driven Dynamic Planning Chain-of-Thought) with XML-structured output: `<inner_monologue>` (hidden reasoning) + `<response>` (visible dialogue).
- `PatientNode` — Patient simulator for sandbox evaluation only; disabled in production. Initialized from PsyQA dataset backgrounds.
- `LLMClient` (`llm_base.py`) — Unified wrapper for any OpenAI Chat Completions-compatible API. Factory: `LLMClient.from_role("therapist" | "supervisor" | "judge" | "baseline")`. Auto-reads role-prefixed config from `.env`.
- `DialogueState` (`state.py`) — Typed `TypedDict` for LangGraph shared state. Fields include `chat_history`, `cbt_form`, `entropy_scores`, `turn_count`.
- `workflow.py` — Builds LangGraph `StateGraph`: Entry → DiagnosticianNode → TherapistNode → END. Each `invoke()` = one complete diagnosis + therapy turn.

**2. Web Layer (`webapp/`)**
- `app.py` — Flask application factory; registers chat blueprint, serves frontend.
- `routes/chat.py` — HTTP endpoints only (`/api/chat/start`, `/message`, `/history`, `/cbt_form`, `/session`). Delegates all logic to `SessionManager`.
- `core/session_manager.py` — Per-browser-session state container. Maintains independent `DialogueState` per user, executes LangGraph workflow per turn, handles safety checks. Thread-safe singleton.
- `core/safety.py` — Crisis detection on user input (suicide, self-harm) and harmful output filtering. Returns care hotline on crisis detection.
- Frontend: vanilla HTML/CSS/JS single-page app (`templates/index.html`, `static/js/app.js`). Welcome screen → chat interface → CBT assessment panel with real-time form updates in right sidebar.

**3. Evaluation Layer**
- `eval_pipeline.py` — Three modules: **IG-PQA** (Shannon entropy reduction across 5 "fact evidence clarity" dimensions), **CTRS** (LLM-as-Judge 0–6 scale for clinical fidelity), **Belief Conviction Decay** (conviction score delta per turn).
- Results saved as JSON to `results/`.

### Key Design Patterns
- **Diagnosis ⊥ Dialogue decoupling**: Diagnostician analyzes, Therapist speaks — no mixed roles.
- **Role-based LLM configuration**: Different models per role via `.env` prefixes (`THERAPIST_*`, `SUPERVISOR_*`, `JUDGE_*`, `BASELINE_*`).
- **State-driven communication**: Agents share state exclusively through `DialogueState`; no direct agent-to-agent messaging.
- **XML-structured TherapistNode output**: Inner monologue is parsed and stripped; only `<response>` content is shown to users.

## Datasets

Located in `datasets/`: PsyQA (psychology QA for patient backgrounds), CBT-Bench, SupervisedVsLLM-EfficacyEval. Primarily Chinese-language content.
