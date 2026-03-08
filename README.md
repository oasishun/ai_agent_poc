# Logistics AI Agent PoC

국제물류 도메인 AI Agent PoC — **Booking Agent** + **Track & Trace Agent** Multi-Agent 협업 시스템

## Overview

- **Booking Agent**: 해상화물 스케줄 조회, 운임 비교, 최적 선사 추천, 부킹 생성
- **Track & Trace Agent**: 화물 위치 추적, 이상 감지(ETA 지연), 이해관계자 알림
- **RAG Pipeline**: ChromaDB + OpenAI Embedding 기반 지식 검색
- **Decision Trace**: 모든 Agent 의사결정 과정 투명하게 기록

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM | OpenAI GPT-4o |
| Agent Framework | LangChain 0.3.x + LangGraph |
| Vector Store | ChromaDB (local) |
| Embeddings | text-embedding-3-small |
| CLI | Rich |
| Config | pydantic-settings |

## Installation

```bash
cd ai_agent_poc

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

## Setup

### 1. Configure Environment

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 2. Ingest Knowledge Base

```bash
python -c "
from src.knowledge.rag import RAGPipeline
r = RAGPipeline()
counts = r.ingest_all()
print('Ingested:', counts)
"
```

### 3. Run Unit Tests

```bash
pytest tests/test_knowledge_store.py -v
pytest tests/test_booking_agent.py -v
pytest tests/test_tracking_agent.py -v
```

### 4. Start CLI

```bash
python -m src.cli.app
```

## CLI Usage

```
╔══════════════════════════════════════════════════════════════╗
║  Logistics AI Agent PoC v1.0          [Session: sess-abc123] ║
║  Active Agent: 🚢 Booking Agent                              ║
╚══════════════════════════════════════════════════════════════╝

You > 부산에서 LA로 40HC 1대 부킹해줘. 4월 초 출항 희망.
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help and available commands |
| `/switch [booking\|tracking]` | Switch active agent manually |
| `/trace [id]` | View specific Decision Trace |
| `/traces` | List all traces for current session |
| `/knowledge` | Show Knowledge Store stats |
| `/log [DEBUG\|INFO\|WARN]` | Change log level |
| `/export` | Export session traces to JSON |
| `/exit` | End session |

## Example Conversations

### Booking Flow

```
You > 부산에서 LA로 40HC 1대 부킹해줘. 4월 초 출항 희망.

🚢 Booking Agent >
부산(KRPUS) → LA(USLAX) 40HC 1대 조회 결과:

#1 HMM — PS7 | ETD: Apr 01 → ETA: Apr 18 (17일) | Total: USD 2,815
   ⚡ 비용 최적. 단, 상하이 경유 지연 리스크 있음.

#2 MAERSK — TP6 | ETD: Apr 03 → ETA: Apr 17 (14일) | Total: USD 3,005
   ⚡ 직항, 최단 transit time. 신뢰도 A등급.

어떤 옵션으로 부킹할까요? (1 또는 2)

You > 1번으로 해줘

🚢 Booking Agent > ✅ 부킹 완료! Booking No: BK-2026-XXX
```

### Tracking Flow

```
You > HDMU1234567 컨테이너 지금 어디있어?

📦 Track & Trace Agent >
현재 위치: Pacific Ocean, East of Japan
상태: IN_TRANSIT
ETA: 2026-04-19 08:00 (원래 Apr 18 대비 18시간 지연)
⚠️ 이상 감지: Shanghai 환적 혼잡으로 18시간 지연
```

## Project Structure

```
ai_agent_poc/
├── src/
│   ├── agents/          # LangGraph Agents (Booking, Tracking, Orchestrator)
│   ├── knowledge/       # RAG Pipeline + Knowledge Store
│   ├── decision/        # Decision Trace system
│   ├── tools/           # LangChain @tool definitions
│   ├── mock_api/        # Mock external API layer
│   └── cli/             # Rich-based CLI interface
├── data/
│   ├── structured/      # JSON: schedules, rates, port info
│   ├── unstructured/    # MD: carrier notices, regulations
│   ├── tribal/          # MD: booking tips, carrier preferences
│   └── decision_traces/ # Persisted trace records
├── tests/               # pytest tests + eval framework
└── docs/                # PRD documentation
```

## Evaluation

```bash
# Run evaluation framework
python tests/eval_framework.py --output reports/eval_result.json
```

Metrics: Task Completion, Decision Quality, Knowledge Utilization, Tool Efficiency, Handoff Accuracy

## Architecture

```
CLI (Rich)
    │
Orchestrator (LangGraph) — intent routing
    ├── Booking Agent (LangGraph StateGraph)
    │       └── Tools: search_schedules, get_freight_rates, create_booking, handoff_to_tracking
    └── Tracking Agent (LangGraph StateGraph)
            └── Tools: track_shipment, get_milestones, check_anomalies, notify_stakeholder

Shared Services:
    ├── Knowledge Store (ChromaDB + OpenAI Embeddings)
    │       └── Collections: structured / unstructured / tribal
    ├── Decision Trace (traces.json)
    └── Mock API Layer (carrier, terminal, tracking)
```
