# PRD: Logistics AI Agent PoC

> **Project**: Logistics AI Agent Proof of Concept  
> **Version**: 1.0  
> **Date**: 2026-03-07  
> **Status**: Draft  
> **Stack**: Python 3.11+ · LangChain/LangGraph · OpenAI API · CLI

---

## 1. Executive Summary

본 문서는 국제물류 도메인에서 AI Agent를 활용한 업무 자동화 PoC 프로젝트의 제품 요구사항을 정의한다.

두 가지 핵심 Agent를 통해 물류 업무의 AI 자동화 가능성을 검증한다:

1. **International Booking Agent** — 해상/항공 화물 부킹 프로세스 자동화
2. **Track & Trace Agent** — 실시간 화물 추적 및 이상 감지

두 Agent는 LangGraph 기반 Multi-Agent 협업을 통해 연동되며, Knowledge Base에 축적된 정형/비정형/Tribal Knowledge를 RAG 파이프라인으로 활용하여 의사결정을 수행한다. 모든 인터랙션은 Python CLI 기반 대화형 인터페이스에서 이루어지며, Agent의 의사결정 과정(Decision Trace)이 투명하게 로깅된다.

---

## 2. Project Overview

### 2.1 Goals & Objectives

- 물류 도메인 특화 AI Agent의 실현 가능성 검증
- 정형 데이터(스케줄, 요율), 비정형 데이터(이메일, 메모), Tribal Knowledge(암묵지)를 Agent가 활용할 수 있는 Knowledge Management 체계 구축
- Agent의 의사결정 과정을 Decision Trace로 기록하여 투명성 및 디버깅 가능성 확보
- Multi-Agent 협업 패턴(Booking ↔ Tracking) 검증
- 간단한 RAG 파이프라인을 통한 지식 검색/활용 검증
- Mock API를 통한 외부 시스템 연동 시뮬레이션

### 2.2 Scope

#### In Scope

- International Booking Agent: 스케줄 조회, 요율 비교, 부킹 생성/확인
- Track & Trace Agent: 화물 위치 추적, 상태 변경 알림, 이상 감지
- Knowledge Store: JSON/MD 포맷의 정형/비정형/Tribal Knowledge 저장소
- Decision Trace: Agent 의사결정 과정 저장 및 갱신 시스템
- RAG Pipeline: 지식 검색을 위한 간단한 Retrieval-Augmented Generation
- Mock External APIs: 선사/항공사/터미널 시스템 시뮬레이션
- CLI Interface: 터미널 기반 대화형 인터페이스 with 실시간 로깅
- Evaluation Framework: Agent 성능 평가를 위한 테스트 시나리오

#### Out of Scope

- 실제 외부 시스템(TMS, 선사 API 등) 연동
- Web/Mobile UI
- 프로덕션 수준의 보안, 인증, 권한 관리
- 대용량 데이터 처리 및 성능 최적화
- 실시간 데이터 스트리밍

### 2.3 Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM Backend | OpenAI API (GPT-4o / GPT-4o-mini) |
| Agent Framework | LangChain 0.3.x + LangGraph |
| RAG / Embeddings | LangChain + ChromaDB (local vector store) |
| Data Format | JSON (structured), Markdown (unstructured/tribal) |
| Interface | Python CLI (Rich library for formatting) |
| Logging | Python logging + Rich console output |
| Testing | pytest + custom evaluation framework |
| Package Manager | Poetry or pip + requirements.txt |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Interface (Rich)                  │
│              User Input / Agent Output / Logs            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Agent Orchestrator (LangGraph)              │
│         Intent Analysis → Agent Routing → State Mgmt    │
├─────────────────────┬───────────────────────────────────┤
│                     │                                   │
│  ┌──────────────────▼────────────┐  ┌────────────────┐ │
│  │     Booking Agent             │  │ Track/Trace    │ │
│  │  ┌─────────────────────────┐  │  │    Agent       │ │
│  │  │ Tools:                  │  │  │ ┌────────────┐ │ │
│  │  │  - search_schedules     │  │  │ │ Tools:     │ │ │
│  │  │  - get_freight_rates    │  │  │ │ - track    │ │ │
│  │  │  - create_booking       │  │  │ │ - mileston │ │ │
│  │  │  - search_knowledge     │  │  │ │ - anomaly  │ │ │
│  │  │  - log_decision         │  │  │ │ - notify   │ │ │
│  │  │  - handoff_to_tracking  │  │  │ └────────────┘ │ │
│  │  └─────────────────────────┘  │  └────────────────┘ │
│  └───────────────────────────────┘                      │
├─────────────────────────────────────────────────────────┤
│                    Shared Services                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐│
│  │ Knowledge    │ │ Decision     │ │ RAG Pipeline     ││
│  │ Store        │ │ Trace        │ │ (Embed→Search    ││
│  │ (JSON/MD)    │ │ (JSON Log)   │ │  →Augment)       ││
│  └──────┬───────┘ └──────────────┘ └────────┬─────────┘│
│         │                                    │          │
│  ┌──────▼────────────────────────────────────▼─────────┐│
│  │              ChromaDB (Local Vector Store)           ││
│  └─────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                   Mock API Layer                         │
│  Carrier Schedule │ Freight Rate │ Booking │ Tracking   │
│  Port/Terminal    │ (JSON-based simulations)             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
logistics-ai-agent-poc/
├── pyproject.toml
├── README.md
├── .env.example                        # OPENAI_API_KEY 등 환경변수 템플릿
├── src/
│   ├── __init__.py
│   ├── config.py                       # App configuration
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py             # LangGraph multi-agent router
│   │   ├── booking_agent.py            # International Booking Agent
│   │   └── tracking_agent.py           # Track & Trace Agent
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── store.py                    # Knowledge CRUD operations
│   │   ├── rag.py                      # RAG pipeline (embed + search)
│   │   └── loader.py                   # JSON/MD file loader
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── trace.py                    # Decision trace logger
│   │   └── models.py                   # Trace data models (Pydantic)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── booking_tools.py            # LangChain tools for booking
│   │   ├── tracking_tools.py           # LangChain tools for tracking
│   │   └── knowledge_tools.py          # LangChain tools for knowledge
│   ├── mock_api/
│   │   ├── __init__.py
│   │   ├── carrier_api.py              # Mock carrier schedule/rate API
│   │   ├── terminal_api.py             # Mock terminal/port API
│   │   └── tracking_api.py             # Mock tracking data provider
│   └── cli/
│       ├── __init__.py
│       ├── app.py                      # Main CLI entrypoint
│       ├── renderer.py                 # Rich console output
│       └── logger.py                   # Structured logging setup
├── data/
│   ├── structured/
│   │   ├── vessel_schedules.json
│   │   ├── freight_rates.json
│   │   ├── port_info.json
│   │   └── booking_templates.json
│   ├── unstructured/
│   │   ├── carrier_notices.md
│   │   ├── trade_regulations.md
│   │   └── port_congestion_reports.md
│   ├── tribal/
│   │   ├── booking_tips.md
│   │   ├── carrier_preferences.md
│   │   └── common_issues.md
│   └── decision_traces/
│       └── traces.json
├── tests/
│   ├── scenarios/
│   │   ├── booking_scenarios.json
│   │   └── tracking_scenarios.json
│   ├── test_booking_agent.py
│   ├── test_tracking_agent.py
│   ├── test_knowledge_store.py
│   └── eval_framework.py
└── docs/
    └── PRD.md                          # 이 문서
```

---

## 4. Knowledge Management System

Agent가 도메인 전문성을 갖추기 위해 세 가지 유형의 지식을 관리한다. 모든 데이터는 PoC 단계에서 JSON 또는 Markdown 파일로 저장한다.

### 4.1 Knowledge Types

| Type | Format | Examples | Storage Path |
|---|---|---|---|
| Structured | JSON | Vessel schedules, freight rates, port data | `data/structured/*.json` |
| Unstructured | Markdown | Carrier notices, regulations, congestion reports | `data/unstructured/*.md` |
| Tribal | Markdown | Booking tips, carrier preferences, common issues | `data/tribal/*.md` |

### 4.2 Sample Data Specifications

#### 4.2.1 Structured — `vessel_schedules.json`

```json
[
  {
    "schedule_id": "VSL-2026-001",
    "vessel_name": "HMM ALGECIRAS",
    "carrier": "HMM",
    "voyage_no": "0125E",
    "route": {
      "origin": { "port_code": "KRPUS", "port_name": "Busan", "terminal": "HBCT" },
      "destination": { "port_code": "USLAX", "port_name": "Los Angeles", "terminal": "APM" },
      "via_ports": [
        { "port_code": "CNSHA", "port_name": "Shanghai", "eta": "2026-04-05", "etd": "2026-04-06" }
      ]
    },
    "etd": "2026-04-01T08:00:00Z",
    "eta": "2026-04-18T14:00:00Z",
    "transit_days": 17,
    "available_space": { "20GP": 45, "40GP": 30, "40HC": 25 },
    "cut_off": {
      "doc_cutoff": "2026-03-29T17:00:00Z",
      "cargo_cutoff": "2026-03-30T12:00:00Z"
    },
    "service_name": "PS7",
    "status": "OPEN"
  },
  {
    "schedule_id": "VSL-2026-002",
    "vessel_name": "MAERSK EDINBURGH",
    "carrier": "MAERSK",
    "voyage_no": "206W",
    "route": {
      "origin": { "port_code": "KRPUS", "port_name": "Busan", "terminal": "BNCT" },
      "destination": { "port_code": "USLAX", "port_name": "Los Angeles", "terminal": "Pier400" },
      "via_ports": []
    },
    "etd": "2026-04-03T06:00:00Z",
    "eta": "2026-04-17T10:00:00Z",
    "transit_days": 14,
    "available_space": { "20GP": 20, "40GP": 15, "40HC": 10 },
    "cut_off": {
      "doc_cutoff": "2026-03-31T17:00:00Z",
      "cargo_cutoff": "2026-04-01T12:00:00Z"
    },
    "service_name": "TP6",
    "status": "OPEN"
  }
]
```

#### 4.2.2 Structured — `freight_rates.json`

```json
[
  {
    "rate_id": "RT-2026-001",
    "carrier": "HMM",
    "origin": "KRPUS",
    "destination": "USLAX",
    "container_type": "40HC",
    "base_rate": 1850,
    "surcharges": {
      "BAF": 350,
      "CAF": 45,
      "THC_origin": 185,
      "THC_dest": 320,
      "doc_fee": 50,
      "seal_fee": 15
    },
    "total_rate": 2815,
    "currency": "USD",
    "valid_from": "2026-03-01",
    "valid_to": "2026-04-30",
    "contract_type": "SPOT",
    "remarks": "Peak season surcharge may apply after Apr 15"
  },
  {
    "rate_id": "RT-2026-002",
    "carrier": "MAERSK",
    "origin": "KRPUS",
    "destination": "USLAX",
    "container_type": "40HC",
    "base_rate": 2100,
    "surcharges": {
      "BAF": 280,
      "CAF": 40,
      "THC_origin": 195,
      "THC_dest": 310,
      "doc_fee": 65,
      "seal_fee": 15
    },
    "total_rate": 3005,
    "currency": "USD",
    "valid_from": "2026-03-01",
    "valid_to": "2026-05-31",
    "contract_type": "CONTRACT",
    "remarks": "Guaranteed space for contract customers"
  }
]
```

#### 4.2.3 Structured — `port_info.json`

```json
[
  {
    "port_code": "KRPUS",
    "port_name": "Busan",
    "country": "KR",
    "timezone": "Asia/Seoul",
    "terminals": [
      { "code": "HBCT", "name": "Hanjin Busan Container Terminal", "operator": "HMM" },
      { "code": "BNCT", "name": "Busan New Container Terminal", "operator": "BNCT" },
      { "code": "PNC", "name": "Pusan Newport Container Terminal", "operator": "DP World" }
    ],
    "congestion_level": "LOW",
    "avg_dwell_time_hours": 48,
    "working_hours": "24/7"
  },
  {
    "port_code": "USLAX",
    "port_name": "Los Angeles",
    "country": "US",
    "timezone": "America/Los_Angeles",
    "terminals": [
      { "code": "APM", "name": "APM Terminals", "operator": "APM" },
      { "code": "Pier400", "name": "Pier 400 Maersk", "operator": "Maersk" }
    ],
    "congestion_level": "MEDIUM",
    "avg_dwell_time_hours": 96,
    "working_hours": "06:00-02:00"
  }
]
```

#### 4.2.4 Unstructured — `carrier_notices.md`

```markdown
# Carrier Notices

## 2026-03-05 — HMM: Peak Season Surcharge Notice
HMM announces a Peak Season Surcharge (PSS) of USD 200/TEU effective April 15, 2026
for all Trans-Pacific Eastbound services. Early booking is strongly recommended.

## 2026-03-01 — MAERSK: Shanghai Transshipment Delay
Due to increased volume at Shanghai port, MAERSK services routing via Shanghai
are experiencing 2-3 day delays. Direct services remain on schedule.

## 2026-02-28 — ONE: Equipment Shortage at Busan
ONE reports limited availability of 40HC containers at Busan. Customers requiring
40HC are advised to book at least 2 weeks in advance or consider 40GP alternatives.
```

#### 4.2.5 Tribal Knowledge — `booking_tips.md`

```markdown
# Booking Tips (Tribal Knowledge)

## Carrier-Specific Tips

### HMM
- 금요일 부킹 마감이 타 선사 대비 12시간 빠름. 목요일까지 부킹 완료 권장.
- PS7 서비스는 상하이 경유이므로 환적 지연 리스크 있음. 긴급화물은 직항 서비스 추천.
- HMM은 자사 터미널(HBCT) 사용 시 THC 할인 가능 (영업팀 별도 협의 필요).

### MAERSK
- TP6 직항 서비스가 transit time 가장 짧지만, 스페이스 확보 어려움.
- 계약 고객은 guaranteed space 혜택 있으므로, 스팟 물량이라도 영업 담당 통해 문의 가치 있음.
- Pier 400 터미널은 주말 게이트 오픈 시간이 짧으므로 금요일 입항 화물 주의.

## Route-Specific Tips

### KRPUS → USLAX
- 4-5월 peak season 진입기로 스페이스 사전 확보 필수.
- Reefer 컨테이너는 최소 2주 전 사전 부킹 필요 (만성적 장비 부족).
- LA항 혼잡 시 Long Beach 대안 검토. 내륙 운송비 차이는 약 USD 50-80/container.

## General Tips
- 부킹 시 정확한 화물 중량 기재 필수. VGM 미제출 시 선적 거부될 수 있음.
- 위험물(DG cargo)은 선사별 접수 가능 Class가 다름. 사전 확인 필수.
- 복합운송(intermodal) 요청 시 내륙 구간 리드타임 최소 3영업일 추가 고려.
```

#### 4.2.6 Tribal Knowledge — `carrier_preferences.md`

```markdown
# Carrier Preferences (Tribal Knowledge)

## Reliability Scores (자체 평가 기준, 최근 6개월)

| Carrier | On-Time Rate | Rollover Rate | Communication | Overall Score |
|---------|-------------|---------------|---------------|---------------|
| MAERSK  | 92%         | 2%            | Excellent     | A             |
| HMM     | 85%         | 5%            | Good          | B+            |
| ONE     | 88%         | 4%            | Good          | B+            |
| MSC     | 80%         | 8%            | Fair          | B             |
| Evergreen | 83%       | 6%            | Fair          | B             |

## Decision Factors (우선순위)
1. Transit Time — 고객이 납기일 지정한 경우 최우선
2. Total Cost — 예산 제한 있는 경우
3. Carrier Reliability — On-time rate + rollover rate 종합
4. Space Availability — Peak season에는 스페이스 확보 자체가 핵심
5. Special Requirements — Reefer, DG, OOG 등 특수화물 대응 가능 여부
```

#### 4.2.7 Tribal Knowledge — `common_issues.md`

```markdown
# Common Issues & Resolutions (Tribal Knowledge)

## Issue: 선적 마감 후 부킹 변경 요청
- **빈도**: 주 2-3회
- **원인**: 고객측 화물 준비 지연, 서류 미비
- **해결**: 선사 영업팀에 직접 연락하여 grace period 요청. HMM은 보통 4시간, MAERSK는 2시간 추가 허용.
- **예방**: 고객에게 마감 48시간 전 알림 발송

## Issue: Rollover (선적 누락)
- **빈도**: 월 1-2회 (peak season 증가)
- **원인**: 선사측 오버부킹, 항만 혼잡
- **해결**: 즉시 다음 가용 선박 확인 후 re-booking. 선사 클레임 접수 (롤오버 보상 가능).
- **예방**: 스페이스 tight한 선박은 early cut-off 적용. 백업 부킹 권장.

## Issue: B/L 발급 지연
- **빈도**: 월 3-4회
- **원인**: 화주 정보 불일치, 선사 시스템 오류
- **해결**: 선사 Doc팀에 draft B/L 기준 발급 요청. Surrendered B/L 또는 Sea Waybill 전환 검토.
- **예방**: 부킹 단계에서 shipper/consignee 정보 정확도 사전 검증
```

### 4.3 Knowledge Store Operations

- **Create**: 새로운 지식 항목 추가 (JSON append / MD 섹션 추가)
- **Read**: 키워드 기반 검색 + RAG 기반 의미 검색
- **Update**: 기존 항목 수정 (JSON diff 기반 변경 이력 관리)
- **Delete**: 항목 삭제 (soft delete with `deprecated` flag)
- **Tagging**: 지식 유형(`structured` / `unstructured` / `tribal`), 도메인(`booking` / `tracking`), 우선순위 태깅
- **Versioning**: 지식 항목 변경 이력 관리

### 4.4 RAG Pipeline

```
Ingestion → Embedding → Indexing → Retrieval → Augmentation
```

1. **Ingestion**: JSON/MD 파일 로드 → 청크 분할 (LangChain `RecursiveCharacterTextSplitter`, chunk_size=500, overlap=50)
2. **Embedding**: OpenAI `text-embedding-3-small` 모델로 벡터 변환
3. **Indexing**: ChromaDB 로컬 벡터 스토어에 저장 (collection per knowledge type)
4. **Retrieval**: 사용자 쿼리 → 유사도 검색 → Top-K(k=5) 결과 반환
5. **Augmentation**: 검색 결과를 Agent 프롬프트 컨텍스트에 주입 (system message에 `<knowledge>` 블록으로)

---

## 5. Decision Trace System

Agent의 의사결정 과정을 추적하고 기록하는 시스템이다. 이를 통해 Agent가 왜 특정 결정을 내렸는지 사후 분석이 가능하며, 향후 Agent 개선의 핵심 데이터로 활용된다.

### 5.1 Trace Data Model

```json
{
  "trace_id": "dt-550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-03-07T14:32:05Z",
  "agent_id": "booking_agent",
  "session_id": "sess-abc123",
  "decision_type": "carrier_selection",
  "input_context": {
    "user_request": "부산에서 LA로 40HC 1대 부킹해줘. 4월 초 출항 희망.",
    "retrieved_knowledge": [
      { "source": "tribal/booking_tips.md", "chunk": "...", "relevance_score": 0.91 },
      { "source": "tribal/carrier_preferences.md", "chunk": "...", "relevance_score": 0.87 }
    ],
    "tool_results": {
      "search_schedules": ["VSL-2026-001", "VSL-2026-002"],
      "get_freight_rates": ["RT-2026-001", "RT-2026-002"]
    }
  },
  "reasoning_steps": [
    "Step 1: 2개 가용 스케줄 확인 — HMM PS7 (17일, $2815) vs MAERSK TP6 (14일, $3005)",
    "Step 2: Tribal knowledge 참조 — MAERSK TP6 직항이 transit time 최단, HMM PS7는 상하이 경유 지연 리스크",
    "Step 3: Carrier reliability 비교 — MAERSK A등급(92% on-time) vs HMM B+등급(85%)",
    "Step 4: 비용 차이 USD 190, transit time 3일 단축. 사용자가 납기일 미지정이므로 비용 가중치 상향.",
    "Step 5: 종합 평가 — HMM을 1순위(비용 우위), MAERSK를 2순위(속도/신뢰도 우위)로 추천"
  ],
  "decision_output": {
    "recommendation_rank": [
      { "rank": 1, "schedule_id": "VSL-2026-001", "carrier": "HMM", "reason": "비용 최적" },
      { "rank": 2, "schedule_id": "VSL-2026-002", "carrier": "MAERSK", "reason": "속도/신뢰도 최적" }
    ]
  },
  "confidence_score": 0.91,
  "tools_used": ["search_schedules", "get_freight_rates", "search_knowledge"],
  "feedback": null
}
```

### 5.2 Trace Data Fields

| Field | Type | Description |
|---|---|---|
| `trace_id` | string | UUID v4 기반 고유 식별자 (`dt-` prefix) |
| `timestamp` | datetime | ISO 8601 형식 의사결정 시각 |
| `agent_id` | string | `booking_agent` / `tracking_agent` |
| `session_id` | string | 대화 세션 식별자 (`sess-` prefix) |
| `decision_type` | string | `carrier_selection` / `route_recommendation` / `anomaly_detection` / `eta_update` 등 |
| `input_context` | object | 의사결정 시 참조한 컨텍스트 (사용자 요청, 검색된 지식, tool 결과) |
| `reasoning_steps` | array[string] | LLM 추론 단계별 기록 |
| `decision_output` | object | 최종 의사결정 결과 |
| `confidence_score` | float | 0.0~1.0 의사결정 확신도 |
| `tools_used` | array[string] | 사용된 도구/API 목록 |
| `feedback` | object\|null | 사용자 피드백 (승인/거부/수정) |

### 5.3 Trace Operations

- **Create**: Agent가 의사결정을 내릴 때마다 자동 생성
- **Append**: 동일 세션 내 후속 결정은 이전 trace를 참조하며 연결 (`parent_trace_id`)
- **Update**: 사용자 피드백 반영 시 `feedback` 필드 갱신
- **Query**: `session_id`, `agent_id`, `decision_type` 등으로 필터링 조회
- **Export**: 분석용 JSON 파일 내보내기

### 5.4 Trace Logging in CLI

모든 Decision Trace는 CLI 터미널에 실시간으로 출력된다. Rich 라이브러리를 활용하여:

- `reasoning_steps` → Rich `Tree` 형태로 렌더링
- Tool 호출 → Rich `Panel` 형태로 렌더링
- 최종 결정 → 색상 강조 + 박스 표시
- `confidence_score` < 0.7 → 경고 색상(yellow)으로 표시

---

## 6. Agent Specifications

### 6.1 International Booking Agent

해상/항공 국제물류 부킹 업무를 수행하는 Agent. 사용자의 화물 정보와 요구사항을 기반으로 최적의 선사/항공사 및 스케줄을 추천하고 부킹을 생성한다.

#### 6.1.1 Capabilities

- **스케줄 조회**: 출발항/도착항, 날짜 범위 기준 가용 스케줄 검색
- **요율 비교**: 복수 선사/항공사 운임 비교 분석 (base rate + surcharges)
- **최적 경로 추천**: transit time, cost, reliability score 기반 종합 평가
- **부킹 생성**: 선택한 스케줄로 부킹 요청 생성 (Mock API 호출)
- **Knowledge 활용**: Tribal knowledge 참조하여 선사별 특이사항 반영
- **Decision Trace 기록**: 왜 특정 선사/경로를 추천했는지 근거 기록

#### 6.1.2 Tools (LangChain)

| Tool Name | Description | Parameters |
|---|---|---|
| `search_schedules` | 출발항/도착항/날짜로 가용 스케줄 검색 | `origin: str, destination: str, date_from: str, date_to: str` |
| `get_freight_rates` | 선사별 운임 요율 조회 | `origin: str, destination: str, container_type: str` |
| `create_booking` | 부킹 요청 생성 및 확인번호 반환 | `schedule_id: str, container_type: str, quantity: int, cargo_info: dict` |
| `search_knowledge` | Knowledge Store에서 관련 지식 검색 (RAG) | `query: str, knowledge_type: str?, top_k: int?` |
| `log_decision` | Decision Trace에 의사결정 기록 | `decision_type: str, reasoning: list, output: dict` |
| `handoff_to_tracking` | 부킹 완료 후 Track/Trace Agent로 handoff | `booking_id: str, shipment_ref: str` |

#### 6.1.3 Workflow (LangGraph State Machine)

```
[INIT] → [SEARCH] → [ANALYZE] → [RECOMMEND] → [CONFIRM] → [HANDOFF]
  │         │           │            │             │           │
  │         │           │            │             │           └─ Track/Trace Agent 연동
  │         │           │            │             └─ 사용자 선택 확인 → 부킹 생성
  │         │           │            └─ 상위 3개 옵션 + 근거 제시
  │         │           └─ Knowledge 검색 → 선사별 평판/특이사항 반영
  │         └─ 스케줄 + 요율 동시 조회
  └─ 사용자 요청 수신 → 화물 정보 파싱
```

**State Schema (LangGraph)**:

```python
from typing import TypedDict, Annotated, Literal

class BookingState(TypedDict):
    messages: Annotated[list, add_messages]
    user_request: str
    cargo_info: dict | None
    schedules: list[dict]
    rates: list[dict]
    knowledge_context: list[dict]
    recommendations: list[dict]
    selected_option: dict | None
    booking_result: dict | None
    decision_traces: list[str]  # trace_ids
    current_step: Literal["init", "search", "analyze", "recommend", "confirm", "handoff", "done"]
```

### 6.2 Track & Trace Agent

부킹된 화물의 실시간 위치 추적, 상태 모니터링, 이상 감지를 수행하는 Agent.

#### 6.2.1 Capabilities

- **화물 추적**: B/L 번호 또는 컨테이너 번호 기반 실시간 위치/상태 조회
- **ETA 업데이트**: 예상 도착일 변동 감지 및 알림
- **이상 감지**: 지연, 경로 이탈, 롤오버 등 이상 상황 식별
- **이력 조회**: 화물의 전체 이동 이력(milestone events) 제공
- **Knowledge 활용**: 항만 혼잡도, 기상 정보 등 참조
- **Booking Agent 연동**: 부킹 정보 기반 자동 추적 설정

#### 6.2.2 Tools (LangChain)

| Tool Name | Description | Parameters |
|---|---|---|
| `track_shipment` | B/L 또는 컨테이너 번호로 현재 위치/상태 조회 | `reference: str, ref_type: Literal["bl", "container"]` |
| `get_milestones` | 화물 이동 이력(milestone events) 전체 조회 | `reference: str` |
| `check_anomalies` | ETA 지연, 경로 이탈 등 이상 감지 로직 실행 | `shipment_id: str` |
| `search_knowledge` | 항만 혼잡도, 규정 변경 등 관련 지식 검색 (RAG) | `query: str, knowledge_type: str?, top_k: int?` |
| `log_decision` | 이상 감지 판단 근거를 Decision Trace에 기록 | `decision_type: str, reasoning: list, output: dict` |
| `notify_stakeholder` | 이상 상황 발생 시 관련자 알림 (Mock) | `shipment_id: str, event_type: str, message: str` |

#### 6.2.3 Sample Tracking Data — `tracking_events.json`

```json
[
  {
    "shipment_id": "SHP-2026-001",
    "booking_id": "BK-2026-001",
    "bl_number": "HDMU1234567",
    "container_number": "HDMU1234567",
    "carrier": "HMM",
    "current_status": "IN_TRANSIT",
    "current_location": {
      "lat": 33.12,
      "lon": 140.56,
      "description": "Pacific Ocean, East of Japan"
    },
    "eta_original": "2026-04-18T14:00:00Z",
    "eta_current": "2026-04-19T08:00:00Z",
    "eta_delay_hours": 18,
    "milestones": [
      { "event": "BOOKING_CONFIRMED", "timestamp": "2026-03-25T10:00:00Z", "location": "KRPUS", "detail": "Booking confirmed: BK-2026-001" },
      { "event": "GATE_IN", "timestamp": "2026-03-30T09:30:00Z", "location": "KRPUS/HBCT", "detail": "Container gated in at HBCT" },
      { "event": "LOADED", "timestamp": "2026-04-01T06:00:00Z", "location": "KRPUS/HBCT", "detail": "Loaded on vessel HMM ALGECIRAS" },
      { "event": "DEPARTED", "timestamp": "2026-04-01T08:00:00Z", "location": "KRPUS", "detail": "Vessel departed Busan" },
      { "event": "TRANSSHIPMENT_ARRIVAL", "timestamp": "2026-04-05T14:00:00Z", "location": "CNSHA", "detail": "Arrived at Shanghai for transshipment" },
      { "event": "TRANSSHIPMENT_DEPARTURE", "timestamp": "2026-04-07T02:00:00Z", "location": "CNSHA", "detail": "Departed Shanghai (delayed 20h due to congestion)" },
      { "event": "IN_TRANSIT", "timestamp": "2026-04-10T00:00:00Z", "location": "Pacific Ocean", "detail": "In transit to Los Angeles" }
    ],
    "anomalies": [
      {
        "type": "ETA_DELAY",
        "detected_at": "2026-04-07T03:00:00Z",
        "severity": "MEDIUM",
        "description": "Shanghai transshipment 20시간 지연으로 ETA 18시간 후연",
        "root_cause": "Shanghai port congestion"
      }
    ]
  }
]
```

### 6.3 Multi-Agent Collaboration

두 Agent 간의 협업은 LangGraph의 상태 기반 라우팅으로 구현한다.

#### Orchestrator Architecture

```python
from langgraph.graph import StateGraph, END

class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: Literal["orchestrator", "booking", "tracking"]
    shared_context: dict  # booking_id, shipment_ref 등 Agent 간 공유 데이터
    handoff_payload: dict | None
    session_id: str

# Graph 구성
graph = StateGraph(OrchestratorState)
graph.add_node("router", route_intent)         # 의도 분석 → Agent 라우팅
graph.add_node("booking", booking_agent_node)   # Booking Agent 실행
graph.add_node("tracking", tracking_agent_node) # Tracking Agent 실행
graph.add_edge("router", "booking")             # conditional edge
graph.add_edge("router", "tracking")            # conditional edge
graph.add_edge("booking", "router")             # handoff 또는 완료 시 복귀
graph.add_edge("tracking", "router")
```

#### Handoff Protocol

- **Booking → Tracking**: 부킹 완료 시 `booking_id`, `shipment_ref`, `schedule_info`를 `shared_context`에 저장
- **Tracking → Booking**: 이상 감지 시 re-booking 필요하면 `anomaly_info`와 함께 Booking Agent로 handoff
- **Context Preservation**: LangGraph State에서 `shared_context` dict로 Agent 간 데이터 공유
- **Cross-reference**: Tracking Agent가 Booking Agent의 Decision Trace를 `parent_trace_id`로 참조 가능

---

## 7. CLI Interface & Logging

### 7.1 CLI Design

Python Rich 라이브러리를 활용한 터미널 기반 대화형 인터페이스를 구현한다.

#### 7.1.1 Main Features

- **Interactive Chat**: 사용자와 Agent 간 자연어 대화
- **Agent Indicator**: 현재 활성 Agent 표시 (색상 구분 — Booking: blue, Tracking: green)
- **Command Support**: `/help`, `/switch`, `/trace` 등 슬래시 커맨드
- **Split View**: 상단에 Agent 응답, 하단에 실시간 로그 (Rich `Layout` 활용)

#### 7.1.2 Slash Commands

| Command | Description |
|---|---|
| `/help` | 사용 가능한 명령어 및 Agent 기능 안내 |
| `/switch [agent]` | Booking/Tracking Agent 간 수동 전환 |
| `/trace [id]` | 특정 Decision Trace 상세 조회 |
| `/traces` | 현재 세션의 모든 Decision Trace 목록 |
| `/knowledge` | Knowledge Store 현황 조회 |
| `/log [level]` | 로그 출력 레벨 변경 (`DEBUG` / `INFO` / `WARN`) |
| `/export` | 현재 세션의 대화 이력 및 Trace 내보내기 |
| `/exit` | 세션 종료 |

#### 7.1.3 CLI Example Session

```
╔══════════════════════════════════════════════════════════════╗
║  Logistics AI Agent PoC v1.0          [Session: sess-abc123] ║
║  Active Agent: 🚢 Booking Agent                             ║
╚══════════════════════════════════════════════════════════════╝

You > 부산에서 LA로 40HC 1대 부킹해줘. 4월 초 출항 희망.

─── Agent Log ─────────────────────────────────────────────────
[14:32:01] [INFO] [booking] [parse] Parsed: origin=KRPUS, dest=USLAX, type=40HC, qty=1, date=2026-04
[14:32:01] [INFO] [booking] [tool]  Calling search_schedules(origin=KRPUS, dest=USLAX, ...)
[14:32:02] [INFO] [booking] [tool]  → Found 2 schedules
[14:32:02] [INFO] [booking] [tool]  Calling get_freight_rates(origin=KRPUS, dest=USLAX, type=40HC)
[14:32:03] [INFO] [booking] [rag]   Searching knowledge: "KRPUS USLAX 40HC booking tips"
[14:32:03] [INFO] [booking] [rag]   → Found 3 relevant items (best score: 0.91)
[14:32:05] [INFO] [booking] [decision] Trace created: carrier_selection (confidence: 0.91)
────────────────────────────────────────────────────────────────

🚢 Booking Agent >
부산(KRPUS) → LA(USLAX) 40HC 1대 부킹을 위해 조회한 결과입니다:

┌─── Recommendation ─────────────────────────────────────────┐
│ #1 HMM — PS7 서비스                                        │
│    Vessel: HMM ALGECIRAS (0125E)                           │
│    ETD: Apr 01 → ETA: Apr 18 (17일)                        │
│    Total: USD 2,815                                         │
│    ⚡ 비용 최적. 단, 상하이 경유 지연 리스크 있음.             │
│                                                             │
│ #2 MAERSK — TP6 서비스                                      │
│    Vessel: MAERSK EDINBURGH (206W)                          │
│    ETD: Apr 03 → ETA: Apr 17 (14일)                        │
│    Total: USD 3,005                                         │
│    ⚡ 직항, 최단 transit time. 신뢰도 A등급.                  │
└─────────────────────────────────────────────────────────────┘

💡 참고: 4월은 peak season 진입기로 스페이스 조기 마감 가능성이 있습니다.
   HMM PS7는 상하이 경유 시 환적 지연 리스크가 있으니 참고하세요.

어떤 옵션으로 부킹할까요? (1 또는 2)

You > 1번으로 해줘

─── Agent Log ─────────────────────────────────────────────────
[14:32:15] [INFO] [booking] [tool]  Calling create_booking(schedule=VSL-2026-001, type=40HC, qty=1)
[14:32:16] [INFO] [booking] [tool]  → Booking confirmed: BK-2026-001
[14:32:16] [INFO] [booking] [decision] Trace updated: feedback=APPROVED
────────────────────────────────────────────────────────────────

🚢 Booking Agent >
✅ 부킹이 완료되었습니다!

  Booking No: BK-2026-001
  Vessel: HMM ALGECIRAS (0125E)
  ETD: 2026-04-01 / ETA: 2026-04-18
  Doc Cut-off: 2026-03-29 17:00
  Cargo Cut-off: 2026-03-30 12:00

화물 추적을 자동으로 설정할까요?
```

### 7.2 Logging System

개발 참조용으로 Agent의 모든 처리 과정을 터미널에 실시간 로깅한다.

#### 7.2.1 Log Levels & Output

| Level | Color | Content |
|---|---|---|
| `DEBUG` | Gray | LLM prompt/response 전문, embedding 벡터, raw API 호출 상세 |
| `INFO` | Cyan | Agent 전환, Tool 호출/결과, Knowledge 검색 결과, Decision Trace 생성 |
| `WARN` | Yellow | API 지연 (>3s), 낮은 confidence score (<0.7), fallback 동작 |
| `ERROR` | Red | API 실패, JSON 파싱 오류, 예외 발생 |

#### 7.2.2 Log Format

```
[{timestamp}] [{level}] [{agent_id}] [{component}] {message}
```

Components: `parse`, `tool`, `rag`, `decision`, `llm`, `handoff`, `system`

#### 7.2.3 Implementation Notes

- 기본 로그 레벨: `INFO` (CLI에서 `/log DEBUG`로 변경 가능)
- Rich `Console` 의 `log()` 메서드 활용하여 자동 타임스탬프 + 색상 처리
- Agent 응답 영역과 로그 영역을 시각적으로 구분 (Rich `Rule` 또는 `Panel`)
- 로그는 파일(`logs/session_{session_id}.log`)에도 동시 저장

---

## 8. Mock API Layer

PoC 단계에서 외부 시스템 연동을 시뮬레이션하기 위한 Mock API를 구현한다. 모든 Mock API는 Python 함수로 구현되며, JSON 데이터 파일을 기반으로 응답한다.

### 8.1 Mock API List

| API | Functions | Data Source |
|---|---|---|
| Carrier Schedule API | `search()`, `get_detail()` | `data/structured/vessel_schedules.json` |
| Freight Rate API | `get_rates()`, `compare()` | `data/structured/freight_rates.json` |
| Booking API | `create()`, `confirm()`, `cancel()` | Runtime generated (in-memory) |
| Tracking API | `get_status()`, `get_milestones()` | `data/structured/tracking_events.json` |
| Port/Terminal API | `get_info()`, `congestion_level()` | `data/structured/port_info.json` |

### 8.2 Simulation Features

- **Latency Simulation**: `asyncio.sleep(random.uniform(0.5, 2.0))` — 실제 API 호출처럼 랜덤 지연 (configurable via `config.py`)
- **Error Injection**: `MOCK_ERROR_RATE` 설정 (default: 5%) — 일정 확률로 에러 응답 반환하여 Agent의 에러 핸들링 검증
- **Data Variability**: 동일 조회에도 약간의 데이터 변동 (가격 ±3%, 스페이스 변동 등)
- **State Management**: 부킹 생성 후 해당 부킹의 tracking 데이터 자동 생성 (in-memory state)

### 8.3 Mock API Interface

```python
from abc import ABC, abstractmethod

class BaseMockAPI(ABC):
    """모든 Mock API의 베이스 클래스"""

    def __init__(self, data_path: str, error_rate: float = 0.05, latency_range: tuple = (0.5, 2.0)):
        self.data = self._load_data(data_path)
        self.error_rate = error_rate
        self.latency_range = latency_range

    async def _simulate_latency(self):
        """실제 API 지연 시뮬레이션"""
        await asyncio.sleep(random.uniform(*self.latency_range))

    def _maybe_raise_error(self):
        """설정된 확률로 에러 발생"""
        if random.random() < self.error_rate:
            raise MockAPIError("Simulated API error")

    @abstractmethod
    def _load_data(self, path: str) -> dict:
        pass
```

---

## 9. Evaluation & Testing Framework

### 9.1 Test Scenarios

Agent의 성능을 평가하기 위한 시나리오 기반 테스트 프레임워크를 구성한다. 각 시나리오는 JSON 파일로 정의된다.

#### 9.1.1 Scenario Format — `booking_scenarios.json`

```json
[
  {
    "scenario_id": "BK-EVAL-001",
    "name": "기본 부킹 — 부산→LA 일반화물",
    "description": "가장 기본적인 부킹 시나리오. Agent가 스케줄 조회, 요율 비교, 추천, 부킹 생성까지 완료해야 함.",
    "user_messages": [
      "부산에서 LA로 40HC 1대 부킹해줘. 4월 초 출항 희망.",
      "1번으로 해줘"
    ],
    "expected_behaviors": [
      "search_schedules 호출",
      "get_freight_rates 호출",
      "search_knowledge 호출 (tribal knowledge 참조)",
      "최소 2개 옵션 추천",
      "각 옵션의 비용, transit time, 특이사항 제시",
      "Decision Trace 생성 (carrier_selection type)",
      "create_booking 호출",
      "부킹 확인 정보 제공"
    ],
    "evaluation_criteria": {
      "task_completion": "부킹 확인번호가 반환되어야 함",
      "decision_quality": "추천 근거에 cost, transit_time, reliability 중 최소 2개 포함",
      "knowledge_utilization": "tribal knowledge에서 관련 팁이 1개 이상 반영",
      "max_tool_calls": 8,
      "max_latency_seconds": 30
    }
  },
  {
    "scenario_id": "BK-EVAL-002",
    "name": "긴급 부킹 — 최단 transit time 우선",
    "description": "고객이 납기가 촉박하여 가장 빠른 출항을 요청하는 시나리오.",
    "user_messages": [
      "급한 화물인데 부산에서 LA까지 가장 빨리 갈 수 있는 방법 찾아줘. 40GP 2대.",
      "가장 빠른 걸로 바로 부킹해줘"
    ],
    "expected_behaviors": [
      "transit_time 기준 정렬된 추천",
      "직항 서비스 우선 추천",
      "긴급 부킹 시 주의사항 안내 (cut-off 임박 여부 등)",
      "Decision Trace에 '속도 우선' 근거 기록"
    ],
    "evaluation_criteria": {
      "task_completion": "가장 짧은 transit time 옵션이 1순위로 추천되어야 함",
      "decision_quality": "transit time이 추천 근거의 핵심 요소여야 함",
      "knowledge_utilization": "optional",
      "max_tool_calls": 8,
      "max_latency_seconds": 30
    }
  },
  {
    "scenario_id": "BK-EVAL-003",
    "name": "Knowledge 활용 — Tribal knowledge 기반 의사결정",
    "description": "Tribal knowledge의 선사별 팁이 추천에 반영되는지 검증.",
    "user_messages": [
      "부산에서 LA로 reefer 컨테이너 1대 부킹 가능할까? 다음 달 출항이면 좋겠어."
    ],
    "expected_behaviors": [
      "search_knowledge에서 reefer 관련 tribal knowledge 검색",
      "'reefer는 최소 2주 전 사전 부킹 필요' 팁 반영",
      "장비 부족 관련 notice 참조",
      "추천 시 reefer 가용성 중심 평가"
    ],
    "evaluation_criteria": {
      "task_completion": "reefer 가용 옵션을 제시해야 함",
      "decision_quality": "reefer 특수사항이 추천 근거에 포함",
      "knowledge_utilization": "tribal knowledge에서 reefer 관련 팁 최소 1개 반영 필수",
      "max_tool_calls": 10,
      "max_latency_seconds": 30
    }
  }
]
```

#### 9.1.2 Tracking Scenarios — `tracking_scenarios.json`

```json
[
  {
    "scenario_id": "TK-EVAL-001",
    "name": "정상 추적 — 스케줄대로 이동 중",
    "user_messages": ["HDMU1234567 컨테이너 지금 어디있어?"],
    "expected_behaviors": ["track_shipment 호출", "현재 위치/상태 제공", "ETA 정보 제공"],
    "evaluation_criteria": { "task_completion": "현재 위치와 ETA가 정확히 반환" }
  },
  {
    "scenario_id": "TK-EVAL-002",
    "name": "이상 감지 — ETA 지연",
    "user_messages": ["BK-2026-001 부킹 화물 추적해줘. 이상 있으면 알려줘."],
    "expected_behaviors": [
      "track_shipment + check_anomalies 호출",
      "상하이 환적 지연 감지",
      "ETA 변경 사항 안내",
      "Decision Trace에 anomaly_detection 기록",
      "지연 원인(port congestion) 설명"
    ],
    "evaluation_criteria": {
      "task_completion": "이상 상황이 감지되고 원인이 설명되어야 함",
      "decision_quality": "anomaly severity와 root cause가 trace에 포함"
    }
  },
  {
    "scenario_id": "TK-EVAL-003",
    "name": "Multi-Agent — Booking→Tracking 자동 연동",
    "user_messages": [
      "부산에서 LA로 40HC 1대 부킹해줘",
      "1번으로 해줘",
      "네, 추적도 설정해줘",
      "지금 상태 어때?"
    ],
    "expected_behaviors": [
      "Booking Agent가 부킹 완료",
      "handoff_to_tracking으로 Tracking Agent 전환",
      "shared_context에 booking_id 전달",
      "Tracking Agent가 자동으로 해당 부킹 추적"
    ],
    "evaluation_criteria": {
      "task_completion": "부킹→추적 전환이 seamless하게 이루어져야 함",
      "handoff_accuracy": "booking_id, shipment_ref가 정확히 전달"
    }
  }
]
```

### 9.2 Evaluation Metrics

| Metric | Description | Measurement |
|---|---|---|
| Task Completion | 시나리오의 목표를 성공적으로 달성했는지 | pass/fail |
| Decision Quality | Decision Trace의 reasoning이 논리적이고 근거 충분한지 | 0-5 score (LLM-as-Judge) |
| Knowledge Utilization | RAG 검색 결과를 의사결정에 적절히 활용했는지 | 0-5 score |
| Response Latency | 사용자 입력 → Agent 응답까지 소요 시간 | seconds |
| Tool Usage Efficiency | 불필요한 도구 호출 없이 효율적 수행 | actual_calls / expected_calls |
| Handoff Accuracy | Agent 간 전환 시 컨텍스트 정확 전달 여부 | pass/fail |

### 9.3 Test Execution

```bash
# 전체 테스트 실행
python -m pytest tests/ -v

# 평가 프레임워크 실행 (시나리오 기반)
python -m pytest tests/eval_framework.py --eval-report

# 특정 Agent만 테스트
python -m pytest tests/test_booking_agent.py -v
python -m pytest tests/test_tracking_agent.py -v

# 평가 리포트 출력 (JSON)
python tests/eval_framework.py --output reports/eval_result.json
```

---

## 10. Implementation Milestones

PoC 프로젝트는 4주 스프린트로 진행하며, 각 마일스톤별 산출물을 정의한다.

| Week | Phase | Deliverables | Key Focus |
|---|---|---|---|
| Week 1 | Foundation | 프로젝트 셋업, Knowledge Store 구현, 샘플 데이터 생성, Mock API 구현 | Data layer & infrastructure |
| Week 2 | Core Agents | Booking Agent, Tracking Agent 개별 구현, RAG pipeline, Decision Trace 시스템 | Single agent functionality |
| Week 3 | Integration | Multi-agent orchestration, CLI interface, Logging system, Agent 간 handoff | Agent collaboration & UX |
| Week 4 | Polish & Eval | 테스트 시나리오 작성, Evaluation framework, 문서화, 데모 준비 | Quality & demonstration |

### Week별 상세 태스크

#### Week 1 — Foundation
- [ ] 프로젝트 초기화 (pyproject.toml, .env, directory structure)
- [ ] `data/` 디렉토리에 모든 샘플 데이터(JSON/MD) 생성
- [ ] `src/knowledge/store.py` — Knowledge CRUD 구현
- [ ] `src/knowledge/loader.py` — JSON/MD 파일 로더 구현
- [ ] `src/mock_api/` — 5개 Mock API 구현 (BaseMockAPI 상속)
- [ ] `src/config.py` — 환경변수 및 설정 관리
- [ ] 기본 pytest 환경 구성

#### Week 2 — Core Agents
- [ ] `src/knowledge/rag.py` — RAG pipeline (ChromaDB + OpenAI Embedding)
- [ ] `src/decision/models.py` — Decision Trace Pydantic 모델
- [ ] `src/decision/trace.py` — Trace CRUD 구현
- [ ] `src/tools/` — LangChain Tool 정의 (booking, tracking, knowledge)
- [ ] `src/agents/booking_agent.py` — Booking Agent (LangGraph StateGraph)
- [ ] `src/agents/tracking_agent.py` — Tracking Agent (LangGraph StateGraph)
- [ ] 단위 테스트 작성

#### Week 3 — Integration
- [ ] `src/agents/orchestrator.py` — Multi-agent router (LangGraph)
- [ ] Agent 간 handoff protocol 구현
- [ ] `src/cli/app.py` — CLI 메인 루프 (Rich 기반)
- [ ] `src/cli/renderer.py` — Agent 응답 렌더링
- [ ] `src/cli/logger.py` — 실시간 로깅 시스템
- [ ] 슬래시 커맨드 구현 (/help, /switch, /trace 등)
- [ ] 통합 테스트 작성

#### Week 4 — Polish & Eval
- [ ] `tests/scenarios/` — 평가 시나리오 JSON 작성
- [ ] `tests/eval_framework.py` — 자동화 평가 프레임워크
- [ ] Edge case 처리 (에러 핸들링, timeout, fallback)
- [ ] README.md 작성 (설치, 실행, 사용법)
- [ ] 데모 시나리오 준비 및 리허설
- [ ] 최종 코드 리뷰 및 정리

---

## 11. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| LLM Hallucination | High | Medium | RAG 기반 Ground Truth 제공, Decision Trace로 검증 가능성 확보, System prompt에 strict instruction |
| OpenAI API Rate Limit | Medium | Low | GPT-4o-mini fallback, response 캐싱, 요청 배치 처리 |
| Agent Infinite Loop | Medium | Medium | LangGraph에 `recursion_limit` 설정 (default: 25), step-level timeout 처리 |
| Knowledge Staleness | Low | Low | PoC에서는 정적 데이터 사용, timestamp 기반 유효성 체크 로직 포함 |
| Context Window Overflow | Medium | Medium | RAG Top-K 제한 (k=5), 대화 이력 summarization 적용, token counting |
| ChromaDB Performance | Low | Low | PoC 규모 데이터는 local ChromaDB로 충분, 인덱스 최적화 불필요 |

---

## 12. Configuration Reference

### `.env.example`

```env
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_MODEL_FALLBACK=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG
CHROMA_PERSIST_DIR=./data/chroma
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=5

# Mock API
MOCK_LATENCY_MIN=0.5
MOCK_LATENCY_MAX=2.0
MOCK_ERROR_RATE=0.05

# Logging
LOG_LEVEL=INFO
LOG_FILE_DIR=./logs

# Agent
AGENT_MAX_ITERATIONS=25
AGENT_TIMEOUT_SECONDS=60
```

---

## 13. Appendix

### 13.1 Glossary

| Term | Definition |
|---|---|
| B/L | Bill of Lading — 선하증권 |
| TEU | Twenty-foot Equivalent Unit — 20피트 컨테이너 기준 단위 |
| 20GP / 40GP / 40HC | 컨테이너 타입 (20ft General Purpose / 40ft GP / 40ft High Cube) |
| Tribal Knowledge | 조직 내 비공식적으로 공유되는 경험 기반 암묵지 |
| Decision Trace | Agent의 의사결정 과정을 추적한 기록 |
| RAG | Retrieval-Augmented Generation — 검색 증강 생성 |
| Rollover | 선적 누락으로 다음 선박으로 이관되는 상황 |
| THC | Terminal Handling Charge — 터미널 화물 처리 비용 |
| BAF / CAF | Bunker / Currency Adjustment Factor — 유류/환율 할증료 |
| ETD / ETA | Estimated Time of Departure / Arrival — 출항/도착 예정일 |
| Cut-off | 서류/화물 마감 시한 |
| VGM | Verified Gross Mass — 컨테이너 총 중량 확인 |

### 13.2 References

- LangChain Documentation: https://python.langchain.com/docs/
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- OpenAI API Reference: https://platform.openai.com/docs/
- ChromaDB Documentation: https://docs.trychroma.com/
- Rich Library: https://rich.readthedocs.io/

---

*End of Document*
