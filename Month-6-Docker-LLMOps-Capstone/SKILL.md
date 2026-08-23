---
name: month-6-docker-llmops-capstone
description: >
  Month 6 of the 6-Month AI Engineering Roadmap.
  Covers Docker, Redis, PostgreSQL, LLMOps, Monitoring, AI System Design,
  Capstone project, and Interview Preparation.
  Goal: Build a Production Multi-LLM AI Platform.
tags:
  - docker
  - redis
  - postgresql
  - llmops
  - monitoring
  - system-design
  - capstone
  - interview
  - month-6
---

# 🚀 Month 6 — Docker · Redis · PostgreSQL · LLMOps · Monitoring · AI System Design · Capstone

> **Goal for this month:** Deploy a production-grade Multi-LLM AI Platform and land the job.

---

## 📅 Learning Path

```
Docker & Containers (Dockerfile, Compose, images)
      ↓
Redis (caching, sessions, queues, rate limiting)
      ↓
PostgreSQL (advanced: JSONB, pgvector, migrations)
      ↓
LLMOps (Langfuse, cost tracking, CI/CD)
      ↓
Monitoring (Prometheus, Grafana, alerting)
      ↓
AI System Design (interview patterns)
      ↓
🏗️ Capstone: Production Multi-LLM AI Platform
      ↓
Interview Preparation
```

---

## 📁 Folder Structure

```
Month-6-Docker-LLMOps-Capstone/
├── SKILL.md              ← You are here
├── concepts/
│   ├── docker.md
│   ├── redis_patterns.md
│   ├── postgresql_advanced.md
│   ├── llmops.md
│   ├── monitoring.md
│   └── ai_system_design.md
└── practice/
    └── production_platform/ ← 🏗️ Capstone project
```

---

## ⚡ Phase 1 — Docker for AI Services

### Optimized Dockerfile
```dockerfile
# Use slim base — minimizes image size
FROM python:3.11-slim

# Security: don't run as root
RUN useradd --create-home appuser

WORKDIR /app

# Install deps first (layer caching — only re-runs if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY --chown=appuser:appuser . .

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### docker-compose for Full AI Stack
```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/aidb
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: aidb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrantdata:/qdrant/storage

  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/langfuse
    depends_on:
      - db

volumes:
  pgdata:
  redisdata:
  qdrantdata:
```

---

## ⚡ Phase 2 — Redis Patterns for AI

### Semantic Cache
```python
import redis.asyncio as redis
import json, hashlib

class SemanticCache:
    def __init__(self, redis_url: str, similarity_threshold: float = 0.95):
        self.redis = redis.from_url(redis_url)
        self.threshold = similarity_threshold

    async def get(self, query: str) -> str | None:
        query_embedding = await embed(query)
        # Search cached queries by cosine similarity
        cached_keys = await self.redis.keys("cache:query:*")
        for key in cached_keys:
            cached = json.loads(await self.redis.get(key))
            sim = cosine_similarity(query_embedding, cached["embedding"])
            if sim >= self.threshold:
                return cached["response"]
        return None

    async def set(self, query: str, response: str, ttl: int = 3600):
        embedding = await embed(query)
        key = f"cache:query:{hashlib.md5(query.encode()).hexdigest()}"
        await self.redis.setex(key, ttl, json.dumps({"embedding": embedding, "response": response}))
```

### Rate Limiting (Token Bucket)
```python
async def check_rate_limit(user_id: str, limit: int = 100, window: int = 3600) -> bool:
    key = f"rate:{user_id}:{int(time.time() // window)}"
    current = await redis_client.incr(key)
    if current == 1:
        await redis_client.expire(key, window)
    return current <= limit

# Session management
async def save_session(session_id: str, messages: list, ttl: int = 86400):
    await redis_client.setex(f"session:{session_id}", ttl, json.dumps(messages))

async def get_session(session_id: str) -> list:
    data = await redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else []
```

### Background Task Queue (Redis + asyncio)
```python
import asyncio
from redis.asyncio import Redis

async def enqueue_task(redis: Redis, queue: str, task: dict):
    await redis.lpush(queue, json.dumps(task))

async def worker(redis: Redis, queue: str):
    while True:
        _, task_data = await redis.brpop(queue)
        task = json.loads(task_data)
        await process_task(task)

# Run workers
asyncio.create_task(worker(redis, "document_ingestion"))
```

---

## ⚡ Phase 3 — PostgreSQL Advanced

### AI-Specific Schema
```sql
-- Users with usage tracking
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    tier VARCHAR(20) DEFAULT 'free',  -- free, pro, enterprise
    monthly_token_budget INTEGER DEFAULT 100000,
    tokens_used_this_month INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversations with full history
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title TEXT,
    messages JSONB DEFAULT '[]'::jsonb,
    model_used VARCHAR(100),
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10,6) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for JSONB queries
CREATE INDEX idx_conversations_user ON conversations(user_id, created_at DESC);

-- LLM request audit log
CREATE TABLE llm_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    provider VARCHAR(50) NOT NULL,     -- openai, claude, gemini
    model VARCHAR(100) NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd DECIMAL(10,6),
    latency_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Partitioned by month for performance
CREATE TABLE llm_requests_2026_01 PARTITION OF llm_requests
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

---

## ⚡ Phase 4 — LLMOps with Langfuse

### What is LLMOps?
The MLOps equivalent for LLM applications:
- **Trace** every LLM call (input, output, tokens, cost, latency)
- **Evaluate** responses automatically (LLM-as-judge)
- **Monitor** quality degradation over time
- **A/B test** prompts in production

### Langfuse Integration
```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host="http://localhost:3000"
)

@observe()  # auto-traces this function
async def generate_response(user_message: str, user_id: str) -> str:
    langfuse_context.update_current_trace(
        user_id=user_id,
        tags=["production", "chat"]
    )
    
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_message}]
    )
    
    # Log usage
    langfuse_context.update_current_observation(
        usage={"input": response.usage.prompt_tokens,
               "output": response.usage.completion_tokens}
    )
    
    return response.choices[0].message.content
```

### CI/CD Prompt Regression Tests
```python
# tests/test_prompts.py — runs in CI
import pytest
from langfuse import Langfuse

@pytest.mark.asyncio
async def test_rag_quality():
    """Ensure RAG answers are grounded in context."""
    questions = load_test_questions("tests/fixtures/rag_questions.json")
    
    for q in questions:
        result = await rag_pipeline(q["question"])
        
        # LLM-as-judge evaluation
        score = await evaluate_faithfulness(result.answer, result.context)
        assert score >= 0.8, f"Low faithfulness score for: {q['question']}"
```

---

## ⚡ Phase 5 — Monitoring

### Production Monitoring Checklist
- [ ] Token usage per user/feature (cost tracking)
- [ ] Latency p50/p95/p99 per provider
- [ ] Error rates per provider (for circuit breaker)
- [ ] Response quality scores (LLM-as-judge)
- [ ] Hallucination detection rate
- [ ] Prompt regression tests in CI

### Prometheus + Grafana Setup
```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Define metrics
llm_requests_total = Counter("llm_requests_total", "Total LLM requests",
                             ["provider", "model", "status"])
llm_latency = Histogram("llm_latency_seconds", "LLM request latency",
                        ["provider", "model"],
                        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
llm_tokens_total = Counter("llm_tokens_total", "Total tokens used",
                           ["provider", "model", "type"])

# Middleware to record metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    # Record to Prometheus...
    return response

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## ⚡ Phase 6 — AI System Design (Interview Framework)

### The 7-Step Framework (Use for Every Design Question)
```
1. Clarify requirements
   - Functional: what must it do?
   - Non-functional: scale, latency, cost, availability?

2. Identify AI components
   - What ACTUALLY needs AI vs what can be rule-based?

3. Draw data flow (end-to-end pipeline)
   - Input → Processing → AI → Output → Storage

4. Model selection
   - Justify: cost vs quality vs latency trade-offs

5. Scalability
   - 10x, 100x load → caching, async, horizontal scaling

6. Cost estimation
   - $/1K requests → model routing strategy

7. Reliability + Monitoring
   - Fallback chain, circuit breaker, SLAs, alerting
```

### Classic Design: AI Customer Support (18 LPA Level)
```
User Message
    ↓
Intent Classifier (fast, cheap model)
    ↓
Router
  ├── High confidence (>0.9) → RAG KB → GPT-4o → Streaming response
  ├── Medium confidence → Human escalation queue (Redis pub/sub)
  └── Low confidence  → Clarifying question (GPT-3.5)

State:
  - Redis: session, rate limiting
  - PostgreSQL: conversation history, audit log
  - Qdrant: knowledge base embeddings

Monitoring:
  - Langfuse: traces per conversation
  - Prometheus: latency, error rate
  - PagerDuty: alerts on p99 > 5s
```

### Design: RAG Platform (Multi-tenant)
```
[Ingestion API] → S3 storage → [Worker Queue] → Chunking → Embedding
                                                         → Qdrant (namespace per tenant)

[Query API] → Auth → Rate limit (Redis) → Hybrid search → Rerank → LLM → Response
           ↘ Semantic cache (Redis) ↗

Multi-tenancy: namespace isolation in Qdrant, row-level security in PostgreSQL
Cost: track per tenant, bill accordingly
```

---

## 🏗️ Capstone: Production Multi-LLM AI Platform

### What to Build
A complete AI platform combining everything from months 1-5:

| Component | Tech Stack |
|---|---|
| API Gateway | FastAPI + async |
| Multi-LLM Router | Month 3 work + circuit breaker |
| RAG Engine | Month 4 work + Qdrant |
| AI Agents | Month 5 LangGraph work |
| Auth & Users | JWT + PostgreSQL |
| Caching | Redis semantic cache |
| Observability | Langfuse + Prometheus + Grafana |
| Deployment | Docker Compose / Kubernetes |

### Platform Features
- [ ] Multi-provider chat (OpenAI, Claude, Gemini, Groq)
- [ ] RAG knowledge base (upload + query documents)
- [ ] AI Agent mode (web search, code execution)
- [ ] User auth + usage tracking + billing
- [ ] Real-time streaming for all responses
- [ ] Semantic cache (Redis)
- [ ] Admin dashboard (cost/usage metrics)
- [ ] Full observability (Langfuse + Prometheus)
- [ ] CI/CD pipeline with prompt tests
- [ ] Kubernetes-ready deployment

---

## 🎯 Interview Preparation

### Behavioral Questions (STAR Format)
- "Tell me about an AI system you built in production" → Capstone project
- "How did you handle LLM hallucinations?" → RAG + faithfulness evaluation
- "How did you optimize LLM costs at scale?" → Multi-LLM routing + semantic cache
- "Describe a time your RAG pipeline failed" → Evaluation-driven debugging

### Technical Deep Dives
- Explain attention mechanism and why transformers replaced RNNs
- How does RAG differ from fine-tuning? When to use each?
- Design a multi-tenant RAG system for 10K users
- How would you add streaming to an LLM API?
- What is MCP and how does it differ from function calling?
- How do you evaluate LLM output quality in production?

### System Design Questions (18 LPA Level)
- "Design an AI-powered customer support system (100K queries/day)"
- "Design a document Q&A platform with multi-tenancy"
- "How would you build a code review AI agent?"
- "Design the infrastructure for serving a fine-tuned LLM at scale"
- "Design a cost-efficient multi-LLM router"

### Portfolio Checklist (Before Interviews)
- [ ] All 6 monthly projects on GitHub with clean READMEs
- [ ] Capstone live demo (deploy to Railway / Render / AWS)
- [ ] 2-minute project walkthrough prepared
- [ ] System design diagram for capstone (Excalidraw)
- [ ] Blog post or LinkedIn article about one project

---

## 📚 Resources
- [Langfuse Docs](https://langfuse.com/docs)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Chip Huyen — AI Engineering (book)](https://www.oreilly.com/library/view/ai-engineering/9781098166298)
- [The System Design Interview — Alex Xu](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF)
- [AI Engineering Job Board — Chip Huyen's newsletter](https://huyenchip.com)
