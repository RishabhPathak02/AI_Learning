---
name: month-1-python-fastapi-sql
description: >
  Month 1 of the 6-Month AI Engineering Roadmap.
  Covers Python fundamentals, Async Python, Pydantic, FastAPI, and SQL.
  Goal: Build a production-ready FastAPI AI backend.
tags:
  - python
  - async
  - pydantic
  - fastapi
  - sql
  - month-1
---

# 🐍 Month 1 — Python · Async Python · Pydantic · FastAPI · SQL

> **Goal for this month:** Build a solid backend engineering foundation so every AI service you write is clean, fast, and production-ready.

---

## 📅 Learning Path

```
Python Fundamentals
      ↓
Async Python (asyncio, httpx)
      ↓
Pydantic (validation, schemas)
      ↓
FastAPI (REST APIs, dependency injection)
      ↓
SQL (PostgreSQL, SQLAlchemy)
      ↓
🏗️ BUILD: FastAPI AI Backend
```

---

## 📁 Folder Structure

```
Month-1-Python-FastAPI-SQL/
├── SKILL.md              ← You are here
├── concepts/             ← Theory, notes, cheatsheets
│   ├── python_basics.md
│   ├── async_python.md
│   ├── pydantic_guide.md
│   ├── fastapi_guide.md
│   └── sql_guide.md
└── practice/             ← Hands-on code & projects
    ├── streamlit/        ← (moved from root — early UI experiments)
    └── fastapi_ai_backend/ ← 🏗️ Month build project
```

---

## ⚡ Phase 1 — Python for AI Engineering

### Must-Know Concepts
- **Type hints** — required for Pydantic + FastAPI
- **Dataclasses & Pydantic models** — structured data everywhere
- **List/dict comprehensions** — concise data transformation
- **Generators** — memory-efficient iteration for large datasets
- **Context managers** — `async with`, `with open()` — resource safety
- **Decorators** — `@app.get`, `@retry`, `@cache` — used constantly

### Key Libraries
| Library | Purpose |
|---|---|
| `httpx` | Async HTTP client (LLM API calls) |
| `aiohttp` | Alternative async HTTP |
| `python-dotenv` | Load `.env` secrets |
| `loguru` | Beautiful structured logging |
| `tqdm` | Progress bars for long jobs |
| `rich` | Pretty terminal output |

---

## ⚡ Phase 2 — Async Python

LLM API calls are I/O-bound — async lets you call 5 providers simultaneously instead of sequentially (5x faster).

```python
import asyncio
import httpx

async def call_all_providers(prompt: str):
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            call_openai(client, prompt),
            call_claude(client, prompt),
            call_gemini(client, prompt),
            return_exceptions=True
        )
    return results

# Timeout handling
async def safe_call(prompt: str, timeout: float = 10.0):
    try:
        return await asyncio.wait_for(call_llm(prompt), timeout=timeout)
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}
```

| Function | Use |
|---|---|
| `asyncio.gather(*coros)` | Run multiple coroutines concurrently |
| `asyncio.wait_for(coro, timeout)` | Add timeout to any coroutine |
| `asyncio.create_task(coro)` | Schedule coroutine without awaiting |
| `asyncio.sleep(n)` | Non-blocking sleep |

---

## ⚡ Phase 3 — Pydantic

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class LLMRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)

class LLMResponse(BaseModel):
    content: str
    model: str
    tokens_used: int
    cost_usd: float
```

---

## ⚡ Phase 4 — FastAPI

```python
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()

app = FastAPI(title="AI Backend", lifespan=lifespan)

@app.post("/chat", response_model=LLMResponse)
async def chat(request: LLMRequest, db=Depends(get_db)):
    return await llm_service.generate(request)

@app.post("/chat/stream")
async def chat_stream(request: LLMRequest):
    async def generate():
        async for chunk in llm_service.stream(request):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## ⚡ Phase 5 — SQL for AI Applications

```sql
-- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}'
);

-- Similarity search
SELECT content, 1 - (embedding <=> '[...]') AS similarity
FROM document_chunks
ORDER BY embedding <=> '[...]'
LIMIT 5;
```

---

## 🏗️ Month 1 Build: FastAPI AI Backend

### Checklist
- [ ] `POST /chat` — sync chat endpoint
- [ ] `POST /chat/stream` — streaming SSE endpoint
- [ ] `GET /conversations/{id}` — retrieve history
- [ ] Pydantic models for all request/response
- [ ] SQLAlchemy async models + migrations
- [ ] `.env` config with pydantic-settings
- [ ] Structured logging with loguru
- [ ] `Dockerfile` + `docker-compose.yml`
- [ ] `/docs` OpenAPI working

### Run It
```bash
uvicorn main:app --reload --port 8000
# Visit http://localhost:8000/docs
```

---

## 📚 Resources
- [FastAPI Official Docs](https://fastapi.tiangolo.com)
- [Pydantic V2 Docs](https://docs.pydantic.dev/latest)
- [SQLAlchemy 2.0 Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Real Python — Async IO](https://realpython.com/async-io-python)
