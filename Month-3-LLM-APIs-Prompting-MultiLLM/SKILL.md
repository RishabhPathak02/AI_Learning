---
name: month-3-llm-apis-prompting-multillm
description: >
  Month 3 of the 6-Month AI Engineering Roadmap.
  Covers LLM APIs, Prompt Engineering, Structured Output, Function Calling,
  Streaming, and Multi-LLM orchestration.
  Goal: Build a Multi-LLM Router.
tags:
  - llm
  - openai
  - claude
  - gemini
  - prompt-engineering
  - function-calling
  - streaming
  - multi-llm
  - month-3
---

# 🤖 Month 3 — LLM APIs · Prompt Engineering · Structured Output · Function Calling · Streaming · Multi-LLM

> **Goal for this month:** Master every major LLM provider API and orchestrate them intelligently in a production router.

---

## 📅 Learning Path

```
LLM APIs (OpenAI, Claude, Gemini, Groq)
      ↓
Prompt Engineering (system prompts, chain-of-thought, few-shot)
      ↓
Structured Output (JSON mode, Pydantic parsing)
      ↓
Function Calling (tool use)
      ↓
Streaming (SSE, real-time responses)
      ↓
Multi-LLM (routing, fallbacks, cost tracking)
      ↓
🏗️ BUILD: Multi-LLM Router
```

---

## 📁 Folder Structure

```
Month-3-LLM-APIs-Prompting-MultiLLM/
├── SKILL.md              ← You are here
├── concepts/
│   ├── llm_apis.md
│   ├── prompt_engineering.md
│   ├── structured_output.md
│   ├── function_calling.md
│   ├── streaming.md
│   └── multi_llm_patterns.md
└── practice/
    └── multi_llm_router/  ← 🏗️ Month build project
```

---

## ⚡ Phase 1 — LLM APIs

### OpenAI
```python
from openai import AsyncOpenAI

client = AsyncOpenAI()  # reads OPENAI_API_KEY from env

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain RAG in one paragraph."}
    ],
    temperature=0.7,
    max_tokens=500
)
print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Claude (Anthropic)
```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

message = await client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system="You are an expert AI engineer.",
    messages=[{"role": "user", "content": "What is attention mechanism?"}]
)
print(message.content[0].text)
print(f"Input tokens: {message.usage.input_tokens}")
```

### Gemini (Google)
```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content("Explain transformers briefly.")
print(response.text)
```

### Groq (Ultra-fast inference)
```python
from groq import AsyncGroq

client = AsyncGroq()
response = await client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Sub-200ms latency for LLaMA 3
```

### LLM Model Routing Matrix
| Query Type | Best Model | Reason |
|---|---|---|
| Simple Q&A | GPT-3.5 / Mistral 7B | Cost |
| Complex reasoning | GPT-4o / Claude Opus | Accuracy |
| Long documents | Claude 3.5 Sonnet | 200K context |
| Code generation | GPT-4o / DeepSeek | Specialized |
| Vision | Gemini 2.0 / GPT-4V | Multimodal |
| Fast inference | Groq (LLaMA 3) | <200ms |

---

## ⚡ Phase 2 — Prompt Engineering

### Core Techniques
```
1. Zero-shot:     Just the question
2. Few-shot:      Question + 2-3 examples
3. Chain-of-thought (CoT): "Think step by step"
4. System prompt: Set persona, constraints, output format
5. ReAct:         Reason + Act (for agents)
```

### Prompt Template
```python
SYSTEM_PROMPT = """You are an expert {role} assistant.

CONSTRAINTS:
- Always respond in {language}
- Keep responses under {max_words} words
- Format output as {format}

EXAMPLES:
User: {example_input}
Assistant: {example_output}
"""

# Dynamic prompt building
def build_prompt(role: str, query: str, context: str = "") -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(role=role, ...)}]
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    else:
        messages.append({"role": "user", "content": query})
    return messages
```

### Chain-of-Thought
```python
cot_prompt = """Solve this step by step:

Problem: {problem}

Let me think through this carefully:
1. First, I identify...
2. Then, I calculate...
3. Finally, I conclude...

Answer:"""
```

---

## ⚡ Phase 3 — Structured Output

### OpenAI JSON Mode
```python
from pydantic import BaseModel

class SentimentResult(BaseModel):
    sentiment: str  # "positive" | "negative" | "neutral"
    confidence: float
    reasoning: str

# Method 1: JSON mode
response = await client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": f"Analyze sentiment: {text}. Respond with JSON."}]
)
result = SentimentResult.model_validate_json(response.choices[0].message.content)

# Method 2: Structured outputs (gpt-4o-2024-08-06+)
from openai.lib._pydantic import to_strict_json_schema
response = await client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=SentimentResult
)
result = response.choices[0].message.parsed
```

### Instructor Library (Best for structured output)
```python
import instructor
from openai import AsyncOpenAI

client = instructor.from_openai(AsyncOpenAI())

result = await client.chat.completions.create(
    model="gpt-4o",
    response_model=SentimentResult,
    messages=[{"role": "user", "content": "Analyze: I love this product!"}]
)
print(result.sentiment)    # "positive"
print(result.confidence)   # 0.95
```

---

## ⚡ Phase 4 — Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Mumbai?"}],
    tools=tools,
    tool_choice="auto"
)

# Handle tool call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = get_weather(**args)  # your actual function
    
    # Send result back to LLM
    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
```

---

## ⚡ Phase 5 — Streaming

```python
# Server-Sent Events with FastAPI
@app.post("/chat/stream")
async def chat_stream(request: LLMRequest):
    async def generate():
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=request.messages,
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'content': delta})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})
```

---

## ⚡ Phase 6 — Multi-LLM Router

### Router Architecture
```
Request → Classifier → Route Decision
              ├── Simple?  → Groq/Mistral (cheap, fast)
              ├── Complex? → GPT-4o / Claude (accurate)
              ├── Long?    → Claude 3.5 (200K ctx)
              └── Vision?  → Gemini 2.0
```

### Production Patterns
| Pattern | Implementation |
|---|---|
| Retry with backoff | `tenacity` + `@retry(wait=wait_exponential())` |
| Rate limiting | Token bucket in Redis |
| Cost tracking | Log `usage.prompt_tokens + completion_tokens` per call |
| Semantic cache | Embed query → cosine search → return if similarity > 0.95 |
| Circuit breaker | Track failure% per provider, open after N failures |
| Fallback chain | GPT-4o → Claude → GPT-3.5 → Cache → Static template |

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def call_with_retry(provider: str, request: LLMRequest):
    return await PROVIDERS[provider].generate(request)

async def route_request(request: LLMRequest) -> LLMResponse:
    provider = classify_request(request)  # your routing logic
    try:
        return await call_with_retry(provider, request)
    except Exception:
        # Fallback chain
        for fallback in FALLBACK_CHAIN:
            try:
                return await call_with_retry(fallback, request)
            except Exception:
                continue
        raise HTTPException(500, "All providers failed")
```

---

## 🏗️ Month 3 Build: Multi-LLM Router

### What to Build
A FastAPI service that:
- Routes requests to the optimal LLM based on complexity/type
- Tracks token usage and cost per provider
- Implements retry + fallback chain
- Supports streaming for all providers
- Has a semantic cache (Redis + embeddings)
- Exposes unified API regardless of underlying LLM

### Checklist
- [ ] Provider clients: OpenAI, Claude, Gemini, Groq
- [ ] `POST /chat` — unified endpoint with routing
- [ ] Request classifier (simple heuristic or LLM-based)
- [ ] Fallback chain with circuit breaker
- [ ] Cost tracking: log per-request costs to DB
- [ ] Semantic cache with Redis
- [ ] `GET /stats` — cost/usage dashboard
- [ ] Rate limiting per user
- [ ] Docker deployment

---

## 📚 Resources
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic Claude Docs](https://docs.anthropic.com)
- [Google Gemini API](https://ai.google.dev/docs)
- [Instructor Library](https://python.useinstructor.com)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
