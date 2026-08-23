---
name: month-5-agents-langgraph-finetuning
description: >
  Month 5 of the 6-Month AI Engineering Roadmap.
  Covers AI Agents, LangGraph, MCP, Fine-tuning, LoRA, and QLoRA.
  Goal: Build an AI Agent + Fine-tuned model.
tags:
  - agents
  - langgraph
  - mcp
  - fine-tuning
  - lora
  - qlora
  - peft
  - month-5
---

# 🤖 Month 5 — Agents · LangGraph · MCP · Fine-tuning · LoRA · QLoRA

> **Goal for this month:** Build autonomous AI agents with complex multi-step reasoning AND fine-tune an LLM for a specific task.

---

## 📅 Learning Path

```
AI Agents (ReAct, tool use, memory)
      ↓
LangGraph (stateful agent graphs)
      ↓
MCP (Model Context Protocol)
      ↓
Fine-tuning Decision Framework
      ↓
LoRA (Low-Rank Adaptation)
      ↓
QLoRA (Quantized LoRA — consumer GPU)
      ↓
🏗️ BUILD: AI Agent + Fine-tuned Model
```

---

## 📁 Folder Structure

```
Month-5-Agents-LangGraph-FineTuning/
├── SKILL.md              ← You are here
├── concepts/
│   ├── ai_agents.md
│   ├── langgraph.md
│   ├── mcp.md
│   ├── fine_tuning_when.md
│   ├── lora.md
│   └── qlora.md
└── practice/
    ├── langgraph_agent/    ← Agent build
    └── finetune_lora/      ← 🏗️ Fine-tuning project
```

---

## ⚡ Phase 1 — AI Agents

### What is an AI Agent?
An LLM that can **observe → think → act → repeat** in a loop until the task is done.

### ReAct Loop
```
Thought: I need to find the weather in Mumbai
Action: search_web("Mumbai weather today")
Observation: Mumbai is 32°C, partly cloudy
Thought: Now I have the answer
Final Answer: It's 32°C and partly cloudy in Mumbai.
```

### Tool Definition (OpenAI)
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout": {"type": "integer", "default": 30}
                },
                "required": ["code"]
            }
        }
    }
]
```

### Basic Agent Loop
```python
async def agent_loop(task: str, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": "Use tools to complete the task. Think step by step."},
        {"role": "user", "content": task}
    ]
    
    for i in range(max_iterations):
        response = await client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if not message.tool_calls:  # Agent decided to stop
            return message.content
        
        messages.append(message)  # Add assistant's tool call
        
        # Execute each tool
        for tool_call in message.tool_calls:
            result = await execute_tool(tool_call.function.name,
                                        json.loads(tool_call.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    
    return "Max iterations reached"
```

---

## ⚡ Phase 2 — LangGraph

### Why LangGraph?
- Complex multi-agent workflows with **state persistence**
- Conditional routing between nodes
- Human-in-the-loop checkpoints
- Built-in memory and streaming

### Core Concepts
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only
    next_step: str
    iteration: int
    final_answer: str

# 2. Define nodes (functions that transform state)
async def planner_node(state: AgentState) -> AgentState:
    # Decide what to do next
    response = await llm.generate(state["messages"])
    return {"messages": [response], "next_step": "executor"}

async def executor_node(state: AgentState) -> AgentState:
    # Execute tool calls from planner
    result = await execute_tools(state["messages"][-1])
    return {"messages": [result], "iteration": state["iteration"] + 1}

async def reviewer_node(state: AgentState) -> AgentState:
    # Check if task is complete
    is_done = await check_completion(state)
    return {"next_step": "done" if is_done else "planner"}

# 3. Routing function
def should_continue(state: AgentState) -> str:
    if state["iteration"] >= 10:
        return "done"
    return state["next_step"]

# 4. Build graph
builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("reviewer", reviewer_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "reviewer")
builder.add_conditional_edges("reviewer", should_continue,
                               {"planner": "planner", "done": END})

# 5. Compile with memory
from langgraph.checkpoint.memory import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())

# 6. Run
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="Research LangGraph and write a summary")],
     "iteration": 0},
    config={"configurable": {"thread_id": "session-123"}}  # persistence
)
```

### Multi-Agent Patterns
```
Supervisor → [Researcher, Writer, Reviewer]    (hierarchical)
Pipeline   → Planner → Executor → Validator   (sequential)
Parallel   → run specialists concurrently → aggregate
```

---

## ⚡ Phase 3 — MCP (Model Context Protocol)

### What is MCP?
Anthropic open standard for AI ↔ tool connectivity.
Build MCP servers that expose any tool to any MCP-compatible client.

### Build a Simple MCP Server
```python
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp import types
import mcp.server.stdio

server = Server("my-ai-tools")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_database",
            description="Search the product database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_database":
        results = await db.search(arguments["query"], arguments.get("limit", 10))
        return [types.TextContent(type="text", text=json.dumps(results))]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                        InitializationOptions(server_name="my-ai-tools", server_version="0.1.0"))
```

---

## ⚡ Phase 4 — Fine-Tuning Decision Framework

### When to Fine-Tune?
```
Task solvable with prompt engineering?   → DON'T fine-tune (save $$$)
Need consistent output format/structure? → Fine-tune
Domain-specific vocabulary/jargon?       → Consider RAG first, then FT
Need latest knowledge updates?           → RAG (FT doesn't update knowledge)
Reducing inference cost (smaller model)? → Fine-tune smaller model
```

### Fine-Tuning vs RAG
| | Fine-Tuning | RAG |
|---|---|---|
| Updates knowledge | ❌ Expensive to retrain | ✅ Just add documents |
| Consistent format | ✅ Learns format | ❌ Prompting needed |
| Cost at inference | ✅ Smaller model possible | ❌ Retrieval overhead |
| Time to production | ❌ Days/weeks | ✅ Hours |

---

## ⚡ Phase 5 — LoRA

### How LoRA Works
Instead of training all model weights (billions!), LoRA trains two small matrices (A and B) and adds their product to the frozen weights:
```
W_new = W_frozen + (B × A)   where rank(A,B) << rank(W)
```

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# Load base model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# LoRA config
lora_config = LoraConfig(
    r=16,                           # rank — higher = more params, more capacity
    lora_alpha=32,                  # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # only ~1% of total params!

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch"
    )
)
trainer.train()
```

---

## ⚡ Phase 6 — QLoRA (Consumer GPU Fine-Tuning)

QLoRA = LoRA + 4-bit quantization → fine-tune 7B+ models on a single 24GB GPU.

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — best for weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True    # saves additional memory
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for LoRA training (important for quantized models)
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# Then apply LoRA as normal...
```

### Dataset Format (Alpaca / ChatML)
```python
# Alpaca format
{"instruction": "Classify sentiment", "input": "I love this!", "output": "positive"}

# ChatML format (preferred)
{"messages": [
    {"role": "system", "content": "Classify sentiment as positive/negative/neutral."},
    {"role": "user", "content": "I love this product!"},
    {"role": "assistant", "content": "positive"}
]}
```

---

## 🏗️ Month 5 Build: AI Agent + Fine-tuned Model

### Part A — LangGraph Agent
- [ ] Research agent: searches web + summarizes findings
- [ ] Multi-step reasoning with LangGraph state machine
- [ ] Human-in-the-loop checkpoint before final action
- [ ] Persistent memory across sessions
- [ ] FastAPI endpoint: `POST /agent/run`
- [ ] Streaming intermediate steps

### Part B — Fine-tuned Model
- [ ] Choose a task: classification, extraction, Q&A, or chatbot
- [ ] Prepare dataset (1000+ examples minimum)
- [ ] Fine-tune Mistral 7B or LLaMA 3.1 8B with QLoRA
- [ ] Evaluate: compare FT model vs base model vs GPT-3.5
- [ ] Deploy with vLLM or Ollama
- [ ] FastAPI endpoint serving the fine-tuned model

---

## 📚 Resources
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph)
- [MCP Python SDK](https://github.com/anthropics/mcp)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [TRL — Transformer Reinforcement Learning](https://huggingface.co/docs/trl)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Axolotl — Fine-tuning Framework](https://github.com/OpenAccess-AI-Collective/axolotl)
