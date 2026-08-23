---
name: month-4-embeddings-vectordb-rag
description: >
  Month 4 of the 6-Month AI Engineering Roadmap.
  Covers Embeddings, Vector Databases, RAG, Hybrid Search, Reranking, and RAG Evaluation.
  Goal: Build a Production RAG Application.
tags:
  - embeddings
  - vector-db
  - rag
  - hybrid-search
  - reranking
  - evaluation
  - month-4
---

# 🔍 Month 4 — Embeddings · Vector DB · RAG · Hybrid Search · Reranking · RAG Evaluation

> **Goal for this month:** Build a production-grade RAG system that retrieves accurate context and generates grounded answers.

---

## 📅 Learning Path

```
Embeddings (semantic representations)
      ↓
Vector Databases (Chroma, Qdrant, Pinecone, pgvector)
      ↓
RAG (Retrieval-Augmented Generation)
      ↓
Hybrid Search (BM25 + Vector + RRF)
      ↓
Reranking (Cohere, BGE)
      ↓
RAG Evaluation (RAGAS, LLM-as-judge)
      ↓
🏗️ BUILD: Production RAG Application
```

---

## 📁 Folder Structure

```
Month-4-Embeddings-VectorDB-RAG/
├── SKILL.md              ← You are here
├── concepts/
│   ├── embeddings.md
│   ├── vector_databases.md
│   ├── rag_architecture.md
│   ├── hybrid_search.md
│   ├── reranking.md
│   └── rag_evaluation.md
└── practice/
    └── production_rag_app/  ← 🏗️ Month build project
```

---

## ⚡ Phase 1 — Embeddings

### What Are Embeddings?
Dense vector representations of text in high-dimensional space.
Semantically similar text → similar vectors → small cosine distance.

```python
from openai import AsyncOpenAI
import numpy as np

client = AsyncOpenAI()

async def embed(text: str) -> list[float]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dims, cheapest
        input=text
    )
    return response.data[0].embedding

# Cosine similarity
def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### Embedding Model Selection
| Model | Dims | Cost | Best For |
|---|---|---|---|
| `text-embedding-3-small` | 1536 | $ | General use, RAG |
| `text-embedding-3-large` | 3072 | $$ | High accuracy |
| `all-MiniLM-L6-v2` | 384 | Free | Local/offline |
| `bge-large-en-v1.5` | 1024 | Free | Best open-source |
| `cohere-embed-v3` | 1024 | $ | Multilingual |

---

## ⚡ Phase 2 — Vector Databases

### Comparison
| DB | Best For | Hosting | Notes |
|---|---|---|---|
| **Chroma** | Local dev, <1M vectors | Local | Zero setup, great for prototyping |
| **Qdrant** | Production, self-hosted | Docker/Cloud | Best perf/cost, rich filtering |
| **Pinecone** | Managed, zero-ops | Cloud | Serverless option available |
| **pgvector** | Already on Postgres | Your DB | SQL + vectors in one place |
| **Weaviate** | Hybrid search native | Self/Cloud | Built-in BM25 + vector |

### Qdrant (Recommended for Production)
```python
from qdrant_client import AsyncQdrantClient, models

client = AsyncQdrantClient(url="http://localhost:6333")

# Create collection
await client.create_collection(
    collection_name="documents",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

# Upsert vectors
await client.upsert(
    collection_name="documents",
    points=[
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"content": chunk, "source": filename, "page": page_num}
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
)

# Search
results = await client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value="report.pdf"))]
    )
)
```

---

## ⚡ Phase 3 — RAG Architecture

### Basic RAG Pipeline
```
INDEXING (offline):
Documents → Chunk (512 tokens, 50 overlap) → Embed → Vector DB

RETRIEVAL (online):
Query → Embed → Top-K search (k=5) → Retrieved chunks

GENERATION (online):
Retrieved chunks + Query → LLM → Answer
```

### Document Processing
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document_text)
```

### RAG Generation
```python
async def rag_query(question: str, collection: str) -> str:
    # 1. Embed the question
    query_embedding = await embed(question)
    
    # 2. Retrieve top-5 chunks
    results = await vector_db.search(collection, query_embedding, limit=5)
    context = "\n\n".join([r.payload["content"] for r in results])
    
    # 3. Generate with LLM
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context. If unsure, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content
```

### Advanced RAG Techniques
| Technique | When to Use | Benefit |
|---|---|---|
| **HyDE** | Vague queries | Generate hypothetical answer, embed that |
| **Multi-query** | Complex questions | Decompose → retrieve for each sub-query |
| **Parent-doc retrieval** | Need more context | Retrieve small chunks, return parent |
| **Contextual compression** | Reduce noise | LLM filters retrieved chunks |
| **Step-back prompting** | Abstract reasoning | Retrieve high-level principles first |

```python
# HyDE Implementation
async def hyde_query(question: str) -> str:
    # Generate hypothetical answer
    hypothetical = await llm.generate(f"Write a hypothetical answer to: {question}")
    # Embed the hypothetical answer (not the question!)
    hyp_embedding = await embed(hypothetical)
    # Retrieve based on hypothetical
    return await vector_db.search(hyp_embedding, limit=5)
```

---

## ⚡ Phase 4 — Hybrid Search

### Why Hybrid?
- **Vector search**: finds semantically similar content (understands meaning)
- **BM25 (keyword)**: finds exact term matches (great for rare words, codes)
- **Hybrid = best of both**

### Reciprocal Rank Fusion (RRF)
```python
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse multiple ranked lists using RRF."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

# With LangChain
from langchain.retrievers import EnsembleRetriever, BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # adjust based on your data
)
```

---

## ⚡ Phase 5 — Reranking

```python
import cohere

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

def rerank(query: str, documents: list[str], top_n: int = 3) -> list[str]:
    results = co.rerank(
        query=query,
        documents=documents,
        model="rerank-english-v3.0",
        top_n=top_n
    )
    return [documents[r.index] for r in results.results]

# Open-source alternative: BGE reranker
from FlagEmbedding import FlagReranker
reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)
scores = reranker.compute_score([[query, doc] for doc in documents])
```

---

## ⚡ Phase 6 — RAG Evaluation

### RAGAS Framework
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Create evaluation dataset
from datasets import Dataset
eval_data = Dataset.from_dict({
    "question": questions,
    "answer": generated_answers,
    "contexts": retrieved_contexts,      # list of lists
    "ground_truth": reference_answers   # for recall metrics
})

result = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
print(result)
```

### Key RAG Metrics
| Metric | Measures | Formula |
|---|---|---|
| **Faithfulness** | Is answer grounded in context? | % claims supported by context |
| **Answer Relevancy** | Does answer address the question? | Embedding similarity |
| **Context Recall** | Did retrieval find all relevant info? | Overlap with ground truth |
| **Context Precision** | Is retrieved context relevant? | % chunks that are relevant |

---

## 🏗️ Month 4 Build: Production RAG Application

### What to Build
A multi-document Q&A system with:
- PDF/text ingestion pipeline
- Chunking + embedding + Qdrant storage
- Hybrid search (BM25 + vector) + Cohere reranking
- Streaming answers via FastAPI
- Citation tracking (which chunk answered the question)
- RAGAS evaluation dashboard

### Checklist
- [ ] `POST /ingest` — upload and process documents
- [ ] `POST /query` — RAG query with citations
- [ ] `POST /query/stream` — streaming RAG response
- [ ] Hybrid search (BM25 + vector)
- [ ] Cohere reranking integration
- [ ] HyDE for improved retrieval
- [ ] RAGAS evaluation on test set
- [ ] Conversation history (multi-turn)
- [ ] `GET /documents` — list ingested docs
- [ ] Docker + docker-compose (FastAPI + Qdrant)

---

## 📚 Resources
- [RAGAS Documentation](https://docs.ragas.io)
- [Qdrant Documentation](https://qdrant.tech/documentation)
- [Cohere Rerank API](https://docs.cohere.com/reference/rerank)
- [LangChain RAG Guide](https://python.langchain.com/docs/tutorials/rag)
- Paper: *RAGAS: Automated Evaluation of RAG Pipelines*
