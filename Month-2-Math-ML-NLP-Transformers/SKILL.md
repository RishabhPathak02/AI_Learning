---
name: month-2-math-ml-nlp-transformers
description: >
  Month 2 of the 6-Month AI Engineering Roadmap.
  Covers Math for AI, Machine Learning, Deep Learning, NLP, and Transformers.
  Goal: Build a small text-classification/embedding project.
tags:
  - math
  - ml
  - deep-learning
  - nlp
  - transformers
  - month-2
---

# 📐 Month 2 — Math · ML · Deep Learning · NLP · Transformers

> **Goal for this month:** Understand *why* AI works under the hood so you can debug, tune, and design better systems.

---

## �� Learning Path

```
Math & Statistics (Linear Algebra, Probability, Calculus)
      ↓
Machine Learning Fundamentals (scikit-learn, evaluation)
      ↓
Deep Learning (PyTorch, neural networks)
      ↓
NLP (text processing, tokenization, embeddings)
      ↓
Transformers (attention, BERT, GPT)
      ↓
🏗️ BUILD: Text Classification / Embedding Project
```

---

## 📁 Folder Structure

```
Month-2-Math-ML-NLP-Transformers/
├── SKILL.md              ← You are here
├── concepts/
│   ├── Basics/           ← (moved from root — ML/NLP foundations)
│   ├── math_for_ai.md
│   ├── ml_fundamentals.md
│   ├── deep_learning.md
│   ├── nlp_guide.md
│   └── transformers.md
└── practice/
    ├── Tokenization/     ← (moved from root)
    ├── OneHotEncoded/    ← (moved from root)
    ├── PartsOfSpeech/    ← (moved from root)
    ├── NamedEntity/      ← (moved from root)
    └── text_classifier/  ← 🏗️ Month build project
```

---

## ⚡ Phase 1 — Math for AI

### Linear Algebra
```
Vectors: [0.2, 0.8, 0.1] — word embeddings are vectors
Matrix multiplication: W @ x + b  — forward pass in NN
Dot product: a·b = Σ(aᵢ * bᵢ) — used in attention scores
Cosine similarity: cos(A,B) = A·B / (|A||B|) — semantic search
```

### Probability & Statistics
```
Softmax: σ(zᵢ) = exp(zᵢ) / Σ exp(zⱼ) — converts logits to probs
Cross-entropy loss: L = -Σ yᵢ log(ŷᵢ) — training loss for classification
Bayes theorem: P(A|B) = P(B|A) * P(A) / P(B)
```

### Calculus for ML
```
Gradient: direction of steepest ascent of loss
Gradient descent: θ = θ - α * ∇L(θ)
Chain rule: enables backpropagation through layers
Adam optimizer: adaptive learning rates + momentum
```

---

## ⚡ Phase 2 — Machine Learning

### scikit-learn Pipeline Pattern
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Key Evaluation Metrics (Know These Cold!)
| Metric | Formula | When to Use |
|---|---|---|
| Accuracy | correct / total | Balanced classes |
| Precision | TP / (TP+FP) | Cost of false positives high |
| Recall | TP / (TP+FN) | Cost of false negatives high |
| F1 | 2 * P * R / (P+R) | Imbalanced classes |
| AUC-ROC | Area under ROC curve | Binary classification quality |

---

## ⚡ Phase 3 — Deep Learning with PyTorch

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)  # average token embeddings
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Training loop
model = TextClassifier(vocab_size=10000, embed_dim=128, num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
```

---

## ⚡ Phase 4 — NLP Fundamentals

### Text Processing Pipeline
```
Raw Text → Tokenize → Remove Stopwords → Stemming/Lemmatization
→ Vectorize (TF-IDF / Embeddings) → Model → Prediction
```

### Tokenization Types (moved practice notebooks are here)
- **Word tokenization** — split by spaces/punctuation
- **BPE (Byte-Pair Encoding)** — GPT uses this (`tiktoken`)
- **WordPiece** — BERT uses this
- **SentencePiece** — T5, LLaMA use this

```python
# Using tiktoken (OpenAI tokenizer)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, AI Engineer!")
print(tokens)        # [9906, 11, 15592, 34004, 0]
print(len(tokens))   # 5

# HuggingFace tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer("Hello world", return_tensors="pt")
```

### Embeddings
```python
# Sentence embeddings with HuggingFace
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["I love AI", "Machine learning is fun"]
embeddings = model.encode(sentences)  # shape: (2, 384)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"Similarity: {sim[0][0]:.3f}")
```

---

## ⚡ Phase 5 — Transformers & Attention

### Self-Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
  Q = Query matrix  (what we're looking for)
  K = Key matrix    (what each token offers)
  V = Value matrix  (what each token contributes)
  d_k = dimension of key vectors (for scaling)
```

### Multi-Head Attention
```
Run H attention heads in parallel (each with different W_Q, W_K, W_V)
Concatenate all heads → linear projection
Captures different types of relationships simultaneously
```

### Using HuggingFace Transformers
```python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Zero-shot — no fine-tuning needed
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "This is a great product!",
    candidate_labels=["positive", "negative", "neutral"]
)

# Sentiment analysis
sentiment = pipeline("sentiment-analysis")
print(sentiment("I love building AI systems!"))

# Text embeddings
from transformers import AutoModel
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

### Transformer Architecture
```
Input → Tokenize → Embed + Positional Encoding
      → [Multi-Head Attention → Add & Norm → FFN → Add & Norm] × N layers
      → Output head (classification / generation)
```

---

## 🏗️ Month 2 Build: Text Classification + Embedding Project

### Option A — Text Classifier
Build a sentiment/topic classifier:
- [ ] Dataset: HuggingFace datasets (e.g., `imdb`, `ag_news`)
- [ ] Tokenize with HuggingFace tokenizer
- [ ] Fine-tune `distilbert-base-uncased` with `Trainer`
- [ ] Evaluate: accuracy, F1, confusion matrix
- [ ] Export model + serve with FastAPI

### Option B — Semantic Search Engine
- [ ] Embed 1000+ documents with `all-MiniLM-L6-v2`
- [ ] Store in Chroma (local vector DB)
- [ ] Query with natural language → return top-5 similar
- [ ] FastAPI endpoint: `POST /search`

### Checklist (Both)
- [ ] Jupyter notebook EDA → understand the data
- [ ] Clean training pipeline with evaluation
- [ ] Model saved to `models/` directory
- [ ] FastAPI inference endpoint
- [ ] README with results and examples

---

## 📚 Resources
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [fast.ai — Practical Deep Learning](https://course.fast.ai)
- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Book: *Hands-On Machine Learning* — Aurélien Géron
