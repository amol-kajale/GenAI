# RAG Integration Guide

## Overview

Your GenAI solution has been successfully converted to a **Retrieval-Augmented Generation (RAG)** system. This guide explains how everything works together.

## What Is RAG?

RAG combines retrieval and generation:
1. **Retrieve**: Find relevant documents from your knowledge base
2. **Augment**: Add retrieved documents to the LLM prompt
3. **Generate**: LLM creates response using both query and context

**Benefits:**
- More accurate answers
- Uses your specific knowledge
- Reduces hallucinations
- Cost efficient

## Architecture

```
┌─────────────────────────────────────────────┐
│         User Application                     │
└────────────────┬────────────────────────────┘
                 │
        ┌────────▼────────┐
        │   FastAPI App   │
        │   (main.py)     │
        └────────┬────────┘
                 │
         ┌───────┴───────┐
         │               │
    ┌────▼────┐    ┌────▼──────┐
    │ Vector  │    │ LLM        │
    │ Store   │    │ Inference  │
    │(Chroma) │    │ (Hugging   │
    └──────┬──┘    │ Face)      │
           │       └────┬───────┘
    ┌──────▼──────┐     │
    │ Documents   │     │
    │ Embeddings  │     │
    └─────────────┘     │
                        │
                   ┌────▼─────┐
                   │ Response  │
                   │ (JSON)    │
                   └───────────┘
```

## Data Flow

### Upload Phase (Document Storage)
```
User Document
    ↓
Chroma Embedding Model
    ↓
Vector Embedding
    ↓
Store in Vector DB
    ↓
Index Built
```

### Query Phase (Retrieval-Augmented Generation)
```
User Question
    ↓
Question Embedding
    ↓
Similarity Search (Top-K)
    ↓
Retrieved Documents
    ↓
Augmented Prompt
    ↓
LLM Inference
    ↓
Structured Response
    ↓
Metrics Recording
```

## Module Structure

### `vector_db.py` - Vector Database Wrapper

```python
class VectorStore:
    - __init__(db_path) - Initialize Chroma
    - add_documents() - Store documents with IDs
    - retrieve(query) - Get relevant docs
    - format_retrieved_docs() - Format for LLM
    - delete_all() - Clear collection
```

### `main.py` - FastAPI Application

**New Endpoints:**
- `POST /documents/upload` - Upload knowledge base
- `POST /ask` with `use_rag` parameter

**Modified Endpoints:**
- `POST /ask` - Enhanced with RAG context

**New Variables:**
- `vector_store` - Global VectorStore instance
- `retrieved_docs_count` - Tracks retrieval

### `prompts.py` - Prompt Engineering

**New Functions:**
- Context parameter in `get_analysis_prompt()`
- Improved JSON parsing with regex
- Better validation

## API Usage

### 1. Upload Documents

**Endpoint:** `POST /documents/upload`

**Request:**
```json
{
  "documents": [
    "Document content here...",
    "Another document..."
  ],
  "ids": ["doc_1", "doc_2"],
  "metadata": [
    {"source": "help_desk", "category": "faq"},
    {"source": "documentation", "category": "api"}
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Uploaded 2 documents to vector store",
  "count": 2
}
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/documents/upload",
    json={
        "documents": [
            "Your company's documentation...",
            "Support knowledge base..."
        ],
        "ids": ["doc1", "doc2"],
        "metadata": [
            {"type": "docs"},
            {"type": "kb"}
        ]
    }
)
print(response.json())
```

### 2. Query with RAG

**Endpoint:** `POST /ask`

**Request (with RAG):**
```json
{
  "question": "How do I reset my password?",
  "use_rag": true,
  "model": "openai/gpt-oss-20b:groq"
}
```

**Response:**
```json
{
  "answer": {
    "summary": "Based on documentation, you can reset your password by...",
    "intent": "support",
    "priority": "medium"
  },
  "confidence_score": 0.92,
  "suggested_actions": [
    "Provide step-by-step resolution guide",
    "Ask for environment and reproducible steps"
  ],
  "rag_enabled": true,
  "retrieved_documents_count": 2
}
```

**Request (without RAG):**
```json
{
  "question": "What is machine learning?",
  "use_rag": false
}
```

**Python Example:**
```python
import requests
import json

# Query with RAG
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "How do I reset my password?",
        "use_rag": True
    }
)

result = response.json()
print(f"Answer: {result['answer']['summary']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Documents retrieved: {result['retrieved_documents_count']}")
```

## Configuration

### Environment Variables (`.env`)

```bash
# Required
HF_API_KEY=your_hugging_face_token

# LLM Settings
HF_DEFAULT_MODEL=openai/gpt-oss-20b:groq
HF_API_URL=https://router.huggingface.co/v1/chat/completions

# Vector DB Settings
CHROMA_DB_PATH=chroma_db

# Cost Estimation
COST_PER_1K_PROMPT=0.0
COST_PER_1K_COMPLETION=0.0
```

### Vector Store Configuration

**Location:** `vector_db.py`, `__init__()` method

```python
# Customize embeddings
# Default: Chroma's default embeddings (all-MiniLM-L6-v2)

# Customize distance metric
metadata={"hnsw:space": "cosine"}  # Options: cosine, l2, ip

# Customize persistence
chroma_db_impl="duckdb+parquet"
persist_directory=db_path
```

## Metrics Tracking

### CSV Columns (metrics.csv)

New columns for RAG:
- `rag_used` - Boolean: was RAG enabled?
- `retrieved_docs_count` - Integer: how many documents retrieved?

**Example Entry:**
```csv
timestamp,question,model,latency_ms,prompt_tokens,completion_tokens,total_tokens,estimated_cost_usd,intent,priority,confidence_score,rag_used,retrieved_docs_count
2024-01-27T10:30:45.123456,How do I reset password?,openai/gpt-oss-20b:groq,2345.67,150,75,225,0.000000,support,medium,0.92,true,2
```

## Performance Considerations

### Latency Breakdown
```
RAG Query Latency = Embedding Time + Search Time + LLM Time

- Embedding: ~50-200ms (per query)
- Search: ~10-50ms (depends on doc count)
- LLM: 1000-5000ms (main bottleneck)
- Total: ~1100-5250ms per query
```

### Memory Usage
```
Vector Store Size ≈ (Doc Count × Embedding Dim × 4 bytes)

Example:
- 1000 docs × 384 dims × 4 bytes ≈ 1.5 MB
- 10000 docs × 384 dims × 4 bytes ≈ 15 MB
- 100000 docs × 384 dims × 4 bytes ≈ 150 MB
```

### Optimization Tips

1. **Batch Uploads**
   - Upload documents in batches of 100-500
   - Reduces transaction overhead

2. **Document Chunking**
   - Split long documents into paragraphs
   - Better retrieval accuracy
   - Lower latency

3. **Selective RAG**
   - Disable RAG for general questions
   - Enable for domain-specific queries
   - Balances cost and accuracy

4. **Caching**
   - Cache retrieved documents
   - Reduces embedding overhead
   - Implement in main.py

## Troubleshooting

### Issue: "chromadb not installed"

**Solution:**
```bash
pip install chromadb
```

If compilation fails:
```bash
# Use pre-built wheel
pip install --only-binary :all: chromadb

# Or install build tools first
pip install cmake wheel
pip install chromadb
```

### Issue: Poor Retrieval Results

**Causes & Solutions:**
- Documents too long → Implement chunking
- Wrong embeddings → Use domain-specific model
- Irrelevant documents → Improve document quality
- Poor query → Use query expansion

**Example Fix:**
```python
# Retrieve more documents for better results
retrieval_results = vector_store.retrieve(question, n_results=5)
```

### Issue: Slow Responses

**Causes & Solutions:**
- First query slow → Normal, embeddings cached after
- Too many documents → Optimize search scope
- Large embeddings → Use smaller embedding model
- Network latency → Check API connectivity

## Advanced Configuration

### Use Custom Embeddings

```python
# In vector_db.py
from chromadb.utils import embedding_functions

# Use OpenAI embeddings (requires API key)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="YOUR_OPENAI_API_KEY",
    model_name="text-embedding-3-small"
)

self.collection = self.client.get_or_create_collection(
    name="documents",
    embedding_function=openai_ef
)
```

### Hybrid Search (Keyword + Semantic)

```python
# Implement BM25 + Vector search
# For better retrieval accuracy
```

### Document Preprocessing

```python
# Add to main.py
def preprocess_document(doc):
    # Remove extra whitespace
    # Convert to lowercase
    # Remove special characters
    # Split into chunks
    return cleaned_doc
```

## Testing

### Unit Tests

```bash
pytest test_main.py -v
pytest test_main.py::test_ask_with_rag -v
```

### Integration Tests

```python
# Manual testing script
import requests

# Test 1: Upload
response = requests.post(...) # upload docs
assert response.status_code == 200

# Test 2: Query with RAG
response = requests.post(...) # ask with RAG
assert response.json()["rag_enabled"] == True
assert response.json()["retrieved_documents_count"] > 0

# Test 3: Query without RAG
response = requests.post(...) # ask without RAG
assert response.json()["rag_enabled"] == False
```

## Security Considerations

1. **Input Validation**
   - Validate document size
   - Sanitize document content
   - Rate limit uploads

2. **API Security**
   - Protect HF_API_KEY
   - Use HTTPS in production
   - Implement authentication

3. **Data Privacy**
   - Encrypt vector store
   - Audit document access
   - GDPR compliance for user data

## Deployment

### Docker Example

```dockerfile
FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install chromadb

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genai-config
data:
  CHROMA_DB_PATH: /data/chroma_db
  HF_DEFAULT_MODEL: openai/gpt-oss-20b:groq
```

## Monitoring & Logging

### Key Metrics to Monitor

```python
# Implement in main.py
- RAG retrieval latency
- Document retrieval count distribution
- Cache hit rate
- Query latency percentiles (p50, p95, p99)
- Error rates by endpoint
```

### Log RAG Operations

```python
import logging

logger = logging.getLogger("rag")
logger.info(f"Retrieved {doc_count} documents for query: {question}")
logger.debug(f"Retrieval took {retrieval_time}ms")
logger.warning(f"No documents found for query: {question}")
logger.error(f"Vector DB error: {error}")
```

---

This RAG implementation provides a complete solution for document-aware question answering. Start with the quick start guide and expand based on your needs.
