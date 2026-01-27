# GenAI Solution with RAG (Retrieval-Augmented Generation)

This is an enhanced version of the GenAI LLM Metrics API that includes **Retrieval-Augmented Generation (RAG)** capabilities using a vector database.

## What's New

### RAG Implementation
- **Vector Database**: Uses Chroma for semantic document storage and retrieval
- **Document Upload**: `/documents/upload` endpoint to add documents to the knowledge base
- **Intelligent Retrieval**: Automatically retrieves relevant documents for each query
- **Optional RAG**: Can disable RAG per request with `use_rag` parameter

### Key Files

- **`vector_db.py`** - Vector database operations using Chroma
- **`main.py`** - FastAPI application with RAG integration
- **`prompts.py`** - Enhanced prompts with RAG context injection
- **`test_main.py`** - Test suite including RAG tests

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install chromadb  # Vector database (optional, for local development)
```

### 2. Configure Environment

Update `.env` with your Hugging Face API key:

```bash
HF_API_KEY=hf_your_actual_api_key_here
HF_DEFAULT_MODEL=openai/gpt-oss-20b:groq
CHROMA_DB_PATH=chroma_db
```

### 3. Run the Application

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```

### Upload Documents
```bash
POST /documents/upload
Content-Type: application/json

{
  "documents": [
    "Document 1 text",
    "Document 2 text"
  ],
  "ids": ["doc1", "doc2"],
  "metadata": [
    {"category": "support"},
    {"category": "incident"}
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

### Ask a Question (with RAG)
```bash
POST /ask
Content-Type: application/json

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
    "summary": "Based on our documentation, you can reset your password by...",
    "intent": "support",
    "priority": "medium"
  },
  "confidence_score": 0.85,
  "suggested_actions": [
    "Provide step-by-step resolution guide",
    "Ask for environment and reproducible steps"
  ],
  "rag_enabled": true,
  "retrieved_documents_count": 2
}
```

## How RAG Works

1. **Document Upload**: Documents are converted to embeddings and stored in Chroma
2. **Query Retrieval**: User questions are converted to embeddings
3. **Semantic Search**: The system finds the most relevant documents based on embeddings
4. **Context Injection**: Retrieved documents are added to the LLM prompt
5. **Enhanced Response**: The LLM generates answers based on both the query and context

## Metrics Tracking

The system tracks detailed metrics in `metrics.csv` including:
- Query latency
- Token usage
- Estimated costs
- Intent and priority classification
- **RAG usage** - whether RAG was enabled
- **Retrieved document count** - how many documents were retrieved

## Testing

Run the test suite:

```bash
pytest test_main.py -v
```

Tests include:
- Document upload tests
- RAG-enabled queries
- RAG-disabled queries  
- Validation of response structure

## Architecture

```
User Query
    ↓
[RAG Enabled?] → Yes → Vector DB Search → Get Relevant Docs
    ↓                          ↓
   No                    Inject into Prompt
    ↓                          ↓
    └─────────────→ LLM Prompt ←─────────────┘
                         ↓
                   LLM Response
                         ↓
                   Parse JSON
                         ↓
                 Structured Answer
                         ↓
                  Save Metrics + Return
```

## Configuration

### Vector Database
- **Type**: Chroma (lightweight vector DB)
- **Storage**: Local disk-based (configurable via `CHROMA_DB_PATH`)
- **Embeddings**: Uses default Chroma embeddings (sentence-transformers)

### LLM Integration
- **API**: Hugging Face Inference API
- **Models**: Any model available via HF router
- **Default**: openai/gpt-oss-20b:groq

## Cost Optimization

- **With RAG**: More relevant documents → better answers → less token waste
- **Optional RAG**: Disable for simple queries to save embedding costs
- **Metrics**: Track cost per query including RAG overhead

## Future Enhancements

- [ ] Multi-modal embeddings (images, tables)
- [ ] Hybrid search (BM25 + semantic)
- [ ] Document summarization before storage
- [ ] Query expansion for better retrieval
- [ ] Feedback loop for ranking optimization
- [ ] Automatic document chunking strategies
- [ ] Vector store persistence and versioning

## Troubleshooting

### Vector DB Connection Issues
```bash
# Reset vector store
rm -rf chroma_db

# Restart application
uvicorn main:app --reload
```

### Memory Issues with Large Document Sets
- Consider implementing document chunking
- Use sparse embeddings for efficiency
- Implement batch uploads

### Poor RAG Results
- Check document quality and relevance
- Verify embeddings are appropriate for domain
- Consider fine-tuning embeddings for specific use case

## License

MIT
