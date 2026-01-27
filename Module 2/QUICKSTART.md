# Quick Start Guide - RAG Solution

## 🎯 What's Been Added

Your GenAI solution has been converted to **RAG (Retrieval-Augmented Generation)** with vector database support.

## 📦 New Files

1. **`vector_db.py`** - Vector database management
2. **`RAG_SETUP.md`** - Detailed RAG documentation
3. **`IMPLEMENTATION.md`** - Implementation summary

## 🔧 Installation

### Step 1: Install Dependencies

```bash
# From your project directory
pip install -r requirements.txt

# For full RAG support (optional - may require build tools)
pip install chromadb
```

### Step 2: Configure Environment

Your `.env` file is ready with:
- `HF_API_KEY` - Your Hugging Face token
- `CHROMA_DB_PATH` - Vector database location
- Other LLM settings

## 🚀 Running the Application

```bash
# Start the server
uvicorn main:app --reload

# The API will be at http://localhost:8000
```

## 📝 How to Use

### 1. Upload Documents (Knowledge Base)

Upload your documents once:

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "How to reset password: Go to settings...",
      "Contact support at support@example.com",
      "API documentation is at docs.example.com"
    ],
    "ids": ["faq1", "faq2", "api_ref"],
    "metadata": [
      {"category": "help"},
      {"category": "support"},
      {"category": "documentation"}
    ]
  }'
```

### 2. Ask Questions (with RAG)

The system will automatically search your documents:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I reset my password?",
    "use_rag": true
  }'
```

Response includes:
- Answer with context from your documents
- Confidence score
- How many documents were retrieved
- Suggested actions

### 3. Ask Questions (without RAG)

For general knowledge questions:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?",
    "use_rag": false
  }'
```

## 🔄 How RAG Works

```
Your Question
    ↓
Search Vector DB → Find Similar Documents
    ↓
Combine with Prompt
    ↓
Send to LLM with Context
    ↓
Get Better Answer
```

**Benefits:**
- More accurate answers
- Uses your own documents
- Reduces hallucinations
- Lower costs

## 📊 Monitor Performance

Your metrics are saved in `metrics.csv` with:
- Response time
- Token usage
- Cost estimation
- **RAG stats** - whether RAG was used
- **Document count** - how many docs were retrieved

## ⚙️ Key Configuration

Edit `.env` to change:

```bash
# Your API token (required)
HF_API_KEY=your_token_here

# Which LLM to use
HF_DEFAULT_MODEL=openai/gpt-oss-20b:groq

# Where to store vector database
CHROMA_DB_PATH=chroma_db
```

## 🧪 Running Tests

```bash
# Run all tests
pytest test_main.py -v

# Test specific feature
pytest test_main.py::test_ask_with_rag -v
```

## 🐛 Troubleshooting

### RAG Not Working?
```bash
# Check if chromadb is installed
python -c "import chromadb; print('✓ chromadb installed')"

# If not, install it
pip install chromadb
```

### Can't Upload Documents?
```bash
# Verify the JSON format is correct
# Ensure document count matches ID count
# Check error message in response
```

### Slow Responses?
- First query will be slower (embedding generation)
- Subsequent queries use cached embeddings
- More documents = slightly slower retrieval

## 📚 Example Python Script

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Upload documents
docs_response = requests.post(f"{BASE_URL}/documents/upload", json={
    "documents": [
        "Your document 1",
        "Your document 2"
    ],
    "ids": ["doc1", "doc2"]
})
print("Upload:", docs_response.json())

# Ask a question
ask_response = requests.post(f"{BASE_URL}/ask", json={
    "question": "Your question here?",
    "use_rag": True
})
result = ask_response.json()
print("Answer:", result["answer"]["summary"])
print("Retrieved docs:", result["retrieved_documents_count"])
```

## 🎓 Next Steps

1. ✅ Start the application
2. ✅ Upload your knowledge base
3. ✅ Test with sample questions
4. ✅ Monitor metrics.csv
5. ✅ Fine-tune based on results

## 📖 Full Documentation

See `RAG_SETUP.md` for:
- Complete API reference
- Architecture details
- Performance tips
- Advanced configuration

---

**Ready to use!** Start with uploading your documents, then ask questions. The system will automatically use RAG by default.
