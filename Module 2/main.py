import os
import time
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from dotenv import load_dotenv
from prompts import get_analysis_prompt, parse_json_response, validate_analysis_response
from vector_db import VectorStore

# Load .env if present
load_dotenv()


class AskRequest(BaseModel):
    question: str
    model: Optional[str] = None
    use_rag: Optional[bool] = True


class DocumentUploadRequest(BaseModel):
    documents: List[str]
    ids: List[str]
    metadata: Optional[List[Dict]] = None


app = FastAPI(title="GenAI - Hugging Face LLM Metrics API with RAG")

# Configure Hugging Face API key and defaults from environment
HF_API_KEY = os.getenv("HF_API_KEY", "")
if not HF_API_KEY or HF_API_KEY == "hf_REPLACE_WITH_YOUR_API_KEY":
    print("⚠️  WARNING: HF_API_KEY not set or still has placeholder value in .env file")
    print("   Set a real Hugging Face API token in Module 1/.env before running requests.")

# Initialize vector store with built-in TF-IDF similarity (no external dependencies)
vector_store = VectorStore(db_path=os.getenv("CHROMA_DB_PATH", "chroma_db"))
print("✓ Vector Store initialized successfully (using TF-IDF in-memory indexing)")

DEFAULT_MODEL = os.getenv("HF_DEFAULT_MODEL", "openai/gpt-oss-20b:groq")

# Cost estimation (per 1k tokens). Set via env vars to reflect your pricing.
# Note: Hugging Face Inference API pricing varies; adjust based on your plan.
COST_PER_1K_PROMPT = float(os.getenv("COST_PER_1K_PROMPT", "0.0"))
COST_PER_1K_COMPLETION = float(os.getenv("COST_PER_1K_COMPLETION", "0.0"))

# Hugging Face Inference API endpoint (configurable via .env)
HF_API_URL = os.getenv("HF_API_URL", "https://router.huggingface.co/v1/chat/completions")


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    cost = (prompt_tokens / 1000.0) * COST_PER_1K_PROMPT + (completion_tokens / 1000.0) * COST_PER_1K_COMPLETION
    return float(cost)


def estimate_tokens(text: str) -> int:
    # Very rough approximation: ~1.3 tokens per word
    words = len(text.split())
    return max(1, int(words * 1.3))


def _metrics_csv_path() -> str:
    return os.path.join(os.path.dirname(__file__), "metrics.csv")


def save_metrics_csv(entry: Dict[str, Any]) -> None:
    path = _metrics_csv_path()
    fieldnames = [
        "timestamp",
        "question",
        "model",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "intent",
        "priority",
        "confidence_score",
        "rag_used",
        "retrieved_docs_count",
    ]
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "question": entry.get("question", ""),
                "model": entry.get("model", ""),
                "latency_ms": entry.get("latency_ms", ""),
                "prompt_tokens": entry.get("tokens", {}).get("prompt_tokens", ""),
                "completion_tokens": entry.get("tokens", {}).get("completion_tokens", ""),
                "total_tokens": entry.get("tokens", {}).get("total_tokens", ""),
                "estimated_cost_usd": entry.get("estimated_cost_usd", ""),
                "intent": entry.get("intent", ""),
                "priority": entry.get("priority", ""),
                "confidence_score": entry.get("confidence_score", ""),
                "rag_used": entry.get("rag_used", False),
                "retrieved_docs_count": entry.get("retrieved_docs_count", 0),
            }
            writer.writerow(row)
    except Exception as e:
        print("Warning: failed to write metrics CSV:", e)


@app.post("/documents/upload")
def upload_documents(req: DocumentUploadRequest) -> Dict[str, Any]:
    """Upload documents to the vector store"""
    try:
        if len(req.documents) != len(req.ids):
            raise ValueError("Number of documents must match number of IDs")
        
        vector_store.add_documents(req.documents, req.ids, req.metadata)
        
        return {
            "status": "success",
            "message": f"Uploaded {len(req.documents)} documents to vector store",
            "count": len(req.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload documents: {str(e)}")


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY not set in environment")

    model = req.model or DEFAULT_MODEL

    # Determine token for router: prefer HF_TOKEN, fall back to HUGGINGFACE_API_KEY
    token = HF_API_KEY
    if not token:
        raise HTTPException(status_code=500, detail="HF_TOKEN or HUGGINGFACE_API_KEY not set in environment")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # RAG: Retrieve relevant documents if enabled
    retrieved_context = ""
    retrieved_docs_count = 0
    
    if req.use_rag:
        try:
            retrieval_results = vector_store.retrieve(req.question, n_results=3)
            retrieved_context = vector_store.format_retrieved_docs(retrieval_results)
            retrieved_docs_count = len(retrieval_results.get("documents", [[]])[0])
        except Exception as e:
            print(f"Warning: RAG retrieval failed: {e}")
    
    # Generate prompt with RAG context
    analysis_prompt = get_analysis_prompt(req.question, retrieved_context)
    
    # Router expects OpenAI-like chat completions payload
    payload = {
        "messages": [
            {"role": "user", "content": analysis_prompt}
        ],
        "model": model
    }

    start = time.time()
    try:
        resp = requests.post(HF_API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face router request failed: {e}")
    latency_ms = (time.time() - start) * 1000.0

    # Parse response similar to OpenAI router-style response
    answer_text = ""
    try:
        # expected shape: { choices: [ { message: { content: "..." } } ], usage: {...} }
        answer_text = data.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception:
        # fallback: try other common fields
        if isinstance(data, list) and len(data) > 0:
            answer_text = data[0].get("generated_text", "")
        else:
            answer_text = str(data)

    # Parse the JSON response from LLM
    parsed_response = parse_json_response(answer_text)
    
    if not parsed_response or not validate_analysis_response(parsed_response):
        # Fallback to structured response if LLM didn't return valid JSON
        parsed_response = {
            "answer": answer_text,
            "intent": "information",
            "priority": "medium",
            "confidence": 0.5
        }

    # Try to read usage if present
    usage = data.get("usage") if isinstance(data, dict) else None
    tokens_info = {}
    if usage and all(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        tokens_info = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }
    else:
        # fallback estimate
        prompt_tokens = estimate_tokens(analysis_prompt)
        completion_tokens = estimate_tokens(answer_text)
        tokens_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "note": "estimated via word count (rough approximation)",
        }

    estimated_cost = estimate_cost(tokens_info["prompt_tokens"], tokens_info["completion_tokens"])

    structured_answer = {
        "summary": parsed_response.get("answer", ""),
        "intent": parsed_response.get("intent", "information"),
        "priority": parsed_response.get("priority", "medium"),
    }

    confidence = float(parsed_response.get("confidence", 0.5))
    
    # Suggest actions based on intent
    def _suggest_actions(intent_label: str) -> list:
        if intent_label == "incident":
            return [
                "Acknowledge incident and open ticket",
                "Request logs and timestamps from user",
                "Escalate to on-call engineer",
            ]
        if intent_label == "support":
            return [
                "Provide step-by-step resolution guide",
                "Ask for environment and reproducible steps",
                "Offer follow-up troubleshooting session",
            ]
        return [
            "Provide documentation links",
            "Offer examples and further reading",
        ]

    actions = _suggest_actions(structured_answer["intent"])

    response = {
        "answer": structured_answer,
        "confidence_score": float(confidence),
        "suggested_actions": actions,
        "rag_enabled": req.use_rag,
        "retrieved_documents_count": retrieved_docs_count,
    }

    # Persist basic metrics to CSV (best-effort)
    try:
        metrics_entry = {
            "question": req.question,
            "model": model,
            "latency_ms": round(latency_ms, 2),
            "tokens": tokens_info,
            "estimated_cost_usd": round(estimated_cost, 6),
            "intent": structured_answer.get("intent"),
            "priority": structured_answer.get("priority"),
            "confidence_score": confidence,
            "rag_used": req.use_rag,
            "retrieved_docs_count": retrieved_docs_count,
        }
        save_metrics_csv(metrics_entry)
    except Exception:
        # do not fail the request if metrics saving fails
        pass

    return response