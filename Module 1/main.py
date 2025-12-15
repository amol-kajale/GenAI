import os
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class AskRequest(BaseModel):
    question: str
    model: Optional[str] = None


app = FastAPI(title="GenAI - Hugging Face LLM Metrics API")

# Configure Hugging Face API key and defaults from environment
HF_API_KEY = os.getenv("HF_API_KEY", "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")

DEFAULT_MODEL = os.getenv("HF_DEFAULT_MODEL", "openai/gpt-oss-20b:groq")

# Cost estimation (per 1k tokens). Set via env vars to reflect your pricing.
# Note: Hugging Face Inference API pricing varies; adjust based on your plan.
COST_PER_1K_PROMPT = float(os.getenv("COST_PER_1K_PROMPT", "0.0"))
COST_PER_1K_COMPLETION = float(os.getenv("COST_PER_1K_COMPLETION", "0.0"))

# Hugging Face Inference API endpoint
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    cost = (prompt_tokens / 1000.0) * COST_PER_1K_PROMPT + (completion_tokens / 1000.0) * COST_PER_1K_COMPLETION
    return float(cost)


def estimate_tokens(text: str) -> int:
    # Very rough approximation: ~1.3 tokens per word
    words = len(text.split())
    return max(1, int(words * 1.3))

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
    print("Question:", req.question, "Model:", model, "Headers:", headers)
    # Router expects OpenAI-like chat completions payload
    payload = {
        "messages": [
            {"role": "user", "content": req.question}
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
    answer = ""
    try:
        # expected shape: { choices: [ { message: { content: "..." } } ], usage: {...} }
        answer = data.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception:
        # fallback: try other common fields
        if isinstance(data, list) and len(data) > 0:
            answer = data[0].get("generated_text", "")
        else:
            answer = str(data)

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
        prompt_tokens = estimate_tokens(req.question)
        completion_tokens = estimate_tokens(answer)
        tokens_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "note": "estimated via word count (rough approximation)",
        }

    estimated_cost = estimate_cost(tokens_info["prompt_tokens"], tokens_info["completion_tokens"])

    result = {
        "question": req.question,
        "answer": answer,
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "tokens": tokens_info,
        "estimated_cost_usd": round(estimated_cost, 6),
        "raw_response": data,
    }

    return result