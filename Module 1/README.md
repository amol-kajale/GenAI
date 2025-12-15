# GenAI - FastAPI Hugging Face LLM Backend

This FastAPI app accepts a user question, calls the Hugging Face Inference API, and returns structured JSON with latency, token estimates, and cost.

## Quick Start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `HUGGINGFACE_API_KEY` (get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

3. Run the server:

```powershell
uvicorn main:app --reload --port 8000
```

4. Example request:

```powershell
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question":"What is machine learning?"}'
```

## Response Keys

- `question`: original question
- `answer`: generated text from Hugging Face model
- `model`: model ID used
- `latency_ms`: round-trip latency to Hugging Face (ms)
- `tokens`: object with `prompt_tokens`, `completion_tokens`, `total_tokens` (estimated via word count)
- `estimated_cost_usd`: cost estimate (default 0 for free tier)
- `raw_response`: raw Hugging Face API response

## Configuration

Environment variables:

- `HUGGINGFACE_API_KEY` (required): Your Hugging Face API token
- `HF_DEFAULT_MODEL`: Model ID (defaults to `meta-llama/Llama-2-7b-chat-hf`)
- `COST_PER_1K_PROMPT`: Cost per 1k prompt tokens (default 0.0)
- `COST_PER_1K_COMPLETION`: Cost per 1k completion tokens (default 0.0)

## Supported Models

Popular free/inference models on Hugging Face:

- `meta-llama/Llama-2-7b-chat-hf` (Llama 2 Chat, 7B)
- `mistralai/Mistral-7B-Instruct-v0.1` (Mistral, 7B)
- `tiiuae/falcon-7b-instruct` (Falcon, 7B)
- `google/flan-t5-base` (FLAN-T5, smaller)

Note: Token counts are **estimated** via word count (~1.3 tokens per word). The Hugging Face Inference API does not always return exact token counts in the response.
