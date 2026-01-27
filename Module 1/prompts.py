"""
Prompt templates for LLM interactions.
These prompts request structured JSON responses for intent classification and priority assessment.
"""

import json
from typing import Dict, Any, Optional


def get_analysis_prompt(question: str) -> str:
    """
    Generate a prompt that requests the LLM to analyze a question and return
    structured JSON with intent and priority.
    
    Args:
        question: The user's question or request
        
    Returns:
        A prompt string that instructs the LLM to respond with JSON
    """
    return f"""You are an AI assistant that analyzes user questions and classifies them.

Analyze the following question and respond ONLY with valid JSON (no markdown, no extra text):

Question: {question}

Classify this question and respond with ONLY a JSON object (no markdown code blocks) in this exact format:
{{
    "answer": "Your detailed answer here",
    "intent": "one of: incident, support, information",
    "priority": "one of: high, medium, low",
    "confidence": 0.0 to 1.0
}}

Guidelines for classification:
- Intent "incident": Contains keywords like urgent, error, fail, down, outage, immediately
- Intent "support": Contains keywords like how, configure, setup, help, install, deploy, guide, performance
- Intent "information": Contains keywords like idea, recommend, suggest, info, what is, explain
- Priority "high": Incident-type issues or urgent support requests
- Priority "medium": Support requests without urgency markers
- Priority "low": Information requests
- Confidence: 0.5 base, +0.25 if detailed answer, +0.15 if clear intent/priority classification"""


def get_summary_prompt(answer: str) -> str:
    """
    Generate a prompt to extract a summary from a longer answer.
    
    Args:
        answer: The full answer text
        
    Returns:
        A prompt string requesting a summary
    """
    return f"""Extract a concise summary (1-2 sentences, max 160 characters) from this text:

{answer}

Respond with ONLY the summary text, no quotes or markdown."""


def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON response from LLM, handling various formats.
    
    Args:
        response_text: The text response from the LLM
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try removing markdown code blocks
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        # Remove markdown code blocks
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    # Try parsing cleaned text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def validate_analysis_response(response: Dict[str, Any]) -> bool:
    """
    Validate that the analysis response has required fields.
    
    Args:
        response: The parsed JSON response
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = {"answer", "intent", "priority"}
    if not isinstance(response, dict):
        return False
    
    if not required_fields.issubset(response.keys()):
        return False
    
    valid_intents = {"incident", "support", "information"}
    valid_priorities = {"high", "medium", "low"}
    
    if response.get("intent") not in valid_intents:
        return False
    
    if response.get("priority") not in valid_priorities:
        return False
    
    return True
