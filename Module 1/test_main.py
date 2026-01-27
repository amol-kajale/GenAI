import pytest
from main import estimate_tokens, estimate_cost, _metrics_csv_path
import os
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
import requests


def test_estimate_tokens():
    """Test token estimation with various text inputs."""
    # Test with empty string
    assert estimate_tokens("") >= 1
    
    # Test with single word
    result = estimate_tokens("hello")
    assert result >= 1
    
    # Test with multiple words
    result = estimate_tokens("hello world test")
    assert result >= 3


def test_estimate_cost():
    """Test cost estimation calculation."""
    # Test with zero tokens
    assert estimate_cost(0, 0) == 0.0
    
    # Test with tokens
    result = estimate_cost(100, 50)
    assert isinstance(result, float)
    assert result >= 0


def test_metrics_csv_path():
    """Test that metrics CSV path is correctly formed."""
    path = _metrics_csv_path()
    assert path.endswith("metrics.csv")
    assert os.path.dirname(path) == os.path.dirname(__file__)


def test_ask_missing_api_key(monkeypatch):
    """Test that ask raises HTTPException when HF_API_KEY is not set."""
    # Patch the module-level HF_API_KEY variable
    monkeypatch.setattr("main.HF_API_KEY", "")
    
    client = TestClient(app)
    response = client.post("/ask", json={"question": "What is AI?"})
    assert response.status_code == 500


def test_ask_success(monkeypatch):
    """Test successful ask request with mocked API response."""
    monkeypatch.setenv("HF_API_KEY", "test_key_12345")
    monkeypatch.setenv("HF_DEFAULT_MODEL", "test-model")
    
    # Mock LLM response with valid JSON containing intent and priority
    mock_response = {
        "choices": [
            {"message": {"content": '{"answer": "This is a test answer.", "intent": "information", "priority": "medium", "confidence": 0.8}'}}
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        client = TestClient(app)
        response = client.post("/ask", json={"question": "What is AI?"})
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence_score" in data
        assert "suggested_actions" in data
        # Verify LLM returned intent and priority
        assert data["answer"]["intent"] == "information"
        assert data["answer"]["priority"] == "medium"


def test_ask_intent_classification(monkeypatch):
    """Test that ask correctly uses LLM to classify intent."""
    monkeypatch.setenv("HF_API_KEY", "test_key")
    monkeypatch.setenv("HF_DEFAULT_MODEL", "test-model")
    
    # Mock LLM response classifying as incident with high priority
    mock_response = {
        "choices": [
            {"message": {"content": '{"answer": "System is down.", "intent": "incident", "priority": "high", "confidence": 0.95}'}}
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8
        }
    }
    
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        client = TestClient(app)
        response = client.post("/ask", json={"question": "urgent error down"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"]["intent"] == "incident"
        assert data["answer"]["priority"] == "high"


def test_ask_request_exception(monkeypatch):
    """Test that ask handles request exceptions gracefully."""
    monkeypatch.setenv("HF_API_KEY", "test_key")
    monkeypatch.setenv("HF_DEFAULT_MODEL", "test-model")
    
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        client = TestClient(app)
        response = client.post("/ask", json={"question": "Test?"})
        
        assert response.status_code == 500
