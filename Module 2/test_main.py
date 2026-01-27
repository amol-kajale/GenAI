import pytest
from main import estimate_tokens, estimate_cost, _metrics_csv_path, vector_store
import os
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
import requests


@pytest.fixture(autouse=True)
def cleanup_vector_store():
    """Clean up vector store before each test"""
    vector_store.delete_all()
    yield
    vector_store.delete_all()


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


def test_upload_documents():
    """Test document upload endpoint"""
    client = TestClient(app)
    payload = {
        "documents": [
            "How to reset your password: Go to settings and click forgot password.",
            "Server down? Check status page and contact support team.",
            "API rate limit exceeded? Upgrade your subscription plan."
        ],
        "ids": ["doc1", "doc2", "doc3"],
        "metadata": [
            {"category": "account"},
            {"category": "incident"},
            {"category": "billing"}
        ]
    }
    response = client.post("/documents/upload", json=payload)
    assert response.status_code == 200
    assert response.json()["count"] == 3


def test_upload_documents_mismatch():
    """Test invalid document upload with mismatched counts"""
    client = TestClient(app)
    payload = {
        "documents": ["doc1", "doc2"],
        "ids": ["id1"]
    }
    response = client.post("/documents/upload", json=payload)
    assert response.status_code == 400


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
        response = client.post("/ask", json={"question": "What is AI?", "use_rag": False})
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence_score" in data
        assert "suggested_actions" in data
        assert "rag_enabled" in data
        assert data["rag_enabled"] is False


def test_ask_with_rag(monkeypatch):
    """Test ask request with RAG enabled"""
    monkeypatch.setenv("HF_API_KEY", "test_key_12345")
    monkeypatch.setenv("HF_DEFAULT_MODEL", "test-model")
    
    # Upload documents first
    client = TestClient(app)
    upload_payload = {
        "documents": [
            "To reset password: Go to settings menu and select 'Reset Password'",
            "Contact support at support@example.com for urgent issues"
        ],
        "ids": ["help1", "help2"]
    }
    client.post("/documents/upload", json=upload_payload)
    
    # Mock LLM response
    mock_response = {
        "choices": [
            {"message": {"content": '{"answer": "Based on our docs, you can reset your password from settings.", "intent": "support", "priority": "medium", "confidence": 0.9}'}}
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30
        }
    }
    
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        response = client.post("/ask", json={"question": "How do I reset my password?", "use_rag": True})
        
        assert response.status_code == 200
        data = response.json()
        assert "rag_enabled" in data
        assert data["rag_enabled"] is True
        assert "retrieved_documents_count" in data
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
