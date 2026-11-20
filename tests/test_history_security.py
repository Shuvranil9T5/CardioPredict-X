from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_history_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_HISTORY", raising=False)
    monkeypatch.delenv("HISTORY_API_TOKEN", raising=False)
    assert client.get("/api/v1/history").status_code == 404
    assert client.delete("/api/v1/history").status_code == 404

def test_history_requires_token(monkeypatch):
    monkeypatch.setenv("ENABLE_HISTORY", "1")
    monkeypatch.setenv("HISTORY_API_TOKEN", "test-secret")
    assert client.get("/api/v1/history").status_code == 401
    assert client.delete("/api/v1/history").status_code == 401
    assert client.get("/api/v1/history", headers={"Authorization": "Bearer wrong"}).status_code == 401
