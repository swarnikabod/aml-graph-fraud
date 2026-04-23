"""Tests for AML Fraud Detection API."""
import pytest
from fastapi.testclient import TestClient
from api.main import app, state
from api.model import load_model

@pytest.fixture(autouse=True)
def load_model_for_tests():
    state["model"] = load_model()
    yield
    state.clear()

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "GraphSAGE"
    assert data["version"] == "1.0.0"

def test_predict_single_transaction():
    response = client.post("/predict", json={
        "transactions": [
            {"src": "70:100428660", "dst": "10:8000EBD30", "amount_paid": 5000}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert len(data["scores"]) > 0
    score = data["scores"][0]
    assert 0 <= score["fraud_score"] <= 1
    assert "threshold_exceeded" in score

def test_predict_batch():
    response = client.post("/predict", json={
        "transactions": [
            {"src": "70:100428660", "dst": "10:8000EBD30", "amount_paid": 5000},
            {"src": "70:100428660", "dst": "12:8000F503",  "amount_paid": 3000},
            {"src": "70:100428660", "dst": "15:8000A100",  "amount_paid": 2000},
            {"src": "1:8000F5AD0",  "dst": "10:8000EBD30", "amount_paid": 100},
            {"src": "2:8000F5AD0",  "dst": "11:8000EBD30", "amount_paid": 200},
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["scores"]) >= 5
    flagged = [s for s in data["scores"] if s["account_id"] == "70:100428660"]
    assert len(flagged) == 1
    assert flagged[0]["fraud_score"] > 0.9
    assert flagged[0]["threshold_exceeded"] == True

def test_predict_empty_transactions():
    response = client.post("/predict", json={"transactions": []})
    assert response.status_code == 400

def test_explain_returns_valid_subgraph():
    response = client.post("/explain", json={"account_id": "70:100428660"})
    assert response.status_code == 501
