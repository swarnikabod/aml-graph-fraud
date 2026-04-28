from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
import json
from pathlib import Path
from api.schemas import PredictRequest, PredictResponse, AccountScore, ExplainRequest, ExplainResponse
from api.model import load_model
from api.graph import build_graph_features
from api.llm import generate_sar_narrative

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()
    # Load explainability report
    report_path = Path("notebooks/explainability_report.json")
    if report_path.exists():
        with open(report_path) as f:
            state["report"] = {r["account_id"]: r for r in json.load(f)}
    else:
        state["report"] = {}
    print("API ready!")
    yield
    state.clear()

app = FastAPI(title="AML Fraud Detection API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model": "GraphSAGE", "version": "1.0.0"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if not payload.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")
    transactions = [t.model_dump() for t in payload.transactions]
    data, nodes, features = build_graph_features(transactions)
    model = state["model"]
    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)[:, 1].numpy()
    THRESHOLD = 0.5
    scores = []
    for i, node in enumerate(nodes):
        scores.append(AccountScore(
            account_id=node,
            fraud_score=round(float(probs[i]), 4),
            threshold_exceeded=bool(probs[i] >= THRESHOLD),
            out_degree=int(features["out_degree"].get(node, 0)),
            in_degree=int(features["in_degree"].get(node, 0)),
            pagerank=round(float(features["pagerank"].get(node, 0)), 6)
        ))
    scores.sort(key=lambda x: x.fraud_score, reverse=True)
    flagged = [s for s in scores if s.threshold_exceeded]
    return PredictResponse(scores=scores, total_flagged=len(flagged), model_version="1.0.0")

@app.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest):
    report = state.get("report", {})
    account_id = payload.account_id

    if account_id in report:
        account_data = report[account_id]
    else:
        # Build minimal account data for unknown accounts
        account_data = {
            "account_id": account_id,
            "fraud_score": 0.5,
            "out_degree": 0,
            "in_degree": 0,
            "pagerank": 0.0,
            "clustering": 0.0,
            "top_features": {"out_degree": 0.0, "in_degree": 0.0, "pagerank": 0.0, "clustering": 0.0}
        }

    narrative = generate_sar_narrative(account_data)

    return ExplainResponse(
        account_id=account_id,
        fraud_score=account_data.get("fraud_score", 0.0),
        top_features=account_data.get("top_features", {}),
        is_real_fraud=account_data.get("is_real_fraud"),
        summary=narrative
    )
