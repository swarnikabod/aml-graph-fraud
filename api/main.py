from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
from api.schemas import PredictRequest, PredictResponse, AccountScore, ExplainRequest, ExplainResponse
from api.model import load_model
from api.graph import build_graph_features

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()
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
    transactions = [t.dict() for t in payload.transactions]
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
    raise HTTPException(status_code=501, detail="Explain endpoint coming in v1.1")
