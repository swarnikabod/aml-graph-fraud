from pydantic import BaseModel
from typing import List, Optional

class Transaction(BaseModel):
    src: str
    dst: str
    amount_paid: float

class PredictRequest(BaseModel):
    transactions: List[Transaction]

class AccountScore(BaseModel):
    account_id: str
    fraud_score: float
    threshold_exceeded: bool
    out_degree: int
    in_degree: int
    pagerank: float

class PredictResponse(BaseModel):
    scores: List[AccountScore]
    total_flagged: int
    model_version: str

class ExplainRequest(BaseModel):
    account_id: str

class ExplainResponse(BaseModel):
    account_id: str
    fraud_score: float
    top_features: dict
    is_real_fraud: Optional[bool]
    summary: str
