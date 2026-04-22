# AML Fraud Detection with GNNs

This project explores **anti-money laundering (AML)** use cases by modeling financial and entity relationships as a **graph** and applying **graph neural networks (GNNs)** for fraud and suspicious-pattern detection. Neo4j stores and queries the graph (with the **Graph Data Science (GDS)** library for analytics), while PyTorch Geometric supports training GNN models on graph-structured data.

# STOP Neo4j (do this when done for the day - keeps data!)
docker compose stop

## Getting started

1. Start Neo4j (with GDS): `docker compose up -d`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Verify the database connection: `python src/test_connection.py`

Default Bolt URL: `bolt://localhost:7687`, user `neo4j`, password as set in `docker-compose.yml`.
# AML Graph Fraud Detection

End-to-end Anti-Money Laundering detection system using Graph Neural Networks.

## Stack
- Neo4j — transaction graph database
- GraphSAGE — inductive GNN classifier (99.99% recall)
- SMOTE — synthetic oversampling for class imbalance
- MLflow — experiment tracking
- GNNExplainer — subgraph explainability (coming)
- FastAPI — model serving (coming)
- LLM — SAR narrative generation (coming)

## Dataset
IBM Synthetic AML Dataset (HI-Small) — 5M transactions

## Results
| Model | Recall | Precision | False Negatives |
|-------|--------|-----------|-----------------|
| GCN (baseline) | 98.7% | 99% | 433 |
| GraphSAGE | 99.99% | 100% | 4 |