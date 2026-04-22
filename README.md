# AML Fraud Detection with GNNs

This project explores **anti-money laundering (AML)** use cases by modeling financial and entity relationships as a **graph** and applying **graph neural networks (GNNs)** for fraud and suspicious-pattern detection. Neo4j stores and queries the graph (with the **Graph Data Science (GDS)** library for analytics), while PyTorch Geometric supports training GNN models on graph-structured data.

## Getting started

1. Start Neo4j (with GDS): `docker compose up -d`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Verify the database connection: `python src/test_connection.py`

Default Bolt URL: `bolt://localhost:7687`, user `neo4j`, password as set in `docker-compose.yml`.
