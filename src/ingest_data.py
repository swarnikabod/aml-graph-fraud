"""Ingest transaction data from CSV into Neo4j."""

from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "password123"

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "HI-Small_Trans.csv"
N_ROWS = 200_000
BATCH_SIZE = 1_000


def load_csv(path: Path, n_rows: int) -> pd.DataFrame:
    """Load and normalize the first n_rows from the AML transaction CSV."""
    df = pd.read_csv(path, nrows=n_rows)

    # The source CSV has duplicate "Account" headers; rename columns by position.
    df.columns = [
        "timestamp",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    ]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int)

    # Build unique account IDs that include bank code + account number.
    df["from_account_id"] = df["from_bank"].astype(str) + ":" + df["from_account"].astype(str)
    df["to_account_id"] = df["to_bank"].astype(str) + ":" + df["to_account"].astype(str)

    return df


def insert_batch(tx, rows: list[dict]) -> None:
    """Insert one batch of transactions and account nodes."""
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (src:Account {id: row.from_account_id})
          ON CREATE SET src.bank = row.from_bank, src.account = row.from_account
        MERGE (dst:Account {id: row.to_account_id})
          ON CREATE SET dst.bank = row.to_bank, dst.account = row.to_account
        CREATE (src)-[:TRANSACTION {
          timestamp: row.timestamp,
          amount_received: row.amount_received,
          receiving_currency: row.receiving_currency,
          amount_paid: row.amount_paid,
          payment_currency: row.payment_currency,
          payment_format: row.payment_format,
          is_laundering: row.is_laundering
        }]->(dst)
        """,
        rows=rows,
    )


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    print(f"Loading first {N_ROWS:,} rows from {CSV_PATH}...")
    df = load_csv(CSV_PATH, N_ROWS)
    total = len(df)
    print(f"Loaded {total:,} rows. Connecting to Neo4j at {URI}...")

    records = df.to_dict(orient="records")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j. Starting ingestion...")

        with driver.session() as session:
            for start in range(0, total, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total)
                batch = records[start:end]
                session.execute_write(insert_batch, batch)
                print(f"Ingested rows {end:,}/{total:,}")

        print("Ingestion complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
