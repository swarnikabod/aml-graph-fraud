"""Verify connectivity to the Neo4j instance (see docker-compose.yml for credentials)."""

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password123"


def main() -> None:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    main()
