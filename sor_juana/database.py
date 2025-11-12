"""DuckDB database operations for corpus storage."""

import json
from typing import List, Dict, Any
import duckdb
from pathlib import Path
from .config import DB_PATH


class CorpusDatabase:
    """Manages DuckDB database for Sor Juana corpus."""

    def __init__(self, db_path: Path = DB_PATH):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        """Create corpus table if it doesn't exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sor_juana_corpus (
                id INTEGER PRIMARY KEY,
                title VARCHAR,
                source VARCHAR,
                genre VARCHAR,
                url VARCHAR,
                text TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    def insert_text(self, text: str, metadata: Dict[str, Any]) -> int:
        """Insert a single text entry into the corpus."""
        result = self.conn.execute(
            """
            INSERT INTO sor_juana_corpus (title, source, genre, url, text, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
        """,
            (
                metadata.get("title", ""),
                metadata.get("source", ""),
                metadata.get("genre", ""),
                metadata.get("url", ""),
                text,
                json.dumps(metadata),
            ),
        ).fetchone()
        return result[0] if result else -1

    def insert_many(self, items: List[Dict[str, Any]]) -> None:
        """Insert multiple text entries."""
        for item in items:
            self.insert_text(item["text"], item["metadata"])

    def get_all_texts(self) -> List[Dict[str, Any]]:
        """Retrieve all texts with metadata."""
        results = self.conn.execute(
            """
            SELECT id, text, metadata FROM sor_juana_corpus
        """
        ).fetchall()

        return [{"id": row[0], "text": row[1], "metadata": json.loads(row[2]) if row[2] else {}} for row in results]

    def delete_by_id(self, text_id: int) -> None:
        """Delete a text entry by ID."""
        self.conn.execute("DELETE FROM sor_juana_corpus WHERE id = ?", (text_id,))

    def delete_many(self, text_ids: List[int]) -> None:
        """Delete multiple text entries by ID."""
        if not text_ids:
            return
        placeholders = ",".join("?" * len(text_ids))
        self.conn.execute(f"DELETE FROM sor_juana_corpus WHERE id IN ({placeholders})", text_ids)

    def count(self) -> int:
        """Get total number of texts in corpus."""
        result = self.conn.execute("SELECT COUNT(*) FROM sor_juana_corpus").fetchone()
        return result[0] if result else 0

    def export_to_jsonl(self, output_path: Path) -> int:
        """Export corpus to JSONL file."""
        texts = self.get_all_texts()
        with open(output_path, "w", encoding="utf-8") as f:
            for item in texts:
                entry = {"text": item["text"], "metadata": item["metadata"]}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return len(texts)

    def export_to_json(self, output_path: Path) -> int:
        """Export corpus to JSON file."""
        texts = self.get_all_texts()
        corpus = [{"text": item["text"], "metadata": item["metadata"]} for item in texts]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        return len(texts)

    def query_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all texts from a specific source."""
        results = self.conn.execute(
            """
            SELECT id, text, metadata FROM sor_juana_corpus
            WHERE source = ?
        """,
            (source,),
        ).fetchall()

        return [{"id": row[0], "text": row[1], "metadata": json.loads(row[2]) if row[2] else {}} for row in results]

    def clear_all(self) -> None:
        """Clear all entries from corpus."""
        self.conn.execute("DELETE FROM sor_juana_corpus")

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
