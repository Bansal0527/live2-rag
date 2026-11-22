import json
import sqlite3
from pathlib import Path
from app.schemas import DocumentDetail, DocumentExtraction, DocumentRecord


class DocumentStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, file_type TEXT NOT NULL,
                uploaded_at TEXT NOT NULL, status TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0, error TEXT
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS extractions (
                document_id TEXT PRIMARY KEY REFERENCES documents(id),
                payload TEXT NOT NULL
            )""")

    def create(self, record: DocumentRecord):
        with self._connect() as db:
            db.execute(
                "INSERT INTO documents (id, name, file_type, uploaded_at, status, chunk_count, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.id, record.name, record.file_type, record.uploaded_at.isoformat(), record.status, record.chunk_count, record.error),
            )

    def mark_ready(self, document_id: str, chunk_count: int, extraction: DocumentExtraction):
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO extractions (document_id, payload) VALUES (?, ?)", (document_id, extraction.model_dump_json()))
            db.execute("UPDATE documents SET status = 'ready', chunk_count = ?, error = NULL WHERE id = ?", (chunk_count, document_id))

    def mark_failed(self, document_id: str, error: str):
        with self._connect() as db:
            db.execute("UPDATE documents SET status = 'failed', error = ? WHERE id = ?", (error, document_id))

    def list(self) -> list[DocumentRecord]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [DocumentRecord.model_validate(dict(row)) for row in rows]

    def get(self, document_id: str) -> DocumentDetail | None:
        with self._connect() as db:
            row = db.execute("""SELECT d.*, e.payload FROM documents d
                LEFT JOIN extractions e ON e.document_id = d.id WHERE d.id = ?""", (document_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["extraction"] = json.loads(data.pop("payload")) if data["payload"] else None
        return DocumentDetail.model_validate(data)

    def ready_ids(self) -> set[str]:
        with self._connect() as db:
            rows = db.execute("SELECT id FROM documents WHERE status = 'ready'").fetchall()
        return {row["id"] for row in rows}
