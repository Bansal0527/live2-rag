from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from langchain_core.documents import Document
from app.schemas import DocumentDetail, DocumentExtraction, DocumentRecord
from app.schemas import Source

STOP_WORDS = {"a", "an", "and", "are", "about", "do", "does", "for", "from", "how",
              "in", "is", "me", "of", "or", "the", "to", "what", "which"}


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
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id UNINDEXED, document_id UNINDEXED, document_name UNINDEXED,
                page UNINDEXED, text, tokenize='unicode61'
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
            db.execute("DELETE FROM chunk_fts WHERE document_id = ?", (document_id,))
            db.execute("UPDATE documents SET status = 'failed', error = ? WHERE id = ?", (error, document_id))

    def index_chunks(self, document_id: str, chunks: list[Document]):
        with self._connect() as db:
            db.execute("DELETE FROM chunk_fts WHERE document_id = ?", (document_id,))
            db.executemany(
                "INSERT INTO chunk_fts (chunk_id, document_id, document_name, page, text) VALUES (?, ?, ?, ?, ?)",
                [(chunk.metadata["chunk_id"], document_id, chunk.metadata["document_name"],
                  chunk.metadata["page"], chunk.page_content) for chunk in chunks],
            )

    def keyword_search(self, query: str, document_ids: list[str] | None = None, limit: int = 10) -> list[Source]:
        tokens = [word for word in re.findall(r"\w+", query.lower()) if word not in STOP_WORDS][:12]
        if not tokens or document_ids == []:
            return []
        match = " OR ".join(f'"{word}"' for word in tokens)
        sql = """SELECT chunk_fts.chunk_id, chunk_fts.document_id, chunk_fts.document_name,
                 chunk_fts.page, chunk_fts.text FROM chunk_fts
                 JOIN documents ON documents.id = chunk_fts.document_id
                 WHERE chunk_fts MATCH ? AND documents.status = 'ready'"""
        params: list = [match]
        if document_ids is not None:
            sql += f" AND chunk_fts.document_id IN ({','.join('?' for _ in document_ids)})"
            params.extend(document_ids)
        sql += " ORDER BY bm25(chunk_fts) LIMIT ?"
        params.append(limit)
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [Source(
            chunk_id=row["chunk_id"], document_id=row["document_id"],
            document_name=row["document_name"], page=int(row["page"]) or None,
            text=row["text"], similarity=None,
        ) for row in rows]

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

    def unindexed_ready_ids(self) -> list[str]:
        with self._connect() as db:
            rows = db.execute("""SELECT id FROM documents WHERE status = 'ready'
                AND NOT EXISTS (SELECT 1 FROM chunk_fts WHERE chunk_fts.document_id = documents.id)""").fetchall()
        return [row["id"] for row in rows]
