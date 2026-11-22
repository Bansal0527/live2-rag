from datetime import datetime, timezone
from langchain_core.documents import Document
from app.schemas import DocumentExtraction, DocumentRecord
from app.storage import DocumentStore


def test_keyword_index_persists_and_hides_failed_documents(tmp_path):
    path = tmp_path / "documents.sqlite3"
    store = DocumentStore(path)
    store.create(DocumentRecord(
        id="doc-1", name="fictional.txt", file_type="txt",
        uploaded_at=datetime.now(timezone.utc), status="processing",
    ))
    store.index_chunks("doc-1", [Document(
        page_content="The invented reference code is quasar42.",
        metadata={"chunk_id": "doc-1:1", "document_name": "fictional.txt", "page": 0},
    )])
    store.mark_ready("doc-1", 1, DocumentExtraction())
    assert store.unindexed_ready_ids() == []

    reopened = DocumentStore(path)
    hits = reopened.keyword_search("quasar42")
    assert len(hits) == 1 and hits[0].chunk_id == "doc-1:1"
    assert reopened.keyword_search("quasar42", ["different-doc"]) == []

    reopened.mark_failed("doc-1", "test failure")
    assert reopened.keyword_search("quasar42") == []


def test_ready_document_without_keyword_index_is_detected(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3")
    store.create(DocumentRecord(
        id="old-doc", name="old.txt", file_type="txt",
        uploaded_at=datetime.now(timezone.utc), status="processing",
    ))
    store.mark_ready("old-doc", 1, DocumentExtraction())
    assert store.unindexed_ready_ids() == ["old-doc"]
