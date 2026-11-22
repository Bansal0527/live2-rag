from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.llm import ProviderFailure
from app.main import create_app
from app.schemas import AnswerDraft, DocumentExtraction, ExtractedFact, Source
from app.service import DocumentService
from app.storage import DocumentStore


class FakeIndex:
    def __init__(self):
        self.docs = {}
        self.fail_add = False

    def add(self, document_id, document_name, chunks):
        if self.fail_add:
            raise RuntimeError("embedding API unavailable")
        self.docs[document_id] = (document_name, chunks)

    def delete(self, document_id):
        self.docs.pop(document_id, None)

    def search(self, query, ready_ids, document_ids, min_similarity, limit=5):
        if "unknown" in query.lower():
            return []
        results = []
        for document_id, (name, chunks) in self.docs.items():
            if document_id not in ready_ids or (document_ids is not None and document_id not in document_ids):
                continue
            for chunk in chunks:
                results.append(Source(document_id=document_id, document_name=name,
                                      page=chunk.metadata["page"] or None,
                                      chunk_id=chunk.metadata["chunk_id"],
                                      text=chunk.page_content, similarity=0.88))
        return results[:limit]


class FakeAI:
    def __init__(self):
        self.fail_extract = False
        self.fail_answer = False
        self.bad_citation = False

    def extract(self, name, pages):
        if self.fail_extract:
            raise ProviderFailure("Gemini extraction request failed")
        return DocumentExtraction(document_type="Policy", facts=[
            ExtractedFact(name="notice_period", value="30 days", evidence="30 days", page=pages[0].number)
        ])

    def answer(self, query, search):
        if self.fail_answer:
            raise ProviderFailure("Simulated Gemini answer outage")
        sources = search(query)
        if self.bad_citation:
            return AnswerDraft(answer="30 days", citation_ids=["fabricated:1"]), sources
        return AnswerDraft(answer="30 days" if sources else None,
                           citation_ids=[s.chunk_id for s in sources]), sources


@pytest.fixture
def setup(tmp_path):
    settings = Settings(tmp_path, "fake", 1024 * 1024, 10000, 0.3)
    index = FakeIndex()
    ai = FakeAI()
    service = DocumentService(settings, DocumentStore(tmp_path / "documents.sqlite3"), index, ai)
    with TestClient(create_app(service)) as client:
        yield client, service, index, ai


def upload(client, name="agreement.txt", text=b"The notice period is 30 days."):
    return client.post("/documents/upload", files={"file": (name, text)})


def test_two_uploads_and_separate_extraction(setup):
    client, service, _, _ = setup
    first = upload(client)
    second = upload(client, "policy.txt", b"A notice period of 30 days applies.")
    assert first.status_code == second.status_code == 201
    listed = client.get("/documents").json()
    assert len(listed) == 2
    assert all("extraction" not in record for record in listed)
    detail = client.get(f"/documents/{first.json()['id']}").json()
    assert detail["status"] == "ready"
    assert detail["extraction"]["facts"][0]["value"] == "30 days"
    assert detail["chunk_count"] >= 1


def test_grounded_answer_has_valid_source(setup):
    client, _, _, _ = setup
    document_id = upload(client).json()["id"]
    response = client.post("/ask", json={"query": "How long is the notice period?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert data["sources"][0]["document_id"] == document_id
    assert data["sources"][0]["chunk_id"].startswith(document_id)


def test_answer_can_cite_two_documents(setup):
    client, _, _, _ = setup
    a = upload(client).json()["id"]
    b = upload(client, "other.txt", b"The notice period is 30 days.").json()["id"]
    data = client.post("/ask", json={"query": "Compare the notice periods"}).json()
    assert data["status"] == "answered"
    assert {source["document_id"] for source in data["sources"]} == {a, b}


def test_missing_answer_abstains(setup):
    client, _, _, _ = setup
    upload(client)
    response = client.post("/ask", json={"query": "Unknown benefit?"})
    assert response.json() == {"answer": None, "sources": [], "status": "insufficient_evidence"}


def test_search_spans_documents_and_filters(setup):
    client, _, _, _ = setup
    a = upload(client).json()["id"]
    b = upload(client, "other.txt", b"The notice period is 30 days.").json()["id"]
    all_hits = client.post("/search", json={"query": "termination notice"}).json()["results"]
    assert {hit["document_id"] for hit in all_hits} == {a, b}
    filtered = client.post("/search", json={"query": "termination notice", "document_ids": [a]}).json()["results"]
    assert {hit["document_id"] for hit in filtered} == {a}


def test_invalid_uploads_and_missing_document(setup):
    client, _, _, _ = setup
    assert upload(client, "bad.docx").status_code == 422
    assert upload(client, text=b" \n ").status_code == 422
    assert upload(client, "bad.pdf", b"plain text").status_code == 422
    assert client.get("/documents/missing").status_code == 404


def test_provider_failure_keeps_failed_status(setup):
    client, _, _, ai = setup
    ai.fail_extract = True
    assert upload(client).status_code == 502
    records = client.get("/documents").json()
    assert len(records) == 1 and records[0]["status"] == "failed"


def test_answer_provider_failure_returns_502(setup):
    client, _, _, ai = setup
    upload(client)
    ai.fail_answer = True
    assert client.post("/ask", json={"query": "notice period"}).status_code == 502


def test_unrecognized_citation_abstains(setup):
    client, _, _, ai = setup
    upload(client)
    ai.bad_citation = True
    response = client.post("/ask", json={"query": "notice period"})
    assert response.json()["status"] == "insufficient_evidence"


def test_answer_source_text_matches_document(setup):
    client, _, _, _ = setup
    content = b"The notice period is 30 days."
    upload(client, text=content)
    sources = client.post("/ask", json={"query": "notice period"}).json()["sources"]
    assert sources and all(source["text"].encode() in content for source in sources)


def test_query_validation(setup):
    client, _, _, _ = setup
    assert client.post("/search", json={"query": "  "}).status_code == 422
