from langchain_core.documents import Document
from app.vector import ChromaIndex


class FixedEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_chroma_persists_chunks_and_source_metadata(tmp_path):
    path = tmp_path / "chroma"
    index = ChromaIndex(path, FixedEmbeddings())
    index.add("doc-1", "policy.pdf", [Document(page_content="Annual leave is 20 days.", metadata={
        "document_id": "doc-1", "document_name": "policy.pdf", "page": 2, "chunk_id": "doc-1:1",
    })])
    reopened = ChromaIndex(path, FixedEmbeddings())
    hits = reopened.search("annual leave", {"doc-1"}, None, 0.3)
    assert len(hits) == 1
    assert hits[0].page == 2
    assert hits[0].document_name == "policy.pdf"
    assert hits[0].chunk_id == "doc-1:1"
    stored_chunks = reopened.chunks_for_document("doc-1")
    assert len(stored_chunks) == 1
    assert stored_chunks[0].metadata["chunk_id"] == "doc-1:1"
    reopened.delete("doc-1")
    assert reopened.search("annual leave", {"doc-1"}, None, 0.3) == []
