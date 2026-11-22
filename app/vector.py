from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.schemas import Source


class ChromaIndex:
    def __init__(self, path: Path, embeddings):
        path.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name="document_chunks", embedding_function=embeddings,
            persist_directory=str(path), collection_metadata={"hnsw:space": "cosine"},
        )

    def add(self, document_id: str, document_name: str, chunks: list[Document]):
        self.store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])

    def delete(self, document_id: str):
        matches = self.store.get(where={"document_id": document_id}, include=[])
        if matches["ids"]:
            self.store.delete(ids=matches["ids"])

    def chunks_for_document(self, document_id: str) -> list[Document]:
        matches = self.store.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        return [Document(page_content=text, metadata=metadata)
                for text, metadata in zip(matches["documents"], matches["metadatas"])]

    def search(self, query: str, ready_ids: set[str], document_ids: list[str] | None,
               min_similarity: float, limit: int = 5) -> list[Source]:
        eligible = ready_ids if document_ids is None else ready_ids.intersection(document_ids)
        if not eligible:
            return []
        where = {"document_id": next(iter(eligible))} if len(eligible) == 1 else {"document_id": {"$in": list(eligible)}}
        matches = self.store.similarity_search_with_score(query, k=max(limit * 3, 10), filter=where)
        found: list[Source] = []
        for document, distance in matches:
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            if similarity < min_similarity:
                continue
            metadata = document.metadata
            found.append(Source(
                document_id=metadata["document_id"], document_name=metadata["document_name"],
                page=metadata["page"] or None, chunk_id=metadata["chunk_id"],
                text=document.page_content, similarity=round(similarity, 4),
            ))
            if len(found) == limit:
                break
        return found
