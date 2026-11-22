from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import logging
from app.config import Settings
from app.llm import GeminiService, InvalidModelOutput, ProviderFailure, ProviderTimeout
from app.parsing import InvalidDocument, chunk_pages, parse_document
from app.schemas import AskResponse, DocumentDetail, DocumentFilters, DocumentRecord, SearchResponse
from app.storage import DocumentStore
from app.vector import ChromaIndex

logger = logging.getLogger(__name__)


def fuse_results(vector_hits, keyword_hits, limit: int = 5):
    """Combine ranked lists with reciprocal rank fusion; no score calibration needed."""
    sources = {}
    scores = {}
    for hits in (vector_hits, keyword_hits):
        for rank, source in enumerate(hits, start=1):
            sources.setdefault(source.chunk_id, source)
            scores[source.chunk_id] = scores.get(source.chunk_id, 0) + 1 / (60 + rank)
    ranked = sorted(sources, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    maximum = 2 / 61
    return [sources[chunk_id].model_copy(update={
        "relevance_score": round(scores[chunk_id] / maximum, 4)
    }) for chunk_id in ranked]


class DocumentService:
    def __init__(self, settings: Settings, store: DocumentStore, index, ai):
        self.settings = settings
        self.store = store
        self.index = index
        self.ai = ai
        self.upload_dir = settings.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, name: str, content: bytes) -> DocumentDetail:
        suffix = Path(name).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            raise InvalidDocument("Only PDF and TXT files are supported")
        if not content:
            raise InvalidDocument("File is empty")
        if len(content) > self.settings.max_upload_bytes:
            raise InvalidDocument("File exceeds upload size limit")
        if suffix == ".pdf" and not content.startswith(b"%PDF-"):
            raise InvalidDocument("File is not a valid PDF")
        pages = parse_document(content, suffix[1:])
        if sum(len(page.text) for page in pages) > self.settings.max_document_chars:
            raise InvalidDocument("Document text exceeds processing limit")

        document_id = str(uuid4())
        safe_name = Path(name).name or f"upload{suffix}"
        record = DocumentRecord(
            id=document_id, name=safe_name, file_type=suffix[1:],
            uploaded_at=datetime.now(timezone.utc), status="processing",
        )
        chunks = chunk_pages(pages, document_id, safe_name)
        self.store.create(record)
        upload_path = self.upload_dir / f"{document_id}{suffix}"
        try:
            upload_path.write_bytes(content)
            extraction = self.ai.extract(safe_name, pages)
            self.index.add(document_id, safe_name, chunks)
            self.store.index_chunks(document_id, chunks)
            self.store.mark_ready(document_id, len(chunks), extraction)
        except Exception as exc:
            logger.error("Document %s processing failed: %s", document_id, type(exc).__name__)
            try:
                self.index.delete(document_id)
            except Exception:
                logger.warning("Could not remove partial vector records for %s", document_id)
            self.store.mark_failed(document_id, type(exc).__name__)
            if isinstance(exc, (ProviderFailure, InvalidModelOutput, InvalidDocument)):
                raise
            raise ProviderFailure("Document processing failed") from exc
        return self.store.get(document_id)

    def search(self, query: str, document_ids: list[str] | None = None,
               filters: DocumentFilters | None = None) -> SearchResponse:
        try:
            eligible_ids = self.store.filtered_ready_ids(document_ids, filters)
            if not eligible_ids:
                return SearchResponse(results=[])
            vector_hits = self.index.search(
                query, eligible_ids, None,
                self.settings.min_similarity, limit=10,
            )
            keyword_hits = self.store.keyword_search(query, list(eligible_ids), limit=10)
        except TimeoutError as exc:
            raise ProviderTimeout("Embedding search timed out") from exc
        except Exception as exc:
            raise ProviderFailure("Document search failed") from exc
        return SearchResponse(results=fuse_results(vector_hits, keyword_hits))

    def ask(self, query: str, document_ids: list[str] | None = None,
            filters: DocumentFilters | None = None) -> AskResponse:
        draft, sources = self.ai.answer(query, lambda q: self.search(q, document_ids, filters).results)
        source_map = {source.chunk_id: source for source in sources}
        citations = list(dict.fromkeys(draft.citation_ids))
        if not draft.answer or not citations or any(identifier not in source_map for identifier in citations):
            return AskResponse(answer=None, sources=[], status="insufficient_evidence")
        return AskResponse(answer=draft.answer, sources=[source_map[i] for i in citations], status="answered")


def build_service(settings: Settings) -> DocumentService:
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before starting the API")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    model = ChatGoogleGenerativeAI(model=settings.chat_model, temperature=0, timeout=45, max_retries=1)
    store = DocumentStore(settings.data_dir / "documents.sqlite3")
    index = ChromaIndex(settings.data_dir / "chroma", embeddings)
    for document_id in store.unindexed_ready_ids():
        store.index_chunks(document_id, index.chunks_for_document(document_id))
    return DocumentService(
        settings, store, index, GeminiService(model),
    )
