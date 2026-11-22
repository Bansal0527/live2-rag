from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ExtractedFact(BaseModel):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Verbatim supporting text")
    page: int | None = Field(default=None, ge=1)


class DocumentExtraction(BaseModel):
    document_type: str | None = None
    summary: str | None = None
    facts: list[ExtractedFact] = Field(default_factory=list)
    key_clauses: list[ExtractedFact] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    id: str
    name: str
    file_type: Literal["pdf", "txt"]
    uploaded_at: datetime
    status: Literal["processing", "ready", "failed"]
    chunk_count: int = 0
    error: str | None = None


class DocumentDetail(DocumentRecord):
    extraction: DocumentExtraction | None = None


class Source(BaseModel):
    document_id: str
    document_name: str
    page: int | None
    chunk_id: str
    text: str
    similarity: float | None = None
    relevance_score: float | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = None

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query must not be blank")
        return value


class SearchResponse(BaseModel):
    results: list[Source]


class AnswerDraft(BaseModel):
    answer: str | None = None
    citation_ids: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str | None
    sources: list[Source]
    status: Literal["answered", "insufficient_evidence"]
