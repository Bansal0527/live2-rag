# Document Intelligence API

A local FastAPI application for PDF/TXT uploads, Pydantic-validated information extraction, Gemini embeddings, semantic search, and grounded question answering.

## Architecture

```mermaid
flowchart LR
  Upload[POST /documents/upload] --> Validate[Validate and parse PDF/TXT]
  Validate --> Pages[LangChain page-aware text splitter]
  Pages --> Embed[Gemini embedding-001]
  Embed --> Chroma[(LangChain Chroma vector store)]
  Pages --> FTS[(SQLite FTS5 keyword index)]
  Validate --> Extract[Gemini structured extraction]
  Extract --> Pydantic[Pydantic evidence validation]
  Pydantic --> SQLite[(SQLite extractions)]
  Upload --> SQLiteMeta[(SQLite documents)]
  Search[POST /search] --> Retrieve[Hybrid retrieval and rank fusion]
  Ask[POST /ask] --> Agent[LangChain agent with search_documents tool]
  Chroma --> Retrieve
  FTS --> Retrieve
  Agent --> Retrieve
  Retrieve --> Agent
  Agent --> Answer[Grounded answer with citations]
```

SQLite stores operational document metadata, validated business facts, and an FTS5 keyword index of chunks. Chroma stores chunk text, embeddings, and source identifiers. An upload is marked `ready` only after extraction and both indexes succeed. Failed uploads stay visible with a `failed` status; partial index entries are removed when possible. On startup, ready documents from older versions are added to the keyword index from their stored Chroma chunks. The original upload is stored locally under `data/uploads/`.

## Setup

Requires Python 3.10+ and a Gemini API key. Gemini calls can incur API charges and send uploaded document text to Google.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env
```

Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in `.env` or your shell. For example:

```bash
export GOOGLE_API_KEY='your-key'
uvicorn app.main:app --reload
```

Open [Swagger UI](http://127.0.0.1:8000/docs). Data is saved in `./data` by default. `GEMINI_CHAT_MODEL` defaults to `gemini-2.5-flash` and can be changed without changing code.

## API

| Method and path | Purpose |
| --- | --- |
| `POST /documents/upload` | Multipart `file`; process PDF or UTF-8 TXT synchronously |
| `GET /documents` | List document IDs, filenames, timestamps, counts, and status |
| `GET /documents/{id}` | Get operational metadata and extracted information |
| `POST /search` | Hybrid keyword and semantic search with optional document filters |
| `POST /ask` | Answer from retrieved chunks and return verified source references |

Example:

```bash
curl -F 'file=@samples/employment_agreement.txt' http://127.0.0.1:8000/documents/upload
curl -H 'Content-Type: application/json' -d '{"query":"What is the notice period?"}' http://127.0.0.1:8000/ask
curl -H 'Content-Type: application/json' -d '{"query":"notice period","filters":{"document_type":"Employment Agreement","file_type":"txt"}}' http://127.0.0.1:8000/search
```

Files in `samples/` are fictional demo fixtures created for this project.

Both `/search` and `/ask` accept `filters` with optional `name`, `file_type` (`pdf` or `txt`), and `document_type` fields. Fields are combined with AND; `name` matches the full filename and both `name` and `document_type` match case-insensitively. `document_type` is the value from validated information extraction. The existing `document_ids` field can be combined with these filters. SQLite selects matching ready documents, then both Chroma and FTS5 search only their IDs. A filter with no matches returns no search results or an insufficient-evidence answer.

`/ask` returns `{"answer": null, "sources": [], "status": "insufficient_evidence"}` when retrieval finds no relevant chunks or the model cannot provide a valid source citation. Each source includes the document ID/name, page (null for TXT), chunk ID, supporting text, optional vector `similarity`, and fused `relevance_score`. HTTP 422 covers invalid or empty files, 413 covers configured size limits, 502 covers Gemini/index failures or invalid model output, and 504 covers timeouts.

## Design choices

- **Gemini embedding-001:** text-only retrieval fits this assignment. The LangChain adapter uses `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` for the respective inputs. Default 3072-dimensional embeddings avoid extra normalization logic. The model's 2,048-token input limit is well above our 1,200-character chunk target.
- **Chunking:** `RecursiveCharacterTextSplitter` splits LangChain `Document` objects inside each PDF page at roughly 1,200 characters with 180-character overlap. This preserves precise page citations and keeps a nearby clause from being cut in two.
- **RAG flow:** A LangChain agent receives a single `search_documents` tool. It searches the uploaded documents, then returns a Pydantic answer with chunk IDs. The API accepts citations only for chunks actually returned by the tool.
- **Relevance:** Chroma uses cosine distance. The API converts it to `1 - distance` and uses `MIN_SIMILARITY` (default `0.30`) as an initial vector cutoff. SQLite FTS5 ranks keyword matches with BM25. Reciprocal rank fusion combines the top results from both lists; a chunk found by both methods receives a higher rank. `relevance_score` is a normalized fusion score, not a probability. `similarity` is null for keyword-only results. Optional metadata and `document_ids` filtering applies to both indexes. The cutoff and fusion ranking should be tuned with representative questions.
- **Hallucination controls:** extraction requires a verbatim evidence quote on the cited page. Q&A uses retrieved chunks only, can abstain, and accepts citations only if their chunk IDs came from retrieval. This checks source identity and evidence location; it cannot prove every sentence is logically entailed by the quote.
- **Failure behavior:** malformed extraction is retried once. Provider and embedding failures return a clear error; document records remain in `failed` state for inspection. There is no distributed transaction across SQLite and Chroma, so cleanup is best effort.

## Scope and limitations

Uploads are synchronous. Maximum file size is 5 MiB and maximum extracted text is 100,000 characters by default. Scanned/image-only PDFs need OCR and are rejected as empty when no text layer exists. Complex tables and multi-column PDFs may have imperfect reading order. There is no authentication or tenant isolation; run this locally with non-sensitive test data. Reranking and background jobs are planned as later enhancements.

## Tests

```bash
pytest -q
```

Tests use fake Gemini and vector components, so they run without an API key. They cover two-document upload and metadata, grounded citations, missing answers, cross-document search and metadata filtering, invalid files, provider failure, fabricated citations, and query validation. An end-to-end Gemini smoke test requires a key and is not part of the automated suite.

Live Gemini checks require available API quota.
