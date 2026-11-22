import logging
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from app.config import Settings
from app.llm import InvalidModelOutput, ProviderFailure, ProviderTimeout
from app.parsing import InvalidDocument
from app.schemas import AskResponse, DocumentDetail, DocumentRecord, QueryRequest, SearchResponse
from app.service import DocumentService, build_service

logging.basicConfig(level=logging.WARNING)


def create_app(service: DocumentService | None = None) -> FastAPI:
    app = FastAPI(title="Document Intelligence API", version="0.1.0")
    app.state.service = service

    def current_service() -> DocumentService:
        if app.state.service is None:
            app.state.service = build_service(Settings.from_env())
        return app.state.service

    @app.exception_handler(InvalidDocument)
    async def invalid_document(_, exc: InvalidDocument):
        return JSONResponse(status_code=413 if "limit" in str(exc) else 422, content={"detail": str(exc)})

    @app.exception_handler(InvalidModelOutput)
    async def invalid_model_output(_, exc: InvalidModelOutput):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ProviderTimeout)
    async def provider_timeout(_, exc: ProviderTimeout):
        return JSONResponse(status_code=504, content={"detail": str(exc)})

    @app.exception_handler(ProviderFailure)
    async def provider_failure(_, exc: ProviderFailure):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.post("/documents/upload", response_model=DocumentDetail, status_code=201)
    async def upload(file: UploadFile = File(...)):
        settings = current_service().settings
        content = await file.read(settings.max_upload_bytes + 1)
        return current_service().upload(file.filename or "", content)

    @app.get("/documents", response_model=list[DocumentRecord])
    def list_documents():
        return current_service().store.list()

    @app.get("/documents/{document_id}", response_model=DocumentDetail)
    def get_document(document_id: str):
        record = current_service().store.get(document_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return record

    @app.post("/search", response_model=SearchResponse)
    def search(request: QueryRequest):
        return current_service().search(request.query, request.document_ids)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: QueryRequest):
        return current_service().ask(request.query, request.document_ids)

    return app


app = create_app()
