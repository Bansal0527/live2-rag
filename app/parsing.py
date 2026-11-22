from dataclasses import dataclass
import re
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class PageText:
    number: int | None
    text: str


class InvalidDocument(ValueError):
    pass


def parse_document(content: bytes, file_type: str) -> list[PageText]:
    if file_type == "txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidDocument("TXT file must be UTF-8 encoded") from exc
        pages = [PageText(None, text)]
    else:
        try:
            with pymupdf.open(stream=content, filetype="pdf") as pdf:
                if pdf.is_encrypted:
                    raise InvalidDocument("Encrypted PDFs are not supported")
                pages = [PageText(i + 1, page.get_text("text", sort=True)) for i, page in enumerate(pdf)]
        except InvalidDocument:
            raise
        except Exception as exc:
            raise InvalidDocument("PDF could not be read") from exc
    pages = [PageText(p.number, re.sub(r"[ \t]+", " ", p.text).strip()) for p in pages]
    if not any(p.text for p in pages):
        raise InvalidDocument("Document contains no extractable text")
    return pages


def chunk_pages(pages: list[PageText], document_id: str, document_name: str,
                size: int = 1200, overlap: int = 180) -> list[Document]:
    """Use LangChain's splitter; one input Document per page preserves citations."""
    page_documents = [Document(page_content=page.text, metadata={
        "document_id": document_id, "document_name": document_name,
        "page": page.number or 0,
    }) for page in pages if page.text]
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(page_documents)
    for number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"{document_id}:{number}"
    return chunks


def extraction_text(pages: list[PageText]) -> str:
    return "\n\n".join(f"[Page {p.number if p.number is not None else 'N/A'}]\n{p.text}" for p in pages if p.text)
