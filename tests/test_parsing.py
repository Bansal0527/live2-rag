from pathlib import Path
import pytest
from app.parsing import InvalidDocument, chunk_pages, parse_document


def test_pdf_chunk_keeps_one_based_page_number():
    sample = Path(__file__).resolve().parents[1] / "samples" / "company_policy.pdf"
    pages = parse_document(sample.read_bytes(), "pdf")
    chunks = chunk_pages(pages, "policy", "company_policy.pdf")
    assert chunks and all(chunk.metadata["page"] == 1 for chunk in chunks)
    assert "annual leave" in chunks[0].page_content.lower()


def test_blank_pdf_has_no_extractable_text():
    import pymupdf
    pdf = pymupdf.open()
    pdf.new_page()
    content = pdf.tobytes()
    pdf.close()
    with pytest.raises(InvalidDocument, match="no extractable text"):
        parse_document(content, "pdf")
