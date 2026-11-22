import json
from pathlib import Path

from evaluation.run import score_case


def test_predefined_evaluation_has_ten_cases():
    cases = json.loads((Path(__file__).resolve().parents[1] / "evaluation/questions.json").read_text())
    assert len(cases) == 10
    assert len({case["id"] for case in cases}) == 10
    assert all("expected_answer" in case for case in cases)
    assert all(case["expected_answer"] for case in cases if case["expected_status"] == "answered")
    assert all(case["expected_answer"] is None for case in cases if case["expected_status"] == "insufficient_evidence")
    assert any(case["expected_status"] == "insufficient_evidence" for case in cases)
    assert any(len(case["expected_documents"]) == 2 for case in cases)


def test_evaluation_scores_answer_and_citations():
    case = {
        "expected_answer": "60 calendar days of written notice",
        "expected_status": "answered", "expected_terms": ["60", "day"],
        "expected_documents": ["employment_agreement.txt"],
    }
    body = {"status": "answered", "answer": "60 calendar days", "sources": [{
        "document_id": "doc-1", "document_name": "employment_agreement.txt", "chunk_id": "doc-1:1",
    }]}
    assert score_case(case, body)
    assert not score_case(case, {**body, "sources": []})
    assert not score_case(case, {**body, "answer": "15 days"})
    assert not score_case({**case, "expected_answer": "15 calendar days"}, body)
