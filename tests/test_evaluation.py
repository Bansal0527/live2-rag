import json
from pathlib import Path

from evaluation.run import score_case


def test_predefined_evaluation_has_ten_cases():
    cases = json.loads((Path(__file__).resolve().parents[1] / "evaluation/questions.json").read_text())
    assert len(cases) == 10
    assert len({case["id"] for case in cases}) == 10
    assert any(case["expected_status"] == "insufficient_evidence" for case in cases)
    assert any(len(case["expected_documents"]) == 2 for case in cases)


def test_evaluation_scores_answer_and_citations():
    case = {
        "expected_status": "answered", "expected_terms": ["60", "day"],
        "expected_documents": ["employment_agreement.txt"],
    }
    body = {"status": "answered", "answer": "60 calendar days", "sources": [{
        "document_id": "doc-1", "document_name": "employment_agreement.txt", "chunk_id": "doc-1:1",
    }]}
    assert score_case(case, body)
    assert not score_case(case, {**body, "sources": []})
    assert not score_case(case, {**body, "answer": "15 days"})
