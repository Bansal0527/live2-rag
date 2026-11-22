"""Run 10 predefined RAG questions against the real API and Gemini services."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.service import build_service


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT.parent / "samples"
DOCUMENTS = ("employment_agreement.txt", "vendor_terms.txt", "company_policy.pdf")


def score_case(case: dict, body: dict) -> bool:
    if case["expected_status"] == "insufficient_evidence":
        return (case["expected_answer"] is None and body.get("status") == "insufficient_evidence"
                and body.get("answer") is None and not body.get("sources"))
    expected = case["expected_answer"].casefold()
    answer = (body.get("answer") or "").casefold()
    sources = body.get("sources") or []
    return (
        body.get("status") == "answered"
        and all(term.casefold() in expected and term.casefold() in answer for term in case["expected_terms"])
        and set(case["expected_documents"]).issubset({source.get("document_name") for source in sources})
        and all(source.get("chunk_id", "").startswith(source.get("document_id", "") + ":") for source in sources)
    )


def run() -> dict:
    questions = json.loads((ROOT / "questions.json").read_text())
    if len(questions) != 10:
        raise ValueError("Evaluation requires exactly 10 questions")
    if any("expected_answer" not in case for case in questions):
        raise ValueError("Every evaluation question needs an expected answer")
    results = []
    with TemporaryDirectory() as directory:
        service = build_service(replace(Settings.from_env(), data_dir=Path(directory)))
        with TestClient(create_app(service)) as client:
            for name in DOCUMENTS:
                with (SAMPLES / name).open("rb") as file:
                    response = client.post("/documents/upload", files={"file": (name, file)})
                if response.status_code != 201:
                    raise RuntimeError(f"Upload failed for {name}: HTTP {response.status_code} {response.text[:200]}")

            for case in questions:
                response = client.post("/ask", json={"query": case["query"]})
                body = response.json()
                passed = response.status_code == 200 and score_case(case, body)
                results.append({
                    "id": case["id"], "passed": passed, "http_status": response.status_code,
                    "status": body.get("status"), "expected_answer": case["expected_answer"],
                    "answer": body.get("answer"),
                    "source_documents": [source.get("document_name") for source in body.get("sources", [])],
                })
                print(f"{'PASS' if passed else 'FAIL'} {case['id']}: {body.get('status', response.status_code)}", flush=True)

    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "results": results,
    }
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = run()
    print(f"{report['passed']}/{report['total']} questions passed")
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
